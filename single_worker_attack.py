# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.
import torch
import tqdm
import numpy as np
import math
import time
import itertools

def hamming_distance(a, b):
    return (a != b).sum()


def brute_force_one_batch(
    secret_combs: torch.Tensor,
    RAs: torch.Tensor,
    RBs: torch.Tensor,
    top_n,
    Q,
    brute_force_dim
):
  keep_n_tops = top_n[0].shape[0]
  batch_size = secret_combs.shape[0]
  with torch.amp.autocast('cuda'):
    secret_cands = torch.zeros(batch_size, brute_force_dim, device=RAs.device, dtype=torch.float16)
    secret_cands.scatter_(1, secret_combs, 1)
    test_stat = ((secret_cands @ RAs.T - RBs) % Q).std(1)
    if len(test_stat) > keep_n_tops:
        biggest_stds = test_stat.topk(keep_n_tops)
    else:
        biggest_stds = test_stat.topk(len(test_stat))
    top_2n = torch.cat([top_n[0], biggest_stds.values])
    top_2n_secrets = torch.cat([top_n[1], secret_cands[biggest_stds.indices]])
    new_topn = top_2n.topk(keep_n_tops)
    top_n = [new_topn.values, new_topn.indices]
    top_n[1] = top_2n_secrets[top_n[1]]

    return top_n


class Annealer:
    def __init__(
        self,
        RA,
        Rb,
        secret_cand,
        total_hw,
        Q,
        brute_force_dim,
        max_steps=None,
        accept_scaling=1.0,
    ):
        # cooling: go down from 1 -> 1e-2 exponentially
        self.cooling = (1e-2) ** (1.0 / max_steps)
        self.device = RA.device
        self.q = Q
        self.RA = RA
        self.RB = Rb
        self.dim = secret_cand.shape[0]
        self.bf_dim = brute_force_dim
        self.bf_hw = int(secret_cand[:self.bf_dim].sum().item())
        self.other_hw = total_hw - self.bf_hw
        self.hamming_weight = total_hw
        self.secret_cand = secret_cand
        # assume i know the target hamming weight
        self.secret_cand[self.bf_dim:self.bf_dim + self.other_hw] = 1
        self.secret_cand[self.bf_dim + self.other_hw:] = 0
        self.temp = 1.0
        self.loss = 1e9
        self.accept_scaling = accept_scaling
        self.step_up_prob = 0
        self.best_secret = self.secret_cand
        self.best_loss = 1e9

    def generate_new_secret_with_hw(self):
        n_changes = 1
        # now we move `n_changes` bits to something new
        # first, we choose which bits to change
        secret_cand_new = self.secret_cand[self.bf_dim:].clone()
        # we choose `n_changes` bits to change
        # random choice from the currently active bits
        make_this_0 = self.secret_cand[self.bf_dim:].multinomial(n_changes)
        # take another one from the inactive bits
        make_this_1 = (1 - self.secret_cand[self.bf_dim:]).multinomial(n_changes)
        secret_cand_new[make_this_0] = 0
        secret_cand_new[make_this_1] = 1
        return torch.cat([self.secret_cand[:self.bf_dim], secret_cand_new])

    def accept_step(self, loss_new):
        # self.temp goes from 1 -> 0.
        # at self.temp = 0, step_up_prob should be 0
        # in between it should be dependent on the loss difference
        loss_diff = self.loss - loss_new
        if loss_diff > 0:
            return True
        else:
            step_up_prob = torch.exp(loss_diff / (self.temp * self.accept_scaling))
            self.step_up_prob = step_up_prob
            return torch.rand(1, device=self.device) < step_up_prob

    def step(self):
        # 1. generate new binary secret (hamming dist depends on temp)
        secret_new = self.generate_new_secret_with_hw()
        # 2. compute loss
        loss_new = -((self.RA @ secret_new - self.RB) % self.q).std()
        # accept with probability
        if self.accept_step(loss_new):
            self.loss = loss_new
            self.secret_cand = secret_new
        self.temp *= self.cooling  # cooling

        if self.loss < self.best_loss:
            self.best_loss = self.loss
            self.best_secret = self.secret_cand



class Attacker:
    def __init__(
        self,
        RAs,
        RBs,
        true_secret,
        Q,
        brute_force_dim,
        n_data_for_brute_force,
        n_data_for_greedy,
        keep_n_tops=100,
        check_every_n_batches=10000,
        batch_size=5000,
        use_tqdm=True,
        compile_bf=True,
    ):

        RAs = RAs / Q * 1000
        RBs = RBs / Q * 1000
        Q = 1000

        self.RAs = torch.tensor(RAs, dtype=torch.float16)
        self.RBs = torch.tensor(RBs, dtype=torch.float16)

        selection_for_bf = torch.randperm(len(RAs))[:n_data_for_brute_force]
        selection_for_G = torch.randperm(len(RAs))[:n_data_for_greedy]

        self.RAs_BF = torch.tensor(
            RAs[selection_for_bf, :brute_force_dim], dtype=torch.float16
        )
        self.RBs_BF = torch.tensor(RBs[selection_for_bf], dtype=torch.float16)

        self.RAs_G = torch.tensor(RAs[selection_for_G], dtype=torch.float16)
        self.RBs_G = torch.tensor(RBs[selection_for_G], dtype=torch.float16)

        self.brute_force_dim = brute_force_dim
        self.true_secret = torch.tensor(true_secret)
        self.Q = Q
        self.secret_dim = len(true_secret)
        self.keep_n_tops = keep_n_tops
        self.check_every_n_batches = check_every_n_batches
        self.batch_size = batch_size
        self.use_tqdm=use_tqdm
        self.compile_bf=compile_bf


    def secret_found(self, cand):
        return (cand.to(device=self.true_secret.device) == self.true_secret).all()

    def check_candidates_via_greedy(self, cands, RAs_G, RBs_G):
        for cand in cands:
            full_cand = self.greedy_secret_completion(cand, RAs_G, RBs_G)
            found = self.secret_found(full_cand)
            if (cand.cpu() == self.true_secret[:self.brute_force_dim]).all() and not found:
              print("brute force worked, greedy failed! increase size of dataset")
              exit()
            if found:
              print("SUCCESS!")
              return True
            # if we passed the secret, abort too
            if (cand.cpu().sum() > self.true_secret[:self.brute_force_dim].sum()):
              print("brute force failed.. increase size of dataset")
              exit()
        return False

    def check_candidates_via_annealing(self, cands, RAs_G, RBs_G):
      for cand in cands:
          full_cand = torch.cat([cand, torch.zeros(self.secret_dim - len(cand), device=cand.device, dtype=torch.float16)])
          annealer = Annealer(
            RAs_G.to(cands.device),
            RBs_G.to(cands.device),
            full_cand,
            self.true_secret.sum().item(),
            self.Q,
            self.brute_force_dim,
            max_steps=100000,
            accept_scaling=10,
          )
          annealer_true_secret_loss = -((annealer.RA @ self.true_secret.to(torch.float16).cuda() - annealer.RB) % annealer.q).std().item()
          for i in (bar:=tqdm.tqdm(range(100000))):
            annealer.step()
            if i % 100 == 0:
                bar.set_description(f"loss {annealer.loss}/{annealer_true_secret_loss}, best loss {annealer.best_loss}, hw diff {hamming_distance(annealer.best_secret.cpu(), self.true_secret)}")

          #check all secrets in queue
          if self.secret_found(annealer.best_secret):
            print("SUCCESS!")
            return True
      return False


    @torch.inference_mode()
    def greedy_secret_completion(self, secret_cand, RAs_G, RBs_G):
        current_idx = secret_cand.shape[0]
        secret_cand = torch.cat(
            [secret_cand, torch.zeros(self.secret_dim - len(secret_cand), device=secret_cand.device, dtype=torch.float16)]
        )
        current_std = ((RAs_G @ secret_cand - RBs_G) % self.Q).std()
        while current_idx < self.secret_dim:
            secret_cand[current_idx] = 1
            new_std = ((RAs_G @ secret_cand - RBs_G) % self.Q).std()
            if new_std > current_std:
                current_std = new_std
            else:
                secret_cand[current_idx] = 0
            current_idx += 1
        return secret_cand

    @staticmethod
    def _generate_in_batches(generator, batch_size):
        while True:
            batch = tuple(itertools.islice(generator, batch_size))
            if not batch:
                break
            yield batch

    @staticmethod
    def generate_from_to_in_batches(n, k, start, end, batch_size):
        combinations = itertools.islice(itertools.combinations(range(n), k), start, end)
        yield from Attacker._generate_in_batches(combinations, batch_size)

    def calculate_idxs_for_each_hw(self, min_HW, max_HW, start_idx, stop_idx):
        """
        calculate which indices between start_idx and stop_idx span which HW
        """
        if stop_idx == -1:
          stop_idx = sum(math.comb(self.brute_force_dim, hw) for hw in range(min_HW, max_HW + 1))
        hw_idxs = {}
        n_total_combs = 0
        for hw in range(min_HW, max_HW + 1):
          n_choose_hw = math.comb(self.brute_force_dim, hw)
          if start_idx < n_total_combs + n_choose_hw:
            if stop_idx <= n_total_combs + n_choose_hw:
              # all in this HW, start from 0
              hw_idxs[hw] = (start_idx - n_total_combs, stop_idx - n_total_combs)
              break
            else:
              # start from 0, end at stop_idx - n_total_combs
              hw_idxs[hw] = (start_idx - n_total_combs, n_choose_hw)
              start_idx = n_total_combs + n_choose_hw
          n_total_combs += n_choose_hw

        return hw_idxs

    def check_if_this_can_work(self, hw_idxs):
      n_cruel_bits = self.true_secret[:self.brute_force_dim].sum().item()
      print(f"checking if this can work, secret has {n_cruel_bits} cruel bits")
      if n_cruel_bits in hw_idxs:
        return True
      return False


    @torch.inference_mode()
    def brute_force_worker(
        self,
        min_HW,
        max_HW,
        start_idx,
        stop_idx,
        device="cpu",
    ):
        hw_idxs = self.calculate_idxs_for_each_hw(min_HW, max_HW, start_idx, stop_idx)
        print(hw_idxs)

        if not self.check_if_this_can_work(hw_idxs):
          print("aborting early, no point in running this, assume it runs for the full time and finds nothing")
          return False

        top_n = [
            torch.zeros(self.keep_n_tops, device=device, dtype=torch.float16),
            torch.zeros((self.keep_n_tops, self.brute_force_dim), device=device, dtype=torch.float16),
        ]

        RAs = self.RAs_BF.to(device)
        RBs = self.RBs_BF.to(device)

        RAs_G = self.RAs_G.to(device)
        RBs_G = self.RBs_G.to(device)

        brute_force_fn = torch.compile(brute_force_one_batch, disable=not self.compile_bf)

        for hamming_weight, (start, stop) in hw_idxs.items():
          print(f"hamming weight: {hamming_weight}, start: {start}, stop: {stop}")
          generator = self.generate_from_to_in_batches(
              self.brute_force_dim, hamming_weight, start, stop, self.batch_size
          )
          length = math.ceil((stop - start) / self.batch_size)

          if self.use_tqdm:
            bar = tqdm.tqdm(generator, mininterval=1, total = length)
          else:
            bar = generator

          batch_counter = 0
          for secret_combs in bar:
            batch_counter += 1
            secret_combs = torch.tensor(secret_combs, device=device, dtype=torch.int64)
            top_n = brute_force_fn(
                secret_combs, RAs, RBs, top_n, self.Q, self.brute_force_dim
            )
            if (batch_counter > 0 and batch_counter % self.check_every_n_batches == 0):
              if self.check_candidates_via_greedy(top_n[1], RAs_G, RBs_G):
                return True

          print(f"finalizing HW {hamming_weight}, last check here, ran through {batch_counter} batches")
          if self.check_candidates_via_greedy(top_n[1], RAs_G, RBs_G):
            return True
        print("done, secret not found")
        return False
