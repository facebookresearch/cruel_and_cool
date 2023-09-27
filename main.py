# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.
import argparse
from single_worker_attack import Attacker
import math
from multiprocessing import Pool
from data import Data
from time import time

def parse_args():
  parser = argparse.ArgumentParser(description='Cool&Cruel bits')
  parser.add_argument('--path', type=str, help='load the data from')
  parser.add_argument('--full_hw', type=int, help='full secret hamming weight', default=20)
  parser.add_argument('--min_bf_hw', type=int, help='minimum brute force hamming weight (going in order)', default=5)
  parser.add_argument('--max_bf_hw', type=int, help='maximum brute force hamming weight (going in order)', default=5)
  parser.add_argument('--seed', type=int, help='seed', default=0)
  parser.add_argument('--bf_max_data', type=int, help='max data', default=30000)
  parser.add_argument('--greedy_max_data', type=int, help='max data', default=None)
  parser.add_argument('--bf_dim', type=int, help='brute force dim', default=20)
  parser.add_argument('--keep_n_tops', type=int, help='keep n tops', default=10)
  parser.add_argument('--check_every_n_batches', type=int, help='check every n batches', default=10000)
  parser.add_argument('--batch_size', type=int, help='batch size', default=10000)
  parser.add_argument('--device', type=str, help='device, if multiple, work is split among them', default="cuda:0")
  parser.add_argument('--work_split_into', type=int, help='work split into how many workers?', default=1)
  parser.add_argument('--which_worker_am_i', type=int, help='which worker am i?', default=0)
  parser.add_argument('--tqdm', type=int, help='use tqdm', default=1)
  parser.add_argument('--exp_name', type=str, help='experiment name', default="test")
  parser.add_argument('--compile_bf', type=int, help='compile bf', default=1)

  args = parser.parse_args()
  return args

def worker(args, device, start, stop):
  data = Data.from_files(
    path = args.path,
    hamming_weight=args.full_hw,
    seed=args.seed,
  )
  if args.bf_max_data is None:
    args.bf_max_data = data.RA.shape[0]
  if args.greedy_max_data is None:
    args.greedy_max_data = data.RA.shape[0]

  attacker = Attacker(
      data.RA,
      data.RB,
      data.secret,
      data.params.Q,
      args.bf_dim,
      args.bf_max_data,
      args.greedy_max_data,
      keep_n_tops=args.keep_n_tops,
      check_every_n_batches=args.check_every_n_batches,
      batch_size=args.batch_size,
      use_tqdm = bool(args.tqdm),
      compile_bf = bool(args.compile_bf),
  )

  found_secret = attacker.brute_force_worker(args.min_bf_hw, args.max_bf_hw, start, stop, device)
  if found_secret:
    #write a file
    with open("found_secret.txt", "w") as f:
      import json
      f.write(json.dumps(vars(args)))

def calculate_work_idxs(args):
  n_work_full = sum(math.comb(args.bf_dim, hw) for hw in range(args.min_bf_hw, args.max_bf_hw + 1))
  n_work_per_worker = math.ceil(n_work_full / args.work_split_into)
  assert args.which_worker_am_i < args.work_split_into
  start = args.which_worker_am_i * n_work_per_worker
  stop = min(start + n_work_per_worker, n_work_full)
  total_in_this_process = stop - start

  # split this work into devices
  devices = args.device.split(",")
  thresholds = [start + total_in_this_process * i // len(devices) for i in range(len(devices) + 1)]
  return devices, thresholds, start, stop


def main(args):

  devices, thresholds, start, stop = calculate_work_idxs(args)
  print(f"It'sa me, worker {args.which_worker_am_i} out of {args.work_split_into} total workers")
  print(f"I have to do work from {start} to {stop}, split into {len(devices)} devices, thats quite a bit eh")
  print(f"Thresholds are {thresholds}")

  start_time = time()
  if len(devices) == 1:
    worker(args, devices[0], thresholds[0], thresholds[1])
  else:
    with Pool(len(devices)) as p:
      p.starmap(worker, [(args, device, thresholds[i], thresholds[i+1]) for i, device in enumerate(devices)])
  end_time = time()
  print(f"Attack took {end_time - start_time} seconds")


if __name__ == '__main__':
  args = parse_args()
  main(args)
