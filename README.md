## The Cool and the Cruel
Code for the paper [https://arxiv.org/abs/2403.10328](https://arxiv.org/abs/2403.10328)

To run everything, you need
- `flatter` (only if you want to actually use that, can also reduce with BKZ)
- `fpylll`
- `pytorch` (nightly, unless you disable compilation with --compile_bf 0)
- `numpy`
- `tqdm`

Note that the data run on in the paper was not produced by this code. However, with `data.py`
one can make smaller datasets in a simplified fashion, see `Data.create_new_A` and `Data.create_data_from_A`

`main.py`: runs the attack, check the file for the options
`data.py`: has methods to create new data (quickly with flatter or BKZ for smaller dimensions)
`reduction.py`: implementation of the reductions, depends on `flatter` and `fpylll`.
`single_worker_attack.py`: called by main and has the implementation of the attack.


See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
Cruel and Cool is CC BY-NC licensed, as found in the LICENSE file.
