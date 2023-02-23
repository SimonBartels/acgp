# [Adaptive Cholesky Gaussian Process (ACGP)](https://arxiv.org/abs/2202.10769)
Code files for the project on sublinear GP regression.

## Setup
In the project folder:
```
conda create --prefix ./env
conda activate ./env
conda install -y -c conda-forge -c gpytorch --file requirements.txt
```

### Installation as python package (optional)
Run the following from the main directory (where this README file is also located) to install the package in development mode (that is, modifications to the source code is directly visible to file importing it without the need for reinstallation).
```
pip install -e .
```

### Using OpenBLAS wrappers (optional)

Copy `/usr/lib/libopenblas.so` into `acgp/blas_wrapper/openblas/lib`.

Test by running `python run_hyper_parameter_tuning.py`. If it throws an error like the following:

```
AttributeError: [...]/blas_wrappers/openblas/lib/libopenblas.so: undefined symbol: dpotrf_
```

copy instead `/usr/lib/liblapack.so` to `blas_wrapper/openblas/lib/libopenblas.so` (yes, to `libopenblas.so`!).

## Example Code
The file ``example_script.py`` guides through the main concepts of ACGP.


## Running large scale experiments on a slurm cluster
Switch to the experiments folder.
```
cd experiments
```
### Bound Experiments
```
python run_ground_truth_experiments.py -m generate_batch_jobs
python run_cglb_experiments.py -m generate_batch_jobs
```
To recreate the plots run
```
python make_bound_plotting.py
python make_llh_plotting.py
```

### Hyper-parameter tuning experiments
```
python run_hyper_parameter_tuning.py -m generate_batch_jobs
```
After all runs have finished execute
```
python local_auxilary_computations.py
```
To recreate the result tables run
```
python make_results_table.py
```

For the plots run
```
python make_optimization_plotting.py
```
