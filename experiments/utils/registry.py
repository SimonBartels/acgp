import logging
import os
import subprocess
from warnings import warn

from acgp.blas_wrappers.openblas.openblas_wrapper import OpenBlasWrapper
from utils.result_management.constants import ENV_CPUS, ENV_PROC

VERBOSE = False

TEMP_FOLDER = os.getcwd() + os.path.sep + "output" + os.path.sep + "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

KERNEL_DICT = dict()
ALGORITHM_DICT = dict()
ENVIRONMENT_DICT = dict()
try:
    ENVIRONMENT_DICT[ENV_CPUS] = len(os.sched_getaffinity(os.getpid()))  # count the number of CPUs
    ENVIRONMENT_DICT[ENV_PROC] = (subprocess.check_output("lscpu | grep 'Model name'", shell=True).strip()).decode()  #platform.processor()
except Exception as e:
    # TODO: print error message
    warn(f"Populating the environment dictionary failed. Falling back to less reliable methods.")
    import multiprocessing
    ENVIRONMENT_DICT[ENV_CPUS] = multiprocessing.cpu_count()
    ENVIRONMENT_DICT[ENV_PROC] = "unknown cpu name"
"""
Add algorithms to registry.
Depending on which packages are installed or not, we might skip a few.
"""
_algorithm_list = []


def get_fast_meta_cholesky(block_size):
    return MetaCholesky(block_size=block_size, blaswrapper=OpenBlasWrapper())


default_block_size = ENVIRONMENT_DICT[ENV_CPUS] * 256
try:
    from acgp.meta_cholesky import MetaCholesky
    try:
        chol = get_fast_meta_cholesky(block_size=default_block_size)
    except Exception as e:
        logging.exception(e)
        chol = MetaCholesky(block_size=default_block_size)
    _algorithm_list.append(chol)
except ImportError as e:
    pass

# try:
#     from blas_wrappers.torch_wrapper import TorchWrapper
#     chol = MetaCholesky(block_size=default_block_size, blaswrapper=TorchWrapper())
#     _algorithm_list.append(chol)
# except ImportError as e:
#     pass


for a in _algorithm_list:
    ALGORITHM_DICT[a.get_signature()] = a
assert(len(ALGORITHM_DICT.keys()) == len(_algorithm_list))  # make sure we didn't by accident use the same signature twice

from utils.mem_efficient_kernels.isotropic_kernel import RBF, OU
KERNEL_DICT[RBF().name] = RBF()
KERNEL_DICT[OU().name] = OU()

assert(len(KERNEL_DICT.keys()) == 2)
