import pathlib
import os
import subprocess
import warnings
import ctypes
from ctypes import cdll, c_int, c_double

if not "blas" in vars():
    ## TODO: this needs to become some routine to search for a blas library
    # blas_path = os.path.join(
    #    os.path.join(os.path.dirname(__file__), "lib"), "libopenblas.so"
    # )  # path to library
    ###num_cpus = mlflow.get_run(mlflow.active_run().info.run_id).data.tags[ENV_CPUS]
    ## num_cpus = ENVIRONMENT_DICT[ENV_CPUS]
    ## blas_path = os.path.join(os.path.dirname(__file__), 'libopenblas_threads_%i.so' % num_cpus)  # path to library

    blas_path = pathlib.Path(__file__).parent / "lib" / "libopenblas.so"

    def whereis(fname):
        if os.name == "nt":
            raise OSError(f"Please copy `{fname}` to {blas_path} and try again.")

        r = subprocess.run(["whereis", fname], capture_output=True, text=True)

        output = r.stdout.strip().split()

        assert len(output) > 1, f"{fname} not found on system."

        blas_path = output[1]

        return blas_path

    if not blas_path.is_file():
        blas_path = whereis("libopenblas.so")
    else:
        pass

    blas = cdll.LoadLibrary(blas_path)  # load library with ctypes

    # Check that the OpenBLAS file is correct:
    try:
        _ = blas.dpotrf_
    except AttributeError:
        blas_path = whereis("liblapack.so")
        blas = cdll.LoadLibrary(blas_path)  # load library with ctypes

    c_double_p = ctypes.POINTER(
        c_double
    )  # convenience definition of a pointer to a double
    c_int_p = ctypes.POINTER(c_int)


def get_blas_object():
    return blas


def check_info(info: int) -> ():
    """
    takes debug information from the BLAS Cholesky decomposition and prints its meaning
    :param info:
        the debug info
    :return:
        nothing
    """
    if info != 0:
        if info < 0:
            warnings.warn(
                "Function call parameters are misspecified. Info value: %i" % info,
                RuntimeWarning,
            )
        else:
            warnings.warn("K stopped being s.p.d. with row %i." % info, RuntimeWarning)
