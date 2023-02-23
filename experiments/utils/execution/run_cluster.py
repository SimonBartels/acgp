import os
from os.path import sep
from time import time
from datetime import datetime
import pickle

from run_single_array_configuration import RUN_SCRIPT_FOLDER, ENV_LOCATION
from run_single_array_configuration import __file__ as EXECUTION_HELPER_LOCATION


def execute_single_configuration_on_slurm_cluster(command: str, cpus: int):
    file_name = RUN_SCRIPT_FOLDER + sep + command.replace(' ', '') + ".sq"
    with open(file_name, "w+") as fh:
        _write_basic_run_script(fh, cpus)
        fh.writelines(command)
        fh.flush()
        cluster_command = "sbatch --exclusive --ntasks=1 --mem=30000M --time=0-02:00:00 --cpus-per-task=%i %s" % (cpus, file_name)
        os.system(cluster_command)
        return cluster_command


def execute_job_array_on_slurm_cluster(commands: [str], cpus: int, exclusive=True, mem=50000, device="cpu",
                                       max_jobs_parallel=5, set_core_affinity=True):
    stamp = str(time())
    f = open(RUN_SCRIPT_FOLDER + sep + stamp + ".pkl", "wb+")
    pickle.dump(commands, f)
    f.flush()
    f.close()

    file_name = RUN_SCRIPT_FOLDER + sep + stamp + ".sq"
    command_template = f"sbatch --ntasks=1 --mem={mem}M --time=0-12:00:00"
    if exclusive:
        command_template += " --exclusive"
    if device != "cpu":
        command_template += " -p gpu --gres gpu:titanrtx:1"
        set_core_affinity = False
    with open(file_name, "w+") as fh:
        _write_basic_run_script(fh, cpus, set_core_affinity=set_core_affinity)
        fh.writelines("python %s -t %s -j $SLURM_ARRAY_TASK_ID" % (EXECUTION_HELPER_LOCATION, stamp))
        fh.flush()
        cluster_command = command_template + " --cpus-per-task=%i --array=0-%i%%%i %s" % (cpus, len(commands)-1, max_jobs_parallel, file_name)
        os.system(cluster_command)
        return cluster_command


def execute_single_configuration_on_lsf_cluster(command: str, cpus: int):

    # TODO: Update this file to run on Spectrum LSF
    raise NotImplementedError

    file_name = "./run_scripts/" + command.replace(' ', '') + ".sq"
    with open(file_name, "w+") as fh:
        _write_basic_run_script(fh, cpus)
        fh.writelines(command)
        fh.flush()
        cluster_command = "sbatch --exclusive --ntasks=1 --mem=30000M --time=0-02:00:00 --cpus-per-task=%i %s" % (cpus, file_name)
        os.system(cluster_command)
        return cluster_command


def execute_job_array_on_lsf_cluster(commands: [str], cpus: int, exclusive=True, mem=50000, device="cpu", max_jobs_parallel=5):
    #stamp = str(time())
    stamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    f = open(os.getcwd() + sep + "run_scripts" + sep + stamp + ".pkl", "wb+")

    # The job index is used to index the commands, but since LSF job indices start at
    # 1, we prepend an empty command, which will never be run.
    pickle.dump([""] + commands, f)
    f.flush()
    f.close()

    set_core_affinity = True

    file_name = "./run_scripts/" + stamp + ".sq"

    lsf_dir = "./lsf_output"
    os.makedirs(lsf_dir, exist_ok=True)

    JOBNAME = "ACGP"
    #JOBNAME = str(stamp)

    # Commands to be written to batch script:
    lsf_options = [
            ["-J", f"{JOBNAME}[1-{len(commands)}]%{max_jobs_parallel}"],
            ["-q", "hpc"],
            ["-W", "24:00"],                      # maximum wall clock
            ["-n", f"{cpus}"],                    # number of cores
            ["-R", "'span[hosts=1]'"],              # run on single host
            ["-R", f"'rusage[mem={mem//cpus}MB]'"], # memory *per core*
            ["-R", "'select[model == XeonE5_2680v2]'"],
            ["-M", f"{mem}MB"],                   # memory allowed *per process*
            ["-o", f"{lsf_dir}/%J_%I.out"],
            ["-e", f"{lsf_dir}/%J_%I.err"],
]

    if device != "cpu":
        raise NotImplementedError

    with open(file_name, "w+") as fh:

        fh.write("#!/bin/sh\n")

        for option, value in lsf_options:
            fh.write(f"#BSUB {option} {value}\n")

        _write_basic_run_script(fh, cpus,
                #set_core_affinity=set_core_affinity,
                set_core_affinity=False,
                add_shell=False)

        fh.writelines(f"python run_single_array_configuration.py -t {stamp} "
                      "-j $LSB_JOBINDEX")
        fh.flush()

    cluster_command = f"bsub < {file_name}"
    os.system(cluster_command)
    return cluster_command


def _write_basic_run_script(file_handle, cpus, set_core_affinity=True,add_shell=True):
    if add_shell:
        file_handle.writelines("#!/bin/bash\n")
    file_handle.writelines("CONDA_BASE=$(conda info --base) \n")
    file_handle.writelines("source $CONDA_BASE/etc/profile.d/conda.sh \n")
    file_handle.writelines("conda activate %s \n" % ENV_LOCATION)
    if set_core_affinity:
        file_handle.writelines("taskset --cpu-list %s " % str([i for i in range(0, cpus)])[1:-1].replace(' ', ''))  # NO \n!
