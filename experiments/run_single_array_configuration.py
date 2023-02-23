"""
This is a little helper script to make use of array jobs.
"""
import pickle
import argparse
import os
from os.path import sep
import subprocess
from pathlib import Path

from utils.registry import TEMP_FOLDER


RUN_SCRIPT_FOLDER = TEMP_FOLDER + sep + "run_scripts"
os.makedirs(RUN_SCRIPT_FOLDER, exist_ok=True)

ENV_LOCATION = str(Path(os.getcwd()).parent) + sep + "env"


cmd = "CONDA_BASE=$(conda info --base);"
cmd += "source $CONDA_BASE/etc/profile.d/conda.sh;"
cmd += "conda activate %s" % ENV_LOCATION


def run_local(command):
    proc = subprocess.Popen(["/bin/bash", "-c", cmd + ";" + command])
    proc.wait()


def main(**args):
    f = open(RUN_SCRIPT_FOLDER + sep + args["timestamp"] + ".pkl", "rb")
    run_command_list = pickle.load(f)
    run_command = run_command_list[args["job"]]
    f.close()
    run_local(run_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-j", "--job", type=int)
    parser.add_argument("-t", "--timestamp", type=str)
    args = parser.parse_args()
    main(**vars(args))
