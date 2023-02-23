import argparse

from utils.registry import KERNEL_DICT


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--mode", type=str, choices=["generate_batch_jobs", "execute_single_batch_job"],
                        default="execute_single_batch_job")
    parser.add_argument("-d", "--dataset", type=str, default="pumadyn")
    parser.add_argument("-sn2", "--sn2", type=float, default=1e-3)
    parser.add_argument("-k", "--kernel", type=str, choices=KERNEL_DICT.keys(), default=list(KERNEL_DICT.keys())[0])
    for k in list(KERNEL_DICT.keys()):
        KERNEL_DICT[k].add_parameters_to_parser(parser)

    parser.add_argument("-v", "--verbose", type=bool, default=False)

    parser.add_argument("-en", "--experiment-name", type=str, default="debug")

    parser.add_argument("-s", "--seed", type=int, default=0)

    parser.add_argument("-bs", "--block-size", type=int, default=8192)

    parser.add_argument("-ps", "--preconditioner-steps", type=int, default=0)
    return parser
