# coding=utf-8
# Copyright (c) 2020, Hicham EL BOUKKOURI.  All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Script for launching a torch.distributed job on multiple gpu/nodes
NOTE: This was adapted from:
https://github.com/aistairc/kirt_bert_on_abci/blob/beta/prepare_args.py
"""
import os
import socket
import sys
import time
import logging

logging.basicConfig(
    format=\
        "%(asctime)s | PID: %(process)d | " \
        "%(filename)s | %(levelname)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)


def main():
    cache_dir, nnodes, nproc_per_node, node_rank, *args = sys.argv[1:]

    nnodes = int(nnodes)
    nproc_per_node = int(nproc_per_node)
    rank_file = os.path.join(cache_dir, f"rank.{node_rank}")
    with open(rank_file, "w") as f:
        f.write(socket.gethostname())

    timeout = 60
    while timeout > 0:
        timeout -= 5
        time.sleep(5)
        if len(os.listdir(cache_dir)) == nnodes:
            break

    # Read master info
    with open(os.path.join(cache_dir, "rank.0"), "r") as f:
        master_address = f.read().strip()
        logging.info('Master address: %s', master_address)
        os.system(
            "python -m "
            "torch.distributed.launch "
            f"--nproc_per_node={nproc_per_node} "
            f"--nnodes={nnodes} "
            f"--node_rank={node_rank} "
            f"--master_addr={master_address} "
            # " ".join(args) is the python command
            "--master_port=9281 " + " ".join(args)
        )


if __name__ == "__main__":
    main()
