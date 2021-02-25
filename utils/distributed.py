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
"""Various tools for handling distributed training with PyTorch."""
import torch.distributed


def get_rank():
    r"""
    Returns the rank of the current process when torch.distributed is used and
    returns 0 otherwise.
    """
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def is_main_process():
    r"""
    Returns `True` when the current process is the main process (i.e. 
    `torch.distributed.get_rank()==0`), or when torch.distributed is not used.
    """
    return get_rank() == 0
