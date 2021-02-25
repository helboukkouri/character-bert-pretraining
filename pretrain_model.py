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
r"""
Script for pre-training BERT / CharacterBERT.
NOTE: this is adapted from an older version of:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
"""
import os
import csv
import math
import random
import logging
import argparse
import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler, Dataset
)

import amp_C
import apex_C
from apex import amp
from apex.optimizers import FusedLAMB
from apex.parallel import DistributedDataParallel
from apex.parallel.distributed import flat_dist_call
from apex.amp import _amp_state

from schedulers import LinearWarmUpScheduler, PolyWarmUpScheduler
from transformers import (
    BertConfig, BertTokenizer, BertForPreTraining,
    CharacterBertConfig, CharacterBertTokenizer, CharacterBertForPreTraining
)

from utils.distributed import is_main_process

warnings.filterwarnings("ignore")

WORKDIR = os.environ['WORKDIR']
LOGGING_FORMAT = "%(asctime)s | PID: %(process)d | %(filename)s | %(levelname)s - %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)

IGNORE_INDEX = torch.nn.CrossEntropyLoss().ignore_index


def set_all_random_seeds(random_seed: int, verbose: bool = True):
    r"""Sets the initial random seed to a specific value."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    if verbose:
        logging.info("Setting random seed to: %d", random_seed)


class PretrainingDataset(Dataset):
    r"""
    PyTorch Dataset subclass that allows easy access to the pre-training
    data previously stored in an .hdf5 file.
    Args:
    hdf5_fpath (:obj:`str`):
        Path to an .hdf5 file contraining the pre-training data.
    max_masked_tokens_per_input (:obj:`int`):
        Hard limit on the number of masked tokens per input sequence.
        This is therefore also a limit on the number of MLM predictions
        per input sequence.
    """
    def __init__(self,
        hdf5_fpath: str,
        max_masked_tokens_per_input
    ):
        self.hdf5_fpath = hdf5_fpath
        self.max_masked_tokens_per_input = max_masked_tokens_per_input
        file_in = h5py.File(hdf5_fpath, "r")
        keys = [
            'input_ids',
            'input_mask',
            'segment_ids',
            'masked_lm_positions',
            'masked_lm_ids',
            'next_sentence_labels'
        ]
        self.inputs = [np.asarray(file_in[key][:]) for key in keys]
        file_in.close()

    def __len__(self):
        """Returns the total number of samples in the pre-training dataset."""
        return len(self.inputs[0])

    def __getitem__(self, index):
        """Returns the sample at the provided index."""
        # Get elements at `index` as torch tensors
        [
            input_ids, input_mask, segment_ids,
            masked_lm_positions, masked_lm_ids, next_sentence_labels
        ] = [
            torch.from_numpy(element[index].astype(np.int64)) if i < 5
            else torch.from_numpy(np.asarray(element[index].astype(np.int64)))
            for i, element in enumerate(self.inputs)
        ]

        # MLM labels is IGNORE_INDEX everywhere and `token_id` at masked positions
        index = self.max_masked_tokens_per_input
        masked_lm_labels = IGNORE_INDEX * torch.ones((input_ids.shape[0],), dtype=torch.long)
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [
            input_ids, segment_ids, input_mask,
            masked_lm_labels, next_sentence_labels
        ]


def create_pretraining_dataloader(
        hdf5_fpath: str,
        max_masked_tokens_per_input: int,
        batch_size: int
    ):
    r"""
    Makes a PyTorch DataLoader for producing random batches of pre-training
    tensors using data stored in an .hdf5 file. This also returns the path
    to the .hdf5 file for ... TODO: figure out why?
    Args:
    hdf5_fpath (:obj:`str`):
        Path to an .hdf5 file contraining the pre-training data.
    max_masked_tokens_per_input (:obj:`int`):
        Hard limit on the number of masked tokens per input sequence.
        This is therefore also a limit on the number of MLM predictions
        per input sequence.
    batch_size (:obj:`int`):
        Batch size of tensors returned by the DataLoader.
    """
    pretraining_data = PretrainingDataset(
        hdf5_fpath=hdf5_fpath,
        max_masked_tokens_per_input=max_masked_tokens_per_input
    )
    train_sampler = RandomSampler(pretraining_data)
    train_dataloader = DataLoader(
        pretraining_data,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    return train_dataloader, hdf5_fpath


def parse_args():
    r"""Parses a number of arguments to set as attributes for `ModelPretrainer`."""

    parser = argparse.ArgumentParser()

    ##################################################################
    # Required parameters:
    # ---------------------------------------------------------------
    # - input/output dirs
    # - model config (BertConfig / CharacterBertConfig)
    # - a flag for pre-training CharacterBERT instead of BERT
    ##################################################################

    parser.add_argument(
        "--hdf5_directory",
        type=str, required=True,
        help=\
            "Path to a directory contraining hdf5 files: training.*.hdf5, "
            "validation.*.hdf5 and test.*.hdf5 files."
    )
    parser.add_argument(
        "--output_directory",
        type=str, required=True,
        help=\
            "Path to a directory where model checkpoints and metrics "
            "will be saved."
    )
    parser.add_argument(
        '--is_character_bert',
        action='store_true',
        help="Pre-train CharacterBERT instead of BERT."
    )

    ##################################################################
    # Other parameters
    ##################################################################
    # Parameters related to checkpoint handling

    parser.add_argument(
        '--random_seed',
        required=False, type=int, default=42,
        help=\
            "An intial seed for controlling some of the randomness."
    )
    parser.add_argument(
        "--local_rank",
        type=int, default=-1,
        help=\
            "Identifier of the current process within the distributed process "
            "group. This is always `-1` when distributed training is deactivated."
    )
    parser.add_argument(
        '--phase1_end_step',
        type=int, default=7038,
        help=\
            "Number of training steps (backprops) in pre-training phase n°1: "
            "`max_input_length=128`and `max_masked_tokens_per_input=20`."
        )
    parser.add_argument(
        '--phase2',
        action='store_true',
        help=\
            "Whether it is pre-training phase n°2: "
            "`max_input_length=512`and `max_masked_tokens_per_input=80`."
        )
    parser.add_argument(
        "--init_checkpoint",
        type=str, default=None,
        help="An initial checkpoint to start pre-training from."
    )
    parser.add_argument(
        "--resume_pretraining",
        action='store_true',
        help="Whether to resume pre-training from a checkpoint."
    )
    parser.add_argument(
        '--resume_step',
        type=int, default=-1,
        help=\
            "Step to resume pre-training from. By default, this is `-1` "
            "which results in resuming from the latest checkpoint available."
        )

    ##################################################################
    # Training hyperparameters

    parser.add_argument(
        '--max_input_length',
        required=False, type=int, default=128,
        help=\
            "Maximum sequence length for the model input. "
            "Set this according to the input .hdf5 files contents."
    )
    parser.add_argument(
        "--max_masked_tokens_per_input",
        type=int, default=20,
        help=\
            "Hard limit on the number of tokens that can be masked. "
            "Set this according to the input .hdf5 files contents."
    )
    parser.add_argument(
        '--num_accumulation_steps',
        type=int, default=512,
        help=\
            "Number of steps (forward passes) during which gradients are "
            "accumulated before running a single model parameters update."
    )
    parser.add_argument(
        "--target_batch_size",
        type=int, default=8192,
        help=\
            "Target batch size post-accumulation (actual batch size is "
            "derived from the number of accumulation steps). For example, if "
            "`target_batch_size=32` and `num_accumulation_steps=4` then the "
            "actual batch size will be `32/4 = 8`. This is useful for "
            "achieving larger batch sizes while keeping an actual batch size "
            "that is small enough to fit in memory."
    )
    parser.add_argument(
        "--learning_rate",
        type=float, default=6e-3,
        help="The initial learning rate for the FusedLAMB optimizer."
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float, default=0.2843,
        help=\
            "A value of X means that learning rate will increase during "
            "(100*X)%% of pre-training steps before reaching the desired value "
            "then decrease to 0 during the rest of pre-training steps."
    )
    parser.add_argument(
        "--total_steps",
        type=float, default=7038,
        help="Total number of pre-training steps to perform."
    )

    ##################################################################
    # fp16 related parameters

    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit"
    )
    parser.add_argument(
        '--loss_scale',
        type=float, default=0.0,
        help='Loss scaling, positive power of 2 values can improve fp16 convergence.'
    )
    parser.add_argument(
        '--allreduce_post_accumulation',
        action='store_true',
        help="Whether to do allreduces during gradient accumulation steps."
    )
    parser.add_argument(
        '--allreduce_post_accumulation_fp16',
        action='store_true',
        help="Whether to do fp16 allreduce post accumulation.")

    ##################################################################
    # Logging and checkpointing

    parser.add_argument(
        '--do_validation',
        action='store_true',
        help="Whether to run a validation step before checkpointing."
    )

    parser.add_argument(
        '--checkpoint_interval',
        type=int, default=200,
        help=\
            "Number of model updates before a model checkpoint is saved."
    )
    parser.add_argument(
        '--num_checkpoints_to_keep',
        type=int, default=3,
        help=\
            "Maximum number of checkpoints to keep."
    )
    parser.add_argument(
        '--log_freq',
        type=float, default=1.0,
        help='Frequency of logging loss.'
    )
    parser.add_argument(
        '--tensorboard_id',
        type=str, default='default',
        help="Name of the directory where Tensorboard logs will be saved."
    )

    args = parser.parse_args()
    return args


class ModelPretrainer:
    r"""A helper class for pre-training BERT and CharacterBERT models."""
    def __init__(self, args):

        self.start_datetime = datetime.datetime.now()

        # Set attributes from parsed arguments
        self.hdf5_directory = args.hdf5_directory
        self.output_directory = args.output_directory
        self.tensorboard_id = args.tensorboard_id
        self.is_character_bert = args.is_character_bert
        self.local_rank = args.local_rank
        self.phase1_end_step = args.phase1_end_step
        self.phase2 = args.phase2
        self.init_checkpoint = args.init_checkpoint
        self.resume_pretraining = args.resume_pretraining
        self.resume_step = args.resume_step
        self.max_input_length = args.max_input_length
        self.max_masked_tokens_per_input = args.max_masked_tokens_per_input
        self.target_batch_size = args.target_batch_size
        self.learning_rate = args.learning_rate
        self.total_steps = args.total_steps
        self.warmup_proportion = args.warmup_proportion
        self.num_accumulation_steps = args.num_accumulation_steps
        self.allreduce_post_accumulation = args.allreduce_post_accumulation
        self.fp16 = args.fp16
        self.loss_scale = args.loss_scale
        self.allreduce_post_accumulation_fp16 = args.allreduce_post_accumulation_fp16
        self.log_freq = args.log_freq
        self.do_validation = args.do_validation
        self.checkpoint_interval = args.checkpoint_interval
        self.num_checkpoints_to_keep = args.num_checkpoints_to_keep
        self.random_seed = args.random_seed
        self.is_main_process = (
            self.local_rank in [-1, 0]) and is_main_process()

        if self.is_main_process:
            logging.info('Preparing to run pre-training using parameters:')
            for argname, argvalue in vars(args).items():
                logging.info(' * %s: %s', argname, argvalue)

        # Set the random seed for reproducibility
        set_all_random_seeds(self.random_seed, verbose=self.is_main_process)

        # Make sure CUDA is available (it won't be if you're not using GPUs):
        assert torch.cuda.is_available(), "CUDA is unavailable (are you using GPUs?)"

        # Set CUDA-related attributes
        self.training_is_distributed = (self.local_rank != -1)
        if self.training_is_distributed:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            # Initialize distributed backend (takes care of sychronizing nodes/GPUs)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.n_gpu = 1
        else:
            # TODO: test this
            self.device = torch.device("cuda")
            self.n_gpu = torch.cuda.device_count()
            self.allreduce_post_accumulation = False
            self.allreduce_post_accumulation_fp16 = False

        if self.num_accumulation_steps == 1:
            self.allreduce_post_accumulation = False
            self.allreduce_post_accumulation_fp16 = False

        logging.info(
            "Distributed Training: %s, Number of GPUs: %d, Device: `%s`, Local Rank: `%s` (is_main: `%s`)",
            self.training_is_distributed, self.n_gpu, self.device, self.local_rank, self.is_main_process,
        )

        # Derive actual batch size from target batch size and accumulation steps:
        assert self.num_accumulation_steps >= 1, \
            "`num_accumulation_steps` should be greater or equal to 1"
        assert self.target_batch_size % self.num_accumulation_steps == 0, \
            "`target_batch_size` should be divisible by `num_accumulation_steps`"
        self.batch_size = self.target_batch_size // self.num_accumulation_steps

        # Make sure self.output_directory is empty when starting a training from scratch:
        if not self.resume_pretraining:
            os.makedirs(self.output_directory, exist_ok=True)
            assert not any([
                fname.startswith('ckpt')
                for fname in os.listdir(self.output_directory)]), \
            "Output directory should be empty when not resuming from a previous checkpoint"

        self.global_step = None  # training step counter
        self.checkpoint = None  # checkpoint for resuming training
        self.model = None  # actual model we are pre-training
        self.optimizer = None  # the optimizer (FusedLAMB)
        self.lr_scheduler = None  # the scheduler (PolyWarmUpScheduler)
        self.tensorboard_writer = None  # helper for logging loss to Tensorboard
        self.best_validation_loss = float(1e6)  # best val. loss achieved so far
        self.most_recent_ckpts_paths = []  # list of most recent ckpt paths


    def prepare_model_optimizer_and_scheduler(self):
        r"""Prepares the model, the optimizer and the learning rate scheduler."""

        ###################################################################
        # MODEL PREPARATION
        # -----------------
        # - step 1: Initialize a random model from config
        # - step 2: Load model weights from checkpoint if any
        # - step 3: Move model to device (GPU)
        ###################################################################

        # Initialize a random model according to a specific config:
        # NOTE: here we load from a physical path instead of using a keyword
        # as compute nodes may not allow downloading from online hubs
        if self.is_character_bert:
            model_config = CharacterBertConfig.from_pretrained(
                os.path.join(WORKDIR, 'data', 'character-bert'))
            model = CharacterBertForPreTraining(model_config)
        else:
            model_config = BertConfig.from_pretrained(
                os.path.join(WORKDIR, 'data', 'bert-base-uncased'))
            model = BertForPreTraining(model_config)
        if self.is_main_process:
            logging.info(
                "Initialized %s using Config:\n%s",
                "CharacterBERT" if self.is_character_bert else "BERT",
                model_config
            )

        # Load checkpoint if any:
        if not self.resume_pretraining:
            # CASE: no checkpoint -> training from scratch
            self.global_step = 0
            if self.is_main_process:
                logging.info("Pre-training from scratch (good luck!)")
        else:
            if self.init_checkpoint:
                # CASE: load checkpoint from direct path
                self.global_step = 0
                init_checkpoint = self.init_checkpoint
                if self.is_main_process:
                    logging.info(
                        "Resuming pre-training from specific checkpoint `%s`",
                        init_checkpoint
                    )
            else:
                # CASE: load checkpoint from resume_step
                if self.is_main_process:
                    logging.info(
                        "Resuming pre-training from step `%s`. "
                        "Looking inside `output_directory` for checkpoints...",
                        self.resume_step
                    )

                if self.resume_step == -1:
                    # CASE: resume_step == -1, load latest checkpoint
                    model_names = [
                        fname
                        for fname in os.listdir(self.output_directory)
                        if fname.endswith(".pt")]
                    assert model_names, "Could not find any checkpoints to resume from."
                    self.resume_step = max([
                        int(x.split('.pt')[0].split('_')[1].strip())
                        for x in model_names])  # TODO: find a better way for this
                    if self.is_main_process:
                        logging.info(
                            "Resuming from latest checkpoint: ckpt_%s.pt",
                            self.resume_step
                        )
                else:
                    # CASE: resume_step == X, load checkpoint: `ckpt_X.pt`
                    if self.is_main_process:
                        logging.info(
                            "Resuming from checkpoint: ckpt_%s.pt",
                            self.resume_step
                        )
                self.global_step = self.resume_step
                init_checkpoint = os.path.join(
                    self.output_directory, f"ckpt_{self.resume_step}.pt")

            # Load the actual checkpoint file
            self.checkpoint = torch.load(
                init_checkpoint, map_location="cpu"
            )

            # NOTE: Keeping these lines below as a reminder that re-training on
            # a different domain with CharacterBERT requires changing the
            # output layer with a topK tokens matrix from the new domain.

            # # Case where we would retrain a general_domain CharacterBERT
            # # on the medical domain. Don't use the general domain output layer:
            # if self.is_medical_domain and self.is_character_bert and (not self.phase2):
            #     model.load_state_dict(
            #         {
            #             k: v for (k, v) in self.checkpoint['model'].items()
            #             # Don't load output matrix from general domain model
            #             if not k.startswith('cls.predictions')  # ignoring the old output layer
            #         },
            #         strict=False)
            #     if self.is_main_process:
            #         logging.warning(
            #             "Loaded model weights from `%s`, "
            #             "but ignored the `cls.predictions` module.",
            #             init_checkpoint)

            # # General case: load weights from checkpoint
            # else:
            #     model.load_state_dict(self.checkpoint['model'], strict=True)
            #     if self.is_main_process:
            #         logging.info('Loaded model weights from `%s`',
            #                      init_checkpoint)

            # General case: load weights from checkpoint
            model.load_state_dict(self.checkpoint['model'], strict=True)
            if self.is_main_process:
                logging.info('Loaded model weights from `%s`', init_checkpoint)

            # Deduce previous steps from phase1 when in phase2
            if self.phase2 and not self.init_checkpoint:
                self.global_step -= self.phase1_end_step

            if self.is_main_process:
                logging.info("Training will start at global_step=%s", self.global_step)

        # Move model to GPU:
        model.to(self.device)
        if self.is_main_process:
            logging.info("Model was moved to device: %s", self.device)

        ###################################################################
        # OPTIMIZER / SCHEDULER PREPARATION
        # ---------------------------------
        # - step 1: Define the optimizer (FusedLAMB w/ some weight decay)
        # - step 2: Define the learning rate scheduler (PolyWarmUpScheduler)
        ###################################################################

        # Initialize an optimizer:
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']  # no weight decay
        optimizer_grouped_parameters = [
            {
                'params': [
                    param for name, param in model.named_parameters()
                    if not any((nd in name) for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [
                    param for name, param in model.named_parameters()
                    if any((nd in name) for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = FusedLAMB(
            optimizer_grouped_parameters, lr=self.learning_rate)
        if self.is_main_process:
            logging.info("Using optimizer: %s", optimizer)

        # Initialize a learning rate scheduler:
        self.lr_scheduler = PolyWarmUpScheduler(
            optimizer,
            warmup=self.warmup_proportion,
            total_steps=self.total_steps
        )
        if self.is_main_process:
            logging.info("Using scheduler: %s", self.lr_scheduler)

        ###################################################################
        # OTHER PREPARATION STEPS
        # -----------------------
        # - step 1: Set up Mixed Precision training (fp16) if required
        # - step 2: Load optimizer stat from checkpoint if any
        # - step 2: Set up DataParallel
        ###################################################################

        # Set up fp16:
        if self.fp16:
            if self.is_main_process:
                logging.info("Setting up `Almost FP16` Mixed Precision...")
            if self.loss_scale == 0:
                model, optimizer = amp.initialize(
                    model, optimizer, opt_level="O2", loss_scale="dynamic")
            else:
                model, optimizer = amp.initialize(
                    model, optimizer, opt_level="O2", loss_scale=self.loss_scale)
            amp._amp_state.loss_scalers[0]._loss_scale = 2**20

        # Load optimizer state from checkpoint
        if self.resume_pretraining:
            if self.is_main_process:
                logging.info("Loading optimizer state from checkpoint...")
            if self.phase2 or self.init_checkpoint:
                keys = list(self.checkpoint['optimizer']['state'].keys())
                # Override hyperparameters from previous self.checkpoint
                for key in keys:
                    self.checkpoint['optimizer']['state'][key]['step'] = self.global_step
                for i, _ in enumerate(self.checkpoint['optimizer']['param_groups']):
                    self.checkpoint['optimizer']['param_groups'][i]['step'] = self.global_step
                    self.checkpoint['optimizer']['param_groups'][i]['t_total'] = self.total_steps
                    self.checkpoint['optimizer']['param_groups'][i]['warmup'] = self.warmup_proportion
                    self.checkpoint['optimizer']['param_groups'][i]['lr'] = self.learning_rate
                if self.is_main_process:
                    logging.info("Overwrote the following parameters with new values:")
                    logging.info("* step: %s", self.global_step)
                    logging.info("* t_total: %s", self.total_steps)
                    logging.info("* warmup: %s", self.warmup_proportion)
                    logging.info("* lr: %s", self.learning_rate)
            optimizer.load_state_dict(self.checkpoint['optimizer'])
            # Restore AMP master parameters
            if self.fp16:
                if self.is_main_process:
                    logging.info("Restoring AMP master parameters (optimizer)...")
                optimizer._lazy_init_maybe_master_weights()
                optimizer._amp_stash.lazy_init_called = True
                optimizer.load_state_dict(self.checkpoint['optimizer'])
                for param, saved_param in zip(amp.master_params(optimizer), self.checkpoint['master params']):
                    param.data.copy_(saved_param.data)

        # Distribute model
        if self.training_is_distributed:
            if not self.allreduce_post_accumulation:
                model = DistributedDataParallel(
                    model,
                    message_size=250000000,
                    gradient_predivide_factor=\
                        torch.distributed.get_world_size()
                )
            else:
                flat_dist_call(
                    [param.data for param in model.parameters()],
                    torch.distributed.broadcast,
                    (0,)
                )
        elif self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Set the values of self.model and self.optimizer
        self.model = model
        self.optimizer = optimizer


    def run_validation(self):
        r"""Runs a validation step and returns a boolean saying if the model has improved."""
        # Build a list of validation .hdf5 file paths:
        files = []
        for fname in os.listdir(self.hdf5_directory):
            fpath = os.path.join(self.hdf5_directory, fname)
            if os.path.isfile(fpath) and fname.startswith('validation.') and fname.endswith('.hdf5'):
                files.append(fpath)
        f_start_id = 0
        files.sort()
        num_files = len(files)

        # Select first .hdf5 file
        if \
                torch.distributed.is_initialized() \
                and torch.distributed.get_world_size() > num_files:

            remainder = torch.distributed.get_world_size() % num_files
            hdf5_fpath = files[
                (
                    f_start_id * torch.distributed.get_world_size()
                    + torch.distributed.get_rank()
                    + remainder * f_start_id
                ) % num_files
            ]
        else:
            hdf5_fpath = files[
                (
                    f_start_id * torch.distributed.get_world_size()
                    + torch.distributed.get_rank()
                ) % num_files
            ]

        # Set previous_file variable for next iteration
        previous_file = hdf5_fpath

        # Load the pre-training data from the .hdf5 file
        pretraining_data = PretrainingDataset(
            hdf5_fpath=hdf5_fpath,
            max_masked_tokens_per_input=self.max_masked_tokens_per_input
        )
        validation_sampler = RandomSampler(pretraining_data)  # This could be SequentialSampler
        validation_dataloader = DataLoader(
            pretraining_data,
            sampler=validation_sampler,
            batch_size=self.batch_size * self.n_gpu,
            num_workers=4, pin_memory=True
        )

        steps = 0
        average_loss = 0.0  # averaged loss every self.log_freq steps

        # Use model in `evaluation mode`
        with torch.no_grad():
            self.model.eval()
            if self.is_main_process:
                logging.info("*************************")
                logging.info("**   Evaluation step   **")
                logging.info("*************************")

            # Loop over the rest of pre-training data files
            pool = ProcessPoolExecutor(1)
            if len(files) == 1:
                f_start_id = -1
            for f_id in range(f_start_id + 1, 1 + len(files)//torch.distributed.get_world_size()):

                # Submit creation of next DataLoader
                if torch.distributed.get_world_size() > num_files:
                    hdf5_fpath = files[
                        (
                            f_id * torch.distributed.get_world_size()
                            + torch.distributed.get_rank()
                            + remainder * f_id
                        ) % num_files
                    ]
                else:
                    hdf5_fpath = files[
                        (
                            f_id * torch.distributed.get_world_size()
                            + torch.distributed.get_rank()
                        ) % num_files
                    ]
                if self.is_main_process:
                    logging.info(
                        "Local rank: %s | File n° %s: %s",
                        self.local_rank, f_id, os.path.basename(previous_file)
                    )
                previous_file = hdf5_fpath
                dataset_future = pool.submit(
                    create_pretraining_dataloader,
                    hdf5_fpath,
                    self.max_masked_tokens_per_input,
                    self.batch_size * self.n_gpu,
                )

                # Iterate over batches (w/ progress bar for main process)
                validation_batches = tqdm(
                    validation_dataloader,
                    desc="Computing loss on the validation set..."
                    ) if self.is_main_process else validation_dataloader
                for batch in validation_batches:
                    steps += 1
                    (
                        input_ids,
                        segment_ids,
                        input_mask,
                        masked_lm_labels,
                        next_sentence_labels
                    ) = [tensor.to(self.device) for tensor in batch]

                    # Forward Pass
                    model_output = self.model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=masked_lm_labels,
                        next_sentence_label=next_sentence_labels)
                    loss = model_output['loss']
                    if self.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.

                    divisor = self.num_accumulation_steps
                    if self.num_accumulation_steps > 1:
                        if not self.allreduce_post_accumulation:
                            # this division was merged into predivision
                            loss = loss / self.num_accumulation_steps
                            divisor = 1.0

                    # Update average
                    average_loss += loss.item()
                
                # Move to next file after using up all batches of current file
                del validation_dataloader
                validation_dataloader, hdf5_fpath = \
                    dataset_future.result(timeout=None)

        del validation_dataloader

        num_steps = max(1, int(steps / self.num_accumulation_steps))
        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
        average_loss = average_loss / (num_steps * divisor)
        if torch.distributed.is_initialized():
            average_loss /= torch.distributed.get_world_size()
            torch.distributed.all_reduce(average_loss)

        # Check if model has improved
        validation_loss = average_loss.item()
        model_has_improved = False
        if validation_loss < self.best_validation_loss:
            model_has_improved = True
            self.best_validation_loss = validation_loss

        # Log
        if self.is_main_process:
            logging.info(
                "\nTotal Validation Steps: %s | Validation Loss = %.3f",
                num_steps, validation_loss
            )
            self.tensorboard_writer.add_scalar(
                "Avg. validation loss", validation_loss,
                global_step=self.global_step
            )

        # NOTE: /!\ Put model back in `training mode`
        self.model.train()

        return model_has_improved


    def run_pretraining(self):
        r"""Runs the pre-training process."""
        if self.is_main_process:
            logging.info("*********************************")
            logging.info("***   Starting pre-training   ***")
            logging.info("*********************************")
            logging.info("Training on GPU: %s", torch.cuda.get_device_name(0))
            logging.info("Target batch size: %s", self.target_batch_size)
            logging.info("Number of accumulation steps: %s", self.num_accumulation_steps)
            logging.info("Actual batch size: %s", self.batch_size)

        self.model.train()
        self.most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every self.log_freq steps
        epoch = 0
        training_steps = 0
        pool = ProcessPoolExecutor(1)
        if self.is_main_process:
            tensorboard_log_fpath = os.path.join(
                    WORKDIR,
                    '.tensorboard_logs',
                    self.tensorboard_id,
                    self.start_datetime.strftime("%d-%m-%Y_%H-%M-%S")
            )
            logging.info(
                "Writing TensorBoard logs in: %s",
                tensorboard_log_fpath.replace(WORKDIR, '$WORKDIR'))
            self.tensorboard_writer = SummaryWriter(tensorboard_log_fpath)

        # NOTE: Infinite loop over epochs, termination is handled via iteration count
        while True:

            # If beginning of pre-training: read files from hdf5_directory and shuffle
            if (not self.resume_pretraining) or (epoch > 0) \
                    or (self.phase2 and self.global_step < 1) or self.init_checkpoint:
                files = []
                for fname in os.listdir(self.hdf5_directory):
                    fpath = os.path.join(self.hdf5_directory, fname)
                    if os.path.isfile(fpath) and fname.startswith('training.') and fname.endswith('.hdf5'):
                        files.append(fpath)
                f_start_id = 0
                files.sort()
                random.Random(self.random_seed + epoch).shuffle(files)
            # Else: get id of next file
            else:
                f_start_id = self.checkpoint['files'][0]
                files = self.checkpoint['files'][1:]
                self.resume_pretraining = False
            num_files = len(files)

            # Get the current process hdf5 file
            # and handle case where there are more processes than files left:
            if \
                    torch.distributed.is_initialized() \
                    and torch.distributed.get_world_size() > num_files:

                remainder = torch.distributed.get_world_size() % num_files
                hdf5_fpath = files[
                    (
                        f_start_id * torch.distributed.get_world_size()
                        + torch.distributed.get_rank()
                        + remainder * f_start_id
                    ) % num_files
                ]
            else:
                hdf5_fpath = files[
                    (
                        f_start_id * torch.distributed.get_world_size()
                        + torch.distributed.get_rank()
                    ) % num_files
                ]

            # Set previous_file variable for next iteration
            previous_file = hdf5_fpath

            # Load the pre-training data from the .hdf5 file
            pretraining_data = PretrainingDataset(
                hdf5_fpath=hdf5_fpath,
                max_masked_tokens_per_input=self.max_masked_tokens_per_input
            )
            train_sampler = RandomSampler(pretraining_data)
            train_dataloader = DataLoader(
                pretraining_data,
                sampler=train_sampler,
                batch_size=self.batch_size * self.n_gpu,
                num_workers=4, pin_memory=True
            )
            overflow_buf = None
            if self.allreduce_post_accumulation:
                overflow_buf = torch.cuda.IntTensor([0])

            # Loop over the rest of pre-training data files
            if len(files) == 1:
                f_start_id = -1
            for f_id in range(f_start_id + 1, len(files)):

                # Submit creation of next DataLoader
                if torch.distributed.get_world_size() > num_files:
                    hdf5_fpath = files[
                        (
                            f_id * torch.distributed.get_world_size()
                            + torch.distributed.get_rank()
                            + remainder * f_id
                        ) % num_files
                    ]
                else:
                    hdf5_fpath = files[
                        (
                            f_id * torch.distributed.get_world_size()
                            + torch.distributed.get_rank()
                        ) % num_files
                    ]
                if self.is_main_process:
                    logging.info(
                        "Local rank: %s | File n° %s: %s",
                        self.local_rank, f_id, os.path.basename(previous_file)
                    )
                previous_file = hdf5_fpath
                dataset_future = pool.submit(
                    create_pretraining_dataloader,
                    hdf5_fpath,
                    self.max_masked_tokens_per_input,
                    self.batch_size * self.n_gpu,
                )

                # Iterate over batches (w/ progress bar for main process)
                training_batches = tqdm(
                    train_dataloader,
                    desc="Pre-training..."
                    ) if self.is_main_process else train_dataloader
                for batch in training_batches:
                    training_steps += 1
                    (
                        input_ids,
                        segment_ids,
                        input_mask,
                        masked_lm_labels,
                        next_sentence_labels
                    ) = [tensor.to(self.device) for tensor in batch]

                    # Forward Pass
                    model_output = self.model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=masked_lm_labels,
                        next_sentence_label=next_sentence_labels)
                    loss = model_output['loss']
                    if self.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.

                    divisor = self.num_accumulation_steps
                    if self.num_accumulation_steps > 1:
                        if not self.allreduce_post_accumulation:
                            # this division was merged into predivision
                            loss = loss / self.num_accumulation_steps
                            divisor = 1.0

                    # Compute gradients
                    if self.fp16:
                        with amp.scale_loss(
                                loss, self.optimizer,
                                delay_overflow_check=self.allreduce_post_accumulation) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    average_loss += loss.item()

                    # Take optimizer/scheduler step every (gradient_acc_steps) steps
                    # This is the model parameter update:
                    if training_steps % self.num_accumulation_steps == 0:
                        self.lr_scheduler.step()  # learning rate warmup
                        self.take_optimizer_step(overflow_buf)

                    # If reached max steps save everything and log final loss:
                    if self.global_step >= self.total_steps:
                        last_num_steps = int(
                            training_steps / self.num_accumulation_steps
                        ) % self.log_freq
                        last_num_steps = self.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / (last_num_steps * divisor)
                        if torch.distributed.is_initialized():
                            average_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        if self.is_main_process:
                            logging.info(
                                "Total Steps: %s | Final Loss = %.3f",
                                int(training_steps / self.num_accumulation_steps),
                                average_loss.item()
                            )
                            self.tensorboard_writer.add_scalar(
                                "Avg. training loss",
                                average_loss.item(), global_step=self.global_step)

                    # If at a logging step:
                    elif training_steps % (self.log_freq * self.num_accumulation_steps) == 0:
                        if self.is_main_process:
                            logging_message = (
                                f"Global step: {self.global_step} | "
                                f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2E} | "
                                f"Step Loss: {loss.item() * self.num_accumulation_steps / divisor:.3f} | "
                                f"Avg. Loss: {average_loss / (self.log_freq * divisor):.3f}"
                            )
                            # Update the tqdm description
                            training_batches.set_description(logging_message, refresh=True)
                            # Log average training loss to TensorBoard:
                            self.tensorboard_writer.add_scalar(
                                "Avg. training loss",
                                average_loss / (self.log_freq * divisor),
                                global_step=self.global_step)
                        average_loss = 0

                    # If reached max steps at log step or reached checkpoint step:
                    if \
                        self.global_step >= self.total_steps \
                        or training_steps % \
                            (self.checkpoint_interval * self.num_accumulation_steps) == 0:

                        # Check if model has improved then save a checkpoint if so
                        if self.do_validation:
                            model_has_improved = self.run_validation()
                        else:
                            model_has_improved = True
                        if self.is_main_process and model_has_improved:
                            self.make_checkpoint(f_id, files)

                        # End pre-training if reached max steps
                        if self.global_step >= self.total_steps:
                            del train_dataloader
                            return  # NOTE: breaks out of the training loop

                # Move to next file after using up all batches of current file
                del train_dataloader
                train_dataloader, hdf5_fpath = \
                    dataset_future.result(timeout=None)

            # Update epoch after going through all .hdf5 files
            epoch += 1


    def take_optimizer_step(self, overflow_buf):
        r"""Takes an optimizer step (updates the model weights)."""
        if self.allreduce_post_accumulation:
            # manually allreduce gradients after all accumulation steps
            # check for Inf/NaN
            # 1. allocate an uninitialized buffer for flattened gradient
            scaler = _amp_state.loss_scalers[0]
            master_grads = [
                p.grad
                for p in amp.master_params(self.optimizer)
                if p.grad is not None
            ]
            flat_grad_size = sum(p.numel() for p in master_grads)
            allreduce_dtype = \
                torch.float16 \
                if self.allreduce_post_accumulation_fp16 \
                else torch.float32
            flat_raw = torch.empty(
                flat_grad_size,
                device='cuda', dtype=allreduce_dtype)
            # 2. combine unflattening and predivision of unscaled 'raw' gradient
            allreduced_views = apex_C.unflatten(flat_raw, master_grads)
            overflow_buf.zero_()
            amp_C.multi_tensor_scale(
                65536,
                overflow_buf,
                [master_grads, allreduced_views],
                scaler.loss_scale() /
                (torch.distributed.get_world_size()
                 * self.num_accumulation_steps)
            )
            # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
            torch.distributed.all_reduce(flat_raw)
            # 4. combine unscaling and unflattening of allreduced gradient
            overflow_buf.zero_()
            amp_C.multi_tensor_scale(
                65536,
                overflow_buf,
                [allreduced_views, master_grads],
                1./scaler.loss_scale()
            )
            # 5. update loss scale
            scaler = _amp_state.loss_scalers[0]
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overfloat_buf = old_overflow_buf
            # 6. call optimizer step function
            if had_overflow == 0:
                self.optimizer.step()
                self.global_step += 1
            else:
                # Overflow detected, print message and clear gradients
                if self.is_main_process:
                    logging.info(
                        f"Rank {torch.distributed.get_rank()} "
                        ":: Gradient overflow.  Skipping step, "
                        f"reducing loss scale to {scaler.loss_scale()}"
                    )
                if _amp_state.opt_properties.master_weights:
                    for param in self.optimizer._amp_stash.all_fp32_from_fp16_params:
                        param.grad = None
            for param in self.model.parameters():
                param.grad = None
        else:
            self.optimizer.step()
            # NOTE: This basically does: optimizer.zero_grad()
            for param in self.model.parameters():
                param.grad = None
            self.global_step += 1


    def make_checkpoint(self, f_id, files):
        r"""Saves a checkpoint of the model."""
        logging.info("Saving a checkpoint of the current model...")

        # NOTE: model may be an instance of apex.parallel.distributed.DistributedDataParallel
        # in this case, model.module is the actual pytorch module
        model_to_save = \
            self.model.module \
            if hasattr(self.model, 'module') \
            else self.model

        # Save model weights, optimizer state, AMP master parameters and
        # the list of .hdf5 that are yet to be used (e.g. for resuming pre-training)
        if self.resume_step < 0 or not self.phase2:
            output_save_file = os.path.join(
                self.output_directory,
                f"ckpt_{self.global_step}.pt")
        else:
            output_save_file = os.path.join(
                self.output_directory,
                f"ckpt_{self.global_step + self.phase1_end_step}.pt")
        torch.save(
            {
                'model': model_to_save.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'master params': list(amp.master_params(self.optimizer)),
                'files': [f_id] + files
            },
            output_save_file
        )

        # Keep only a specific number of 'best' checkpoints
        self.most_recent_ckpts_paths.append(output_save_file)
        if len(self.most_recent_ckpts_paths) > self.num_checkpoints_to_keep:
            checkpoint_to_remove = \
                self.most_recent_ckpts_paths.pop(0)
            os.remove(checkpoint_to_remove)


def main():
    args = parse_args()
    pretrainer = ModelPretrainer(args)
    pretrainer.prepare_model_optimizer_and_scheduler()
    pretrainer.run_pretraining()

    end_datetime = datetime.datetime.now()
    if pretrainer.is_main_process:
        logging.info(
            "The complete pre-training took approx. %s seconds",
            end_datetime - pretrainer.start_datetime
        )

if __name__ == "__main__":
    main()
