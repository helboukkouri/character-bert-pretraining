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
"""Script for converting corpus shards into tensors that can be fed to a model."""
import os
import logging
import argparse
import concurrent.futures

from tqdm import tqdm
from glob import glob

from utils.pretraining.data import PreTrainingDataGenerator

LOGGING_FORMAT = "%(asctime)s | PID: %(process)d | %(filename)s | %(levelname)s - %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)


parser = argparse.ArgumentParser(
    description=\
        'Converts corpus shards into tensors that can be fed to a model.'
)
parser.add_argument(
    '--shards_path',
    required=True, type=str,
    help=\
        'Path to a directory contraining corpus shards: training.*.txt, '
        'validation.*.txt and test.*.txt files',
)
parser.add_argument(
    '--output_directory',
    required=True, type=str,
    help=\
        'Path to a directory for saving output .hdf5 files.'
)
parser.add_argument(
    '--is_character_bert',
    action='store_true',
    help=\
        'Whether to create pre-training for CharacterBERT of BERT. '
        'When this is True, data is generated for CharacterBERT.',
)
parser.add_argument(
    '--duplication_factor',
    required=False, type=int, default=5,
    help=\
        'How many iterations over the shard contents. The higher this value the '
        'more we randomly generate examples from the same sentences/documents.',
)
parser.add_argument(
    '--max_input_length',
    required=False, type=int, default=128,
    help='Maximum sequence length for the model input. This is usually 128 or 512.',
)
parser.add_argument(
    '--short_input_probability',
    required=False, type=float, default=0.10,
    help=\
        'Probability of generating an exemple with a shorter input. When this '
        'happens, a random number between 2 and `max_input_length` is picked and '
        'a short input of that size is generated.',
)
parser.add_argument(
    '--max_masked_tokens_per_input',
    required=False, type=int, default=20,
    help='Hard limit on the number of tokens that can be masked.'
)
parser.add_argument(
    '--masked_tokens_ratio',
    required=False, type=float, default=0.15,
    help=\
        'Proportion of input tokens that we try to mask. If this is higher than '
        '`max_masked_tokens_per_input` then we only mask the allowed maximum number.'
)
parser.add_argument(
    '--random_seed',
    required=False, type=int, default=42,
    help='A random seed to set before randomly generating pre-training examples.'
)
args = parser.parse_args()

logging.info('Preparing to convert shards to pre-training data using parameters:')
logging.info(' * shards_path: %s', args.shards_path)
logging.info(' * output_directory: %s', args.output_directory)
logging.info(' * is_character_bert: %s', args.is_character_bert)
logging.info(' * duplication_factor: %s', args.duplication_factor)
logging.info(' * max_input_length: %s', args.max_input_length)
logging.info(' * short_input_probability: %s', args.short_input_probability)
logging.info(' * max_masked_tokens_per_input: %s', args.max_masked_tokens_per_input)
logging.info(' * masked_tokens_ratio: %s', args.masked_tokens_ratio)
logging.info(' * random_seed: %s', args.random_seed)


def pretraining_data_generation_job(shard_fpath):
    """Generates pre-training data for a specific shard"""
    if not os.path.exists(shard_fpath):
        logging.warning('Could not find file: %s', shard_fpath)
        return
    elif not os.path.isfile(shard_fpath):
        logging.error(("%s is not a valid path", shard_fpath))
        return
    else:
        pass
    generator = PreTrainingDataGenerator(
        shard_fpath=shard_fpath,
        output_directory=args.output_directory,
        is_character_bert=args.is_character_bert,
        duplication_factor=args.duplication_factor,
        max_input_length=args.max_input_length,
        short_input_probability=args.short_input_probability,
        max_masked_tokens_per_input=args.max_masked_tokens_per_input,
        masked_tokens_ratio=args.masked_tokens_ratio,
        random_seed=args.random_seed,
        verbose=False
    )
    generator.generate_pretraining_examples()
    generator.write_examples_to_hdf5()

# Get all shards paths
all_shard_fpaths = list(glob(args.shards_path + '/*'))

# Run conversion in parallel
logging.info('Concurrently converting shards into pre-training data...')
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Submit jobs
    jobs = []
    for shard_fpath in all_shard_fpaths:
        job = executor.submit(pretraining_data_generation_job, shard_fpath)
        jobs.append(job)
    # Update a progress bar as jobs complete
    n_shards = len(all_shard_fpaths)
    with tqdm(total=n_shards, desc='Iterating over shards') as pbar:
        for i, job in enumerate(concurrent.futures.as_completed(jobs)):
            _ = job.result()  # make sure the job has finished
            pbar.update((i+1) - pbar.n)  # update the bar

    # NOTE: all the complexity above could be reduced using:
    # executor.map(pretraining_data_generation_job, shard_fpaths)
    # But then we wouldn't have a nice progress bar :)
