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
"""Script for splitting a formatted corpus into train/val shards of approx. equal size."""
import logging
import argparse

from utils.sharding import CorpusSharder

LOGGING_FORMAT = "%(asctime)s | PID: %(process)d | %(filename)s | %(levelname)s - %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description=\
            'Splits a formatted corpus into train/val shards of approx. equal size.'
    )
    parser.add_argument(
        '--formatted_corpus_path',
        required=True, type=str,
        help='Path to a formatted corpus (.formatted.txt file).',
    )
    parser.add_argument(
        '--n_training_shards',
        required=False, type=int, default=1024,
        help='Number of training shards.',
    )
    parser.add_argument(
        '--n_validation_shards',
        required=False, type=int, default=16,
        help='Number of validation shards.',
    )
    parser.add_argument(
        '--random_seed',
        required=False, type=int, default=42,
        help='A random seed to set before randomly splitting into shards.',
    )
    args = parser.parse_args()

    logging.info('Preparing to make shards using parameters:')
    logging.info(' * formatted_corpus_path: %s', args.formatted_corpus_path)
    logging.info(' * n_training_shards: %s', args.n_training_shards)
    logging.info(' * n_validation_shards: %s', args.n_validation_shards)
    logging.info(' * random_seed: %s', args.random_seed)
    sharder = CorpusSharder(
        formatted_corpus_path=args.formatted_corpus_path,
        n_training_shards=args.n_training_shards,
        n_validation_shards=args.n_validation_shards,
        random_seed=args.random_seed
    )
    sharder.make_shards()

if __name__ == "__main__":
    main()
