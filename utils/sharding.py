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
"""Tools for splitting a formatted corpus into train/val shards of approx. equal size."""
from typing import Optional

import os
import random
import logging

from tqdm import tqdm

WORKDIR = os.environ['WORKDIR']
SHARDS_DIRECTORY = os.path.join(WORKDIR, 'data', 'shards')
os.makedirs(SHARDS_DIRECTORY, exist_ok=True)


class CorpusSharder:
    r"""
    Splits a formatted corpus into train/val shards of approx. equal size.
    Args:
    formatted_corpus_path (:obj:`str`):
        Path to a formatted corpus (.formatted.txt file).
    n_training_shards (:obj:`int`, `optional`, defaults to :obj:`1024`):
        Number of training shards.
    n_validation_shards (:obj:`int`, `optional`, defaults to :obj:`16`):
        Number of validation shards.
    random_seed (:obj:`int`, `optional`, defaults to :obj:`42`):
        A random seed to set before randomly splitting into shards.
    """
    def __init__(
            self,
            formatted_corpus_path: str,
            n_training_shards: Optional[int] = 1024,
            n_validation_shards: Optional[int] = 16,
            random_seed: Optional[int] = 42
        ):

        random.seed(random_seed) # set random seet

        assert n_training_shards > 0, \
            'There must be at least one training shard.'
        assert n_validation_shards > 0, \
            'There must be at least one validation shard.'

        self.input_file = formatted_corpus_path
        # Recover corpus name from formatted corpus path
        prefix = os.path.basename(os.path.dirname(self.input_file))
        save_path = os.path.join(SHARDS_DIRECTORY, prefix)
        os.makedirs(save_path, exist_ok=True)            
        self.save_path = save_path

        self.n_training_shards = n_training_shards
        self.n_validation_shards = n_validation_shards

        self.sentences = []
        self.n_sentences = 0
        self.n_documents = 0
        self.n_sentences_per_corpus = {}
        self.n_documents_per_corpus = {}
        self.n_sentences_per_document_per_corpus = {}

    def make_shards(self):
        text_files_in_save_path = [
            p.endswith('.txt') for p in os.listdir(self.save_path)]
        if text_files_in_save_path:
            logging.warning('Found existing shards in: %s', self.save_path)
            return

        logging.info('Distributing sentences over shards...')

        # Compute total number of shards
        total_n_of_shards = \
            self.n_training_shards \
            + self.n_validation_shards

        # Initialise n_sentences counters for each shard
        n_sentences_per_shard = {i: 0 for i in range(total_n_of_shards)}

        # Get id of smallest shard
        smallest_shard = min(
            n_sentences_per_shard.items(), key=lambda x: x[1])[0]

        # Open the smallest shard file
        f_out = open(
            os.path.join(
                self.save_path, f'shard_{smallest_shard}.txt'),
            "a", encoding="utf-8")

        # Iterate trough the corpus files
        n_written_lines = 0
        with open(self.input_file, "r", encoding="utf-8") as f_in:
            for line in tqdm(f_in, desc='Number of sentences so far...'):
                line = line.strip()

                # If line is not blank (we are within a document)
                if line != '':
                    # Write the line in the shard file
                    f_out.write(line + '\n')
                    # Update count
                    n_written_lines += 1

                # If line is blank (we are outsite a document)
                else:
                    f_out.write('\n')  # Add a blank line to shard file
                    n_sentences_per_shard[smallest_shard] += n_written_lines  # Update shard counters

                    # Find smallest shard
                    smallest_shard = min(n_sentences_per_shard.items(), key=lambda x: x[1])[0]

                    # Close previous shard and open smallest shard file
                    f_out.close()
                    f_out = open(
                        os.path.join(
                            self.save_path, f'shard_{smallest_shard}.txt'),
                        "a", encoding="utf-8")

                    # Re-init count
                    n_written_lines = 0

        # NOTE: this if statement may be useless. It was usefule when the code
        # looping used to loop over multiple formatted corpora. This version
        # only takes a single input file (which can be the result of multiple
        # formatted corpora combined). I keep the code here so that it can be
        # easilly adapted to the case of multiple formatted corpus files
        if n_written_lines != 0:
            f_out.write('\n')  # Add a blank line to shard file
            n_sentences_per_shard[smallest_shard] += n_written_lines  # Update shard counters

            # Find smallest shard
            smallest_shard = min(n_sentences_per_shard.items(), key=lambda x: x[1])[0]

            # Close previous shard file and open smallest shard file
            f_out.close()
            f_out = open(
                os.path.join(
                    self.save_path,
                    f'shard_{smallest_shard}.txt'),
                "a", encoding="utf-8")

            # Re-init counter
            n_written_lines = 0

        f_out.close()

        logging.info('Renaming shards...')

        all_shards = list(range(total_n_of_shards))
        random.shuffle(all_shards)
        for i in range(self.n_training_shards):
            shard_id = all_shards.pop(0)
            os.rename(
                os.path.join(self.save_path, f'shard_{shard_id}.txt'),
                os.path.join(self.save_path, f'training.{i}.txt'))
        for i in range(self.n_validation_shards):
            shard_id = all_shards.pop(0)
            os.rename(
                os.path.join(self.save_path, f'shard_{shard_id}.txt'),
                os.path.join(self.save_path, f'validation.{i}.txt'))
