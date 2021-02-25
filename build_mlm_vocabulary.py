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
"""Script for building a Masked Language Modeling vocabulary for CharacterBERT."""
import os
import logging
import argparse
from collections import Counter

from tqdm import tqdm

WORKDIR = os.environ['WORKDIR']
MLM_VOCABULARY_DIRECTORY = os.path.join(WORKDIR, 'data', 'mlm_vocabularies')
os.makedirs(MLM_VOCABULARY_DIRECTORY, exist_ok=True)

LOGGING_FORMAT = "%(asctime)s | PID: %(process)d | %(filename)s | %(levelname)s - %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description=\
            'Builds a Masked Language Modeling vocabulary for CharacterBERT.'
    )
    parser.add_argument(
        '--formatted_corpus_path',
        required=True, type=str,
        help='Path to a formatted and pre-tokenized corpus (.formatted.txt file).',
    )
    parser.add_argument(
        '--max_vocabulary_size',
        required=False, type=int, default=100000,
        help='Max number of tokens (from most to less frequent) to keep in the vocabulary.',
    )
    args = parser.parse_args()

    # Recover corpus name from corpus path
    prefix = os.path.basename(os.path.dirname(args.formatted_corpus_path))
    save_path = os.path.join(MLM_VOCABULARY_DIRECTORY, prefix)
    os.makedirs(save_path, exist_ok=True)

    mlm_vocabulary_fpath = os.path.join(save_path, 'mlm_vocab.txt')
    logging.info('Preparing to build a MLM vocabulary using parameters:')
    logging.info(' * formatted_corpus_path: %s', args.formatted_corpus_path)
    logging.info(' * max_vocabulary_size: %s', args.max_vocabulary_size)
    if os.path.exists(mlm_vocabulary_fpath):
        logging.warning('Found existing vocabulary file: %s', mlm_vocabulary_fpath)
        return

    # Count all the tokens in the corpus
    counter = Counter()
    logging.info('Reading corpus file: %s', args.formatted_corpus_path)
    with open(args.formatted_corpus_path, 'r', encoding="utf-8") as stream:
        for line in tqdm(stream, desc='Reading lines...'):
            line = line.strip()
            if line:
                counter.update(line.split())

    # Most frequent tokens
    topk_tokens = counter.most_common()[:args.max_vocabulary_size]
    logging.info('Final vocabulary size: %s', len(topk_tokens))
    logging.info('Most frequent token: \'%s\' (%s)', topk_tokens[0][0], topk_tokens[0][1])
    logging.info('Least frequent token: \'%s\' (%s)', topk_tokens[-1][0], topk_tokens[-1][1])

    # Save Masked Language Modeling vocabulary
    with open(mlm_vocabulary_fpath, 'w', encoding="utf-8") as f:
        for token, count in topk_tokens:
            f.write(f"{count} {token}\n")

if __name__ == "__main__":
    main()
