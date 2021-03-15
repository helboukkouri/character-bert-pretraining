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
"""
Script for formatting document-level corpora: single file with one sentence
per line and a blank line between sentences from different documents.
"""
import os
import logging
import argparse

import nltk
from tqdm import tqdm
from transformers import BasicTokenizer

WORKDIR = os.environ['WORKDIR']
FORMATTED_DATA_DIRECTORY = os.path.join(WORKDIR, 'data', 'formatted')
os.makedirs(FORMATTED_DATA_DIRECTORY, exist_ok=True)

LOGGING_FORMAT = "%(asctime)s | PID: %(process)d | %(filename)s | %(levelname)s - %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)


def main():
    """Script for formatting document-level corpora."""

    parser = argparse.ArgumentParser(
        description=\
            'Formats a document-level corpus.'
    )
    parser.add_argument(
        '--document_level_corpus_path',
        required=True, type=str,
        help=\
            'Path to the document level corpus: one document per line '
            '+ blank line between documents.',
    )
    args = parser.parse_args()

    logging.info('Preparing to format a document-level corpus using parameters:')
    for argname, argvalue in vars(args).items():
        logging.info(' * %s: %s', argname, argvalue)

    # Make sure document-level corpus exists
    assert os.path.exists(args.document_level_corpus_path)

    # Make output directory
    formatted_corpus_path = os.path.join(
        FORMATTED_DATA_DIRECTORY,
        os.path.basename(
            os.path.dirname(
                args.document_level_corpus_path)),
        'formatted.txt'
    )
    os.makedirs(
        os.path.dirname(formatted_corpus_path),
        exist_ok=True
    )

    # Make sur output corpus does not already exist
    if os.path.exists(formatted_corpus_path):
        logging.warning(
            'Found corpus file: %s',
            formatted_corpus_path.replace(WORKDIR, '$WORKDIR'))
        logging.warning('Aborted formatting.')
        return

    # Tokenizer & sentence segmenter
    tokenizer = BasicTokenizer()
    split_into_sentences = nltk.tokenize.sent_tokenize
    logging.info('Using NLTK sentence segmenter.')
    split_into_tokens = tokenizer.tokenize
    logging.info('Using huggingface/transformers BasicTokenizer.')

    # Actual formatting
    logging.info('Formatting corpus...')
    input_file_path = args.document_level_corpus_path
    output_file_path = formatted_corpus_path
    n_tokens = 0
    n_sentences = 0
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            for line in tqdm(input_file, desc='Segmenting corpus'):
                if line.strip():  # if document
                    sentences = split_into_sentences(line)
                    for sentence in sentences:
                        tokens = split_into_tokens(sentence.strip())
                        new_line = ' '.join(tokens) + '\n'
                        output_file.write(new_line)
                        n_sentences += 1
                        n_tokens += len(tokens)
                else:  # if blank line
                    output_file.write('\n')

    logging.info('Done formatting.')
    logging.info('* Total number of sentences: %s', n_sentences)
    logging.info('* Total number of tokens: %s', n_tokens)


if __name__ == "__main__":
    main()
