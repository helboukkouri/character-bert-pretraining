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
Script for formatting Wikipedia Corpora: single file with one sentence per line
and a blank line between sentences from different articles.
"""
import os
import logging
import argparse

import nltk
from tqdm import tqdm
from transformers import BasicTokenizer
from utils.format.wikipedia import WikipediaCorpusFormatter

LOGGING_FORMAT = "%(asctime)s | PID: %(process)d | %(filename)s | %(levelname)s - %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)


# TODO: this does both document extaction from a Wikipedia dump AND
# the actual formatting (one doc per line -> one tokenized sentence per line)
# Ideally the formatting script should be independent from the
# Wikipedia-specific extraction step.
def main():
    parser = argparse.ArgumentParser(
        description=\
            'Formats a Wikipedia dump (.xml.bz2 archive).'
    )
    parser.add_argument(
        '--archive_path',
        required=True, type=str,
        help='Path to Wikipedia dump to format.',
    )
    parser.add_argument(
        '--delete_document_level_corpus',
        action='store_true',
        help='Whether to keep both the formatted and original document-level corpora.',
    )
    args = parser.parse_args()

    formatter = WikipediaCorpusFormatter(archive_path=args.archive_path)
    logging.info('Preparing to format Wikipedia dump using parameters:')
    logging.info(' * archive_path: %s', args.archive_path)

    # NOTE: this is an extraction step (dump -> single .txt file with documents)
    fpath = formatter.format_as_one_document_per_line()
    if fpath is None:
        return  # Aborting formatting as formatted corpus already exists

    # NOTE: this is the actual formatting step:
    # --> one tokenized sentence per line + blank line between documents
    logging.info('Formatting extracted documents...')
    tokenizer = BasicTokenizer()
    split_into_sentences = nltk.tokenize.sent_tokenize
    split_into_tokens = tokenizer.tokenize
    input_file_path = fpath
    output_file_path = fpath.replace('.txt', '.formatted.txt')
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            for line in tqdm(input_file, desc='Segmenting corpus'):
                if line.strip():  # if document
                    sentences = split_into_sentences(line)
                    for sentence in sentences:
                        tokens = split_into_tokens(sentence.strip())
                        new_line = ' '.join(tokens) + '\n'
                        output_file.write(new_line)
                else:  # if blank line
                    output_file.write('\n')

    if args.delete_document_level_corpus:
        logging.info('Deleting document level corpus...')
        os.remove(input_file_path)


if __name__ == "__main__":
    main()
