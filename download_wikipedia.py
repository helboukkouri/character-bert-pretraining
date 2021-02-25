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
"""Script for downloading Wikipedia corpora."""
import logging
import argparse

from utils.download.wikipedia import WikipediaCorpusDownloader

LOGGING_FORMAT = "%(asctime)s | PID: %(process)d | %(filename)s | %(levelname)s - %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description=\
            'Downloads the Wikipedia corpus for a specific language.'
    )
    parser.add_argument(
        '--language',
        required=True, type=str, choices=['en'],
        help='The language of the Wikipedia corpus to download.',
    )
    parser.add_argument(
        '--debug',
        action="store_true",
        help='Whether to download a sample corpus for debugging.',
    )
    args = parser.parse_args()

    downloader = WikipediaCorpusDownloader(
        language=args.language, download_sample=args.debug)
    logging.info('Preparing to download Wikipedia corpus using parameters:')
    logging.info(' * language: %s', args.language)
    logging.info(' * download_sample: %s', args.debug)
    downloader.download()


if __name__ == "__main__":
    main()
