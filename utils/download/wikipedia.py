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
"""Tools for downloading Wikipedia corpora."""
from typing import Optional

import os
import logging
import urllib.request

WORKDIR = os.environ['WORKDIR']
DATA_DOWNLOAD_DIRECTORY = os.path.join(WORKDIR, 'data', 'downloaded')
os.makedirs(DATA_DOWNLOAD_DIRECTORY, exist_ok=True)


class WikipediaCorpusDownloader:
    r"""
    Helper class for downloading Wikipedia corpora.
    Args:
        language (:obj:`str`):
            Short string specifying the language of the Wikipedia corpus to
            download (e.g. 'en' for English).
        download_debug_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to override the corpus choice with a sample corpus. This
            is useful for debugging as downloading an actual corpus may increase
            the processing times (corpus preparation & training)
    """
    
    def __init__(self, language: str, download_sample: Optional[bool] = False):

        self.language = language if (not download_sample) else 'sample'
        self.save_path = os.path.join(
            DATA_DOWNLOAD_DIRECTORY, f'wikipedia_{self.language}')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=False)

        # Add more links as needed.
        # Supported languages are: https://dumps.wikimedia.org/other/static_html_dumps/current/
        # Actual dumps can be found at: https://dumps.wikimedia.org/{{language}}wiki/
        # change {{language}} with your language of choice.
        self.download_urls = {
            'en': \
                "https://dumps.wikimedia.org/enwiki/latest/" \
                "enwiki-latest-pages-articles.xml.bz2",
            'sample': \
                "https://dumps.wikimedia.org/enwiki/latest/" \
                "enwiki-latest-pages-articles1.xml-p1p41242.bz2",

        }
        self.output_files = {
            # I you add more languages, make sure the archive name is 'wikipedia_{language}.xml.bz2'
            'en': 'wikipedia_en.xml.bz2',
            'sample': 'sample.xml.bz2',
        }

    def download(self):
        if (self.language in self.download_urls) and (self.language in self.output_files):
            url = self.download_urls[self.language]
            filename = self.output_files[self.language]
            saved_file_path = os.path.join(self.save_path, filename)
        else:
            raise NotImplementedError(
                f'Language not supported: {self.language}\n'
                'For a list of supported languages, have a look at'
                '`self.download_urls` and `self.output_files` mappings '
                'in: WORKDIR/utils/download/wikipedia.py')

        # Download archive
        if os.path.exists(saved_file_path):
            logging.warning('Found existing corpus file: %s', saved_file_path)
            logging.info('Skipping download...')
        else:
            logging.info('Downloading: %s', url)
            urllib.request.urlretrieve(url, saved_file_path)
