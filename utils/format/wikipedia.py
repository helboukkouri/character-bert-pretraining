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
"""Tools for formatting Wikipedia Corpora: single file with one document per line."""
import os
import re
import glob
import logging
import subprocess
import multiprocessing

import tqdm
from bs4 import BeautifulSoup

WORKDIR = os.environ['WORKDIR']
DATA_FORMAT_DIRECTORY = os.path.join(WORKDIR, 'data', 'formatted')
os.makedirs(DATA_FORMAT_DIRECTORY, exist_ok=True)


class WikipediaCorpusFormatter:
    r"""
    Class for formatting Wikipedia Corpora: single file with one document per line.
    Args:
        archive_path (:obj:`str`):
            Path to Wikipedia dump archive (.xml.bz2 file).
    """
    def __init__(self, archive_path: str):
        self.archive_path = archive_path
        self.read_path = os.path.dirname(archive_path)
        self.save_path = os.path.join(
            DATA_FORMAT_DIRECTORY, os.path.basename(self.read_path))
        self.input_filename = os.path.basename(archive_path)
        self.output_filename = self.input_filename.replace(
            '.xml.bz2', '.txt')
        os.makedirs(self.save_path, exist_ok=True)

    def format_as_one_document_per_line(self):
        """Creates a single file with one document per line."""

        output_file_path = os.path.join(self.save_path, self.output_filename)
        path = output_file_path.replace('.txt', '.formatted.txt')
        if os.path.exists(path):
            logging.warning('Found existing formatted corpus: %s', path)
            return

        # Extract archive
        input_xml_file = self.archive_path.split(".bz2")[0]
        if os.path.exists(input_xml_file):
            logging.warning('Found existing extracted archive. Skipping extraction...')
        else:
            logging.info('Extracting XML file from archive...')
            subprocess.run(f"bzip2 -dk {self.archive_path}", shell=True, check=True)

        # Run Wikiextractor
        os.chdir(WORKDIR)  # just in case
        path_to_wikiextractor = '.'.join((
            'external', 'wikiextractor', 'wikiextractor', 'WikiExtractor'))
        output_dir = os.path.join(self.save_path, "extracted")
        n_processes = max(1, int(0.8 * multiprocessing.cpu_count()))
        wikiextractor_command = ' '.join([
            'python', '-m',
            path_to_wikiextractor,
            input_xml_file,
            '-b', '20M',  # max 20Mb per file
            '--processes', str(n_processes),
            '-o', output_dir
        ])
        logging.info('Running WikiExtractor:\n%s', wikiextractor_command)
        subprocess.run(wikiextractor_command, shell=True, check=True)
        logging.info('Done! Removing XML file...')
        os.remove(input_xml_file)  # Removing .xml file after extraction

        # Merge all wikipedia articles into a single text file: one document per line
        logging.info('Merging articles (documents) into a single file...')
        with open(output_file_path, mode='w', encoding='utf-8') as ofile:
            for filename in tqdm.tqdm(
                    glob.glob(output_dir + '/**', recursive=True),
                    desc=f'Reading files from: {output_dir}'):
                if os.path.isfile(filename):
                    with open(filename, mode='r', encoding='utf-8') as f:
                        parsed_html = BeautifulSoup(
                            f.read(), features="html.parser")
                        for result in parsed_html.find_all('doc'):
                            doc = result.text
                            if doc.strip() != '':
                                doc = re.sub(r'\s+', ' ', doc)
                                ofile.write(doc.strip() + "\n\n")

        # Remove extraction directory
        logging.info('Clean up: removing XML extraction directory...')
        for dirpath in glob.glob(output_dir + '/*'):
            for path in glob.glob(dirpath + '/*'):
                os.remove(path)
            os.rmdir(dirpath)
        os.rmdir(output_dir)

        return output_file_path  # return file path for further processing
