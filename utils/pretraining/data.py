# coding=utf-8
# Copyright (c) 2020, Hicham EL BOUKKOURI.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
Tools for converting corpus shards into tensors that can be fed to a model.
NOTE: this is adapted from an older version of
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/create_pretraining_data.py
"""
from typing import List, Optional

import os
import random
import logging
from collections import OrderedDict

import h5py
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, CharacterBertTokenizer

from utils.text import convert_to_unicode, truncate_seq_pair

WORKDIR = os.environ['WORKDIR']


class PreTrainingExample:
    r"""
    A single pre-training example: sentence pair with targets for NSP and MLM.
    Args:
    tokens (:obj:`list(str)`):
        Input tokens: [CLS] A A A [SEP] B B B [SEP]
    segment_ids (:obj:`list(int)`):
        Segment ids corresponding to the input tokens: 0 0 0 0 0 1 1 1 1
    mlm_positions (:obj:`list(int)`):
        Indices of masked (altered) tokens.
    mlm_labels (:obj:`list(str)`):
        Original tokens for masked (altered) positions.
    is_random_next (:obj:`bool`):
        Whether text span B is random (NSP target is False when this is True).
    """

    def __init__(self,
        tokens: List[str],
        segment_ids: List[int],
        mlm_positions: List[int],
        mlm_labels: List[str],
        is_random_next: bool
    ):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.mlm_positions = mlm_positions
        self.mlm_labels = mlm_labels

    def __str__(self):
        string = ""
        string += "tokens: %s\n" % \
            (" ".join([convert_to_unicode(x) for x in self.tokens]))
        string += "segment_ids: %s\n" % \
            (" ".join([str(x) for x in self.segment_ids]))
        string += "is_random_next: %s\n" % \
            self.is_random_next
        string += "mlm_positions: %s\n" % \
            (" ".join([str(x) for x in self.mlm_positions]))
        string += "mlm_labels: %s\n" % \
            (" ".join([convert_to_unicode(x) for x in self.mlm_labels]))
        string += "\n"
        return string

    def __repr__(self):
        return self.__str__()


class PreTrainingDataGenerator:
    r"""
    Generates MLM/NSP examples for BERT/CharacterBERT from a single shard.
    Args:
    shard_fpath (:obj:`str`):
        Path to a corpus shard contraining one sentence per line and a blank
        line between sentences from different documents.
    output_directory (:obj:`str`)
        'Path to a directory for saving the output .hdf5 file.'
    is_character_bert (:obj:`bool`):
        Whether to create pre-training for CharacterBERT of BERT. When this is
        True, data is generated for CharacterBERT.
    duplication_factor (:obj:`int`):
        How many iterations over the shard contents. The higher this value the
        more we randomly generate examples from the same sentences/documents.
    max_input_length (:obj:`int`):
        Maximum sequence length for the model input. Usually this is set to 512.
    short_input_probability (:obj:`float`):
        Probability of generating an exemple with a shorter input. When this
        happens, a random number between 2 and `max_input_length` is picked and
        a short input of that size is generated.
    max_masked_tokens_per_input (:obj:`int`):
        Hard limit on the number of tokens that can be masked.
    masked_tokens_ratio (:obj:`float`):
        Proportion of input tokens that we attempt to mask. If this is higher than
        `max_masked_tokens_per_input` then we only mask the allowed maximum number.
    random_seed (:obj:`float`):
        A random seed to set before randomly generating pre-training examples.
    verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
        Whether to print logs and progress bars. It is useful to turn this off
        if running multiple processes (too many logs)
    """
    def __init__(self,
        shard_fpath: str,
        output_directory: str,
        is_character_bert: bool,
        duplication_factor: int,
        max_input_length: int,
        short_input_probability: float,
        max_masked_tokens_per_input: int,
        masked_tokens_ratio: float,
        random_seed: int,
        verbose: Optional[bool] = True,
    ):
        self.verbose = verbose
        if not self.verbose:
            logging.disable(logging.WARNING)

        self.random_seed = random_seed
        self.set_all_random_seeds()

        self.shard_fpath = shard_fpath
        self.output_directory = output_directory
        self.is_character_bert = is_character_bert
        self.duplication_factor = duplication_factor
        self.max_input_length = max_input_length
        self.short_input_probability = short_input_probability
        self.max_masked_tokens_per_input = max_masked_tokens_per_input
        self.masked_tokens_ratio = masked_tokens_ratio

        self.pretraining_examples = []

        # Load tokenizer / masked language modeling vocabulary:
        if self.is_character_bert:
            # Here, MLM vocabulary is a list of most frequent tokens in the corpus
            logging.info('Using a "word" vocabulary for MLM.')
            self.tokenizer = CharacterBertTokenizer()
            corpus_name = os.path.basename(os.path.dirname(self.shard_fpath))
            self.mlm_vocabulary = [
                line.strip().split()[-1]
                for line in open(
                    os.path.join(
                        WORKDIR, 'data', 'mlm_vocabularies',
                        corpus_name, 'mlm_vocab.txt'),
                    'r', encoding='utf-8')
            ]
        else:
            # Here, MLM vocabulary is the same as the wordpiece vocabulary
            logging.info('Using a WordPiece vocabulary for MLM.')
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(WORKDIR, 'data', 'bert-base-uncased'))
            self.mlm_vocabulary = list(self.tokenizer.vocab)
        logging.info('Example tokens: %s', random.sample(self.mlm_vocabulary, 10))


    def set_all_random_seeds(self):
        r"""Sets all random seeds to `self.random_seed`"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        logging.info("Random seed set to: %d", self.random_seed)


    def generate_pretraining_examples(self):
        r"""Generates pre-training examples (MLM/NSP) from the input shard."""
        logging.info('BEGIN: Generate Pre-training Examples')

        # Gather token sequences by document
        documents = self.read_documents_from_shard()

        # We run through the corpus as many times as `self.duplication_factor`
        iterator = range(self.duplication_factor)
        if self.verbose:
            iterator = tqdm(iterator, desc='Iterations over all documents')
        for _ in iterator:
            for doc_id in range(len(documents)):
                new_pretraining_examples = \
                    self.generate_examples_from_document(
                        document_id=doc_id,
                        all_documents=documents
                    )
                self.pretraining_examples.extend(new_pretraining_examples)

        # Shuffle the examples within the shard's pre-training data
        logging.info("Shuffling examples...")
        random.shuffle(self.pretraining_examples)

        logging.info('END: Generate Pre-training Examples')


    def read_documents_from_shard(self):
        r"""
        Groups token sequences by document for the input shard then returns
        a list of documents where each document is a set of token sequences.
        """
        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        counter = 0
        documents = [[]]
        logging.info("Grouping sentences by document from: %s", self.shard_fpath)
        with open(self.shard_fpath, "r", encoding='utf-8') as f:
            for line in f:
                line = convert_to_unicode(line)
                line = line.strip()
                if line == '':
                    documents.append([])  # New document
                else:
                    # BERT: basic tokenization + wordpiece tokenization
                    # CharacterBERT: basic tokenization only
                    tokens = self.tokenizer.tokenize(line)
                    if tokens:
                        # Add tokens to last document
                        documents[-1].append(tokens)
                        if counter < 2:
                            logging.info("Example of a sentence: %s", tokens)
                    counter += 1

        logging.info("Removing empty documents (if any)...")
        # There shouldn't be any empty documents, but just to be extra safe
        documents = [x for x in documents if x]

        logging.info("Shuffling documents...")
        random.shuffle(documents)
        return documents


    def generate_examples_from_document(self, document_id, all_documents):
        """Returns a number of `PreTrainingExample` objects from a single document."""
        document = all_documents[document_id]

        # Account for [CLS], [SEP], [SEP]
        max_num_tokens = self.max_input_length - 3

        # We *usually* want to fill up the entire input since we are padding
        # to `max_input_length` anyways, so short input sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_input_probability = 0.1 = 10% of the time) want to use shorter
        # input sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_input_length` is just a rough target however, whereas `max_input_length`
        # is a hard limit.
        target_input_length = max_num_tokens
        if random.random() < self.short_input_probability:
            target_input_length = random.randint(2, max_num_tokens)

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user input.
        examples = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            sentence = document[i]
            current_chunk.append(sentence)
            current_length += len(sentence)

            no_more_sentences_in_document = (i == (len(document) - 1))
            exceeded_target_length = (current_length >= target_input_length)
            if no_more_sentences_in_document or exceeded_target_length:
                if current_chunk:

                    # Building the segment `A`:

                    tokens_a = []
                    # `a_end` is how many sentences from `current_chunk` go into `A`.
                    if len(current_chunk) > 1:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    else:
                        a_end = 1
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    # Building the segment `B`:
                    tokens_b = []
                    is_random_next = False  # By default, `B` follows `A`

                    # In 50% of the cases,
                    # or if the chunk only has one sentence,
                    # segment `B` is from another random document
                    # (does not follow `A`, i.e. NSP target is False)
                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = True
                        target_b_length = target_input_length - len(tokens_a)

                        # Since we didn't actually use these sentences, we
                        # "put them back" so they don't go to waste.
                        num_unused_sentences = len(current_chunk) - a_end
                        i -= num_unused_sentences

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be safe, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        # We don't use a while loop to avoid looping indefinitely
                        # if there is only one document (shouldn't happen)
                        for _ in range(10):
                            random_document_id = random.randint(0, len(all_documents) - 1)
                            if random_document_id != document_id:
                                break

                        # Add tokens using sentences from a random document
                        random_document = all_documents[random_document_id]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break

                    # In all other cases, use the actual next sentences
                    # (j >= a_end) to build segment `B`
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    # Truncate sentence pair to the target length
                    truncate_seq_pair(tokens_a, tokens_b, target_input_length)
                    assert (len(tokens_a) >= 1) and (len(tokens_b) >= 1)

                    # Building the actual input for the model:
                    # 1 - Adding CLS/SEP to [A, B]
                    # 2 - Building segment ids
                    # 3 - Masking tokens
                    tokens = []
                    segment_ids = []

                    tokens.append("[CLS]")
                    segment_ids.append(0)

                    for token in tokens_a:
                        tokens.append(token)
                        segment_ids.append(0)

                    tokens.append("[SEP]")
                    segment_ids.append(0)

                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids.append(1)

                    tokens.append("[SEP]")
                    segment_ids.append(1)

                    (transformed_tokens, mlm_positions, mlm_labels) = \
                        self.generate_mlm_instances_from_tokens(
                            input_tokens=tokens)

                    examples.append(
                        PreTrainingExample(
                            tokens=transformed_tokens,  # Original tokens with random tokens changed
                            segment_ids=segment_ids,  # The input's segment ids
                            is_random_next=is_random_next,  # For Next Sentence Prediction
                            mlm_positions=mlm_positions,  # For Masked Language Modeling
                            mlm_labels=mlm_labels  # For Masked Language Modeling
                        )
                    )
                current_chunk = []
                current_length = 0
            i += 1

        return examples


    def generate_mlm_instances_from_tokens(self, input_tokens):
        """Generates instances for the Masked Language Modelling objective."""
        candidate_tokens = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            candidate_tokens.append((i, token))
        random.shuffle(candidate_tokens)

        # Make a copy of the original input.
        # This will be transformed by randomly altering its tokens
        # and will be the actual input for the model
        output_tokens = input_tokens.copy()

        num_to_predict = max(
            1, int(round(len(input_tokens) * self.masked_tokens_ratio)))
        num_to_predict = min(
            num_to_predict, self.max_masked_tokens_per_input)

        # Create MLM instances
        mlm_instances = []
        covered_indices = set()
        for index, token in candidate_tokens:
            if len(mlm_instances) >= num_to_predict:
                break
            if index in covered_indices:
                continue
            if self.is_character_bert:
                # CharacterBERT: only mask tokens that are in the
                # MLM vocabulary (i.e. most frequent tokens in the corpus)
                if token not in self.mlm_vocabulary:
                    continue
            else:
                # BERT: all tokens are in the MLM vocabulary anyway
                # as MLM vocabulary == WordPiece vocabulary and all tokens
                # are WordPieces
                pass
            covered_indices.add(index)

            # Compute the token that will be placed at `index`
            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = token
                # 10% of the time, replace with random word
                else:
                    masked_token = random.choice(self.mlm_vocabulary)

            # Replace token at `index`
            output_tokens[index] = masked_token

            # Add MLM instance
            original_token = input_tokens[index]
            mlm_instances.append((index, original_token))

        #  Sort instances according to the masked tokens indices
        mlm_instances = sorted(mlm_instances, key=lambda x: x[0])

        mlm_positions, mlm_labels = [], []
        for (index, original_token) in mlm_instances:
            mlm_positions.append(index)
            mlm_labels.append(original_token)

        # Returns:
        #     - output_tokens: transformed tokens
        #     - mlm_positions: indices of masked tokens
        #     - mlm_labels: original token for each masked token
        return (output_tokens, mlm_positions, mlm_labels)


    def write_examples_to_hdf5(self):
        """Converts examples to tensors and saves it in hdf5 format."""

        logging.info('BEGIN: Write Pre-Training Examples to hdf5 File')

        total_written = 0
        features = OrderedDict()
        token_to_id = {w: i for i, w in enumerate(self.mlm_vocabulary)}

        num_instances = len(self.pretraining_examples)

        #### Initializing empty tensors ####

        # NOTE: (!important) here we assume that padding index is 0
        # for both BERT's and CharacterBERT's input ids, which is the default.
        if self.is_character_bert:
            features["input_ids"] = np.zeros(
                [
                    num_instances,
                    self.max_input_length,
                    self.tokenizer._mapper.max_word_length
                ],
                dtype="int32"
            )
        else:
            features["input_ids"] = np.zeros(
                [
                    num_instances,
                    self.max_input_length,
                ],
                dtype="int32")
        features["input_mask"] = np.zeros(
            [num_instances, self.max_input_length], dtype="int32")
        features["segment_ids"] = np.zeros(
            [num_instances, self.max_input_length], dtype="int32")
        features["masked_lm_positions"] = np.zeros(
            [num_instances, self.max_masked_tokens_per_input], dtype="int32")
        features["masked_lm_ids"] = np.zeros(
            [num_instances, self.max_masked_tokens_per_input], dtype="int32")
        features["next_sentence_labels"] = np.zeros(
            num_instances, dtype="int32")

        #### Filling the tensors ####

        iterator = self.pretraining_examples
        if self.verbose:
            iterator = tqdm(
                iterator, total=num_instances,
                desc='Converting pre-training examples to tensors'
            )
        for instance_id, instance in enumerate(iterator):
            input_ids = self.tokenizer.convert_tokens_to_ids(instance.tokens)
            segment_ids = list(instance.segment_ids)
            input_mask = [1] * len(segment_ids)
            assert len(input_ids) <= self.max_input_length
            assert len(segment_ids) <= self.max_input_length

            # Padding:
            while len(segment_ids) < self.max_input_length:
                if self.is_character_bert:
                    input_ids.append([0] * self.tokenizer._mapper.max_word_length)
                else:
                    input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == self.max_input_length
            assert len(input_mask) == self.max_input_length
            assert len(segment_ids) == self.max_input_length


            mlm_positions = list(instance.mlm_positions)
            if self.is_character_bert:
                masked_lm_ids = [
                    token_to_id[label]
                    for label in instance.mlm_labels]
            else:
                # Since the MLM output layer is a copy of the input Embedding matrix
                masked_lm_ids = \
                    self.tokenizer.convert_tokens_to_ids(
                        instance.mlm_labels)

            # NOTE: I'm not very sure about this part...
            # it seems that we fill up the mlm positions with the CLS position
            while len(mlm_positions) < self.max_masked_tokens_per_input:
                mlm_positions.append(0)
                masked_lm_ids.append(0)

            # NOTE: in the original code `is_random_next` is 1, we change it here
            # so that NSP target is True when not `is_random_next`
            next_sentence_label = 1 if (not instance.is_random_next) else 0

            features["input_ids"][instance_id] = input_ids
            features["input_mask"][instance_id] = input_mask
            features["segment_ids"][instance_id] = segment_ids
            features["masked_lm_positions"][instance_id] = mlm_positions
            features["masked_lm_ids"][instance_id] = masked_lm_ids
            features["next_sentence_labels"][instance_id] = next_sentence_label

            total_written += 1

        # Saving pretraining data as an .hdf5 file
        os.makedirs(self.output_directory, exist_ok=True)
        basename = os.path.basename(self.shard_fpath)
        output_hdf5_fpath = os.path.join(
            self.output_directory, basename.replace('.txt', '.hdf5'))

        logging.info("Saving data...")
        f = h5py.File(output_hdf5_fpath, 'w')
        f.create_dataset(
            "input_ids",
            data=features["input_ids"],
            dtype='i4', compression='gzip'
        )
        f.create_dataset(
            "input_mask",
            data=features["input_mask"],
            dtype='i1', compression='gzip'
        )
        f.create_dataset(
            "segment_ids",
            data=features["segment_ids"],
            dtype='i1', compression='gzip'
        )
        f.create_dataset(
            "masked_lm_positions",
            data=features["masked_lm_positions"],
            dtype='i4', compression='gzip'
        )
        f.create_dataset(
            "masked_lm_ids",
            data=features["masked_lm_ids"],
            dtype='i4', compression='gzip'
        )
        f.create_dataset(
            "next_sentence_labels",
            data=features["next_sentence_labels"],
            dtype='i1', compression='gzip'
        )
        f.flush()
        f.close()

        logging.info('END: Write Pre-Training Examples to hdf5 File')
