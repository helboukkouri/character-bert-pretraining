# Pre-training CharacterBERT (and BERT)

This is a repository for pre-training **BERT** and **CharacterBERT**.
> DISCLAIMER: The code was largely adapted from an older version of
[NVIDIA's repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
for benchmarking the pre-training of BERT using Automatic Mixed Precision.
The original code was tweaked to include **CharacterBERT** and other minor elements.

## Python Environment

First of all, we will need to have a Python environment with the required packages
installed.
> NOTE: This was tested with NVIDIA V100 (16GB/32GB) GPUs and a cuda 10.2 installation.

```bash
# Create a Python 3.8 environment via conda 
conda create --name pretraining python=3.8 -y
conda activate pretraining

# For showing progress bars
pip install tqdm

# For extracting raw text from Wikipedia dumps
cd external/
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor/
git checkout 6490f5361d7658208ad7f8e5deeb56ee0fe9e02f
cd ../..

# For parsing Wikiextractor outputs to get Wikipedia text
pip install beautifulsoup4

# For segmenting documents into sentences
pip install nltk
python -c "import nltk;nltk.download('punkt')"

# For saving pre-training data into .hdf5 files
pip install h5py

# For basic tokenization and BERT/CharacterBERT models in PyTorch
cd external/
git clone https://github.com/helboukkouri/transformers.git
cd transformers/
git checkout add-character-bert
pip install -e .
cd ../..

# Actual PyTorch installation
conda install pytorch=1.7.1 cudatoolkit=10.2 -c pytorch -y

# For monitoring training progress
pip install tensorboard

# For using Automatic Mixed Precision (speeds up training) 
# NOTE: this will require some space in /tmp/ during compilation
cd external/
git clone https://github.com/NVIDIA/apex.git
cd apex/
git checkout a78ccf0b3e3f7130b3f157732dc8e8e651389922
pip install \
    -v --disable-pip-version-check --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../..

# Configuration and vocabulary files for BERT (base, uncased)
mkdir ./data/bert-base-uncased/
cd ./data/bert-base-uncased/
wget https://huggingface.co/bert-base-uncased/resolve/main/config.json
wget https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
cd ../..

# Configuration file for CharacterBERT (base, uncased)
mkdir ./data/character-bert/
cd ./data/character-bert/
wget https://huggingface.co/helboukkouri/character-bert/resolve/main/config.json
cd ../..
```

Now we can move on to the corpus preparation step.

## Corpus Preparation

### Downloading the corpus

First, we will need to get a corpus of texts. Let's download and use the
2021-01-01 dump of English Wikipedia.

Assuming that:

- the environment variable `$WORKDIR` contains a path to this repository

- we have already activated our conda environment using: `conda activate pretraining`

We can run this command to download the wikipedia dump

```bash
WORKDIR=$WORKDIR \
    python download_wikipedia.py --language='en'
```

If you don't want to download the whole dump you can experiment with
a sample first by adding the `--debug` flag.

```bash
WORKDIR=$WORKDIR \
    python download_wikipedia.py --language='en' --debug
```

> NOTE: if you use `--debug` then you will need to adapt all the commands
by replacing the corpus name `wikipedia_en` with `wikipedia_sample`.

### Extraction, Tokenization & Formatting

Since Wikipedia dumps come as a large .xml archives, we need to extract the
process the file we just downloaded into a single .txt file with raw text.
Then, we will need to format it in a specific way (one sentence per
line and a blank line between sentences from different Wikipedia articles)
to allow for generating examples for the Next Sentence Prediction task (NSP).
We also tokenize each sentence to be able to easily generate Masked Language Modeling
examples as well in future steps.

The following command runs both the extraction and formatting steps then removes
the extracted document-level corpus only keeps the formatted sentence-level we need.

> NOTE: you can remove the `--delete_document_level_corpus` flag to keep the original
extracted corpus.

```bash
WORKDIR=$WORKDIR \
    python format_wikipedia.py \
        --archive_path=$WORKDIR/data/downloaded/wikipedia_en/wikipedia_en.xml.bz2 \
        --delete_document_level_corpus
```

After running the command above, you should be able to find a single text file
in `data/formatted/wikipedia_en/`.

> NOTE 1: if you want to use a corpus other than Wikipedia then add a component in `utils/format`
and re-use the formatting step from `format_wikipedia.py`.

> NOTE 2: if you want to process the whole English Wikipedia then run the command
and forget about it, it will take a while... 😴 This process can probably be better optimized.

### Pre-training Data generation

Now that we have our formatted Wikipedia corpus, there is one more step before we
can actually run the pre-training. In fact, in order to pre-train models in a reasonable
amount of time, we will be relying on multiple GPUs,
[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
and the `torch.distributed` module. This will effectively have a copy of the model on
each GPU, so we will need to split our corpus into shards (or chunks) so that each GPUs
can process its own shard while others do the same.

First, we will split the corpus into 4096 training and 16 validation shards of equal size.

> NOTE: you can change the amount of shards as you like. These numbers were chosen so
to keep the memory requirement for loading a single shard relatively low as well as
to keep validation steps (a single epoch through the validation shards) relatively fast.

```bash
WORKDIR=$WORKDIR \
    python make_shards.py \
        --formatted_corpus_path=$WORKDIR/data/formatted/wikipedia_en/wikipedia_en.formatted.txt \
        --n_training_shards=4096 \
        --n_validation_shards=16 \
        --random_seed=42
```

The command above creates multiple training{n}.txt and validation.{n}.txt files in
`data/shards/wikipedia_en/`.

The second and last step is to convert each shard into and .hdf5 file containing
the actual pre-training data. But before doing that, we need to define a
vocabulary for the Masked Language Modelling if we want to pre-train CharacterBERT.

> NOTE 1: In fact, when pre-training BERT, since all original tokens from the corpus
are split into elements of the WordPiece vocabulary, we can simply re-use
this same vocabulary as a target space for MLM. And since CharacterBERT
does not rely on WordPieces, it cannot do that and requires a fresh token vocabulary
for Masked Language Modeling.

> NOTE 2: Be careful when re-training a model from a domain A on texts from a domain B
as this will require a new MLM vocabulary (for B) which in turn would require replacing
the output layer in prior checkpoints (from A) before resuming pre-training. As for this
version of the code, you will need to do that manually before running the pre-training.

```bash
WORKDIR=$WORKDIR \
    python build_mlm_vocabulary.py \
        --formatted_corpus_path=$WORKDIR/data/formatted/wikipedia_en/wikipedia_en.formatted.txt
```

After building a MLM vocabulary (in cases where we want to pre-train CharacterBERT)
we can now generate the data for both phases 1 and 2 of the pre-training process.

- phase 1: maximum input length of 128 and maximum number of masked tokens per input of 20.

```bash
WORKDIR=$WORKDIR \
    python make_hdf5.py \
        --shards_path=$WORKDIR/data/shards/wikipedia_en/ \
        --output_directory=$WORKDIR/data/hdf5/wikipedia_en/character_bert/128_20/ \
        --max_input_length=128 \
        --max_masked_tokens_per_input=20 \
        --is_character_bert  # change this accordingly
```

- phase 2: maximum input length of 512 and maximum number of masked tokens per input of 80.

```bash
WORKDIR=$WORKDIR \
    python make_hdf5.py \
        --shards_path=$WORKDIR/data/shards/wikipedia_en/ \
        --output_directory=$WORKDIR/data/hdf5/wikipedia_en/character_bert/512_80/ \
        --max_input_length=512 \
        --max_masked_tokens_per_input=80 \
        --is_character_bert  # change this accordingly
```

> NOTE: if you want to generate data for BERT instead of CharacterBERT,
remove the `--is_character_bert` flag and adapt the `output_directory` path.

## Pre-training

### Launching the pre-training

At this point we are all set to start pre-training models. For that, we can
simply run the following bash scripts.

> NOTE: you may need to change the value of WORKDIR in the pre-training scripts.

- For phase 1:

```bash
bash $WORKDIR/bash_scripts/run_pretraining.character_bert.step_1.sh
```

or

```bash
bash $WORKDIR/bash_scripts/run_pretraining.bert.step_1.sh
```

- For phase 2:

```bash
bash $WORKDIR/bash_scripts/run_pretraining.character_bert.step_2.sh
```

or

```bash
bash $WORKDIR/bash_scripts/run_pretraining.bert.step_2.sh
```

> NOTE 1: you should change the NUM_GPUs variable inside the bash scripts
to match the number of GPUs on your machine. The parallelization will be handled automatically.

> NOTE 2: the bash scripts support distributed training on multiple gpus within
a single node. Bash scripts that can run on multiple nodes with `sbatch` will be available soon.

Running the bash scripts on large enough corpora should produce good results.
However, if you want to customize the pre-training process, you could change
any of the parameters from `pretrain_model.py`.

For a complete list of parameters, run the following command.

```bash
WORKDIR=$WORKDIR python pretrain_model.py --help
```

In particular, if you don't want to run a validation step before each checkpoint
you can remove the `--do_validation` flag. This will make the pre-training process
faster but as a result you will not be able to tell if the language models are
overfitting the training data.

### Monitoring the pre-training

While the models are pre-training you can monitor the average training and
validation losses (MLM + NSP loss) using TensorBoard.

```bash
tensorboard --logdir=$WORKDIR/.tensorboard_logs/
```

## References

Please cite [our paper](https://www.aclweb.org/anthology/2020.coling-main.609/) if you use CharacterBERT in your work.

```text
@inproceedings{el-boukkouri-etal-2020-characterbert,
    title = "{C}haracter{BERT}: Reconciling {ELM}o and {BERT} for Word-Level Open-Vocabulary Representations From Characters",
    author = "El Boukkouri, Hicham  and
      Ferret, Olivier  and
      Lavergne, Thomas  and
      Noji, Hiroshi  and
      Zweigenbaum, Pierre  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.609",
    doi = "10.18653/v1/2020.coling-main.609",
    pages = "6903--6915",
    abstract = "Due to the compelling improvements brought by BERT, many recent representation models adopted the Transformer architecture as their main building block, consequently inheriting the wordpiece tokenization system despite it not being intrinsically linked to the notion of Transformers. While this system is thought to achieve a good balance between the flexibility of characters and the efficiency of full words, using predefined wordpiece vocabularies from the general domain is not always suitable, especially when building models for specialized domains (e.g., the medical domain). Moreover, adopting a wordpiece tokenization shifts the focus from the word level to the subword level, making the models conceptually more complex and arguably less convenient in practice. For these reasons, we propose CharacterBERT, a new variant of BERT that drops the wordpiece system altogether and uses a Character-CNN module instead to represent entire words by consulting their characters. We show that this new model improves the performance of BERT on a variety of medical domain tasks while at the same time producing robust, word-level, and open-vocabulary representations.",
}
```
