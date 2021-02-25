#!/bin/bash

# If you're in a cluster with modules
# # Load environment
# module purge
# module load cuda/10.2

# # NOTE: You may want to adapt this part according to your env. settings
# source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate pretraining

set -v

# NOTE: Change the path below so that it points to the root of the repository
export WORKDIR="$WORK/character-bert-pretraining"

# Number of available GPUs (change accordingly)
export NUM_GPUs=1

# Pre-training arguments
CORPUS="wikipedia_en"
MODEL_TYPE="bert"
export INPUT_DIR="$WORKDIR/data/hdf5/$CORPUS/$MODEL_TYPE/128_20/"
export OUTPUT_DIR="$WORKDIR/output_directory/$CORPUS/$MODEL_TYPE/"
export EXPERIMENT_NAME=$MODEL_TYPE"_128_20"

# Actual python command (STEP 1: 128 / 20)
CMD="$WORKDIR/pretrain_model.py"
CMD+=" --hdf5_directory=$INPUT_DIR"
CMD+=" --output_directory=$OUTPUT_DIR"
CMD+=" --tensorboard_id=$EXPERIMENT_NAME"
CMD+=" --max_input_length=128"
CMD+=" --max_masked_tokens_per_input=20"
CMD+=" --target_batch_size=8192"
CMD+=" --num_accumulation_steps=512"
CMD+=" --learning_rate=6e-3"
CMD+=" --warmup_proportion=0.2843"
CMD+=" --total_steps=7038"
CMD+=" --fp16"
CMD+=" --allreduce_post_accumulation"
CMD+=" --allreduce_post_accumulation_fp16"
CMD+=" --do_validation"
CMD+=" --random_seed=42"

# Distributed launch (Single node, multiple GPUs)
python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUs \
  $CMD
