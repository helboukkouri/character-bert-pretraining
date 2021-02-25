#!/bin/bash

# Load environment
module purge
module load cuda/10.2

# NOTE: You may want to adapt this part according to your env. settings
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate pretraining

set -v

# NOTE: Change the path below to point to the characterbert-pretraining repo
export WORKDIR="$WORK/temporary/character-bert-pretraining"

# Number of available GPUs (change accordingly)
export NUM_GPUs=1

# Pre-training arguments
CORPUS="wikipedia_en"
MODEL_TYPE="character_bert"
export INPUT_DIR="$WORKDIR/data/hdf5/$CORPUS/$MODEL_TYPE/512_80/"
export OUTPUT_DIR="$WORKDIR/output_directory/$CORPUS/$MODEL_TYPE/"
export EXPERIMENT_NAME=$MODEL_TYPE"_512_80"
export TOTAL_STEPS_PHASE1=7038

# Actual python command (STEP 2: 512 / 80)
CMD="$WORKDIR/pretrain_model.py"
CMD+=" --hdf5_directory=$INPUT_DIR"
CMD+=" --output_directory=$OUTPUT_DIR"
CMD+=" --tensorboard_id=$EXPERIMENT_NAME"
CMD+=" --max_input_length=512"
CMD+=" --max_masked_tokens_per_input=80"
CMD+=" --target_batch_size=4096"
CMD+=" --num_accumulation_steps=2048"
CMD+=" --learning_rate=4e-3"
CMD+=" --warmup_proportion=0.128"
CMD+=" --total_steps=1563"
CMD+=" --fp16"
CMD+=" --allreduce_post_accumulation"
CMD+=" --allreduce_post_accumulation_fp16"
CMD+=" --is_character_bert"
CMD+=" --do_validation"
CMD+=" --random_seed=42"
CMD+=" --resume_pretraining --phase2 --phase1_end_step=$TOTAL_STEPS_PHASE1"

# Distributed launch (Single node, multiple GPUs)
python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUs \
  $CMD
