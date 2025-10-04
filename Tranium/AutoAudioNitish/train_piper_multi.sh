#!/bin/bash
# Multi-worker Piper TTS Training for AWS Trainium
# For trn1.32xlarge instances with multiple NeuronCores

set -e

# Configuration
VOICE_NAME="my_voice"
CSV_PATH="/path/to/your/metadata.csv"
AUDIO_DIR="/path/to/your/audio/files"
CACHE_DIR="/path/to/cache"
OUTPUT_DIR="/path/to/checkpoints"
CONFIG_PATH="/path/to/voice_config.json"
BATCH_SIZE=8  # Smaller batch size per worker
MAX_EPOCHS=100
NUM_WORKERS=8  # Number of workers (adjust based on instance)

# Optional: Path to pretrained checkpoint
CKPT_PATH="/path/to/pretrained_checkpoint.ckpt"

# Environment setup
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export NEURON_RT_NUM_CORES=$NUM_WORKERS

# Create directories
mkdir -p $CACHE_DIR
mkdir -p $OUTPUT_DIR

echo "Starting multi-worker Piper TTS training on AWS Trainium..."
echo "Workers: $NUM_WORKERS"
echo "Batch Size per Worker: $BATCH_SIZE"

# Pre-compilation step
echo "Starting pre-compilation..."
export COMPILE=1
torchrun --nproc_per_node=$NUM_WORKERS train_piper_trainium.py \
    --voice_name "$VOICE_NAME" \
    --csv_path "$CSV_PATH" \
    --audio_dir "$AUDIO_DIR" \
    --espeak_voice "en-us" \
    --cache_dir "$CACHE_DIR" \
    --config_path "$CONFIG_PATH" \
    --sample_rate 22050 \
    --batch_size $BATCH_SIZE \
    --num_workers 2 \
    --max_epochs 1 \
    --output_dir "$OUTPUT_DIR" \
    ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"}

echo "Pre-compilation completed!"

# Actual training
echo "Starting actual training..."
export COMPILE=0
torchrun --nproc_per_node=$NUM_WORKERS train_piper_trainium.py \
    --voice_name "$VOICE_NAME" \
    --csv_path "$CSV_PATH" \
    --audio_dir "$AUDIO_DIR" \
    --espeak_voice "en-us" \
    --cache_dir "$CACHE_DIR" \
    --config_path "$CONFIG_PATH" \
    --sample_rate 22050 \
    --batch_size $BATCH_SIZE \
    --num_workers 2 \
    --max_epochs $MAX_EPOCHS \
    --output_dir "$OUTPUT_DIR" \
    ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"}

echo "Multi-worker training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
