#!/bin/bash
# Piper TTS Training Script for AWS Trainium
# Based on AWS Trainium training patterns

set -e

# Configuration
VOICE_NAME="my_voice"
CSV_PATH="/path/to/your/metadata.csv"
AUDIO_DIR="/path/to/your/audio/files"
CACHE_DIR="/path/to/cache"
OUTPUT_DIR="/path/to/checkpoints"
CONFIG_PATH="/path/to/voice_config.json"
BATCH_SIZE=16
MAX_EPOCHS=100

# Optional: Path to pretrained checkpoint for fine-tuning
CKPT_PATH="/path/to/pretrained_checkpoint.ckpt"

# Environment setup
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export NEURON_RT_NUM_CORES=2  # Adjust based on your instance

# Create directories
mkdir -p $CACHE_DIR
mkdir -p $OUTPUT_DIR

echo "Starting Piper TTS training on AWS Trainium..."
echo "Voice: $VOICE_NAME"
echo "CSV: $CSV_PATH"
echo "Audio Dir: $AUDIO_DIR"
echo "Batch Size: $BATCH_SIZE"

# Pre-compilation step (optional but recommended)
echo "Starting pre-compilation..."
export COMPILE=1
python train_piper_trainium.py \
    --voice_name "$VOICE_NAME" \
    --csv_path "$CSV_PATH" \
    --audio_dir "$AUDIO_DIR" \
    --espeak_voice "en-us" \
    --cache_dir "$CACHE_DIR" \
    --config_path "$CONFIG_PATH" \
    --sample_rate 22050 \
    --batch_size $BATCH_SIZE \
    --num_workers 4 \
    --max_epochs 1 \
    --output_dir "$OUTPUT_DIR" \
    ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"}

echo "Pre-compilation completed!"

# Actual training
echo "Starting actual training..."
export COMPILE=0
python train_piper_trainium.py \
    --voice_name "$VOICE_NAME" \
    --csv_path "$CSV_PATH" \
    --audio_dir "$AUDIO_DIR" \
    --espeak_voice "en-us" \
    --cache_dir "$CACHE_DIR" \
    --config_path "$CONFIG_PATH" \
    --sample_rate 22050 \
    --batch_size $BATCH_SIZE \
    --num_workers 4 \
    --max_epochs $MAX_EPOCHS \
    --output_dir "$OUTPUT_DIR" \
    ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"}

echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
