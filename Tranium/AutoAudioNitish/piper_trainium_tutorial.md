# Training Piper TTS on AWS Trainium

In this tutorial, we show how to fine-tune a Piper TTS model using PyTorch Lightning
and the AWS Neuron SDK. This example fine-tunes a Piper TTS model for text-to-speech
synthesis on a custom dataset with audio files and corresponding text.

## Table of Contents

- [Setup and compilation](#setup-and-compilation)
- [Dataset preparation](#dataset-preparation)
- [Single-worker training](#single-worker-training)
- [Multi-worker Training](#multi-worker-training)
- [Download pretrained checkpoint](#download-pretrained-checkpoint)
- [Model export](#model-export)
- [Monitoring Training](#monitoring-training)
- [Known issues and limitations](#known-issues-and-limitations)
- [Performance considerations](#performance-considerations)

## Setup and compilation

Before running the tutorial please follow the installation instructions at:

[Install PyTorch Neuron on Trn1](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx)

Please set the storage of instance to **512GB** or more if you want to train with large datasets.

For all the commands below, make sure you are in the virtual environment that you have created above before you run the commands:

```bash
source ~/aws_neuron_venv_piper/bin/activate
```

First we install the necessary packages and clone the Piper repository:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build espeak-ng

# Install PyTorch Neuron
pip install torch-neuronx torchvision --extra-index-url https://pip.repos.neuron.amazonaws.com

# Install Piper and dependencies
pip install piper-tts[train] librosa soundfile pandas

# Clone Piper repository
git clone https://github.com/OHF-voice/piper1-gpl.git
cd piper1-gpl
```

## Dataset preparation

Your dataset should be in CSV format with the following columns:
- `segment_index`: Integer index for audio file naming
- `text`: Text content for speech synthesis
- `word_count`: Number of words (optional)
- `estimated_duration_seconds`: Duration estimate (optional)

Audio files should be named as: `{prefix}_{segment_index:04d}.wav`

Example dataset format:

```csv
segment_index,text,word_count,estimated_duration_seconds
403,"the lever with the reward of escaping the box and getting to the food. After twenty to thirty trials, this behavior became so automatic and habitual that the cat could",30,12
```

Prepare your dataset for Piper training:

```bash
python prepare_dataset.py \
  --input_csv /path/to/your/dataset.csv \
  --audio_dir /path/to/audio/files \
  --output_csv /path/to/prepared_dataset.csv \
  --sample_rate 22050 \
  --audio_prefix "Atomic Habits"
```

## Single-worker training

We will run Piper TTS fine-tuning following the standard Piper training workflow
adapted for AWS Trainium.

We use BF16 precision with stochastic rounding enabled for best performance.
First, paste the following script into your terminal to create a "run_piper.sh" file:

```bash
#!/bin/bash
set -e

# Configuration
VOICE_NAME="my_voice"
CSV_PATH="/path/to/prepared_dataset.csv"
AUDIO_DIR="/path/to/audio/files"
CACHE_DIR="/path/to/cache"
OUTPUT_DIR="/path/to/checkpoints"
CONFIG_PATH="/path/to/voice_config.json"
BATCH_SIZE=16
MAX_EPOCHS=100

# Optional: Path to pretrained checkpoint for fine-tuning
CKPT_PATH="/path/to/pretrained_checkpoint.ckpt"

# Environment setup
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export NEURON_RT_NUM_CORES=2

# Create directories
mkdir -p $CACHE_DIR
mkdir -p $OUTPUT_DIR

echo "Starting Piper TTS training on AWS Trainium..."

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
```

We optionally precompile the model and training script using
[neuron_parallel_compile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile.html?highlight=neuron_parallel_compile)
to warm up the persistent graph cache (Neuron Cache) such that the actual run has fewer compilations (faster run time).

Precompilation is optional and only needs to be done once unless
hyperparameters such as batch size are modified. After the optional
precompilation, the actual run will be faster with minimal additional
compilations.

If precompilation was not done, the first execution of ./run_piper.sh will be
slower due to serial compilations. Rerunning the same script a second
time would show quicker execution as the compiled graphs will be already
cached in persistent cache.

Running the above script will run the Piper TTS fine-tuning on a single
process.

**Note:** Piper TTS uses VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture, which is different from transformer-based models. The training process involves both text-to-phoneme conversion and audio generation, making it suitable for high-quality speech synthesis.

## Multi-worker Training

The above script will run one worker on one NeuronCore. To run on
multiple cores, first modify the training script to support distributed training.

Then launch the training script with torchrun using
--nproc_per_node=N option to specify the number of workers (N=2 for
trn1.2xlarge, and N=2, 8, or 32 for trn1.32xlarge). The following
example runs 8 workers. Paste the following script into your terminal to
create a "run_piper_multi.sh" file:

```bash
#!/bin/bash
set -e

# Configuration
VOICE_NAME="my_voice"
CSV_PATH="/path/to/prepared_dataset.csv"
AUDIO_DIR="/path/to/audio/files"
CACHE_DIR="/path/to/cache"
OUTPUT_DIR="/path/to/checkpoints"
CONFIG_PATH="/path/to/voice_config.json"
BATCH_SIZE=8  # Smaller batch size per worker
MAX_EPOCHS=100
NUM_WORKERS=8  # Number of workers

# Optional: Path to pretrained checkpoint
CKPT_PATH="/path/to/pretrained_checkpoint.ckpt"

# Environment setup
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export NEURON_RT_NUM_CORES=$NUM_WORKERS

# Create directories
mkdir -p $CACHE_DIR
mkdir -p $OUTPUT_DIR

echo "Starting multi-worker Piper TTS training on AWS Trainium..."

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
```

Again, we optionally precompile the model and training script using
neuron_parallel_compile to warm up the persistent graph cache (Neuron
Cache), ignoring the results from this precompile run as it is only for
extracting and compiling the XLA graphs.

Precompilation is optional and only needs to be done once unless
hyperparameters such as batch size are modified. After the optional
precompilation, the actual run will be faster with minimal additional
compilations.

During run, you will notice that the "Total train batch size" is now
increased by the number of workers and the training steps are distributed
across workers. Also, if you open ``neuron-top`` in a separate terminal, 
you should see multiple cores being utilized.

## Download pretrained checkpoint

It is highly recommended to start with a pretrained checkpoint for faster convergence:

```bash
# Download a pretrained Piper checkpoint
python3 -m piper.download_voices en_US-lessac-medium
```

This will download the checkpoint files to the current directory. You can then
use the downloaded checkpoint as your starting point by setting the CKPT_PATH
variable in your training script.

## Model export

When your model is finished training, export it to ONNX format:

```bash
python3 -m piper.train.export_onnx \
  --checkpoint /path/to/best_checkpoint.ckpt \
  --output-file /path/to/your_voice.onnx
```

To make this compatible with other Piper voices, rename the files:
- `your_voice.onnx` → `en_US-yourvoice-medium.onnx`
- `voice_config.json` → `en_US-yourvoice-medium.onnx.json`

## Monitoring Training

### Tensorboard monitoring

In addition to the text-based job monitoring described in the previous section,
you can also use standard tools such as TensorBoard to monitor training job progress.
To view an ongoing training job in TensorBoard, you first need to identify the
experiment directory associated with your ongoing job.
This will typically be the most recently created directory under
your output directory. Once you have identified the directory, cd into it, and then launch TensorBoard:

```bash
cd /path/to/checkpoints
tensorboard --logdir ./
```

With TensorBoard running, you can then view the TensorBoard dashboard by browsing to
`http://localhost:6006` on your local machine. If you cannot access TensorBoard at this address,
please make sure that you have port-forwarded TCP port 6006 when SSH'ing into the instance,

```bash
ssh -i YOUR_KEY.pem ubuntu@INSTANCE_IP_ADDRESS -L 6006:127.0.0.1:6006
```

### neuron-top / neuron-monitor / neuron-ls

The [neuron-top](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-top-user-guide.html)
tool can be used to view useful information about NeuronCore utilization, vCPU and RAM utilization,
and loaded graphs on a per-node basis. To use neuron-top during an ongoing training job, run `neuron-top`:

```bash
neuron-top
```

Similarly, you can also use other Neuron tools such as
[neuron-monitor](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html)
and [neuron-ls](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html)
to capture performance and utilization statistics and to understand NeuronCore allocation.

## Known issues and limitations

The following are currently known issues:

-  Long compilation times: this can be alleviated with
   `neuron_parallel_compile` tool to extract graphs from a short trial run and
   compile them in parallel ahead of the actual run, as shown above.
- Audio preprocessing: Make sure your audio files are in the correct format and sample rate
- Memory usage: Piper TTS models can be memory-intensive; adjust batch size accordingly
- Checkpoint compatibility: Only medium quality checkpoints are supported without additional configuration

## PyTorch NeuronX Best Practices

Based on the official PyTorch NeuronX Developer Guide, follow these best practices:

### **Lazy Mode Understanding**
- PyTorch NeuronX runs in lazy mode - tensors are placeholders until `xm.mark_step()` is called
- Minimize compilation-and-execution cycles for best performance
- Use `xm.add_step_closure()` to wrap tensor printing/logging to avoid unnecessary compilations

### **Device Management**
```python
import torch_xla.core.xla_model as xm

# Get device
device = xm.xla_device()
# or
device = 'xla'

# Check device kind
devkind = xm.xla_device_kind()  # Returns NC_v2 for Trainium1
print(f"Device kind: {devkind}")

# Check number of devices
devices = xm.get_xla_supported_devices()
print(f"Available devices: {len(devices)}")
```

### **Model Saving**
Use PyTorch/XLA's save function for proper checkpointing:
```python
# For single device
xm.save(model.state_dict(), checkpoint_path)

# For multi-device (to avoid high host memory usage)
from torch_xla.utils.serialization import save, load
save(model.state_dict(), checkpoint_path)
```

## Performance considerations

To achieve better performance, consider the following:

**Batch Size Optimization:**
- Start with smaller batch sizes (8-16) and increase if memory allows
- Monitor memory usage with neuron-top during training

**Audio Preprocessing:**
- Ensure consistent sample rate (22050 Hz recommended)
- Pre-process audio files to remove silence and normalize volume

**Checkpoint Usage:**
- Always use pretrained checkpoints for faster convergence
- Fine-tuning from existing checkpoints is much faster than training from scratch

**Data Loading:**
- Use appropriate number of workers for data loading
- Cache preprocessed audio files to speed up subsequent training runs
