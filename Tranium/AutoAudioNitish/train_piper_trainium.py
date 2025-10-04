#!/usr/bin/env python3
"""
Piper TTS Training Script for AWS Trainium
Adapted from original Piper training to work with PyTorch Neuron
"""

import os
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from piper.train import fit as piper_fit
from piper.train.data import PiperDataModule
from piper.train.model import PiperModel
import argparse

def setup_trainium_environment():
    """Setup environment variables for Trainium training"""
    # Enable stochastic rounding for BF16 (recommended for faster convergence)
    os.environ['NEURON_RT_STOCHASTIC_ROUNDING_EN'] = '1'
    # Get number of available devices
    devices = torch_xla.core.xla_model.get_xla_supported_devices()
    print(f"Available XLA devices: {len(devices)}")
    
def create_trainium_data_module(args):
    """Create data module compatible with Trainium"""
    data_module = PiperDataModule(
        voice_name=args.voice_name,
        csv_path=args.csv_path,
        audio_dir=args.audio_dir,
        espeak_voice=args.espeak_voice,
        cache_dir=args.cache_dir,
        config_path=args.config_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    return data_module

def create_trainium_model(args):
    """Create model compatible with Trainium"""
    model = PiperModel(
        sample_rate=args.sample_rate,
        checkpoint_path=args.ckpt_path
    )
    return model

def train_piper_on_trainium(args):
    """Main training function for Trainium"""
    
    # Setup Trainium environment
    setup_trainium_environment()
    
    # Get device
    device = xm.xla_device()
    print(f"Training on device: {device}")
    
    # Create data module
    data_module = create_trainium_data_module(args)
    
    # Create model
    model = create_trainium_model(args)
    model = model.to(device)
    
    # Convert model to BF16 for better performance
    model = model.to(torch.bfloat16)
    
    # Setup trainer with Trainium-specific configurations
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='piper-trainium-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=1,  # Single device for now
        accelerator='xla',
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        val_check_interval=0.25
    )
    
    # Start training
    print("Starting Piper training on Trainium...")
    trainer.fit(model, data_module)
    
    print(f"Training completed! Checkpoints saved to: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train Piper TTS on AWS Trainium')
    
    # Data arguments
    parser.add_argument('--voice_name', type=str, required=True, help='Name of the voice')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with audio|text pairs')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--espeak_voice', type=str, default='en-us', help='espeak voice name')
    parser.add_argument('--cache_dir', type=str, required=True, help='Cache directory for training artifacts')
    parser.add_argument('--config_path', type=str, required=True, help='Path to write voice config JSON')
    
    # Model arguments
    parser.add_argument('--sample_rate', type=int, default=22050, help='Audio sample rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--ckpt_path', type=str, help='Path to pretrained checkpoint (optional)')
    
    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    if not os.path.exists(args.audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {args.audio_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Start training
    train_piper_on_trainium(args)

if __name__ == '__main__':
    main()
