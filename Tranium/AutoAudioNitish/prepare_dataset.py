#!/usr/bin/env python3
"""
Dataset preparation script for Piper TTS training on AWS Trainium
Converts your dataset to the required format
"""

import os
import pandas as pd
import librosa
import argparse
from pathlib import Path

def prepare_dataset(input_csv, audio_dir, output_csv, target_sample_rate=22050, audio_prefix="Atomic Habits"):
    """
    Prepare dataset for Piper training
    
    Args:
        input_csv: Path to your original dataset CSV
        audio_dir: Directory containing audio files
        output_csv: Path to save the prepared CSV
        target_sample_rate: Target sample rate for audio files
        audio_prefix: Prefix for audio files (e.g., "Atomic Habits")
    """
    
    print(f"Preparing dataset from {input_csv}")
    print(f"Audio directory: {audio_dir}")
    print(f"Target sample rate: {target_sample_rate}")
    
    # Read your dataset
    df = pd.read_csv(input_csv)
    
    # Prepare output data
    prepared_data = []
    
    for idx, row in df.iterrows():
        # Handle your specific dataset format
        segment_index = row['segment_index']
        text = row['text']
        
        # Construct audio filename based on segment_index
        # Format: "Atomic Habits_0403.wav" -> segment_index 403
        audio_file = f"{audio_prefix}_{segment_index:04d}.wav"
        audio_path = os.path.join(audio_dir, audio_file)
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
            
        # Validate audio file
        try:
            # Load audio to check if it's valid
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Resample if necessary
            if sr != target_sample_rate:
                print(f"Resampling {audio_file} from {sr}Hz to {target_sample_rate}Hz")
                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
                # Save resampled audio
                librosa.output.write_wav(audio_path, audio_resampled, target_sample_rate)
            
            # Add to prepared data
            prepared_data.append({
                'audio_file': audio_file,
                'text': text.strip()
            })
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    # Create output CSV in Piper format
    output_df = pd.DataFrame(prepared_data)
    output_df['combined'] = output_df['audio_file'] + '|' + output_df['text']
    
    # Save in Piper format (audio_file|text)
    with open(output_csv, 'w', encoding='utf-8') as f:
        for _, row in output_df.iterrows():
            f.write(row['combined'] + '\n')
    
    print(f"Dataset preparation completed!")
    print(f"Total samples: {len(prepared_data)}")
    print(f"Output saved to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for Piper TTS training')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input dataset CSV')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save prepared CSV')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Target sample rate')
    parser.add_argument('--audio_prefix', type=str, default='Atomic Habits', help='Prefix for audio files')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if not os.path.exists(args.audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {args.audio_dir}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare dataset
    prepare_dataset(args.input_csv, args.audio_dir, args.output_csv, args.sample_rate, args.audio_prefix)

if __name__ == '__main__':
    main()
