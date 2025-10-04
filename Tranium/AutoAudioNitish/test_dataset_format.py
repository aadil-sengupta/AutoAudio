#!/usr/bin/env python3
"""
Test script to verify dataset preparation for your specific format
"""

import pandas as pd
import os

def test_dataset_format():
    """Test the dataset format with your sample data"""
    
    # Read your test CSV
    df = pd.read_csv('test.csv')
    print("Your dataset format:")
    print(df.head())
    print(f"Columns: {df.columns.tolist()}")
    
    # Test the audio file naming logic
    for idx, row in df.iterrows():
        segment_index = row['segment_index']
        text = row['text']
        
        # Construct audio filename
        audio_file = f"Atomic Habits_{segment_index:04d}.wav"
        print(f"\nRow {idx}:")
        print(f"  Segment Index: {segment_index}")
        print(f"  Audio File: {audio_file}")
        print(f"  Text: {text[:50]}...")
        
        # Check if audio file exists
        if os.path.exists(audio_file):
            print(f"  ✓ Audio file found: {audio_file}")
        else:
            print(f"  ✗ Audio file not found: {audio_file}")
    
    # Show what the Piper format will look like
    print("\n" + "="*50)
    print("Piper format preview:")
    print("="*50)
    
    for idx, row in df.iterrows():
        segment_index = row['segment_index']
        text = row['text']
        audio_file = f"Atomic Habits_{segment_index:04d}.wav"
        
        piper_line = f"{audio_file}|{text}"
        print(piper_line)

if __name__ == '__main__':
    test_dataset_format()
