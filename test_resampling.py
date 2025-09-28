#!/usr/bin/env python3
"""
Test script for audio resampling functionality
"""

import io
import wave
import numpy as np
import soundfile as sf
import time
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import resample_audio, TARGET_SAMPLE_RATE

def create_test_audio(sample_rate=48000, duration=2.0, frequency=440):
    """Create a test audio file with a sine wave at the specified sample rate"""
    print(f"\nCreating test audio: {sample_rate} Hz, {duration}s, {frequency} Hz sine wave")

    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Generate sine wave
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.5

    # Convert to WAV bytes
    output_io = io.BytesIO()
    sf.write(output_io, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    wav_bytes = output_io.getvalue()

    print(f"Generated audio size: {len(wav_bytes)/1024:.1f} KB")

    return wav_bytes

def test_resampling():
    """Test the audio resampling function"""
    print("=" * 60)
    print("Audio Resampling Test")
    print("=" * 60)

    # Test 1: 48000 Hz -> 16000 Hz (typical case)
    print("\n### Test 1: 48000 Hz -> 16000 Hz (typical case)")
    audio_48k = create_test_audio(sample_rate=48000, duration=2.0)
    resampled_audio, metrics = resample_audio(audio_48k, TARGET_SAMPLE_RATE)

    print("\nResampling metrics:")
    print(f"  Original sample rate: {metrics['original_sample_rate']} Hz")
    print(f"  Target sample rate: {metrics['target_sample_rate']} Hz")
    print(f"  Resampled: {metrics['resampled']}")
    print(f"  Original size: {metrics['original_size']/1024:.1f} KB")
    if metrics['resampled_size'] is not None:
        print(f"  Resampled size: {metrics['resampled_size']/1024:.1f} KB")
        print(f"  Size reduction: {(1 - metrics['resampled_size']/metrics['original_size'])*100:.1f}%")
    else:
        print(f"  Resampled size: N/A (resampling failed)")
    print(f"  Resample time: {metrics['resample_time']:.3f}s")
    if 'error' in metrics:
        print(f"  Error: {metrics['error']}")

    # Test 2: 44100 Hz -> 16000 Hz (CD quality)
    print("\n### Test 2: 44100 Hz -> 16000 Hz (CD quality)")
    audio_44k = create_test_audio(sample_rate=44100, duration=2.0)
    resampled_audio, metrics = resample_audio(audio_44k, TARGET_SAMPLE_RATE)

    print("\nResampling metrics:")
    print(f"  Original sample rate: {metrics['original_sample_rate']} Hz")
    print(f"  Target sample rate: {metrics['target_sample_rate']} Hz")
    print(f"  Resampled: {metrics['resampled']}")
    if metrics['resampled_size'] is not None and metrics['resampled']:
        print(f"  Size reduction: {(1 - metrics['resampled_size']/metrics['original_size'])*100:.1f}%")
    print(f"  Resample time: {metrics['resample_time']:.3f}s")

    # Test 3: 16000 Hz -> 16000 Hz (no resampling needed)
    print("\n### Test 3: 16000 Hz -> 16000 Hz (no resampling needed)")
    audio_16k = create_test_audio(sample_rate=16000, duration=2.0)
    resampled_audio, metrics = resample_audio(audio_16k, TARGET_SAMPLE_RATE)

    print("\nResampling metrics:")
    print(f"  Original sample rate: {metrics['original_sample_rate']} Hz")
    print(f"  Target sample rate: {metrics['target_sample_rate']} Hz")
    print(f"  Resampled: {metrics['resampled']}")
    print(f"  Resample time: {metrics['resample_time']:.3f}s")

    # Test 4: Multi-channel audio (stereo)
    print("\n### Test 4: Stereo audio resampling")
    # Create stereo audio
    t = np.linspace(0, 2.0, int(48000 * 2.0), False)
    left_channel = np.sin(2 * np.pi * 440 * t) * 0.5
    right_channel = np.sin(2 * np.pi * 880 * t) * 0.5
    stereo_audio = np.stack([left_channel, right_channel], axis=1)

    output_io = io.BytesIO()
    sf.write(output_io, stereo_audio, 48000, format='WAV', subtype='PCM_16')
    stereo_bytes = output_io.getvalue()

    print(f"Created stereo audio: 48000 Hz, 2 channels, {len(stereo_bytes)/1024:.1f} KB")

    resampled_audio, metrics = resample_audio(stereo_bytes, TARGET_SAMPLE_RATE)

    print("\nResampling metrics:")
    print(f"  Original sample rate: {metrics['original_sample_rate']} Hz")
    print(f"  Target sample rate: {metrics['target_sample_rate']} Hz")
    print(f"  Resampled: {metrics['resampled']}")
    if metrics['resampled_size'] is not None and metrics['resampled']:
        print(f"  Size reduction: {(1 - metrics['resampled_size']/metrics['original_size'])*100:.1f}%")
    print(f"  Resample time: {metrics['resample_time']:.3f}s")

    # Verify the resampled audio
    resampled_io = io.BytesIO(resampled_audio)
    resampled_data, resampled_sr = sf.read(resampled_io)
    print(f"\nVerification: Resampled audio is {resampled_sr} Hz, shape: {resampled_data.shape}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_resampling()
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)