#!/usr/bin/env python3
"""
Simple test to verify both backends work and compare basic performance
"""

import io
import time
import numpy as np
import soundfile as sf
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set FFmpeg as default
os.environ['AUDIO_RESAMPLING_BACKEND'] = 'ffmpeg'

import audio_config
from audio_processor import resample_audio_ffmpeg, resample_audio_librosa, resample_audio_auto

def create_test_audio(duration=5.0):
    """Create 5-second test audio"""
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    left = np.sin(2 * np.pi * 440 * t) * 0.5
    right = np.sin(2 * np.pi * 880 * t) * 0.5
    audio_data = np.stack([left, right], axis=1)

    output_io = io.BytesIO()
    sf.write(output_io, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    return output_io.getvalue()

def test_backend(name, func, audio_data):
    """Test a single backend"""
    print(f"\nTesting {name}...")
    start = time.time()
    try:
        resampled, metrics = func(audio_data, target_sr=16000, force_mono=True)
        elapsed = time.time() - start

        print(f"  ✓ Success in {elapsed:.3f}s")
        print(f"  Input:  {len(audio_data)/1024:.1f} KB, {metrics.get('original_sample_rate')} Hz, {metrics.get('original_channels')} ch")
        print(f"  Output: {len(resampled)/1024:.1f} KB, 16000 Hz, 1 ch (mono)")
        print(f"  Size reduction: {(1 - len(resampled)/len(audio_data))*100:.1f}%")
        return elapsed
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None

def main():
    print("="*60)
    print("AUDIO BACKEND TEST")
    print("="*60)

    # Create 5-second test audio
    print("Creating 5-second 48kHz stereo test audio...")
    test_audio = create_test_audio(5.0)
    print(f"Test audio size: {len(test_audio)/1024:.1f} KB")

    # Test FFmpeg
    ffmpeg_time = test_backend("FFmpeg", resample_audio_ffmpeg, test_audio)

    # Test Librosa
    librosa_time = test_backend("Librosa", resample_audio_librosa, test_audio)

    # Test Auto (should use FFmpeg by default)
    print("\nTesting Auto mode (configured backend)...")
    print(f"Current config: {audio_config.RESAMPLING_BACKEND}")
    auto_time = test_backend("Auto", resample_audio_auto, test_audio)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if ffmpeg_time:
        print(f"FFmpeg:  {ffmpeg_time:.3f}s {'✅' if ffmpeg_time else '❌'}")
    else:
        print(f"FFmpeg:  Failed ❌")

    if librosa_time:
        print(f"Librosa: {librosa_time:.3f}s {'✅' if librosa_time else '❌'}")
    else:
        print(f"Librosa: Failed ❌")

    if ffmpeg_time and librosa_time:
        if ffmpeg_time < librosa_time:
            ratio = librosa_time / ffmpeg_time
            print(f"\n🚀 FFmpeg is {ratio:.1f}x faster than Librosa")
        else:
            ratio = ffmpeg_time / librosa_time
            print(f"\n🚀 Librosa is {ratio:.1f}x faster than FFmpeg")

    print(f"\n✅ Default backend is set to: {audio_config.RESAMPLING_BACKEND}")
    print("   To change, set AUDIO_RESAMPLING_BACKEND environment variable")

if __name__ == "__main__":
    main()