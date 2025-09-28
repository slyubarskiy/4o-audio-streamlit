#!/usr/bin/env python3
"""
Quick benchmark for 1-minute 48kHz stereo audio
"""

import io
import time
import numpy as np
import soundfile as sf
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_config
from audio_processor import resample_audio_ffmpeg, resample_audio_librosa

def create_test_audio(sample_rate=48000, duration=60.0, channels=2):
    """Create test audio"""
    print(f"Creating test audio: {sample_rate} Hz, {duration}s, {channels} channels")
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    if channels == 1:
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.5
    else:
        left = np.sin(2 * np.pi * 440 * t) * 0.5
        right = np.sin(2 * np.pi * 880 * t) * 0.5
        audio_data = np.stack([left, right], axis=1)

    output_io = io.BytesIO()
    sf.write(output_io, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    wav_bytes = output_io.getvalue()

    print(f"Generated audio size: {len(wav_bytes)/1024/1024:.2f} MB")
    return wav_bytes

def benchmark_backend(name, func, audio_data):
    """Benchmark a backend with 3 runs"""
    print(f"\nBenchmarking {name}...")
    times = []

    # Warm-up
    try:
        func(audio_data, target_sr=16000, force_mono=True)
    except:
        pass

    for i in range(3):
        gc.collect()
        start = time.time()
        try:
            resampled, metrics = func(audio_data, target_sr=16000, force_mono=True)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s")
            if i == 0:
                print(f"  Output size: {len(resampled)/1024/1024:.2f} MB")
                print(f"  Metrics: SR {metrics.get('original_sample_rate')}→16000, "
                      f"Ch {metrics.get('original_channels')}→1")
        except Exception as e:
            print(f"  Run {i+1}: Failed - {e}")

    if times:
        avg_time = np.mean(times)
        print(f"  Average: {avg_time:.3f}s")
        return avg_time
    return None

def main():
    print("="*60)
    print("QUICK BENCHMARK: 1-minute 48kHz stereo audio")
    print("="*60)

    # Create test audio
    test_audio = create_test_audio(48000, 60.0, 2)

    # Test both backends
    ffmpeg_time = benchmark_backend("FFmpeg", resample_audio_ffmpeg, test_audio)
    librosa_time = benchmark_backend("Librosa", resample_audio_librosa, test_audio)

    # Report
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    if ffmpeg_time and librosa_time:
        print(f"FFmpeg:  {ffmpeg_time:.3f}s")
        print(f"Librosa: {librosa_time:.3f}s")

        if ffmpeg_time < librosa_time:
            speedup = librosa_time / ffmpeg_time
            print(f"\n✅ FFmpeg is {speedup:.2f}x faster!")
            print("Recommendation: Use FFmpeg as default backend")
        else:
            speedup = ffmpeg_time / librosa_time
            print(f"\n✅ Librosa is {speedup:.2f}x faster!")
            print("Recommendation: Use Librosa as default backend")
    else:
        print("⚠️ One or both backends failed")

if __name__ == "__main__":
    main()