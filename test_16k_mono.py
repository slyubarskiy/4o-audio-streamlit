#!/usr/bin/env python3
"""
Test script to verify that 16kHz mono audio is not resampled
and that output format consistency is maintained
"""

import io
import numpy as np
import soundfile as sf
import sys
import os
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_processor import resample_audio_librosa, resample_audio_ffmpeg

def create_test_audio(sample_rate=16000, channels=1, duration=1.0):
    """Create test audio with specific parameters"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    if channels == 1:
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.5
    else:
        left = np.sin(2 * np.pi * 440 * t) * 0.5
        right = np.sin(2 * np.pi * 880 * t) * 0.5
        audio_data = np.stack([left, right], axis=1)

    # Create WAV with specific format
    output_io = io.BytesIO()
    sf.write(output_io, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    wav_bytes = output_io.getvalue()

    return wav_bytes

def analyze_wav_format(wav_bytes):
    """Analyze WAV file format details"""
    wav_io = io.BytesIO(wav_bytes)

    # Use wave module to get format details
    with wave.open(wav_io, 'rb') as wav_file:
        params = wav_file.getparams()

    # Also get soundfile info
    wav_io.seek(0)
    audio_data, sample_rate = sf.read(wav_io)

    return {
        'channels': params.nchannels,
        'sample_width': params.sampwidth,
        'framerate': params.framerate,
        'nframes': params.nframes,
        'comptype': params.comptype,
        'compname': params.compname,
        'size_bytes': len(wav_bytes),
        'sample_rate_sf': sample_rate,
        'shape': audio_data.shape if hasattr(audio_data, 'shape') else None
    }

def test_16k_mono_passthrough():
    """Test that 16kHz mono audio passes through unchanged"""
    print("="*60)
    print("TEST 1: 16kHz Mono Audio (Should NOT be resampled)")
    print("="*60)

    # Create 16kHz mono audio
    test_audio = create_test_audio(sample_rate=16000, channels=1)
    original_format = analyze_wav_format(test_audio)

    print(f"Original audio format:")
    print(f"  Sample rate: {original_format['framerate']} Hz")
    print(f"  Channels: {original_format['channels']}")
    print(f"  Sample width: {original_format['sample_width']} bytes")
    print(f"  Size: {original_format['size_bytes']} bytes")

    # Test Librosa backend
    print("\n--- Librosa Backend ---")
    output_audio, metrics = resample_audio_librosa(test_audio, target_sr=16000, force_mono=True)
    output_format = analyze_wav_format(output_audio)

    print(f"Resampled: {metrics['resampled']}")
    print(f"Converted to mono: {metrics['converted_to_mono']}")
    print(f"Processing time: {metrics['resample_time']:.3f}s")

    if output_audio == test_audio:
        print("✅ Output is identical to input (no processing)")
    else:
        print(f"❌ Output differs from input")
        print(f"  Output sample rate: {output_format['framerate']} Hz")
        print(f"  Output channels: {output_format['channels']}")
        print(f"  Output size: {output_format['size_bytes']} bytes")

    # Test FFmpeg backend
    print("\n--- FFmpeg Backend ---")
    output_audio_ff, metrics_ff = resample_audio_ffmpeg(test_audio, target_sr=16000, force_mono=True)
    output_format_ff = analyze_wav_format(output_audio_ff)

    print(f"Resampled: {metrics_ff['resampled']}")
    print(f"Converted to mono: {metrics_ff['converted_to_mono']}")
    print(f"Processing time: {metrics_ff['resample_time']:.3f}s")

    if output_audio_ff == test_audio:
        print("✅ Output is identical to input (no processing)")
    else:
        print(f"⚠️ Output differs from input")
        print(f"  Output sample rate: {output_format_ff['framerate']} Hz")
        print(f"  Output channels: {output_format_ff['channels']}")
        print(f"  Output size: {output_format_ff['size_bytes']} bytes")

def test_format_consistency():
    """Test that output format is consistent with input except for channels and sample rate"""
    print("\n" + "="*60)
    print("TEST 2: Format Consistency (48kHz Stereo -> 16kHz Mono)")
    print("="*60)

    # Create 48kHz stereo audio
    test_audio = create_test_audio(sample_rate=48000, channels=2, duration=0.5)
    original_format = analyze_wav_format(test_audio)

    print(f"Original audio format:")
    print(f"  Sample rate: {original_format['framerate']} Hz")
    print(f"  Channels: {original_format['channels']}")
    print(f"  Sample width: {original_format['sample_width']} bytes (bits per sample: {original_format['sample_width']*8})")
    print(f"  Compression: {original_format['comptype']} / {original_format['compname']}")

    # Test Librosa backend
    print("\n--- Librosa Backend ---")
    output_audio, metrics = resample_audio_librosa(test_audio, target_sr=16000, force_mono=True)
    output_format = analyze_wav_format(output_audio)

    print(f"Output audio format:")
    print(f"  Sample rate: {output_format['framerate']} Hz (target: 16000)")
    print(f"  Channels: {output_format['channels']} (target: 1)")
    print(f"  Sample width: {output_format['sample_width']} bytes (bits per sample: {output_format['sample_width']*8})")
    print(f"  Compression: {output_format['comptype']} / {output_format['compname']}")

    # Check consistency
    consistency_checks = []
    consistency_checks.append(("Sample width preserved", original_format['sample_width'] == output_format['sample_width']))
    consistency_checks.append(("Compression type preserved", original_format['comptype'] == output_format['comptype']))

    for check_name, passed in consistency_checks:
        print(f"  {check_name}: {'✅' if passed else '❌'}")

    # Test FFmpeg backend
    print("\n--- FFmpeg Backend ---")
    output_audio_ff, metrics_ff = resample_audio_ffmpeg(test_audio, target_sr=16000, force_mono=True)
    output_format_ff = analyze_wav_format(output_audio_ff)

    print(f"Output audio format:")
    print(f"  Sample rate: {output_format_ff['framerate']} Hz (target: 16000)")
    print(f"  Channels: {output_format_ff['channels']} (target: 1)")
    print(f"  Sample width: {output_format_ff['sample_width']} bytes (bits per sample: {output_format_ff['sample_width']*8})")
    print(f"  Compression: {output_format_ff['comptype']} / {output_format_ff['compname']}")

    # Check consistency
    for check_name, passed in consistency_checks:
        passed_ff = False
        if "Sample width" in check_name:
            passed_ff = original_format['sample_width'] == output_format_ff['sample_width']
        elif "Compression" in check_name:
            passed_ff = original_format['comptype'] == output_format_ff['comptype']
        print(f"  {check_name}: {'✅' if passed_ff else '❌'}")

if __name__ == "__main__":
    test_16k_mono_passthrough()
    test_format_consistency()