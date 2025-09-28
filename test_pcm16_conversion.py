#!/usr/bin/env python3
"""
Test that verifies PCM16 conversion logic
"""

import io
import numpy as np
import soundfile as sf
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_processor import resample_audio_librosa, resample_audio_ffmpeg

def create_audio_with_format(sample_rate=16000, channels=1, bit_depth=16, duration=0.5):
    """Create test audio with specific format"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    if channels == 1:
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.5
    else:
        channels_data = []
        for ch in range(channels):
            freq = 440 * (1 + ch * 0.25)
            channel_data = np.sin(2 * np.pi * freq * t) * 0.5
            channels_data.append(channel_data)
        audio_data = np.stack(channels_data, axis=1)

    # Determine subtype based on bit depth
    subtype_map = {
        8: 'PCM_U8',
        16: 'PCM_16',
        24: 'PCM_24',
        32: 'PCM_32',
        'float32': 'FLOAT',
    }
    subtype = subtype_map.get(bit_depth, 'PCM_16')

    output_io = io.BytesIO()

    # For floating point, use float32 dtype
    if bit_depth == 'float32':
        sf.write(output_io, audio_data.astype(np.float32), sample_rate, format='WAV', subtype=subtype)
    else:
        sf.write(output_io, audio_data, sample_rate, format='WAV', subtype=subtype)

    return output_io.getvalue()

def test_bit_depth_conversion(name, sample_rate, channels, bit_depth, should_process=True):
    """Test a specific bit depth scenario"""
    print(f"\n{name}")
    print("-" * len(name))

    # Create test audio
    test_audio = create_audio_with_format(sample_rate, channels, bit_depth)
    original_size = len(test_audio)

    # Test both backends
    for backend_name, backend_func in [("Librosa", resample_audio_librosa),
                                       ("FFmpeg", resample_audio_ffmpeg)]:
        output, metrics = backend_func(test_audio, target_sr=16000, force_mono=True)

        # Check processing status
        processed = metrics['resampled'] or metrics['converted_to_mono'] or metrics.get('bit_depth_converted', False)

        if should_process:
            if processed:
                result = "✅ Processed"
            else:
                result = "❌ Not processed (should have been)"
        else:
            if output == test_audio:
                result = "✅ Unchanged (passthrough)"
            elif processed:
                result = "❌ Unnecessarily processed"
            else:
                result = "⚠️ Output changed without processing flag"

        print(f"  {backend_name:8}: {result} | "
              f"Resampled: {metrics['resampled']} | "
              f"Mono: {metrics['converted_to_mono']} | "
              f"BitDepth: {metrics.get('bit_depth_converted', False)} | "
              f"Size: {original_size} → {len(output)} bytes")

def main():
    print("="*70)
    print("PCM16 CONVERSION TESTS")
    print("="*70)

    # Test various bit depth scenarios
    test_bit_depth_conversion(
        "16kHz Mono PCM16 (Should NOT process - exact target)",
        16000, 1, 16, should_process=False
    )

    test_bit_depth_conversion(
        "16kHz Mono PCM24 (Should convert bit depth to PCM16)",
        16000, 1, 24, should_process=True
    )

    test_bit_depth_conversion(
        "16kHz Mono PCM32 (Should convert bit depth to PCM16)",
        16000, 1, 32, should_process=True
    )

    test_bit_depth_conversion(
        "16kHz Mono FLOAT32 (Should convert to PCM16)",
        16000, 1, 'float32', should_process=True
    )

    test_bit_depth_conversion(
        "16kHz Mono PCM_U8 (Should convert to PCM16)",
        16000, 1, 8, should_process=True
    )

    test_bit_depth_conversion(
        "48kHz Stereo PCM24 (Should resample + mono + bit depth)",
        48000, 2, 24, should_process=True
    )

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✅ Expected behavior:")
    print("  - 16kHz mono PCM16 passes through unchanged (exact target format)")
    print("  - Any other bit depth is converted to PCM16")
    print("  - All outputs are PCM16 little-endian WAV format")
    print("  - Processing only occurs when input differs from target specs")

if __name__ == "__main__":
    main()