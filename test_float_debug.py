#!/usr/bin/env python3
"""
Debug FLOAT format detection
"""

import io
import numpy as np
import soundfile as sf
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_format_utils import detect_wav_format

def create_float_audio():
    """Create float32 test audio"""
    sample_rate = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

    output_io = io.BytesIO()
    sf.write(output_io, audio_data, sample_rate, format='WAV', subtype='FLOAT')
    return output_io.getvalue()

def main():
    print("Testing FLOAT format detection...")

    # Create float32 audio
    float_audio = create_float_audio()
    print(f"Created float32 audio: {len(float_audio)} bytes")

    # Detect format
    format_info = detect_wav_format(float_audio)

    print("\nDetected format:")
    for key, value in format_info.items():
        print(f"  {key}: {value}")

    print(f"\nSubtype detected correctly: {'FLOAT' == format_info.get('subtype')}")

if __name__ == "__main__":
    main()