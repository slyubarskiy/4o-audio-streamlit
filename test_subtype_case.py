#!/usr/bin/env python3
"""
Test what soundfile returns for subtype strings
"""

import io
import numpy as np
import soundfile as sf

def test_soundfile_subtype():
    """Test what soundfile returns for subtype"""
    print("Testing soundfile subtype string format...")

    # Create PCM16 audio
    sample_rate = 16000
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.5

    # Test different subtypes
    subtypes_to_test = ['PCM_16', 'PCM_24', 'PCM_32', 'FLOAT']

    for subtype in subtypes_to_test:
        try:
            output_io = io.BytesIO()
            sf.write(output_io, audio_data, sample_rate, format='WAV', subtype=subtype)

            # Read it back and check info
            output_io.seek(0)
            info = sf.info(output_io)

            print(f"\nWritten with subtype: '{subtype}'")
            print(f"Read back as: '{info.subtype}'")
            print(f"Are they equal? {subtype == info.subtype}")
            print(f"All available subtypes from soundfile: {sf.available_subtypes('WAV')}")
            break  # Just need to check once

        except Exception as e:
            print(f"Error with {subtype}: {e}")

if __name__ == "__main__":
    test_soundfile_subtype()