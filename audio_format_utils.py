"""
Audio format detection and preservation utilities
"""

import io
import struct
import soundfile as sf
import wave
from typing import Dict, Any, Optional


def detect_wav_format(audio_data: bytes) -> Dict[str, Any]:
    """
    Detect the format details of a WAV file

    Args:
        audio_data: Raw WAV file bytes

    Returns:
        Dictionary with format information
    """
    format_info = {
        'format': 'WAV',
        'subtype': None,
        'sample_rate': None,
        'channels': None,
        'sample_width': None,
        'bit_depth': None,
        'endian': 'LITTLE',  # WAV is little-endian by default
    }

    try:
        # First, check the WAV format code directly from header
        wav_io = io.BytesIO(audio_data)
        wav_data = wav_io.read(100)  # Read header
        fmt_pos = wav_data.find(b'fmt ')

        format_code = 1  # Default to PCM
        if fmt_pos != -1:
            # Format code is 2 bytes after 'fmt ' + 4 bytes chunk size
            format_code_pos = fmt_pos + 8
            if format_code_pos < len(wav_data) - 1:
                format_code = struct.unpack('<H', wav_data[format_code_pos:format_code_pos+2])[0]

        # Try using soundfile for detailed info
        wav_io.seek(0)
        try:
            info = sf.info(wav_io)
            format_info['sample_rate'] = info.samplerate
            format_info['channels'] = info.channels
            format_info['subtype'] = info.subtype

            # Map subtype to bit depth
            if 'PCM_U8' in info.subtype:
                format_info['bit_depth'] = 8
                format_info['sample_width'] = 1
            elif 'PCM_16' in info.subtype:
                format_info['bit_depth'] = 16
                format_info['sample_width'] = 2
            elif 'PCM_24' in info.subtype:
                format_info['bit_depth'] = 24
                format_info['sample_width'] = 3
            elif 'PCM_32' in info.subtype:
                format_info['bit_depth'] = 32
                format_info['sample_width'] = 4
            elif 'FLOAT' in info.subtype:
                format_info['bit_depth'] = 32
                format_info['sample_width'] = 4
            elif 'DOUBLE' in info.subtype:
                format_info['bit_depth'] = 64
                format_info['sample_width'] = 8

        except Exception as sf_error:
            # Fallback to wave module
            wav_io.seek(0)
            try:
                with wave.open(wav_io, 'rb') as wav_file:
                    params = wav_file.getparams()
                    format_info['channels'] = params.nchannels
                    format_info['sample_width'] = params.sampwidth
                    format_info['sample_rate'] = params.framerate
                    format_info['bit_depth'] = params.sampwidth * 8

                    # Determine subtype based on format code and sample width
                    if format_code == 3:  # IEEE float
                        if params.sampwidth == 4:
                            format_info['subtype'] = 'FLOAT'
                        elif params.sampwidth == 8:
                            format_info['subtype'] = 'DOUBLE'
                    else:  # PCM
                        if params.sampwidth == 1:
                            format_info['subtype'] = 'PCM_U8'
                        elif params.sampwidth == 2:
                            format_info['subtype'] = 'PCM_16'
                        elif params.sampwidth == 3:
                            format_info['subtype'] = 'PCM_24'
                        elif params.sampwidth == 4:
                            format_info['subtype'] = 'PCM_32'
            except Exception:
                # Use format code detection as last resort
                if format_code == 3:
                    format_info['subtype'] = 'FLOAT'
                    format_info['bit_depth'] = 32
                    format_info['sample_width'] = 4
                else:
                    format_info['subtype'] = 'PCM_16'  # Default
                    format_info['bit_depth'] = 16
                    format_info['sample_width'] = 2

    except Exception as e:
        # If all else fails, assume standard 16-bit PCM
        format_info['subtype'] = 'PCM_16'
        format_info['bit_depth'] = 16
        format_info['sample_width'] = 2

    return format_info


def get_soundfile_subtype(bit_depth: int, is_float: bool = False) -> str:
    """
    Get the soundfile subtype string for a given bit depth

    Args:
        bit_depth: Bits per sample
        is_float: Whether the format is floating point

    Returns:
        Soundfile subtype string
    """
    if is_float and bit_depth == 32:
        return 'FLOAT'
    elif is_float and bit_depth == 64:
        return 'DOUBLE'

    subtype_map = {
        8: 'PCM_U8',
        16: 'PCM_16',
        24: 'PCM_24',
        32: 'PCM_32'
    }

    return subtype_map.get(bit_depth, 'PCM_16')  # Default to 16-bit


def preserve_format_params(
    input_format: Dict[str, Any],
    preserve_bit_depth: bool = True
) -> Dict[str, Any]:
    """
    Determine output format parameters that preserve input characteristics

    Args:
        input_format: Input format information
        preserve_bit_depth: Whether to preserve the original bit depth

    Returns:
        Dictionary of output format parameters for soundfile
    """
    output_params = {
        'format': 'WAV',
        'subtype': 'PCM_16'  # Default
    }

    if preserve_bit_depth and input_format.get('subtype'):
        # Preserve the original subtype if possible
        output_params['subtype'] = input_format['subtype']
    elif preserve_bit_depth and input_format.get('bit_depth'):
        # Convert bit depth to subtype
        output_params['subtype'] = get_soundfile_subtype(
            input_format['bit_depth'],
            input_format.get('subtype', '').startswith('FLOAT')
        )

    return output_params