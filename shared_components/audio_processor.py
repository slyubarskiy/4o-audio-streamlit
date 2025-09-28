"""
Audio processing module with multiple backend support
Provides FFmpeg and Librosa implementations for audio resampling
"""

import io
import time
import subprocess
import tempfile
import os
import logging
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Dict, Optional, Any

from shared_components import audio_config
from shared_components.audio_format_utils import detect_wav_format

logger = logging.getLogger(__name__)


def resample_audio_ffmpeg(
    audio_data: bytes,
    target_sr: int = audio_config.TARGET_SAMPLE_RATE,
    force_mono: bool = audio_config.FORCE_MONO
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Resample audio using FFmpeg (faster, lower memory usage)

    Args:
        audio_data: Raw WAV audio bytes
        target_sr: Target sample rate (default 16000 Hz)
        force_mono: Convert multi-channel audio to mono (default True)

    Returns:
        Tuple of (resampled_audio_bytes, resample_metrics)
    """
    resample_start = time.time()
    resample_metrics = {
        'original_sample_rate': None,
        'target_sample_rate': target_sr,
        'resample_time': 0,
        'resampled': False,
        'original_size': len(audio_data),
        'resampled_size': None,
        'original_channels': None,
        'converted_to_mono': False,
        'bit_depth_converted': False,
        'backend': 'ffmpeg',
        'error': None
    }

    temp_input = None
    temp_output = None

    try:
        # Detect input format first
        input_format = detect_wav_format(audio_data)
        resample_metrics['input_format'] = input_format

        # Create temporary files for FFmpeg processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input_file:
            temp_input = temp_input_file.name
            temp_input_file.write(audio_data)

        # First, get input audio info using ffprobe
        probe_cmd = [
            audio_config.FFMPEG_PATH.replace('ffmpeg', 'ffprobe'),
            '-v', 'error',
            '-show_entries', 'stream=sample_rate,channels',
            '-of', 'json',
            temp_input
        ]

        original_sr = None
        original_channels = None

        try:
            probe_result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            if probe_result.returncode == 0:
                import json
                probe_data = json.loads(probe_result.stdout)
                if probe_data.get('streams'):
                    stream = probe_data['streams'][0]
                    original_sr = int(stream.get('sample_rate', 0))
                    original_channels = int(stream.get('channels', 1))
                    resample_metrics['original_sample_rate'] = original_sr
                    resample_metrics['original_channels'] = original_channels
                    logger.info(f"FFmpeg probe: sample_rate={original_sr} Hz, "
                               f"channels={original_channels}")
        except Exception as e:
            logger.warning(f"FFprobe failed: {e}, continuing with FFmpeg processing")

        # Check if we need to process at all
        # We only skip processing if input is EXACTLY mono/16kHz/PCM16
        needs_processing = False
        if original_sr and original_sr != target_sr:
            needs_processing = True
            resample_metrics['resampled'] = True
        if force_mono and original_channels and original_channels > 1:
            needs_processing = True
            resample_metrics['converted_to_mono'] = True

        # Check bit depth - we need PCM16 specifically
        if input_format.get('subtype') != 'PCM_16':
            needs_processing = True
            resample_metrics['bit_depth_converted'] = True
            logger.info(f"FFmpeg: Converting from {input_format.get('subtype', 'unknown')} to PCM_16")

        # If no processing needed, return original
        if not needs_processing and original_sr and original_channels:
            logger.info(f"FFmpeg: Audio already at exact target specs (16kHz mono PCM16), skipping processing")
            resample_metrics['resample_time'] = time.time() - resample_start
            resample_metrics['resampled_size'] = len(audio_data)
            return audio_data, resample_metrics

        # Otherwise, continue with FFmpeg processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output_file:
            temp_output = temp_output_file.name

        # Always output PCM16 little-endian as required
        audio_codec = 'pcm_s16le'  # Force 16-bit PCM output

        # Build FFmpeg command
        ffmpeg_cmd = [
            audio_config.FFMPEG_PATH,
            '-i', temp_input,        # Input file
            '-ar', str(target_sr),   # Target sample rate
            '-f', 'wav',             # Output format
            '-acodec', audio_codec,  # Preserve bit depth format
            '-y',                    # Overwrite output file
        ]

        # Add mono conversion if needed (only if not already flagged)
        if force_mono:
            ffmpeg_cmd.extend(['-ac', '1'])  # Convert to 1 channel (mono)

        ffmpeg_cmd.append(temp_output)

        # Log the command
        logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        # Execute FFmpeg
        ffmpeg_start = time.time()
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            timeout=audio_config.FFMPEG_TIMEOUT
        )

        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg failed with return code {result.returncode}: {error_msg}")

        # Read the processed audio
        with open(temp_output, 'rb') as f:
            resampled_bytes = f.read()

        # Update metrics (resampled and converted_to_mono already set above)
        resample_metrics['resampled_size'] = len(resampled_bytes)
        resample_metrics['resample_time'] = time.time() - resample_start
        resample_metrics['ffmpeg_time'] = time.time() - ffmpeg_start

        # Log the conversion
        size_reduction = (1 - len(resampled_bytes) / len(audio_data)) * 100
        conversion_info = []
        if resample_metrics.get('resampled'):
            conversion_info.append(f"{resample_metrics['original_sample_rate']} Hz → {target_sr} Hz")
        if resample_metrics.get('converted_to_mono'):
            conversion_info.append(f"{resample_metrics['original_channels']} ch → mono")
        if resample_metrics.get('bit_depth_converted'):
            conversion_info.append(f"{input_format.get('subtype', 'unknown')} → PCM_16")

        if conversion_info:
            logger.info(f"Audio processed (FFmpeg): {', '.join(conversion_info)}, "
                       f"Size: {len(audio_data)/1024:.1f} KB → {len(resampled_bytes)/1024:.1f} KB "
                       f"({size_reduction:.1f}% reduction), Time: {resample_metrics['resample_time']:.3f}s")
        else:
            logger.info(f"Audio (FFmpeg): Processing completed, Time: {resample_metrics['resample_time']:.3f}s")

        return resampled_bytes, resample_metrics

    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timeout after {audio_config.FFMPEG_TIMEOUT}s")
        resample_metrics['error'] = 'timeout'
        resample_metrics['resample_time'] = time.time() - resample_start
        raise

    except Exception as e:
        logger.error(f"FFmpeg audio processing failed: {str(e)}")
        resample_metrics['error'] = str(e)
        resample_metrics['resample_time'] = time.time() - resample_start
        raise

    finally:
        # Clean up temporary files
        for temp_file in [temp_input, temp_output]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")


def resample_audio_librosa(
    audio_data: bytes,
    target_sr: int = audio_config.TARGET_SAMPLE_RATE,
    force_mono: bool = audio_config.FORCE_MONO
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Resample audio using Librosa (higher quality, more control)
    This is essentially the original resample_audio function

    Args:
        audio_data: Raw WAV audio bytes
        target_sr: Target sample rate (default 16000 Hz)
        force_mono: Convert multi-channel audio to mono (default True)

    Returns:
        Tuple of (resampled_audio_bytes, resample_metrics)
    """
    resample_start = time.time()
    resample_metrics = {
        'original_sample_rate': None,
        'target_sample_rate': target_sr,
        'resample_time': 0,
        'resampled': False,
        'original_size': len(audio_data),
        'resampled_size': None,
        'original_channels': None,
        'converted_to_mono': False,
        'bit_depth_converted': False,
        'backend': 'librosa',
        'error': None
    }

    try:
        # First detect input format
        input_format = detect_wav_format(audio_data)
        resample_metrics['input_format'] = input_format

        # Read the audio data from bytes
        audio_io = io.BytesIO(audio_data)

        # Read audio with soundfile to get the sample rate and data
        audio_array, original_sr = sf.read(audio_io, dtype='float32')
        resample_metrics['original_sample_rate'] = original_sr

        # Determine number of channels
        if len(audio_array.shape) > 1:
            num_channels = audio_array.shape[1]
            resample_metrics['original_channels'] = num_channels
        else:
            num_channels = 1
            resample_metrics['original_channels'] = 1

        logger.info(f"Original audio (Librosa): sample_rate={original_sr} Hz, channels={num_channels}, shape={audio_array.shape}")

        # Convert to mono if multi-channel and force_mono is True
        if force_mono and num_channels > 1:
            logger.info(f"Converting {num_channels}-channel audio to mono")
            # Average all channels to create mono
            audio_array = np.mean(audio_array, axis=1)
            resample_metrics['converted_to_mono'] = True

        # Check if any processing is needed
        # We only skip processing if input is EXACTLY mono/16kHz/PCM16
        needs_resampling = original_sr != target_sr
        needs_bit_depth_conversion = input_format.get('subtype') != 'PCM_16'

        if needs_bit_depth_conversion:
            resample_metrics['bit_depth_converted'] = True
            logger.info(f"Librosa: Converting from {input_format.get('subtype', 'unknown')} to PCM_16")

        # Check if ANY processing is needed
        if not needs_resampling and not resample_metrics['converted_to_mono'] and not needs_bit_depth_conversion:
            logger.info(f"Audio already at exact target specs (16kHz mono PCM16), skipping processing")
            resample_metrics['resample_time'] = time.time() - resample_start
            resample_metrics['resampled_size'] = len(audio_data)
            return audio_data, resample_metrics

        # Resample if needed
        if needs_resampling:
            # Single channel audio (or already converted to mono)
            resampled_audio = librosa.resample(
                audio_array,
                orig_sr=original_sr,
                target_sr=target_sr,
                res_type=audio_config.LIBROSA_RESAMPLE_TYPE
            )
            resample_metrics['resampled'] = True
        else:
            # No resampling needed, just use the mono-converted audio
            resampled_audio = audio_array
            target_sr = original_sr  # Keep original sample rate
            resample_metrics['resampled'] = False  # Explicitly set to False

        # Convert back to WAV bytes - always use PCM16 as required
        output_io = io.BytesIO()

        # Always output as PCM16 little-endian WAV
        sf.write(output_io, resampled_audio, target_sr,
                format='WAV',
                subtype='PCM_16')
        resampled_bytes = output_io.getvalue()

        # Update metrics - only mark as processed if something changed
        resample_metrics['resampled_size'] = len(resampled_bytes)
        resample_metrics['resample_time'] = time.time() - resample_start

        # Log the conversion
        size_reduction = (1 - len(resampled_bytes) / len(audio_data)) * 100
        conversion_info = []
        if needs_resampling and resample_metrics['resampled']:
            conversion_info.append(f"{original_sr} Hz → {target_sr} Hz")
        if resample_metrics['converted_to_mono']:
            conversion_info.append(f"{resample_metrics['original_channels']} ch → mono")
        if resample_metrics['bit_depth_converted']:
            conversion_info.append(f"{input_format.get('subtype', 'unknown')} → PCM_16")

        if conversion_info:
            logger.info(f"Audio processed (Librosa): {', '.join(conversion_info)}, "
                       f"Size: {len(audio_data)/1024:.1f} KB → {len(resampled_bytes)/1024:.1f} KB "
                       f"({size_reduction:.1f}% reduction), Time: {resample_metrics['resample_time']:.3f}s")
        else:
            logger.info(f"Audio (Librosa): No processing needed, already at target specs")

        return resampled_bytes, resample_metrics

    except Exception as e:
        logger.error(f"Librosa audio processing failed: {str(e)}")
        resample_metrics['error'] = str(e)
        resample_metrics['resample_time'] = time.time() - resample_start
        raise


def resample_audio_auto(
    audio_data: bytes,
    target_sr: int = audio_config.TARGET_SAMPLE_RATE,
    force_mono: bool = audio_config.FORCE_MONO,
    backend: Optional[str] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Automatically select and use the best available audio processing backend

    Args:
        audio_data: Raw WAV audio bytes
        target_sr: Target sample rate (default 16000 Hz)
        force_mono: Convert multi-channel audio to mono (default True)
        backend: Override backend selection (optional)

    Returns:
        Tuple of (resampled_audio_bytes, resample_metrics)
    """
    # Validate and select backend
    selected_backend = audio_config.validate_backend(backend)

    logger.info(f"Using audio processing backend: {selected_backend}")

    try:
        if selected_backend == "ffmpeg":
            return resample_audio_ffmpeg(audio_data, target_sr, force_mono)
        else:
            return resample_audio_librosa(audio_data, target_sr, force_mono)

    except Exception as e:
        logger.error(f"Primary backend ({selected_backend}) failed: {str(e)}")

        # Try fallback if enabled
        if audio_config.FALLBACK_TO_LIBROSA and selected_backend == "ffmpeg":
            logger.info("Attempting fallback to Librosa backend")
            try:
                return resample_audio_librosa(audio_data, target_sr, force_mono)
            except Exception as fallback_e:
                logger.error(f"Fallback to Librosa also failed: {str(fallback_e)}")

        # If all processing fails and fallback to original is enabled
        if audio_config.FALLBACK_TO_ORIGINAL:
            logger.warning("All audio processing failed, using original audio")
            return audio_data, {
                'original_sample_rate': None,
                'target_sample_rate': target_sr,
                'resample_time': 0,
                'resampled': False,
                'original_size': len(audio_data),
                'resampled_size': len(audio_data),
                'original_channels': None,
                'converted_to_mono': False,
                'backend': 'none',
                'error': 'All backends failed, using original'
            }

        raise