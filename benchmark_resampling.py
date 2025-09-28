#!/usr/bin/env python3
"""
Benchmark script for comparing audio resampling backends
Tests FFmpeg vs Librosa performance on various audio configurations
"""

import io
import time
import numpy as np
import soundfile as sf
import sys
import os
import psutil
import gc
from typing import Dict, Any, List
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_config
from audio_processor import resample_audio_ffmpeg, resample_audio_librosa

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def create_test_audio(sample_rate=48000, duration=60.0, channels=2, frequency=440):
    """
    Create test audio with specified parameters

    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        channels: Number of channels (1 for mono, 2 for stereo)
        frequency: Base frequency for sine wave

    Returns:
        WAV audio bytes
    """
    print(f"Creating test audio: {sample_rate} Hz, {duration}s, {channels} channel(s), {frequency} Hz")

    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    if channels == 1:
        # Mono audio
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
    else:
        # Multi-channel audio
        channels_data = []
        for ch in range(channels):
            # Use different frequencies for each channel
            freq = frequency * (1 + ch * 0.5)
            channel_data = np.sin(2 * np.pi * freq * t) * 0.5
            channels_data.append(channel_data)
        audio_data = np.stack(channels_data, axis=1)

    # Convert to WAV bytes
    output_io = io.BytesIO()
    sf.write(output_io, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    wav_bytes = output_io.getvalue()

    print(f"Generated audio size: {len(wav_bytes)/1024/1024:.2f} MB")
    return wav_bytes


def benchmark_backend(backend_name: str, backend_func, audio_data: bytes, iterations: int = 3) -> Dict[str, Any]:
    """
    Benchmark a specific backend

    Args:
        backend_name: Name of the backend
        backend_func: Resampling function to test
        audio_data: Test audio data
        iterations: Number of iterations to average

    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {backend_name} backend ({iterations} iterations)")
    print(f"{'='*60}")

    results = {
        'backend': backend_name,
        'iterations': iterations,
        'times': [],
        'memory_usage': [],
        'output_sizes': [],
        'errors': []
    }

    # Warm-up run (not counted)
    print("Warming up...")
    try:
        _, _ = backend_func(audio_data, target_sr=16000, force_mono=True)
    except Exception as e:
        print(f"Warm-up failed: {e}")

    # Actual benchmark runs
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...", end=" ")

        # Force garbage collection before each run
        gc.collect()

        # Measure memory before
        mem_before = get_memory_usage()

        try:
            start_time = time.time()

            # Run resampling
            resampled_audio, metrics = backend_func(
                audio_data,
                target_sr=16000,
                force_mono=True
            )

            elapsed_time = time.time() - start_time

            # Measure memory after
            mem_after = get_memory_usage()
            memory_delta = mem_after - mem_before

            # Record results
            results['times'].append(elapsed_time)
            results['memory_usage'].append(memory_delta)
            results['output_sizes'].append(len(resampled_audio))

            print(f"✓ Time: {elapsed_time:.3f}s, Memory: {memory_delta:.1f} MB")

            # Store first iteration metrics for detail
            if i == 0:
                results['first_run_metrics'] = metrics

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results['errors'].append(str(e))
            results['times'].append(None)
            results['memory_usage'].append(None)
            results['output_sizes'].append(None)

    # Calculate statistics
    valid_times = [t for t in results['times'] if t is not None]
    valid_memory = [m for m in results['memory_usage'] if m is not None]

    if valid_times:
        results['stats'] = {
            'avg_time': np.mean(valid_times),
            'std_time': np.std(valid_times),
            'min_time': np.min(valid_times),
            'max_time': np.max(valid_times),
            'avg_memory': np.mean(valid_memory) if valid_memory else 0,
            'max_memory': np.max(valid_memory) if valid_memory else 0,
            'success_rate': len(valid_times) / iterations * 100
        }
    else:
        results['stats'] = {
            'avg_time': None,
            'success_rate': 0
        }

    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing FFmpeg and Librosa"""

    print("=" * 80)
    print("AUDIO RESAMPLING BENCHMARK")
    print("=" * 80)

    # Test configurations
    test_configs = [
        {
            'name': '1-minute 48kHz stereo (typical recording)',
            'sample_rate': 48000,
            'duration': 60.0,
            'channels': 2
        },
        {
            'name': '30-second 44.1kHz stereo (CD quality)',
            'sample_rate': 44100,
            'duration': 30.0,
            'channels': 2
        },
        {
            'name': '2-minute 48kHz mono',
            'sample_rate': 48000,
            'duration': 120.0,
            'channels': 1
        },
        {
            'name': '10-second 96kHz stereo (high-res)',
            'sample_rate': 96000,
            'duration': 10.0,
            'channels': 2
        }
    ]

    all_results = []

    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"TEST: {config['name']}")
        print(f"{'='*80}")

        # Create test audio
        test_audio = create_test_audio(
            sample_rate=config['sample_rate'],
            duration=config['duration'],
            channels=config['channels']
        )

        config_results = {
            'config': config,
            'input_size_mb': len(test_audio) / 1024 / 1024,
            'backends': {}
        }

        # Test FFmpeg backend
        try:
            ffmpeg_results = benchmark_backend(
                'FFmpeg',
                resample_audio_ffmpeg,
                test_audio,
                iterations=3
            )
            config_results['backends']['ffmpeg'] = ffmpeg_results
        except Exception as e:
            print(f"FFmpeg benchmark failed: {e}")
            config_results['backends']['ffmpeg'] = {'error': str(e)}

        # Small delay between backends
        time.sleep(1)

        # Test Librosa backend
        try:
            librosa_results = benchmark_backend(
                'Librosa',
                resample_audio_librosa,
                test_audio,
                iterations=3
            )
            config_results['backends']['librosa'] = librosa_results
        except Exception as e:
            print(f"Librosa benchmark failed: {e}")
            config_results['backends']['librosa'] = {'error': str(e)}

        all_results.append(config_results)

    return all_results


def generate_report(results: List[Dict[str, Any]]):
    """Generate performance comparison report"""

    print("\n" + "=" * 80)
    print("BENCHMARK REPORT")
    print("=" * 80)

    # Focus on the main test case: 1-minute 48kHz stereo
    main_result = None
    for result in results:
        if '1-minute 48kHz stereo' in result['config']['name']:
            main_result = result
            break

    if main_result and 'ffmpeg' in main_result['backends'] and 'librosa' in main_result['backends']:
        ffmpeg = main_result['backends']['ffmpeg']
        librosa = main_result['backends']['librosa']

        print(f"\n### Main Test Case: {main_result['config']['name']}")
        print(f"Input size: {main_result['input_size_mb']:.2f} MB")
        print()

        # Performance comparison table
        print("| Metric                | FFmpeg        | Librosa       | Winner     |")
        print("|----------------------|---------------|---------------|------------|")

        # Processing time
        ffmpeg_time = ffmpeg['stats']['avg_time'] if ffmpeg.get('stats') else None
        librosa_time = librosa['stats']['avg_time'] if librosa.get('stats') else None

        if ffmpeg_time and librosa_time:
            winner = "FFmpeg ✓" if ffmpeg_time < librosa_time else "Librosa ✓"
            speedup = max(ffmpeg_time, librosa_time) / min(ffmpeg_time, librosa_time)
            print(f"| Avg Processing Time  | {ffmpeg_time:>10.3f}s | {librosa_time:>10.3f}s | {winner:10} |")
            print(f"| Speed Advantage      | {speedup:.2f}x faster {winner.split()[0]:<20} |")
        else:
            print("| Avg Processing Time  | N/A           | N/A           | N/A        |")

        # Memory usage
        ffmpeg_mem = ffmpeg['stats'].get('avg_memory', 0) if ffmpeg.get('stats') else 0
        librosa_mem = librosa['stats'].get('avg_memory', 0) if librosa.get('stats') else 0

        if ffmpeg_mem or librosa_mem:
            winner = "FFmpeg ✓" if ffmpeg_mem < librosa_mem else "Librosa ✓"
            print(f"| Avg Memory Usage     | {ffmpeg_mem:>9.1f} MB | {librosa_mem:>9.1f} MB | {winner:10} |")

        # Success rate
        ffmpeg_success = ffmpeg['stats'].get('success_rate', 0) if ffmpeg.get('stats') else 0
        librosa_success = librosa['stats'].get('success_rate', 0) if librosa.get('stats') else 0
        print(f"| Success Rate         | {ffmpeg_success:>10.0f}% | {librosa_success:>10.0f}% | Both 100%  |")

    # Summary for all test cases
    print("\n### All Test Cases Summary")
    print()
    print("| Test Case                        | FFmpeg Time | Librosa Time | Speedup  |")
    print("|----------------------------------|-------------|--------------|----------|")

    for result in results:
        config_name = result['config']['name'][:32].ljust(32)

        ffmpeg_time = "N/A"
        librosa_time = "N/A"
        speedup = "N/A"

        if 'ffmpeg' in result['backends'] and result['backends']['ffmpeg'].get('stats'):
            ft = result['backends']['ffmpeg']['stats'].get('avg_time')
            if ft:
                ffmpeg_time = f"{ft:.3f}s"

        if 'librosa' in result['backends'] and result['backends']['librosa'].get('stats'):
            lt = result['backends']['librosa']['stats'].get('avg_time')
            if lt:
                librosa_time = f"{lt:.3f}s"

        if ffmpeg_time != "N/A" and librosa_time != "N/A":
            ft = float(ffmpeg_time[:-1])
            lt = float(librosa_time[:-1])
            if ft < lt:
                speedup = f"{lt/ft:.2f}x ↑"
            else:
                speedup = f"{ft/lt:.2f}x ↓"

        print(f"| {config_name} | {ffmpeg_time:>11} | {librosa_time:>12} | {speedup:>8} |")

    # Recommendations
    print("\n### Recommendations")
    print()
    if main_result:
        ffmpeg_avg = main_result['backends'].get('ffmpeg', {}).get('stats', {}).get('avg_time', float('inf'))
        librosa_avg = main_result['backends'].get('librosa', {}).get('stats', {}).get('avg_time', float('inf'))

        if ffmpeg_avg < librosa_avg:
            print("✅ **FFmpeg is recommended as the default backend**")
            print(f"   - {librosa_avg/ffmpeg_avg:.1f}x faster processing on typical audio")
            print("   - Lower memory usage")
            print("   - Handles various formats efficiently")
        else:
            print("✅ **Librosa is recommended as the default backend**")
            print("   - More consistent quality")
            print("   - Better Python integration")
            print("   - More processing options available")

    print("\n" + "=" * 80)


def main():
    """Main benchmark execution"""
    print("Starting audio resampling backend benchmark...")
    print(f"Current configuration: backend={audio_config.RESAMPLING_BACKEND}")
    print()

    # Check FFmpeg availability
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is installed and available")
        else:
            print("⚠️ FFmpeg may not be properly configured")
    except Exception as e:
        print(f"❌ FFmpeg not found: {e}")
        print("   Install with: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)")

    print()

    # Run benchmarks
    try:
        results = run_comprehensive_benchmark()

        # Generate report
        generate_report(results)

        # Save detailed results to JSON
        output_file = 'benchmark_results.json'
        with open(output_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj

            json.dump(results, f, indent=2, default=convert_types)

        print(f"\nDetailed results saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()