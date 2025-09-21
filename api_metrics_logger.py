"""
API Metrics and Performance Logging Module
Tracks OpenAI API usage, performance metrics, and provides analytics
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import streamlit as st
from dataclasses import dataclass, asdict
import pandas as pd
import io
import base64

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionMetrics:
    """Data class for transcription metrics"""
    segment_id: int
    timestamp: float
    audio_size_bytes: int
    audio_duration_estimate: float  # seconds
    encoded_size_bytes: int
    encoding_time: float  # seconds
    api_call_time: float  # seconds
    total_time: float  # seconds
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    status: str  # success, error, timeout
    error_type: Optional[str] = None
    retry_count: int = 0
    cached_response: bool = False
    context_segments_used: int = 0
    response_length: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second"""
        if self.api_call_time > 0:
            return self.total_tokens / self.api_call_time
        return 0
    
    @property
    def audio_to_text_ratio(self) -> float:
        """Calculate ratio of audio duration to processing time"""
        if self.api_call_time > 0:
            return self.audio_duration_estimate / self.api_call_time
        return 0
    
    @property
    def cost_estimate(self) -> float:
        """Estimate API cost (adjust rates as needed)"""
        # Example rates - UPDATE WITH ACTUAL RATES
        input_cost_per_1k = 0.01  # $0.01 per 1K input tokens
        output_cost_per_1k = 0.03  # $0.03 per 1K output tokens
        
        input_cost = (self.prompt_tokens / 1000) * input_cost_per_1k
        output_cost = (self.completion_tokens / 1000) * output_cost_per_1k
        return input_cost + output_cost

class APIMetricsLogger:
    """Centralized metrics logging for API calls"""
    
    def __init__(self):
        self.metrics_key = "api_metrics_log"
        self.summary_key = "api_metrics_summary"
        self.initialize_storage()
    
    def initialize_storage(self):
        """Initialize session state storage for metrics"""
        if self.metrics_key not in st.session_state:
            st.session_state[self.metrics_key] = []
        
        if self.summary_key not in st.session_state:
            st.session_state[self.summary_key] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'total_audio_bytes': 0,
                'total_processing_time': 0.0,
                'start_time': time.time()
            }
    
    def log_transcription(self, metrics: TranscriptionMetrics):
        """Log a transcription event"""
        # Add to detailed log
        st.session_state[self.metrics_key].append(metrics.to_dict())
        
        # Update summary
        summary = st.session_state[self.summary_key]
        summary['total_calls'] += 1
        
        if metrics.status == 'success':
            summary['successful_calls'] += 1
            summary['total_tokens'] += metrics.total_tokens
            summary['total_cost'] += metrics.cost_estimate
        else:
            summary['failed_calls'] += 1
        
        summary['total_audio_bytes'] += metrics.audio_size_bytes
        summary['total_processing_time'] += metrics.total_time
        
        # Log to standard logger
        logger.info(f"Transcription Segment {metrics.segment_id}: "
                   f"Status={metrics.status}, "
                   f"Tokens={metrics.total_tokens}, "
                   f"Time={metrics.api_call_time:.2f}s, "
                   f"Audio={metrics.audio_size_bytes/1024:.1f}KB")
        
        # Keep only last 100 detailed logs to prevent memory issues
        if len(st.session_state[self.metrics_key]) > 100:
            st.session_state[self.metrics_key] = st.session_state[self.metrics_key][-100:]
    
    def get_current_session_metrics(self) -> Dict[str, Any]:
        """Get metrics for current session"""
        summary = st.session_state.get(self.summary_key, {})
        
        if summary and summary.get('total_calls', 0) > 0:
            session_duration = time.time() - summary.get('start_time', time.time())
            
            return {
                'session_duration_min': session_duration / 60,
                'total_calls': summary['total_calls'],
                'success_rate': (summary['successful_calls'] / summary['total_calls']) * 100,
                'avg_processing_time': summary['total_processing_time'] / summary['total_calls'],
                'total_tokens': summary['total_tokens'],
                'estimated_cost': summary['total_cost'],
                'total_audio_mb': summary['total_audio_bytes'] / (1024 * 1024),
                'avg_tokens_per_call': summary['total_tokens'] / max(summary['successful_calls'], 1)
            }
        
        return {}
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance patterns"""
        if self.metrics_key not in st.session_state or not st.session_state[self.metrics_key]:
            return {}
        
        df = pd.DataFrame(st.session_state[self.metrics_key])
        
        # Calculate statistics
        analysis = {
            'avg_api_latency': df['api_call_time'].mean(),
            'p50_api_latency': df['api_call_time'].quantile(0.5),
            'p95_api_latency': df['api_call_time'].quantile(0.95),
            'avg_retry_count': df['retry_count'].mean(),
            'error_rate': (len(df[df['status'] != 'success']) / len(df)) * 100,
            'avg_audio_size_kb': df['audio_size_bytes'].mean() / 1024,
            'avg_tokens_per_segment': df['total_tokens'].mean(),
            'cache_hit_rate': (df['cached_response'].sum() / len(df)) * 100,
            'total_retries': df['retry_count'].sum()
        }
        
        # Error analysis
        if 'error_type' in df.columns:
            error_distribution = df[df['error_type'].notna()]['error_type'].value_counts().to_dict()
            analysis['error_distribution'] = error_distribution
        
        # Performance over time
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')
        df_recent = df[df['timestamp_dt'] > datetime.now() - timedelta(hours=1)]
        
        if not df_recent.empty:
            analysis['recent_performance'] = {
                'last_hour_calls': len(df_recent),
                'last_hour_success_rate': (len(df_recent[df_recent['status'] == 'success']) / len(df_recent)) * 100,
                'last_hour_avg_latency': df_recent['api_call_time'].mean()
            }
        
        return analysis
    
    def export_metrics_csv(self) -> str:
        """Export metrics as CSV"""
        if self.metrics_key not in st.session_state or not st.session_state[self.metrics_key]:
            return ""
        
        df = pd.DataFrame(st.session_state[self.metrics_key])
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Add calculated fields
        df['tokens_per_second'] = df['total_tokens'] / df['api_call_time'].replace(0, 1)
        df['cost_estimate'] = df.apply(
            lambda row: (row['prompt_tokens'] / 1000 * 0.01) + (row['completion_tokens'] / 1000 * 0.03),
            axis=1
        )
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
    
    def display_metrics_dashboard(self):
        """Display metrics dashboard in Streamlit sidebar"""
        st.sidebar.markdown("### 📊 API Metrics")
        
        metrics = self.get_current_session_metrics()
        
        if metrics:
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                st.metric("Total Calls", metrics['total_calls'])
                st.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
                st.metric("Total Tokens", f"{metrics['total_tokens']:,}")
            
            with col2:
                st.metric("Avg Latency", f"{metrics['avg_processing_time']:.2f}s")
                st.metric("Est. Cost", f"${metrics['estimated_cost']:.3f}")
                st.metric("Audio Size", f"{metrics['total_audio_mb']:.1f}MB")
            
            # Performance analysis
            if st.sidebar.checkbox("Show Detailed Analytics"):
                analysis = self.get_performance_analysis()
                
                if analysis:
                    st.sidebar.markdown("#### Performance Analysis")
                    st.sidebar.json({
                        'Latency P50': f"{analysis.get('p50_api_latency', 0):.2f}s",
                        'Latency P95': f"{analysis.get('p95_api_latency', 0):.2f}s",
                        'Error Rate': f"{analysis.get('error_rate', 0):.1f}%",
                        'Cache Hit Rate': f"{analysis.get('cache_hit_rate', 0):.1f}%",
                        'Total Retries': analysis.get('total_retries', 0)
                    })
                    
                    if 'error_distribution' in analysis:
                        st.sidebar.markdown("#### Error Distribution")
                        st.sidebar.json(analysis['error_distribution'])
            
            # Export option
            if st.sidebar.button("📥 Export Metrics CSV"):
                csv_data = self.export_metrics_csv()
                if csv_data:
                    st.sidebar.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"api_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.sidebar.info("No metrics collected yet")

def measure_audio_metrics(audio_data: bytes) -> Dict[str, Any]:
    """Measure audio file metrics"""
    import wave
    import io
    
    metrics = {
        'size_bytes': len(audio_data),
        'size_kb': len(audio_data) / 1024,
        'duration_seconds': 0
    }
    
    try:
        # Try to read as WAV to get duration
        with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            metrics['duration_seconds'] = frames / float(rate)
            metrics['sample_rate'] = rate
            metrics['channels'] = wav_file.getnchannels()
    except:
        # Estimate duration based on file size (rough approximation)
        # Assuming ~16kbps for speech audio
        metrics['duration_seconds'] = (len(audio_data) * 8) / (16 * 1000)
    
    return metrics

def track_api_call(func):
    """Decorator to track API call metrics"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        retry_count = kwargs.get('retry_count', 0)
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log successful call
            logger.info(f"API call {func.__name__} succeeded in {duration:.2f}s")
            
            # Store metrics if in session state
            if 'api_call_metrics' not in st.session_state:
                st.session_state.api_call_metrics = []
            
            st.session_state.api_call_metrics.append({
                'function': func.__name__,
                'duration': duration,
                'status': 'success',
                'timestamp': time.time(),
                'retry_count': retry_count
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log failed call
            logger.error(f"API call {func.__name__} failed after {duration:.2f}s: {str(e)}")
            
            # Store failure metrics
            if 'api_call_metrics' not in st.session_state:
                st.session_state.api_call_metrics = []
            
            st.session_state.api_call_metrics.append({
                'function': func.__name__,
                'duration': duration,
                'status': 'error',
                'error': str(e),
                'timestamp': time.time(),
                'retry_count': retry_count
            })
            
            raise
    
    return wrapper

def log_token_usage(completion):
    """Extract and log token usage from OpenAI completion"""
    usage_data = {}
    
    if hasattr(completion, 'usage'):
        usage = completion.usage
        usage_data = {
            'prompt_tokens': usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
            'completion_tokens': usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
            'total_tokens': usage.total_tokens if hasattr(usage, 'total_tokens') else 0
        }
        
        logger.info(f"Token usage - Prompt: {usage_data['prompt_tokens']}, "
                   f"Completion: {usage_data['completion_tokens']}, "
                   f"Total: {usage_data['total_tokens']}")
    
    return usage_data

def create_performance_report() -> str:
    """Generate a performance optimization report"""
    if 'api_metrics_log' not in st.session_state or not st.session_state['api_metrics_log']:
        return "No data available for performance report"
    
    df = pd.DataFrame(st.session_state['api_metrics_log'])
    
    report = []
    report.append("# API Performance Optimization Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    report.append(f"- Total API Calls: {len(df)}\n")
    report.append(f"- Success Rate: {(df['status'] == 'success').mean() * 100:.1f}%\n")
    report.append(f"- Average Latency: {df['api_call_time'].mean():.2f}s\n")
    report.append(f"- Total Tokens Used: {df['total_tokens'].sum():,}\n")
    report.append(f"- Estimated Total Cost: ${df.apply(lambda x: (x['prompt_tokens']/1000*0.01 + x['completion_tokens']/1000*0.03), axis=1).sum():.2f}\n\n")
    
    # Performance Insights
    report.append("## Performance Insights\n")
    
    # Audio size impact
    if len(df) > 10:
        correlation = df['audio_size_bytes'].corr(df['api_call_time'])
        report.append(f"- Audio Size vs Latency Correlation: {correlation:.2f}\n")
        
        # Optimal audio size
        df['size_bucket'] = pd.cut(df['audio_size_bytes'], bins=5)
        size_performance = df.groupby('size_bucket')['api_call_time'].mean()
        report.append(f"- Optimal Audio Size Range: {size_performance.idxmin()}\n")
    
    # Token efficiency
    report.append(f"- Average Tokens per Second: {(df['total_tokens'] / df['api_call_time']).mean():.1f}\n")
    report.append(f"- Average Audio-to-Text Ratio: {(df['audio_duration_estimate'] / df['api_call_time']).mean():.2f}x\n")
    
    # Error patterns
    errors = df[df['status'] != 'success']
    if not errors.empty:
        report.append("\n## Error Analysis\n")
        error_types = errors['error_type'].value_counts()
        for error_type, count in error_types.items():
            report.append(f"- {error_type}: {count} occurrences\n")
    
    # Optimization Recommendations
    report.append("\n## Optimization Recommendations\n")
    
    avg_audio_size = df['audio_size_bytes'].mean()
    if avg_audio_size > 1024 * 1024:  # 1MB
        report.append("- ⚠️ Consider compressing audio before sending (average size > 1MB)\n")
    
    avg_retry = df['retry_count'].mean()
    if avg_retry > 0.5:
        report.append(f"- ⚠️ High retry rate ({avg_retry:.1f} avg), consider increasing initial delay\n")
    
    p95_latency = df['api_call_time'].quantile(0.95)
    if p95_latency > 30:
        report.append(f"- ⚠️ P95 latency is {p95_latency:.1f}s, consider implementing timeouts\n")
    
    cache_rate = df['cached_response'].mean() * 100
    if cache_rate < 10:
        report.append(f"- 💡 Low cache utilization ({cache_rate:.1f}%), consider caching more responses\n")
    
    return '\n'.join(report)