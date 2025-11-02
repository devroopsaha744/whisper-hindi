"""
Utility script to analyze the downloaded dataset and generate statistics.

Usage:
    python scripts/analyze_dataset.py --data_dir data
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

def load_data(data_dir):
    """Load downloaded data and results."""
    data_dir = Path(data_dir)
    results_file = data_dir / 'download_results.csv'
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    return df


def analyze_downloads(df):
    """Analyze download success rates."""
    print("\n" + "="*60)
    print("DOWNLOAD STATISTICS")
    print("="*60)
    
    total = len(df)
    audio_success = df['audio_success'].sum()
    trans_success = df['transcription_success'].sum()
    both_success = ((df['audio_success']) & (df['transcription_success'])).sum()
    meta_success = df['metadata_success'].sum() if 'metadata_success' in df.columns else 0
    
    print(f"Total recordings: {total}")
    print(f"Audio downloads: {audio_success} ({audio_success/total*100:.1f}%)")
    print(f"Transcription downloads: {trans_success} ({trans_success/total*100:.1f}%)")
    print(f"Complete pairs (audio + transcription): {both_success} ({both_success/total*100:.1f}%)")
    if 'metadata_success' in df.columns:
        print(f"Metadata downloads: {meta_success} ({meta_success/total*100:.1f}%)")
    
    return both_success


def analyze_transcriptions(data_dir):
    """Analyze transcription texts."""
    print("\n" + "="*60)
    print("TRANSCRIPTION ANALYSIS")
    print("="*60)
    
    trans_dir = Path(data_dir) / 'transcriptions'
    
    if not trans_dir.exists():
        print("Transcriptions directory not found")
        return
    
    transcriptions = []
    word_counts = []
    char_counts = []
    
    for trans_file in trans_dir.glob('*.txt'):
        try:
            with open(trans_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                transcriptions.append(text)
                word_counts.append(len(text.split()))
                char_counts.append(len(text))
        except Exception as e:
            print(f"Error reading {trans_file}: {e}")
    
    if not transcriptions:
        print("No transcriptions found")
        return
    
    print(f"Total transcriptions analyzed: {len(transcriptions)}")
    print(f"\nWord count statistics:")
    print(f"  Mean: {np.mean(word_counts):.1f} words")
    print(f"  Median: {np.median(word_counts):.1f} words")
    print(f"  Min: {np.min(word_counts)} words")
    print(f"  Max: {np.max(word_counts)} words")
    print(f"  Std: {np.std(word_counts):.1f} words")
    
    print(f"\nCharacter count statistics:")
    print(f"  Mean: {np.mean(char_counts):.1f} characters")
    print(f"  Median: {np.median(char_counts):.1f} characters")
    print(f"  Min: {np.min(char_counts)} characters")
    print(f"  Max: {np.max(char_counts)} characters")
    
    # Sample transcriptions
    print(f"\nSample transcriptions:")
    for i, text in enumerate(transcriptions[:3], 1):
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"  {i}. {preview}")
    
    return {
        'word_counts': word_counts,
        'char_counts': char_counts,
        'transcriptions': transcriptions[:10]  # Save only first 10
    }


def analyze_audio_files(data_dir):
    """Analyze audio file metadata."""
    print("\n" + "="*60)
    print("AUDIO FILE ANALYSIS")
    print("="*60)
    
    audio_dir = Path(data_dir) / 'audio'
    
    if not audio_dir.exists():
        print("Audio directory not found")
        return
    
    audio_files = list(audio_dir.glob('*.wav'))
    
    if not audio_files:
        print("No audio files found")
        return
    
    print(f"Total audio files: {len(audio_files)}")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in audio_files)
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"Average file size: {total_size / len(audio_files) / 1e6:.2f} MB")
    
    # Try to analyze audio properties using librosa (if available)
    try:
        import librosa
        import soundfile as sf
        
        durations = []
        sample_rates = []
        
        # Sample 10 files for detailed analysis
        sample_files = np.random.choice(audio_files, min(10, len(audio_files)), replace=False)
        
        for audio_file in sample_files:
            try:
                info = sf.info(audio_file)
                durations.append(info.duration)
                sample_rates.append(info.samplerate)
            except Exception as e:
                print(f"  Warning: Could not read {audio_file.name}: {e}")
        
        if durations:
            print(f"\nAudio properties (sampled from {len(sample_files)} files):")
            print(f"  Duration - Mean: {np.mean(durations):.1f}s, "
                  f"Min: {np.min(durations):.1f}s, "
                  f"Max: {np.max(durations):.1f}s")
            print(f"  Sample rates: {set(sample_rates)}")
            
            # Estimate total duration
            avg_duration = np.mean(durations)
            total_duration = avg_duration * len(audio_files)
            print(f"\nEstimated total duration: {total_duration / 3600:.2f} hours")
    
    except ImportError:
        print("\nNote: Install librosa and soundfile for detailed audio analysis")
        print("  pip install librosa soundfile")


def generate_visualizations(data_dir, stats):
    """Generate visualization plots."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    output_dir = Path(data_dir) / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Word count distribution
    if stats and 'word_counts' in stats:
        plt.figure(figsize=(10, 6))
        plt.hist(stats['word_counts'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.title('Distribution of Transcription Word Counts')
        plt.grid(True, alpha=0.3)
        
        output_file = output_dir / 'word_count_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
        
        # Character count distribution
        plt.figure(figsize=(10, 6))
        plt.hist(stats['char_counts'], bins=50, edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('Number of Characters')
        plt.ylabel('Frequency')
        plt.title('Distribution of Transcription Character Counts')
        plt.grid(True, alpha=0.3)
        
        output_file = output_dir / 'char_count_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


def save_report(data_dir, complete_pairs, stats):
    """Save analysis report."""
    output_dir = Path(data_dir) / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    report = {
        'complete_pairs': int(complete_pairs),
        'transcription_stats': {
            'word_counts': {
                'mean': float(np.mean(stats['word_counts'])) if stats and 'word_counts' in stats else None,
                'median': float(np.median(stats['word_counts'])) if stats and 'word_counts' in stats else None,
                'min': int(np.min(stats['word_counts'])) if stats and 'word_counts' in stats else None,
                'max': int(np.max(stats['word_counts'])) if stats and 'word_counts' in stats else None,
            },
            'char_counts': {
                'mean': float(np.mean(stats['char_counts'])) if stats and 'char_counts' in stats else None,
                'median': float(np.median(stats['char_counts'])) if stats and 'char_counts' in stats else None,
                'min': int(np.min(stats['char_counts'])) if stats and 'char_counts' in stats else None,
                'max': int(np.max(stats['char_counts'])) if stats and 'char_counts' in stats else None,
            }
        },
        'sample_transcriptions': stats.get('transcriptions', []) if stats else []
    }
    
    report_file = output_dir / 'dataset_analysis.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Analysis report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze downloaded Hindi ASR dataset"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing downloaded data (default: data)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    
    # Load data
    df = load_data(args.data_dir)
    
    # Analyze downloads
    complete_pairs = analyze_downloads(df)
    
    # Analyze transcriptions
    trans_stats = analyze_transcriptions(args.data_dir)
    
    # Analyze audio files
    analyze_audio_files(args.data_dir)
    
    # Generate visualizations
    if trans_stats:
        generate_visualizations(args.data_dir, trans_stats)
    
    # Save report
    save_report(args.data_dir, complete_pairs, trans_stats)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.data_dir}/analysis/")


if __name__ == "__main__":
    main()
