"""
Script to download audio files and transcriptions from the Hindi ASR dataset.

Usage:
    python download_dataset.py --input_file dataset.xlsx --output_dir data --max_workers 4
"""

import os
import argparse
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Downloads audio files and transcriptions from URLs."""
    
    def __init__(self, output_dir, max_retries=3, timeout=30):
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio"
        self.transcription_dir = self.output_dir / "transcriptions"
        self.metadata_dir = self.output_dir / "metadata"
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcription_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directories created at: {self.output_dir}")
    
    def download_file(self, url, output_path, file_type="file"):
        """Download a single file with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # Write file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                return True, None
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return False, str(e)
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return False, "Max retries exceeded"
    
    def download_recording(self, row):
        """Download audio, transcription, and metadata for a single recording."""
        recording_id = row['recording_id']
        user_id = row['user_id']
        
        results = {
            'recording_id': recording_id,
            'user_id': user_id,
            'audio_success': False,
            'transcription_success': False,
            'metadata_success': False,
            'audio_path': None,
            'transcription_path': None,
            'metadata_path': None,
            'errors': []
        }
        
        # Download audio file
        if pd.notna(row.get('rec_url_gcp')):
            audio_filename = f"{recording_id}.wav"
            audio_path = self.audio_dir / audio_filename
            
            if not audio_path.exists():
                success, error = self.download_file(row['rec_url_gcp'], audio_path, "audio")
                results['audio_success'] = success
                if success:
                    results['audio_path'] = str(audio_path)
                else:
                    results['errors'].append(f"Audio download failed: {error}")
            else:
                results['audio_success'] = True
                results['audio_path'] = str(audio_path)
        
        # Download transcription
        if pd.notna(row.get('transcription_url_gcp')):
            trans_filename = f"{recording_id}.txt"
            trans_path = self.transcription_dir / trans_filename
            
            if not trans_path.exists():
                success, error = self.download_file(row['transcription_url_gcp'], trans_path, "transcription")
                results['transcription_success'] = success
                if success:
                    results['transcription_path'] = str(trans_path)
                else:
                    results['errors'].append(f"Transcription download failed: {error}")
            else:
                results['transcription_success'] = True
                results['transcription_path'] = str(trans_path)
        
        # Download metadata (optional)
        if pd.notna(row.get('metadata_url_gcp')):
            meta_filename = f"{recording_id}_metadata.json"
            meta_path = self.metadata_dir / meta_filename
            
            if not meta_path.exists():
                success, error = self.download_file(row['metadata_url_gcp'], meta_path, "metadata")
                results['metadata_success'] = success
                if success:
                    results['metadata_path'] = str(meta_path)
                else:
                    results['errors'].append(f"Metadata download failed: {error}")
            else:
                results['metadata_success'] = True
                results['metadata_path'] = str(meta_path)
        
        return results
    
    def download_dataset(self, df, max_workers=4):
        """Download all files in the dataset using parallel workers."""
        logger.info(f"Starting download of {len(df)} recordings with {max_workers} workers...")
        
        results = []
        failed_downloads = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_row = {
                executor.submit(self.download_recording, row): idx 
                for idx, row in df.iterrows()
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(df), desc="Downloading") as pbar:
                for future in as_completed(future_to_row):
                    result = future.result()
                    results.append(result)
                    
                    if not (result['audio_success'] and result['transcription_success']):
                        failed_downloads.append(result)
                    
                    pbar.update(1)
        
        # Create summary
        successful_audio = sum(1 for r in results if r['audio_success'])
        successful_trans = sum(1 for r in results if r['transcription_success'])
        successful_meta = sum(1 for r in results if r['metadata_success'])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Download Summary:")
        logger.info(f"  Total recordings: {len(df)}")
        logger.info(f"  Successful audio downloads: {successful_audio}/{len(df)}")
        logger.info(f"  Successful transcription downloads: {successful_trans}/{len(df)}")
        logger.info(f"  Successful metadata downloads: {successful_meta}/{len(df)}")
        logger.info(f"  Failed downloads: {len(failed_downloads)}")
        logger.info(f"{'='*60}\n")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_path = self.output_dir / "download_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Download results saved to: {results_path}")
        
        # Save failed downloads
        if failed_downloads:
            failed_df = pd.DataFrame(failed_downloads)
            failed_path = self.output_dir / "failed_downloads.csv"
            failed_df.to_csv(failed_path, index=False)
            logger.warning(f"Failed downloads saved to: {failed_path}")
        
        return results


def load_dataset(input_file):
    """Load dataset from CSV or Excel file."""
    file_path = Path(input_file)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    logger.info(f"Loading dataset from: {input_file}")
    
    if file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    elif file_path.suffix.lower() == '.csv':
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded {len(df)} recordings")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Validate required columns
    required_cols = ['recording_id', 'rec_url_gcp', 'transcription_url_gcp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download Hindi ASR dataset from URLs"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to input CSV/Excel file containing dataset URLs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for downloaded files (default: data)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help='Number of parallel download workers (default: 4)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to download (default: all)'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    df = load_dataset(args.input_file)
    
    # Limit samples if specified
    if args.max_samples:
        df = df.head(args.max_samples)
        logger.info(f"Limited to {args.max_samples} samples")
    
    # Create downloader and start downloading
    downloader = DatasetDownloader(args.output_dir)
    results = downloader.download_dataset(df, max_workers=args.max_workers)
    
    logger.info("Download complete!")


if __name__ == "__main__":
    main()
