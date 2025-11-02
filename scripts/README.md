# Scripts Directory

This directory contains utility scripts for data preparation and analysis.

## Available Scripts

### 1. download_dataset.py

Downloads audio files and transcriptions from URLs provided in a dataset file.

**Purpose**: Fetch all training data from cloud storage to local disk

**Usage**:
```bash
python scripts/download_dataset.py --input_file dataset.xlsx --output_dir data --max_workers 4
```

**Arguments**:
- `--input_file` (required): Path to CSV/Excel file with dataset URLs
- `--output_dir` (default: 'data'): Directory to save downloaded files
- `--max_workers` (default: 4): Number of parallel download threads
- `--max_samples` (optional): Limit number of samples to download

**Input File Format**:
Must contain these columns:
- `recording_id`: Unique identifier for each recording
- `rec_url_gcp`: URL to audio file
- `transcription_url`: URL to transcription text
- `metadata_url_gcp` (optional): URL to metadata

**Output Structure**:
```
data/
├── audio/              # Audio files (.wav)
├── transcriptions/     # Transcription texts (.txt)
├── metadata/          # Metadata files (optional)
└── download_results.csv  # Download status log
```

**Examples**:
```bash
# Download full dataset
python scripts/download_dataset.py --input_file dataset.xlsx

# Download only 100 samples for testing
python scripts/download_dataset.py --input_file dataset.xlsx --max_samples 100

# Use more workers for faster download
python scripts/download_dataset.py --input_file dataset.xlsx --max_workers 8

# Custom output directory
python scripts/download_dataset.py --input_file dataset.xlsx --output_dir my_data
```

**Features**:
- ✅ Parallel downloads with configurable workers
- ✅ Automatic retry on failure (3 attempts)
- ✅ Progress tracking with tqdm
- ✅ Skip already downloaded files
- ✅ Detailed logging of success/failure
- ✅ Exponential backoff for retries

---

### 2. analyze_dataset.py

Analyzes downloaded dataset and generates statistics.

**Purpose**: Understand dataset characteristics before training

**Usage**:
```bash
python scripts/analyze_dataset.py --data_dir data
```

**Arguments**:
- `--data_dir` (default: 'data'): Directory containing downloaded data

**Analyses Performed**:
1. **Download Statistics**
   - Success rate for audio/transcriptions
   - Failed downloads summary

2. **Transcription Analysis**
   - Word count distribution
   - Character count distribution
   - Sample transcriptions

3. **Audio Analysis**
   - Total number of files
   - Total size and average file size
   - Duration statistics (if librosa available)

4. **Visualizations**
   - Word count histogram
   - Character count histogram

**Output**:
```
data/analysis/
├── dataset_analysis.json        # Detailed statistics
├── word_count_distribution.png  # Visualization
└── char_count_distribution.png  # Visualization
```

**Examples**:
```bash
# Analyze default data directory
python scripts/analyze_dataset.py

# Analyze custom directory
python scripts/analyze_dataset.py --data_dir my_data
```

**Requirements**:
- `pandas`, `numpy` (required)
- `matplotlib`, `seaborn` (for visualizations)
- `librosa`, `soundfile` (optional, for audio analysis)

---

## Common Workflows

### Workflow 1: Initial Data Download

```bash
# 1. Download data
python scripts/download_dataset.py --input_file dataset.xlsx

# 2. Analyze downloaded data
python scripts/analyze_dataset.py --data_dir data

# 3. Check results
cat data/download_results.csv
cat data/analysis/dataset_analysis.json
```

### Workflow 2: Incremental Download

```bash
# Download first batch
python scripts/download_dataset.py --input_file dataset.xlsx --max_samples 100

# Analyze
python scripts/analyze_dataset.py

# Download full dataset
python scripts/download_dataset.py --input_file dataset.xlsx
```

### Workflow 3: Handle Failed Downloads

```bash
# Check failed downloads
cat data/failed_downloads.csv

# Retry with fewer workers (more stable)
python scripts/download_dataset.py --input_file dataset.xlsx --max_workers 2
```

---

## Troubleshooting

### Issue: Download Failures

**Symptoms**: Many entries in `failed_downloads.csv`

**Solutions**:
1. Check internet connection
2. Verify URLs are accessible
3. Reduce `--max_workers` to 2 or 1
4. Check for firewall/proxy issues

### Issue: Out of Memory During Analysis

**Symptoms**: Script crashes when analyzing large datasets

**Solutions**:
1. Analyze subset of files
2. Disable visualizations (edit script)
3. Increase system RAM

### Issue: Encoding Errors

**Symptoms**: Cannot read transcription files

**Solutions**:
1. Check file encoding (should be UTF-8)
2. Verify download completed successfully
3. Re-download problematic files

---

## Script Details

### download_dataset.py

**Key Functions**:
- `DatasetDownloader`: Main class for downloading
  - `download_file()`: Downloads single file with retry
  - `download_recording()`: Downloads all files for one recording
  - `download_dataset()`: Orchestrates parallel downloads

**Error Handling**:
- Automatic retry (3 attempts)
- Exponential backoff between retries
- Graceful failure (continues with other downloads)
- Detailed error logging

**Performance**:
- Parallel downloads using ThreadPoolExecutor
- Configurable number of workers
- Skip already downloaded files
- Efficient for large datasets

### analyze_dataset.py

**Key Functions**:
- `load_data()`: Loads download results
- `analyze_downloads()`: Download statistics
- `analyze_transcriptions()`: Text analysis
- `analyze_audio_files()`: Audio file analysis
- `generate_visualizations()`: Create plots
- `save_report()`: Export JSON report

**Statistics Calculated**:
- Download success rates
- Text length distributions
- Audio duration estimates
- File size summaries

---

## Adding Custom Scripts

To add your own scripts to this directory:

1. Create a new Python file
2. Add argparse for command-line arguments
3. Follow the same structure as existing scripts
4. Document usage in this README

Example template:
```python
"""
My custom script for [purpose].

Usage:
    python scripts/my_script.py --arg value
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="My script")
    parser.add_argument('--arg', type=str, help='Argument description')
    args = parser.parse_args()
    
    # Your code here
    
    print("Done!")

if __name__ == "__main__":
    main()
```

---

## Dependencies

Both scripts require:
- Python 3.8+
- pandas
- numpy
- requests
- tqdm

Optional dependencies:
- matplotlib, seaborn (for visualizations)
- librosa, soundfile (for audio analysis)

Install with:
```bash
pip install pandas numpy requests tqdm matplotlib seaborn librosa soundfile
```

Or use the project requirements:
```bash
pip install -r requirements.txt
```

---

## Best Practices

1. **Always check download_results.csv** after downloading
2. **Run analysis before training** to understand your data
3. **Start with small sample** (use --max_samples) for testing
4. **Save analysis results** for future reference
5. **Monitor disk space** during downloads

---

## Notes

- Scripts are designed to be idempotent (safe to rerun)
- Failed downloads can be retried without re-downloading successes
- All paths are relative to script location
- Progress bars show real-time status
- Logs are comprehensive for debugging

---

For questions or issues with these scripts, refer to the main project documentation in the root directory.
