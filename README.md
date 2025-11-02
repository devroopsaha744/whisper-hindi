# Whisper-Small Hindi ASR Fine-tuning — Experiment Report

This report captures the full fine-tuning and evaluation journey for adapting OpenAI's Whisper-small model to conversational Hindi ASR. The focus is on the data-centric design choices, training configuration, evaluation workflow, and measurable gains over the baseline.

## Project Goals
- Build a Hindi-specific ASR model by fine-tuning `openai/whisper-small` on curated speech segments.
- Preserve reproducibility by scripting data preparation and sharing all artifacts (models, metrics, and visualizations) on Hugging Face.
- Quantify improvements against the Hindi split of the FLEURS benchmark and highlight where the tuned model excels.

## Data Preparation Strategy
- **Segment-level supervision:** `scripts/download_dataset.py` pulls full-call audio and structured transcripts. In `notebooks/training-pipeline.ipynb` we explode each transcription JSON into per-segment examples (text with `start`/`end` timestamps).
- **Quality filters:** Segments longer than 30 seconds or labels exceeding 448 tokens are dropped before feature extraction. Empty or redacted transcriptions are skipped.
- **Consistent sampling:** All audio is resampled to 16 kHz and clipped to the exact segment window using `librosa`. This keeps Whisper's log-mel features aligned with the segment text.
- **Train/validation split:** 95% of the filtered segments feed the fine-tuning loop, with the remainder reserved for validation metrics during training.

## Fine-tuning Configuration
- **Base model:** `openai/whisper-small`
- **Objective:** Speech-to-text transcription (`task="transcribe"`, language hint set to Hindi).
- **Optimizer setup:** Learning rate `1e-5`, batch size 4, gradient accumulation 2, 3 epochs, 500 warmup steps.
- **Trainer:** `Seq2SeqTrainer` with fp16 enabled on GPU, evaluation + checkpointing every 500 steps, best model selected by lowest validation WER.
- **Outputs:** Model weights and processor are stored under `models/` and pushed to Hugging Face at [`datafreak/whisper-hindi`](https://huggingface.co/datafreak/whisper-hindi).

These choices are documented and reproducible inside `notebooks/training_pipeline.ipynb`, which also handles the Hugging Face authentication via the configured token.

## Evaluation Workflow
- **Benchmark:** Hindi (`hi_in`) split of the Google FLEURS dataset, loaded in `notebooks/evaluation-pipeline.ipynb`.
- **Baselines compared:**
	- Baseline Whisper-small (zero-shot).
	- Fine-tuned `datafreak/whisper-hindi` checkpoint.
- **Metric:** Word Error Rate (WER) via `evaluate.load("wer")`.
- **Automation:** The notebook saves a comparison table and detailed per-utterance predictions, then publishes them to the Hugging Face dataset repo [`datafreak/whisper-hindi-eval-results`](https://huggingface.co/datasets/datafreak/whisper-hindi-eval-results).
- **Rehydration:** Downstream analysis reloads the published CSVs from the Hub to ensure the visuals reflect the shared artifacts, not local state.

## Results Overview
- The fine-tuned checkpoint consistently outperforms the baseline in aggregate WER, as illustrated below.
- Per-sample analysis shows the largest gains on medium-length utterances with conversational phrasing, while extremely short or noisy clips remain challenging.
- The improvement correlates with segment length: longer utterances (within the 30s cap) benefit most from Hindi-specific language modeling during decoding.

### Visual Summary
- ![WER Comparison](graphs/wer_comparison.png)
	The aggregate WER chart highlights the gap between the baseline and fine-tuned models on FLEURS Hindi.
- ![WER Distribution](graphs/wer_distribution.png)
	Histograms reveal the shift in per-utterance WER distribution after fine-tuning, with fewer high-error outliers.
- ![Top Improvements](graphs/top_wer_improvements.png)
	The top-gains plot surfaces the segments where Hindi adaptation makes the most difference, often colloquial speech with code-switching.
- ![Length vs Improvement](graphs/length_vs_improvement.png)
	Scatter view relating reference word counts to WER deltas, capturing the sweet spot where the tuned decoder stabilizes.

All four figures are generated from Hub-hosted CSVs and automatically uploaded alongside them in the dataset repository.

## Published Artifacts
- **Fine-tuned model:** [`datafreak/whisper-hindi`](https://huggingface.co/datafreak/whisper-hindi)
- **Evaluation dataset + assets:** [`datafreak/whisper-hindi-eval-results`](https://huggingface.co/datasets/datafreak/whisper-hindi-eval-results)
- **Detailed predictions:** `results/detailed_predictions.csv` (mirrored on the Hub), containing side-by-side reference, baseline, and fine-tuned outputs.
- **Comparison metrics:** `results/wer_comparison.csv`, the authoritative WER snapshot published to the dataset repo.

## Key Takeaways
- Targeted segment filtering (duration, token length) keeps Whisper within its optimal context window, lifting accuracy without architectural changes.
- Hindi-specific fine-tuning reduces WER across the board, particularly for conversational speech with regional vocabulary that the baseline misrecognizes.
- Hosting metrics and visualizations on Hugging Face ensures the evaluation narrative is auditable and immediately consumable for stakeholders.

## Next Steps
- Broaden coverage with accented Hindi and code-mixed Hinglish segments to stress-test robustness.
- Incorporate confidence scoring or semi-supervised augmentation for low-resource speakers.
- Explore larger Whisper variants (base/medium) using the same pipeline to trade latency for accuracy.

_Last updated: November 2025_
