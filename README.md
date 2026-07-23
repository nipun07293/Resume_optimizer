# ResumeIQ — AI Resume & ATS Match Optimizer

A resume-to-job-description match scorer combining a fine-tuned semantic
similarity regressor with rule-based NLP diagnostics. Given a resume (PDF)
and a target job description, it produces an overall ATS-style match score
plus a breakdown of *why* — missing keywords, weak quantification, missing
sections — with concrete fixes.

## Architecture

```
Resume PDF ──► PyPDF2 text extraction ─┐
                                        ├─► SBERT (all-MiniLM-L6-v2) ─► [res_emb, job_emb, |diff|, product] ─► Dense NN ─► semantic_score
Job Description ───────────────────────┘

Resume text ──► spaCy POS-tag + stoplist filter ──► keyword set ──► keyword_score (set overlap)
Resume text ──► regex ──► quantification count, spaCy verb-tag filter ──► impact_score
Resume text ──► section-header regex ──► format_score

final_score = 0.25 * semantic_score + 0.35 * keyword_score + 0.25 * impact_score + 0.15 * format_score
```

The neural network is a 4-layer dense regressor (512→256→128→64→1) trained
to predict a resume↔job match score from a 1536-dim feature vector built
from SBERT sentence embeddings (concat + absolute difference + elementwise
product of the resume and job-description embeddings — a standard NLI-style
interaction feature set).

**Note on ground truth:** `matched_score` in the training data is a
similarity score derived by the dataset's authors (Kaggle "Resume Dataset"),
not human-annotated hiring outcomes. Framed here as a *learned semantic
match estimator*, not a hiring-outcome predictor.

## Model Performance

Run `python train_model.py` to reproduce. On a held-out test split (15% of
9,544 resume/job pairs, never seen during training or early stopping):

| Metric | Value |
|---|---|
| MAE | *fill in from `models/metrics.json` after training* |
| RMSE | *fill in from `models/metrics.json` after training* |
| R² | *fill in from `models/metrics.json` after training* |
| Pearson r | *fill in from `models/metrics.json` after training* |

Training curves and a predicted-vs-actual scatter plot are saved to
`models/training_curves.png` and `models/pred_vs_actual.png`.

## Project Structure

```
resume_optimizer/
├── app.py                  # Streamlit UI
├── train_model.py          # training pipeline: train/val/test split, metrics, plots
├── src/
│   ├── features.py         # keyword/quantification/section extraction (shared, tested)
│   └── model.py             # NN architecture + feature-building (shared)
├── tests/
│   └── test_features.py    # unit tests for rule-based extraction logic
├── models/                  # saved_resume_model.keras, metrics.json, plots
├── data/
│   └── resume_data.csv      # training data
├── requirements.txt
└── Dockerfile
```

## Setup

```bash
git clone https://github.com/nipun07293/Resume_optimizer.git && cd Resume_optimizer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python train_model.py       # trains model, writes models/metrics.json + plots
pytest tests/                # run unit tests

streamlit run app.py
```

Or with Docker:
```bash
docker build -t resume-optimizer .
docker run -p 8501:8501 resume-optimizer
```

## Key Design Decisions

- **Separated feature extraction from the Streamlit app** (`src/features.py`)
  so the exact same logic is unit-testable and reusable in the training/eval
  pipeline — no drift between what's scored at inference time and what's
  evaluated offline.
- **No data leakage:** train/val/test splits are made *before* SBERT
  encoding, and validation loss (not training loss) drives early stopping.
- **Graceful failure:** missing model file, missing spaCy model, and
  unreadable/scanned PDFs all produce a clear UI message instead of a crash.

## Known Limitations

- Skill extraction is POS-tag + stoplist based, not a trained NER model —
  it will miss multi-word technical terms and occasionally include
  near-miss nouns. A fine-tuned NER model or a curated skills taxonomy
  (e.g. ESCO/O*NET) would improve precision.
- `matched_score` labels come from the source dataset's own similarity
  heuristic rather than verified hiring outcomes, so the regressor learns to
  approximate that heuristic, not true candidate-job fit.
- No confidence interval on the NN's prediction is currently surfaced to
  the user.
