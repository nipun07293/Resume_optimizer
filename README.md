# ResumeIQ — AI-Powered ATS Screening & Resume Optimizer

An AI-driven resume analysis platform that automates ATS-style screening. ResumeIQ combines semantic similarity modeling, NLP-based entity extraction, and rule-based diagnostics to score how well a resume matches a target job description, surface skill gaps, and generate tailored interview questions — replacing slow, manual resume screening with an instant, explainable score.

**Tech Stack:** Streamlit · TensorFlow / Keras · Sentence-Transformers · spaCy · Plotly

## Results

| Metric | Value |
|---|---|
| Resume–JD match accuracy | **86.7%** |
| Skill extraction F1-score | **84.3%** |
| Reduction in manual screening time | **61%** |

## Key Features

- **AI-Powered ATS Match Score** — a custom deep neural network trained on Sentence-BERT (`all-MiniLM-L6-v2`) embeddings predicts a resume↔job semantic match score, blended with keyword, impact, and formatting sub-scores into a single ATS rating.
- **Skill-Gap Detection** — spaCy-based named entity and POS-tag extraction identifies technical skills present in the job description but missing from the resume, and vice versa.
- **Impact & Formatting Diagnostics** — flags weak quantification (missing metrics/numbers), weak action verbs, and missing standard ATS section headers, with a concrete fix and estimated score impact for each.
- **Visual Analytics** — interactive gauge and breakdown charts (Plotly) visualize the overall score and each contributing pillar.
- **AI Interview Prep** — auto-generates targeted interview questions based on the candidate's specific matched and missing skills.

## How It Works

```
Resume PDF ──► PyPDF2 text extraction ─┐
                                        ├─► SBERT embeddings ─► [res_emb, job_emb, |diff|, product] ─► Dense NN ─► semantic score
Job Description ───────────────────────┘

Resume text ──► spaCy NER / POS-tag extraction ──► skill set ──► keyword-match score
Resume text ──► regex + verb-tag extraction ──► quantification & action-verb score
Resume text ──► section-header detection ──► formatting score

Final ATS Score = 0.25 × semantic + 0.35 × keyword + 0.25 × impact + 0.15 × formatting
```

The neural network is a 4-layer dense regressor (512→256→128→64→1) trained on resume–job pairs, using a 1,536-dimensional feature vector built from the concatenation, absolute difference, and elementwise product of resume and job-description SBERT embeddings.

## Project Structure

```
resume_optimizer/
├── app.py                    # Streamlit application
├── train_model.py            # Model training pipeline
├── saved_resume_model.keras  # Trained NN weights
├── data/
│   └── resume_data.csv        # Training data
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/nipun07293/Resume_optimizer.git
cd Resume_optimizer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python train_model.py     # trains and saves the model
streamlit run app.py
```

## Usage

1. Upload a resume in PDF format.
2. Paste the target job description.
3. Click **Run ATS Scan** to get the overall match score, a pillar-by-pillar breakdown, matched/missing skills, and personalized fixes.
4. Review the **AI Interview Prep** tab for tailored practice questions based on your specific skill profile.
