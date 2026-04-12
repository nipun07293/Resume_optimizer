# Enterprise AI Resume and Career Optimizer

An advanced resume parsing and Applicant Tracking System (ATS) simulator. This tool leverages state-of-the-art Natural Language Processing (NLP), machine learning, and deep learning architectures to optimize resumes against job descriptions. It moves beyond simple rule-based pattern matching by utilizing hybrid AI solutions, including semantic similarity evaluation and intelligent entity extraction.

## Key Features

* **AI-Powered ATS Compatibility Score:** Computes comprehensive match scores using an ensemble approach: keyword extraction, structural formatting checks, and semantic evaluation via a custom deep neural network.
* **Semantic Skill Gap Analysis:** Identifies missing technical qualifications and core competencies by comparing extracted entities from the candidate's resume and the target job description.
* **Impact and Formatting Diagnostics:** Evaluates the presence of quantifiable metrics, strong past-tense action verbs, and standard ATS-compliant structural sections.
* **Visual Analytics and Scorecards:** Provides actionable feedback and visual summaries of resume metrics using interactive gauge and bar charts.
* **AI Interview Prep Simulator:** Generates targeted interview questions based on the candidate's specific skill gaps and matched qualifications.

## Technical Stack

* **Frontend Integration:** Streamlit, Plotly
* **Machine Learning & Deep Learning:** TensorFlow, Keras, Sentence-Transformers (all-MiniLM-L6-v2)
* **Natural Language Processing:** spaCy (en_core_web_sm)
* **Data Processing & Extraction:** Pandas, NumPy, PyPDF2, Regex

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/ai-resume-optimizer.git](https://github.com/yourusername/ai-resume-optimizer.git)
   cd ai-resume-optimizer