import streamlit as st
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import PyPDF2
import io
import os
import re
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ultimate AI Resume ATS", layout="wide")

@st.cache_resource
def load_all_models():
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "saved_resume_model.keras")
    
    try:
        nn_model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Neural Network not found. Run train_model.py first! Error: {e}")
        nn_model = None
        
    return encoder, nlp, nn_model

encoder, nlp, nn_model = load_all_models()

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def extract_skills(text):
    doc = nlp(text.lower())
    
    ignore_words = {
        'engineer', 'developer', 'analyst', 'scientist', 'manager', 'director', 
        'administrator', 'architect', 'designer', 'consultant', 'specialist', 
        'expert', 'lead', 'executive', 'officer', 'coordinator', 'supervisor', 'worker',
        'plus', 'advantage', 'understanding', 'knowledge', 'expertise', 'proficiency', 
        'familiarity', 'exposure', 'background', 'foundation', 'concept', 'mindset', 
        'passion', 'drive', 'motivation', 'enthusiasm', 'attitude', 'track', 'record', 
        'candidate', 'opportunity', 'environment', 'field', 'industry', 'requirement',
        'ability', 'capability', 'capacity', 'talent', 'competency', 'qualification',
        'behavior', 'test', 'growth', 'retention', 'responsibility', 'product', 
        'title', 'business', 'strategy', 'manipulation', 'dashboard', 'experience', 
        'years', 'team', 'work', 'job', 'project', 'skills', 'using', 'impact', 'results', 
        'data', 'user', 'role', 'company', 'development', 'management', 'system', 
        'process', 'support', 'design', 'application', 'performance', 'solution', 
        'tool', 'client', 'technology', 'service', 'issue', 'quality', 'time', 'value', 
        'degree', 'bachelor', 'master', 'phd', 'certification', 'training', 'course', 
        'class', 'program', 'workshop', 'seminar', 'conference', 'student', 'graduate',
        'report', 'document', 'file', 'record', 'database', 'network', 'software', 
        'hardware', 'platform', 'framework', 'library', 'module', 'component', 'interface', 
        'marketing', 'finance', 'accounting', 'operations', 'logistics', 'supply', 'chain', 
        'production', 'safety', 'security', 'compliance', 'legal', 'policy', 'procedure', 
        'testing', 'maintenance', 'troubleshooting', 'analysis', 'research', 'innovation', 
        'planning', 'execution', 'leadership', 'collaboration', 'teamwork', 'creativity', 
        'communication', 'presentation', 'deployment', 'implementation'
    }
    
    keywords = set()
    for token in doc:
        clean_word = token.lemma_.strip()
        if (token.pos_ in ['NOUN', 'PROPN'] and 
            len(clean_word) > 2 and 
            clean_word not in ignore_words and 
            not token.is_stop and 
            re.search('[a-zA-Z]', clean_word)):
            
            keywords.add(clean_word)
            
    return keywords

def check_quantification(text):
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    percentages = re.findall(r'\b\d+%', text)
    return len(numbers) + len(percentages)

def extract_action_verbs(text):
    doc = nlp(text)
    verbs = set([token.text.capitalize() for token in doc if token.pos_ == 'VERB' and token.tag_ in ['VBD', 'VBN']])
    return list(verbs)

def check_sections(text):
    standard_sections = ['EDUCATION', 'EXPERIENCE', 'SKILLS', 'PROJECTS', 'CERTIFICATIONS']
    found_sections = []
    text_upper = text.upper()
    for sec in standard_sections:
        if sec in text_upper:
            found_sections.append(sec)
    return found_sections

def create_gauge_chart(score):
    color = "#e74c3c" if score < 60 else "#f1c40f" if score < 80 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Overall ATS Match Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': '#fadbd8'},
                {'range': [60, 80], 'color': '#fcf3cf'},
                {'range': [80, 100], 'color': '#d5f5e3'}
            ]
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_breakdown_bar_chart(keyword, semantic, impact, format_score):
    categories = ['Keyword Match (35%)', 'Semantic Context (25%)', 'Impact/Metrics (25%)', 'Formatting (15%)']
    scores = [keyword, semantic, impact, format_score]
    
    fig = px.bar(
        x=scores, 
        y=categories, 
        orientation='h', 
        text=[f"{s:.1f}%" for s in scores],
        color=scores,
        color_continuous_scale=['#e74c3c', '#f1c40f', '#2ecc71'],
        range_color=[0, 100]
    )
    fig.update_layout(
        title="Score Breakdown by Pillar",
        xaxis_title="Score out of 100",
        yaxis_title="",
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_traces(textposition='inside')
    return fig

st.title("Enterprise AI Resume & Career Optimizer")
st.markdown("Featuring Predictive Impact Analytics, Actionable Feedback, and Smart Skill Extraction.")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    resume_text = ""
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("Resume Parsed Successfully!")

with col2:
    job_description = st.text_area("Paste Target Job Description", height=130)

st.markdown("---")

if st.button("Run Enterprise ATS Scan", type="primary", use_container_width=True):
    if not resume_text or not job_description:
        st.warning("Please provide both a Resume and a Job Description.")
    elif nn_model is None:
        st.error("Model missing.")
    else:
        with st.spinner("Analyzing semantic overlap, parsing metrics, and calculating impact..."):
            
            res_vec = encoder.encode([resume_text])
            job_vec = encoder.encode([job_description])
            diff_vec = np.abs(res_vec - job_vec)
            prod_vec = res_vec * job_vec
            X_input = np.hstack((res_vec, job_vec, diff_vec, prod_vec))
            
            nn_score_pred = nn_model.predict(X_input, verbose=0)[0][0]
            semantic_score = max(0.0, min(nn_score_pred, 1.0)) * 100 
            
            res_keywords = extract_skills(resume_text)
            job_keywords = extract_skills(job_description)
            missing_keywords = job_keywords - res_keywords
            matched_keywords = job_keywords.intersection(res_keywords)
            keyword_score = (len(matched_keywords) / len(job_keywords)) * 100 if len(job_keywords) > 0 else 100

            quant_count = check_quantification(resume_text)
            verbs = extract_action_verbs(resume_text)
            
            target_quants = 8
            target_verbs = 10
            
            quant_score = min(100, (quant_count / target_quants) * 100)
            verb_score = min(100, (len(verbs) / target_verbs) * 100)
            impact_score = (quant_score * 0.6) + (verb_score * 0.4)

            word_count = len(resume_text.split())
            sections_found = check_sections(resume_text)
            length_score = 100 if 300 <= word_count <= 800 else (50 if word_count < 300 else 70)
            section_score = (len(sections_found) / 4) * 100
            format_score = (length_score * 0.4) + (min(100.0, section_score) * 0.6)

            final_ats_score = (semantic_score * 0.25) + (keyword_score * 0.35) + (impact_score * 0.25) + (format_score * 0.15)

        dash_col1, dash_col2 = st.columns([1, 1.5])
        with dash_col1:
            st.plotly_chart(create_gauge_chart(final_ats_score), use_container_width=True)
        with dash_col2:
            st.plotly_chart(create_breakdown_bar_chart(keyword_score, semantic_score, impact_score, format_score), use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Personalized Diagnostics & Fixes", "Technical Skill Gaps", "AI Interview Prep"])

        with tab1:
            st.subheader("Your Specific Weaknesses & Score Impact")
            st.write("We calculated exactly how much your ATS score will increase if you fix the following issues:")
            issues_found = False
            
            if quant_count < target_quants:
                issues_found = True
                missing_quants = target_quants - quant_count
                potential_boost = ((missing_quants / target_quants) * 100) * 0.6 * 0.25
                st.error(f"Weakness: Lack of Quantifiable Results (You only have {quant_count} numbers/metrics in your resume).")
                st.info(f"How to improve: Add {missing_quants} more numbers or percentages to your bullet points (e.g., 'Improved speed by 20%').")
                st.success(f"Impact: Fixing this will increase your overall ATS Score by +{potential_boost:.1f}%")
                st.markdown("---")
                
            if len(verbs) < target_verbs:
                issues_found = True
                missing_verbs = target_verbs - len(verbs)
                potential_boost = ((missing_verbs / target_verbs) * 100) * 0.4 * 0.25
                st.error(f"Weakness: Weak Action Verbs (We only found {len(verbs)} strong past-tense verbs).")
                st.info(f"How to improve: Start {missing_verbs} more bullet points with words like: Spearheaded, Architected, Optimized, Deployed.")
                st.success(f"Impact: Fixing this will increase your overall ATS Score by +{potential_boost:.1f}%")
                st.markdown("---")
                
            if len(sections_found) < 4:
                issues_found = True
                st.error(f"Weakness: Missing Standard ATS Sections. (Found: {', '.join(sections_found) if sections_found else 'None'})")
                st.info("How to improve: Ensure your resume has exact, capitalized headers like 'EXPERIENCE', 'EDUCATION', 'SKILLS', and 'PROJECTS'.")
                potential_boost = ((4 - len(sections_found)) / 4) * 100 * 0.6 * 0.15
                st.success(f"Impact: Fixing this will increase your overall ATS Score by +{potential_boost:.1f}%")
                
            if not issues_found:
                st.success("Your Structural Formatting and Impact metrics are flawless! Focus purely on Keyword Matching.")

        with tab2:
            st.subheader("Technical Skill Gap Analysis")
            col_match, col_miss = st.columns(2)
            with col_match:
                st.success(f"Matched Tools & Tech ({len(matched_keywords)})")
                if matched_keywords:
                    st.write(", ".join([k.capitalize() for k in matched_keywords]))
                else:
                    st.write("None found.")
                    
            with col_miss:
                st.error(f"Missing Tools & Tech ({len(missing_keywords)})")
                if missing_keywords:
                    st.write(", ".join([k.capitalize() for k in missing_keywords]))
                    
                    if len(job_keywords) > 0:
                        kw_boost = (len(missing_keywords) / len(job_keywords)) * 100 * 0.35
                        st.success(f"Impact: Weaving these missing skills into your resume will boost your score by +{kw_boost:.1f}%")
                    
                    st.markdown("**Suggested Real-World Learning:**")
                    top_missing = list(missing_keywords)[:3]
                    for tm in top_missing:
                        if len(tm) > 2:
                            st.markdown(f"- Search for **'{tm.capitalize()} for beginners'** on YouTube or take a crash course on Udemy/Coursera to confidently add this to your resume.")
                else:
                    st.success("You possess all the required technical skills for this role!")

        with tab3:
            st.subheader("AI-Generated Interview Simulator")
            st.write("Based on your resume and this job description, be prepared to answer these questions:")
            
            if matched_keywords:
                skill = list(matched_keywords)[0].capitalize()
                st.markdown(f"**1.** Can you walk me through a specific project where you heavily utilized **{skill}**? What was your exact contribution?")
            
            if missing_keywords:
                mskill = list(missing_keywords)[0].capitalize()
                st.markdown(f"**2.** We rely heavily on **{mskill}** for this role, which I don't see prominently on your resume. How quickly can you get up to speed with it, and do you have parallel experience?")
            
            st.markdown("**3.** I noticed some impressive metrics on your resume. Can you describe a time when a project didn't go as planned and how you pivoted to ensure its success?")
            st.info("Pro Tip: Use the STAR method (Situation, Task, Action, Result) to answer these!")