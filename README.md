# Support Ticket Root Cause Analyzer

An agentic NLP pipeline that automatically classifies 97,000+ 
unstructured customer support tweets into root cause categories 
and generates AI-powered postmortem analyses — mimicking what 
customer experience teams do manually every day.

## Problem It Solves
Support teams manually read thousands of complaints to identify 
systemic patterns. This pipeline automates that process end to end.

## Pipeline
Raw tweets → Text cleaning → TF-IDF vectorization → 
KMeans clustering → LLM postmortem generation → Interactive dashboard

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (TF-IDF, KMeans)
- Groq API (LLaMA 3.3)
- Streamlit

## How To Run
pip install -r requirements.txt
streamlit run app.py

## Features
- Classifies 97K+ complaints into 6 root cause categories
- Interactive dashboard with complaint volume by category
- AI-generated postmortem and operational recommendation per category
- One-click analysis for any complaint cluster

# Download dataset from: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
