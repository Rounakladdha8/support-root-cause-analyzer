import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from groq import Groq
import re

client = Groq(api_key="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")

@st.cache_data
def load_and_cluster_data():
    df = pd.read_csv("twcs.csv")
    df = df[df["inbound"] == True].copy()
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    df["cleaned_text"] = df["text"].apply(clean_text)
    keywords = ['order', 'food', 'items', 'dasher', 'refund', 'doordash']
    mask = df['cleaned_text'].str.contains('|'.join(keywords), case=False, na=False)
    df = df[mask].copy()
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df["cleaned_text"])
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans.fit(X)
    df["cluster"] = kmeans.labels_
    return df

df_food = load_and_cluster_data()

cluster_names = {
    0: "Refund Requests",
    1: "Late or Missing Delivery",
    2: "Pre-order Issues",
    3: "Food Delivery Complaints",
    4: "Poor Customer Service",
    5: "Delivery Status Issues"
}

st.title("🍕 Food Delivery Support — Root Cause Dashboard")
st.markdown("Analyzing customer complaints using NLP clustering and AI-generated insights")

col1, col2, col3 = st.columns(3)
col1.metric("Total Complaints Analyzed", f"{len(df_food):,}")
col2.metric("Clusters Found", "6")
col3.metric("Data Source", "Twitter Support")

st.subheader("Complaint Volume by Category")
chart_data = df_food['cluster'].value_counts().reset_index()
chart_data.columns = ['cluster', 'count']
chart_data['category'] = chart_data['cluster'].map(cluster_names)
st.bar_chart(chart_data.set_index('category')['count'])

st.subheader("🤖 AI-Generated Root Cause Analysis")

selected_cluster = st.selectbox(
    "Select a complaint category to analyze:",
    options=list(cluster_names.keys()),
    format_func=lambda x: cluster_names[x]
)

st.write("**Sample complaints from this category:**")
samples = df_food[df_food['cluster'] == selected_cluster]['cleaned_text'].sample(5, random_state=42).tolist()
for s in samples:
    st.write(f"• {s}")

if st.button("Generate AI Postmortem"):
    with st.spinner("Analyzing complaints..."):
        def generate_postmortem(cluster_num, n_samples=50):
            samples = df_food[df_food['cluster'] == cluster_num]['cleaned_text'].sample(n_samples, random_state=42).tolist()
            complaints_text = "\n".join(samples)
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": f"""You are a customer experience analyst at a food delivery company.
Here are {n_samples} real customer complaints from the same category:
{complaints_text}
In 3 sentences:
1. What is the root cause?
2. What operational fix would you recommend?"""
                }]
            )
            return chat_completion.choices[0].message.content
        
        postmortem = generate_postmortem(selected_cluster)
        st.success("**Root Cause Analysis:**")
        st.write(postmortem)