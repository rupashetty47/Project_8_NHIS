import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(page_title="Job Market Explorer", layout="wide")
st.title("Job Market Analysis & Job Recommendations")
st.caption("Exploratory analysis of the supplied Upwork postings dataset. Rates and forecasts are dataset-specific.")


import gdown

# Download the dataset from Google Drive if it is not present
path = "all_upwork_jobs_2024-02-07-2024-03-24.csv"
file_id = "1Jk49jo4pp0qtiKWxQ-fGyh4GAXNmq67G"

if not os.path.exists(path):
    with st.spinner("Downloading job dataset..."):
        gdown.download(
            id=file_id,
            output=path,
            quiet=False
        )

upload = st.file_uploader("Upload the job CSV", type=["csv"])

if upload is not None:
    df = pd.read_csv(upload)
elif os.path.exists(path):
    df = pd.read_csv(path)
else:
    st.info("Upload the CSV or place it beside app.py.")
    st.stop()

required = {"title","published_date","is_hourly","hourly_low","hourly_high","budget","country","link"}
if not required.issubset(df.columns):
    st.error(f"Missing columns: {sorted(required-set(df.columns))}"); st.stop()
df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
for c in ["hourly_low","hourly_high","budget"]: df[c] = pd.to_numeric(df[c], errors="coerce")
df["title"] = df["title"].fillna("").astype(str)
df["country"] = df["country"].fillna("Unknown").astype(str)
df["hourly_mid"] = df[["hourly_low","hourly_high"]].mean(axis=1)
df["hourly_mid"] = df["hourly_mid"].where(df["is_hourly"] == True)
df["month"] = df["published_date"].dt.strftime("%Y-%m")

c1,c2,c3 = st.columns(3)
c1.metric("Job postings", f"{len(df):,}")
c2.metric("Hourly postings", f"{int(df['is_hourly'].sum()):,}")
c3.metric("Countries listed", f"{df['country'].nunique():,}")

st.subheader("Posting trends")
weekly = df.dropna(subset=["published_date"]).set_index("published_date").resample("W").size().rename("postings").reset_index()
st.plotly_chart(px.line(weekly,x="published_date",y="postings",markers=True,title="Weekly postings"), use_container_width=True)

st.subheader("Country hourly rates")
valid = df[(df["is_hourly"] == True) & df["hourly_mid"].between(1,300) & (df["country"]!="Unknown")]
country = valid.groupby("country").agg(jobs=("hourly_mid","size"),median_rate=("hourly_mid","median")).query("jobs >= 30").reset_index().sort_values("median_rate",ascending=False)
st.plotly_chart(px.bar(country.head(25),x="country",y="median_rate",hover_data=["jobs"],title="Median hourly rate (minimum 30 records)"), use_container_width=True)

st.subheader("Find matching jobs")
profile = st.text_input("Enter your skills or desired role", "Python SQL data analysis")
country_choices = ["Any"] + sorted(df["country"].unique().tolist())
country_filter = st.selectbox("Country",country_choices)
hourly_only = st.checkbox("Hourly jobs only")
if st.button("Recommend jobs"):
    vectorizer = TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=1,max_features=30000)
    matrix = vectorizer.fit_transform(df["title"].fillna(""))
    scores = cosine_similarity(vectorizer.transform([profile]),matrix).ravel()
    result = df.copy(); result["relevance"] = scores
    if hourly_only: result = result[result["is_hourly"] == True]
    if country_filter != "Any": result = result[result["country"] == country_filter]
    cols = ["title","country","is_hourly","hourly_low","hourly_high","budget","published_date","link","relevance"]
    st.dataframe(result.sort_values("relevance",ascending=False)[cols].head(30),use_container_width=True)
