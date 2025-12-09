import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import os
from datetime import datetime

# ============================================================
# ✅ SECURE GEMINI CONFIG (Matches your new template format)
# ============================================================
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("⚠️ Gemini API Key missing. Add it in Streamlit Secrets.")
    st.stop()

# ============================================================
# ✅ UNIVERSAL AI RESPONSE HANDLER (Template Requirement)
# ============================================================
def get_ai_response(prompt, fallback="⚠️ AI error. Try again."):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") else fallback
    except Exception as e:
        return f"{fallback}\nError: {str(e)}"

# ============================================================
# 🌍 PAGE CONFIG (Template Style)
# ============================================================
st.set_page_config(
    page_title="🌍 GlobeTrack AI — Cultural Tourism",
    layout="wide"
)

st.title("🌍 GlobeTrack AI — Cultural Tourism Platform")
st.write("✨ Personalized trips, recommendations, analytics & AI chatbot — now in a clean template structure!")

# ============================================================
# 📂 Load Datasets (Template: Simple, Cached)
# ============================================================
@st.cache_data
def load_data():
    try:
        t1 = pd.read_csv("Tourist_Destinations.csv")
        t2 = pd.read_csv("tourism_dataset_5000.csv")
        t3 = pd.read_csv("Worldwide-Travel-Cities-Dataset-Ratings-and-Climate.csv")
        return t1, t2, t3
    except Exception as e:
        st.error(f"Dataset loading error: {e}")
        st.stop()

tourist_df, travel_df, cities_df = load_data()

# ============================================================
# 🧠 RAG SYSTEM (Simplified into Template Structure)
# ============================================================
@st.cache_resource
def init_rag():
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        embed = SentenceTransformer("all-MiniLM-L6-v2")
        return client, embed
    except:
        return None, None

client, embed_model = init_rag()

def build_rag_collection():
    if client is None:
        return None
    try:
        col = client.get_or_create_collection("tourism_rag")
        docs, meta, ids = [], [], []
        for i, row in tourist_df.head(800).iterrows():
            text = f"{row['Destination Name']} in {row['Country']} (Rating {row['Avg Rating']})"
            docs.append(text)
            meta.append({"destination": row["Destination Name"]})
            ids.append(f"id{i}")
        col.add(documents=docs, embeddings=embed_model.encode(docs).tolist(), metadatas=meta, ids=ids)
        return col
    except:
        return None

rag = build_rag_collection()

# ============================================================
# 📌 SIDEBAR NAVIGATION
# ============================================================
page = st.sidebar.selectbox(
    "📌 Choose Module",
    [
        "📊 Dashboard",
        "✈️ Itinerary Generator",
        "⭐ Recommendations",
        "📄 Export PDF",
        "🤖 AI Chatbot",
        "📈 Analytics",
    ]
)

# ============================================================
# 📊 DASHBOARD PAGE (Template: Clean)
# ============================================================
if page == "📊 Dashboard":
    st.header("📊 Executive Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("🌍 Destinations", len(tourist_df))
    col2.metric("👥 Travelers", len(travel_df))
    col3.metric("⭐ Avg Rating", round(tourist_df["Avg Rating"].mean(), 2))

    top = tourist_df.nlargest(10, "Avg Rating")
    fig = px.bar(top, x="Destination Name", y="Avg Rating", title="Top Rated Destinations")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ✈️ ITINERARY GENERATOR (Template Style)
# ============================================================
elif page == "✈️ Itinerary Generator":
    st.header("✈️ AI-Powered Cultural Itinerary")

    interests = st.multiselect("🎯 Interests", ["Culture", "History", "Food", "Art"])
    days = st.slider("📅 Trip Length", 3, 14, 7)
    budget = st.selectbox("💰 Budget", ["Low", "Mid", "High"])
    region = st.selectbox("🌍 Region", ["Europe", "Asia", "Africa", "Americas"])

    if st.button("✨ Generate Itinerary"):
        rag_context = ""
        if rag:
            q = rag.query(query_texts=[region], n_results=5)
            rag_context = " • " + "\n • ".join(q["documents"][0])

        prompt = f"""
        Create a detailed {days}-day itinerary:
        Interests: {interests}
        Region: {region}
        Budget: {budget}
        Include: culture, history, food, travel, timing, tips.
        RAG Destinations:
        {rag_context}
        """

        result = get_ai_response(prompt)
        st.subheader("📜 Your Itinerary")
        st.write(result)

# ============================================================
# ⭐ RECOMMENDATIONS
# ============================================================
elif page == "⭐ Recommendations":
    st.header("⭐ Smart Destination Recommendations")

    interest = st.selectbox("🎯 Primary Interest", ["Culture", "History", "Nature"])
    season = st.selectbox("🌤️ Best Season", ["Summer", "Winter", "Autumn", "Spring"])
    n = st.slider("Results", 3, 10, 5)

    df = tourist_df.copy()
    df = df[df["Best Season"].str.contains(season, case=False, na=False)]
    df = df.nlargest(n, "Avg Rating")

    st.subheader("🔥 Top Picks")
    st.dataframe(df[["Destination Name", "Country", "Avg Rating"]])

# ============================================================
# 📄 PDF EXPORT
# ============================================================
elif page == "📄 Export PDF":
    st.header("📄 Generate PDF Itinerary")

    text = st.text_area("Paste itinerary text:")
    if st.button("📥 Create PDF"):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()

        doc.build([
            Paragraph("🌍 GlobeTrack AI Itinerary", styles["Title"]),
            Spacer(1, 12),
            Paragraph(text, styles["Normal"])
        ])
        buf.seek(0)

        st.download_button("⬇️ Download PDF", buf, "itinerary.pdf")

# ============================================================
# 🤖 AI CHATBOT
# ============================================================
elif page == "🤖 AI Chatbot":
    st.header("🤖 Travel Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])

    if q := st.chat_input("Ask anything about travel..."):
        st.session_state.chat.append({"role": "user", "content": q})

        rag_context = ""
        if rag:
            r = rag.query(query_texts=[q], n_results=3)
            rag_context = "\n".join(r["documents"][0])

        answer = get_ai_response(f"Context:\n{rag_context}\n\nUser question: {q}")
        st.session_state.chat.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

# ============================================================
# 📈 ANALYTICS
# ============================================================
elif page == "📈 Analytics":
    st.header("📈 Usage Analytics (Simulated)")

    fake = pd.DataFrame({
        "Module": ["Itinerary", "Chatbot", "Recommendations", "PDF"],
        "Rating": [4.8, 4.7, 4.5, 4.6]
    })
    fig = px.bar(fake, x="Module", y="Rating", color="Rating")
    st.plotly_chart(fig)
