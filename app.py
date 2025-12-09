import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -----------------------------------------------------------
# 🔐 GEMINI API
# -----------------------------------------------------------
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Add GEMINI_API_KEY in Streamlit Secrets")
    st.stop()

def ai(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        out = model.generate_content(prompt)
        return out.text
    except:
        return "AI Error. Try again."

# -----------------------------------------------------------
# 🌍 PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="GlobeTrack AI", layout="wide")
st.title("🌍 GlobeTrack AI — Cultural Tourism Platform")
st.write("Simple, clean and powerful AI-driven travel app.")

# -----------------------------------------------------------
# 📂 LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    t1 = pd.read_csv("Tourist_Destinations.csv")
    t2 = pd.read_csv("tourism_dataset_5000.csv")
    t3 = pd.read_csv("Worldwide-Travel-Cities-Dataset-Ratings-and-Climate.csv")
    return t1, t2, t3

tour_df, travel_df, city_df = load_data()

# -----------------------------------------------------------
# 🧠 SIMPLE RAG SETUP
# -----------------------------------------------------------
@st.cache_resource
def setup_rag():
    try:
        client = chromadb.PersistentClient(path="./rag_db")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        col = client.get_or_create_collection("tourism")

        # Insert only small sample to keep simple
        docs = []
        ids = []
        for i, r in tour_df.head(300).iterrows():
            text = f"{r['Destination Name']} in {r['Country']}"
            docs.append(text)
            ids.append(str(i))

        col.add(
            documents=docs,
            embeddings=model.encode(docs).tolist(),
            ids=ids
        )
        return col, model
    except:
        return None, None

rag, embed = setup_rag()

def rag_search(query):
    if rag is None:
        return ""
    result = rag.query(
        query_embeddings=embed.encode([query]).tolist(),
        n_results=3
    )
    return "\n".join(result["documents"][0])

# -----------------------------------------------------------
# 📌 SIDEBAR
# -----------------------------------------------------------
page = st.sidebar.radio(
    "Choose Section",
    ["Dashboard", "Itinerary", "Recommendations", "Chatbot", "Export PDF"]
)

# -----------------------------------------------------------
# 📊 DASHBOARD
# -----------------------------------------------------------
if page == "Dashboard":
    st.header("📊 Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Destinations", len(tour_df))
    c2.metric("Travellers", len(travel_df))
    c3.metric("Avg Rating", round(tour_df["Avg Rating"].mean(), 2))

    top = tour_df.nlargest(10, "Avg Rating")
    fig = px.bar(top, x="Destination Name", y="Avg Rating", title="Top Rated Places")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# ✈️ ITINERARY
# -----------------------------------------------------------
elif page == "Itinerary":
    st.header("✈️ Create Itinerary")

    interests = st.multiselect("Interests", ["Culture", "Food", "History", "Art"])
    days = st.slider("Days", 3, 14)
    region = st.selectbox("Region", ["Asia", "Europe", "Africa", "Americas"])

    if st.button("Generate"):
        ctx = rag_search(region)
        prompt = f"""
        Make a {days}-day trip itinerary.
        Interests: {interests}
        Region: {region}
        Suggested places:
        {ctx}
        """

        out = ai(prompt)
        st.write(out)

# -----------------------------------------------------------
# ⭐ RECOMMENDATIONS
# -----------------------------------------------------------
elif page == "Recommendations":
    st.header("⭐ Recommendations")

    season = st.selectbox("Season", ["Summer", "Winter", "Spring", "Autumn"])
    count = st.slider("How many?", 3, 10)

    df = tour_df[tour_df["Best Season"].str.contains(season, case=False, na=False)]
    df = df.nlargest(count, "Avg Rating")

    st.dataframe(df[["Destination Name", "Country", "Avg Rating"]])

# -----------------------------------------------------------
# 🤖 CHATBOT
# -----------------------------------------------------------
elif page == "Chatbot":
    st.header("🤖 AI Travel Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])

    q = st.chat_input("Ask something...")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        ctx = rag_search(q)
        answer = ai(f"Context: {ctx}\n\nQuestion: {q}")
        st.session_state.chat.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

# -----------------------------------------------------------
# 📄 PDF EXPORT
# -----------------------------------------------------------
elif page == "Export PDF":
    st.header("📄 Export to PDF")

    text = st.text_area("Paste your itinerary here")

    if st.button("Create PDF"):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()

        doc.build([
            Paragraph("GlobeTrack AI Itinerary", styles["Title"]),
            Spacer(1, 12),
            Paragraph(text.replace("\n", "<br/>"), styles["Normal"])
        ])

        buf.seek(0)
        st.download_button("Download PDF", buf, "itinerary.pdf")
