# ===============================================
# 🌟 AI Cultural Tourism Insights & Engagement Platform
# ===========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Gemini GenAI configuration
# -------------------------
import google.generativeai as genai

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # use a modern model name — change if required by your account
    AI_MODEL_NAME = "gemini-1.5-pro"
else:
    st.error("⚠️ GEMINI_API_KEY not found in Streamlit Secrets. Add it and reload.")
    st.stop()

def get_ai_response(prompt, fallback="⚠️ AI response unavailable."):
    try:
        model = genai.GenerativeModel(AI_MODEL_NAME)
        resp = model.generate_content(prompt)
        # some SDKs provide .text or .response; handle safely
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        if hasattr(resp, "content") and resp.content:
            return str(resp.content).strip()
        return fallback
    except Exception as e:
        return f"⚠️ AI Error: {e}\n{fallback}"

# -------------------------
# Page config & session
# -------------------------
st.set_page_config(page_title="🌟 AI Cultural Tourism Pro 2025", page_icon="🌍", layout="wide")
st.title("🌍 AI Cultural Tourism Pro 2025")

if "points" not in st.session_state:
    st.session_state.points = 0
    st.session_state.level = 1
    st.session_state.achievements = []
if "datasets" not in st.session_state:
    st.session_state.datasets = {}  # name -> dataframe
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def award_points(n):
    st.session_state.points += n
    # level-up threshold (simple)
    if st.session_state.points >= st.session_state.level * 1000:
        st.session_state.level += 1
        st.session_state.achievements.append(f"Level {st.session_state.level} unlocked!")
        st.balloons()

# -------------------------
# Sidebar - trending features & upload
# -------------------------
st.sidebar.title("🚀 Cultural Tourism AI Pro")
st.sidebar.markdown("### ✅ Trending features (AR removed)")
st.sidebar.markdown("""
- 🎙️ Voice (upload voice / placeholder)
- 📱 Social sharing (auto links)
- 🎮 Gamification (points & badges)
- 📈 Live Trends (computed from uploaded CSVs)
- 🌐 Multilingual GenAI responses
""")
c1, c2 = st.sidebar.columns(2)
c1.metric("🌟 Points", st.session_state.points)
c2.metric("🏆 Level", st.session_state.level)

if st.sidebar.button("🎁 Daily Bonus (+250)"):
    award_points(250)
    st.sidebar.success("Bonus awarded!")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Upload reference CSVs (upload all at once)")
uploaded_files = st.sidebar.file_uploader("Upload one or more CSVs", accept_multiple_files=True, type=["csv"])
if uploaded_files:
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
        except Exception:
            # try with other encodings / separators
            try:
                df = pd.read_csv(f, encoding='latin1')
            except Exception:
                st.sidebar.warning(f"Could not read {f.name} — skipping.")
                continue
        st.session_state.datasets[f.name] = df
    st.sidebar.success(f"Loaded {len(st.session_state.datasets)} dataset(s).")
    # auto-select first uploaded dataset
    if st.session_state.selected_dataset is None and len(st.session_state.datasets) > 0:
        st.session_state.selected_dataset = list(st.session_state.datasets.keys())[0]

# -------------------------
# Main Navigation
# -------------------------
page = st.sidebar.radio("Navigate", [
    "🔎 EDA (Plotly) & Data Prep",
    "✈️ AI Itinerary",
    "⭐ Smart Recommendations",
    "📱 Social Share",
    "📄 PDF Export",
    "🎬 Video Maker",
    "💬 Smart Chat (Multilingual)",
    "📊 Analytics & Live Trends"
])

# =========================
# EDA SECTION (first)
# =========================
if page == "🔎 EDA (Plotly) & Data Prep":
    st.header("🔎 Interactive EDA with Plotly — choose a dataset")
    if not st.session_state.datasets:
        st.info("Upload one or more CSVs from the sidebar to start EDA. Example datasets: destinations, travelers, cities.")
        st.stop()

    dataset_names = list(st.session_state.datasets.keys())
    ds_choice = st.selectbox("Select dataset", dataset_names, index=dataset_names.index(st.session_state.selected_dataset) if st.session_state.selected_dataset in dataset_names else 0)
    st.session_state.selected_dataset = ds_choice
    df = st.session_state.datasets[ds_choice]
    st.markdown(f"**{ds_choice}** — rows: {df.shape[0]}, columns: {df.shape[1]}")

    # show head & sample
    with st.expander("Preview data (first 10 rows)"):
        st.dataframe(df.head(10))

    # basic info & describe
    with st.expander("Dataset summary"):
        st.write("Columns & dtypes")
        meta = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes], "n_missing": df.isna().sum().values})
        st.dataframe(meta, use_container_width=True)
        st.write("Numeric describe")
        st.dataframe(df.select_dtypes(include='number').describe().T)

    # Choose column for plot
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("### 📊 Quick Plots")
    plot_type = st.selectbox("Plot type", ["Histogram", "Bar (top categories)", "Scatter", "Correlation heatmap"])
    if plot_type == "Histogram":
        if not numeric_cols:
            st.warning("No numeric columns available for histogram.")
        else:
            col = st.selectbox("Numeric column", numeric_cols)
            nbins = st.slider("Bins", 5, 200, 30)
            fig = px.histogram(df, x=col, nbins=nbins, title=f"Histogram: {col}")
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Bar (top categories)":
        if not categorical_cols:
            st.warning("No categorical columns available.")
        else:
            cat = st.selectbox("Categorical column", categorical_cols)
            top_n = st.slider("Top N categories", 3, 50, 10)
            vc = df[cat].fillna("N/A").value_counts().nlargest(top_n).reset_index()
            vc.columns = [cat, "count"]
            fig = px.bar(vc, x=cat, y="count", title=f"Top {top_n} categories: {cat}")
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Scatter":
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns for scatter.")
        else:
            xcol = st.selectbox("X axis", numeric_cols, index=0)
            ycol = st.selectbox("Y axis", numeric_cols, index=1)
            color = st.selectbox("Color by (optional)", [None] + categorical_cols)
            fig = px.scatter(df, x=xcol, y=ycol, color=color, title=f"Scatter: {ycol} vs {xcol}", hover_data=df.columns)
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Correlation heatmap":
        if not numeric_cols:
            st.warning("No numeric columns for correlation.")
        else:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
            st.plotly_chart(fig, use_container_width=True)

    # Allow user to pick a "master" dataset for other tabs
    st.markdown("---")
    if st.button("🔖 Use this dataset for the app (set as master)"):
        st.session_state.master_name = ds_choice
        st.success(f"{ds_choice} set as master dataset for the rest of the app.")
        award_points(50)

# =========================
# ITINERARY (Gemini)
# =========================
elif page == "✈️ AI Itinerary":
    st.header("✈️ Premium AI Itinerary (Gemini-powered)")
    master_df = st.session_state.datasets.get(getattr(st.session_state, "master_name", None), None)
    # fallback: pick any uploaded dataset that looks like destinations
    if master_df is None:
        st.info("No master dataset set — using first uploaded dataset as reference (may not be travel-specific).")
        master_df = next(iter(st.session_state.datasets.values()))

    with st.form("it_form"):
        interests = st.multiselect("Interests", ['Culture','History','Nature','Food','Architecture','Adventure'], default=['Culture'])
        days = st.slider("Days", 1, 21, 7)
        budget = st.selectbox("Budget", ["Low","Mid","High"])
        language = st.selectbox("Language", ["English","Spanish","French","Hindi","Japanese"])
        submit = st.form_submit_button("Generate Itinerary")
    if submit:
        # build a safe sample of destination names if available
        dest_col_candidates = [c for c in master_df.columns if "dest" in c.lower() or "name" in c.lower() or "site" in c.lower()]
        dest_list = master_df[dest_col_candidates[0]].dropna().unique().tolist()[:30] if dest_col_candidates else []
        prompt = (
            f"Create a professional {days}-day itinerary in {language} for interests {', '.join(interests)}.\n"
            f"Budget level: {budget}.\n"
            f"Use these destinations as inspiration: {dest_list}.\n"
            "Include daily schedule, transport tips, cultural highlights, sustainability tips, packing tips, and emojis."
        )
        out = get_ai_response(prompt)
        st.markdown("### ✨ Generated Itinerary")
        st.text_area("Itinerary", out, height=400)
        award_points(150)

# =========================
# SMART RECOMMENDATIONS
# =========================
elif page == "⭐ Smart Recommendations":
    st.header("⭐ Smart Recommendations (data-driven + GenAI)")
    master_df = st.session_state.datasets.get(getattr(st.session_state, "master_name", None), None)
    if master_df is None:
        st.info("Set a master dataset (from EDA) for meaningful recommendations.")
        st.stop()

    # try to find rating and type-like columns
    rating_cols = [c for c in master_df.columns if "rating" in c.lower() or "score" in c.lower()]
    type_cols = [c for c in master_df.columns if c.lower() in ("type","category","tag","tags","genre") or "type" in c.lower()]

    interest = st.selectbox("Interest (approx)", ['Culture','History','Nature','Food','Architecture'])
    season_opt = st.selectbox("Season (if available)", sorted(master_df['Best Season'].dropna().unique()) if 'Best Season' in master_df.columns else ["Any"])

    # filter by season if present
    df_candidates = master_df.copy()
    if 'Best Season' in df_candidates.columns and season_opt != "Any":
        df_candidates = df_candidates[df_candidates['Best Season'] == season_opt]

    # filter by interest if there is a Type-like column
    if type_cols:
        type_col = type_cols[0]
        recs = df_candidates[df_candidates[type_col].str.contains(interest, case=False, na=False)] \
               if df_candidates[type_col].dtype == object else df_candidates
    else:
        recs = df_candidates

    # sort by rating if available
    if rating_cols:
        recs = recs.sort_values(by=rating_cols[0], ascending=False)
    recs = recs.head(8)

    st.markdown(f"### Top recommendations ({len(recs)})")
    for i, row in recs.iterrows():
        cols = st.columns([1,4,1])
        rating_display = f"{row[rating_cols[0]]:.1f}" if rating_cols else "-"
        with cols[0]:
            st.metric("⭐", rating_display)
        with cols[1]:
            name_candidates = [c for c in row.index if "name" in c.lower() or "destination" in c.lower() or "site" in c.lower()]
            title = row[name_candidates[0]] if name_candidates else row.index[0]
            st.markdown(f"**{title}**")
            info = []
            for c in ['Country','Type','Best Season']:
                if c in row.index:
                    info.append(f"{c}: {row[c]}")
            st.caption(" | ".join(info))
        with cols[2]:
            if st.button("❤️ Like", key=f"like_{i}"):
                award_points(25)

# =========================
# SOCIAL SHARE
# =========================
elif page == "📱 Social Share":
    st.header("📱 Social Sharing")
    story = st.text_area("Write your travel story / caption", "Planned my trip with AI Tourism Pro! #TravelAI")
    share_buttons = st.columns(3)
    if share_buttons[0].button("Share to Facebook"):
        st.success("Facebook share link created below — copy & paste to a browser.")
        st.code(f"https://facebook.com/sharer.php?u={st.session_state.get('app_url','https://your-app-url.example')}&quote={story}")
        award_points(75)
    if share_buttons[1].button("Share to Twitter/X"):
        st.success("Twitter share link generated.")
        st.code(f"https://twitter.com/intent/tweet?text={story[:250]}")
        award_points(75)
    if share_buttons[2].button("Instagram Story Tip"):
        st.info("For Instagram Story: use the snippet above as caption and upload an eye-catching photo. (Streamlit cannot post directly.)")
        award_points(50)

# =========================
# PDF EXPORT
# =========================
elif page == "📄 PDF Export":
    st.header("📄 Professional PDF Export")
    itinerary_text = st.text_area("Paste itinerary or report to export as PDF", "Day 1: ...\nDay 2: ...")
    if st.button("Generate PDF"):
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = [Paragraph("🌟 AI Cultural Itinerary Pro", styles['Title']), Spacer(1, 12),
                 Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']), Spacer(1,12),
                 Paragraph(itinerary_text.replace("\n","<br/>"), styles['Normal'])]
        doc.build(story)
        buffer.seek(0)
        st.download_button("📥 Download PDF", data=buffer.getvalue(), file_name="itinerary.pdf", mime="application/pdf")
        award_points(75)

# =========================
# VIDEO MAKER (optional moviepy)
# =========================
elif page == "🎬 Video Maker":
    st.header("🎬 Trip Recap Video (image slideshow -> video)")
    uploaded_imgs = st.file_uploader("Upload images to create a short recap (max 10)", accept_multiple_files=True, type=["png","jpg","jpeg"])
    duration_per_slide = st.slider("Seconds per slide", 1, 6, 2)
    if uploaded_imgs and st.button("Generate Video"):
        try:
            from moviepy.editor import ImageClip, concatenate_videoclips
        except Exception:
            st.error("moviepy not installed in this environment. To enable video generation, install moviepy.")
            st.stop()
        clips = []
        for f in uploaded_imgs[:10]:
            img = Image.open(f).convert("RGB")
            clip = ImageClip(np.asarray(img)).set_duration(duration_per_slide)
            clips.append(clip)
        final = concatenate_videoclips(clips, method="compose")
        tmpfile = "recap.mp4"
        final.write_videofile(tmpfile, fps=24, codec="libx264", audio=False, verbose=False, logger=None)
        video_bytes = open(tmpfile, "rb").read()
        st.video(video_bytes)
        award_points(150)

# =========================
# SMART CHAT (Gemini)
# =========================
elif page == "💬 Smart Chat (Multilingual)":
    st.header("💬 Multilingual Smart Chat — powered by Gemini")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    if prompt := st.chat_input("Ask about destinations, itineraries, or datasets..."):
        st.session_state.messages.append({'role':'user','content':prompt})
        with st.chat_message("assistant"):
            reply = get_ai_response(prompt)
            st.markdown(reply)
            st.session_state.messages.append({'role':'assistant','content':reply})
        award_points(25)

# =========================
# ANALYTICS & LIVE TRENDS
# =========================
elif page == "📊 Analytics & Live Trends":
    st.header("📊 Analytics & Live Trends (from uploaded CSVs)")
    if not st.session_state.datasets:
        st.info("Upload datasets to compute live trends.")
        st.stop()

    # simple combined trend computation: if datasets contain 'Destination' or 'Searches' or 'Popularity' fields, use them
    combined = pd.concat(
        [df.assign(__source=name) for name, df in st.session_state.datasets.items()],
        ignore_index=True, sort=False
    )

    # Attempt to find a destination/name column
    name_cols = [c for c in combined.columns if any(k in c.lower() for k in ("destination","name","site","city"))]
    metric_cols = [c for c in combined.columns if any(k in c.lower() for k in ("popul","search","count","rating","score"))]

    if name_cols and metric_cols:
        name_col = name_cols[0]; metric_col = metric_cols[0]
        trend_df = combined.groupby(name_col)[metric_col].agg(['mean','count']).reset_index().sort_values('mean', ascending=False).head(10)
        st.markdown(f"Top 10 by `{metric_col}`")
        fig = px.bar(trend_df, x=name_col, y='mean', hover_data=['count'], title=f"Top destinations by {metric_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # fallback: show counts of sources
        src_counts = combined['__source'].value_counts().reset_index(); src_counts.columns = ['dataset','rows']
        fig = px.pie(src_counts, names='dataset', values='rows', title="Uploaded datasets: row distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Live Feature Metrics (simulated / sample)")
    metrics_df = pd.DataFrame({
        "Feature": ["Itinerary","Recommendations","Voice","Social","Chatbot"],
        "Usage": [int(np.random.randint(400,1600)) for _ in range(5)],
        "Rating": [round(np.random.uniform(4.2,4.9),2) for _ in range(5)],
        "Growth": [int(np.random.randint(5,60)) for _ in range(5)]
    })
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(metrics_df, x="Feature", y="Usage", color="Growth", title="Feature Usage & Growth")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.scatter(metrics_df, x="Usage", y="Rating", size="Growth", text="Feature", title="Usage vs Rating")
        st.plotly_chart(fig2, use_container_width=True)

    # reward user for exploring analytics
    if st.button("✨ Claim analytics explorer reward (+100 pts)"):
        award_points(100)
        st.success("Points added.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built with Gemini GenAI • Plotly • Streamlit — Upload CSVs to begin. Add GEMINI_API_KEY in Streamlit Secrets.")
