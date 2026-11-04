"""AI or Human? A web app that estimates whether an essay was written by a person or by AI.

Mini project for Programming for Data Science, by Seyran Gurbanov.
Run locally with:  streamlit run app.py
"""
import html
import json
import os
import re
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from text_features import FEATURE_NAMES, text_features
except ImportError:
    from src.text_features import FEATURE_NAMES, text_features

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "model", "ai_detector.joblib")
STATS_PATH = os.path.join(ROOT, "model", "model_stats.json")
DATA_PATH = os.path.join(ROOT, "data", "essays.csv.gz")
DB_PATH = os.path.join(ROOT, "data", "history.db")
FIG_DIR = os.path.join(ROOT, "reports", "figures")

HUMAN_COLOR, AI_COLOR = "#2E86AB", "#E4572E"

st.set_page_config(page_title="AI or Human? Essay Detector", page_icon="🔍", layout="wide")

SAMPLE_AI = (
    "In today's rapidly evolving world, technology plays a crucial role in shaping the way students learn. "
    "Online learning offers numerous benefits, such as flexibility and access to a wide range of resources. "
    "Additionally, it allows students to learn at their own pace, which can lead to a deeper understanding "
    "of the material. However, it is important to consider the potential drawbacks. Students may feel "
    "isolated without face-to-face interaction, and some may struggle with self-discipline. Furthermore, "
    "not all students have reliable internet access, which can create inequalities in education. "
    "In conclusion, while online learning provides valuable opportunities, a balanced approach that "
    "combines digital tools with traditional teaching is essential to ensure that every student can thrive."
)
SAMPLE_HUMAN = (
    "I think students should not be forced to do online classes from home. Last year my school tried it for "
    "two weeks when the heating broke and honestly it was a mess. My little brother kept using the laptop for "
    "games and the wifi in our house is really bad, so half the time i couldnt even hear the teacher. Also "
    "its way harder to focus when your in your bedroom and your phone is right there. Some of my friends "
    "liked it because they could sleep longer, but most of us didnt learn much. When we went back to school "
    "everyone was behind in math and our teacher had to explain everything again. So my opinion is that "
    "online classes are okay for emergencies but they cant replace a real classroom with a real teacher."
)


# ----------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading the AI detector model...")
def load_model():
    """Load the trained model. If it cannot be loaded (e.g. library version change), train it again."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        df = pd.read_csv(DATA_PATH)
        df["text"] = df["text"].str.strip()
        df = df.drop_duplicates(subset="text")
        df = df[df["text"].str.split().str.len() >= 50]
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=5),
        )
        model.fit(df["text"], df["generated"])
        return model


@st.cache_data
def load_stats():
    with open(STATS_PATH) as f:
        return json.load(f)


def analyse(model, text):
    """Return the AI probability and the word-level contributions for one text."""
    vectorizer = model.named_steps["tfidfvectorizer"]
    classifier = model.named_steps["logisticregression"]
    vector = vectorizer.transform([text])
    probability = float(classifier.predict_proba(vector)[0, 1])
    names = vectorizer.get_feature_names_out()
    coef = classifier.coef_[0]
    idx = vector.nonzero()[1]
    contrib = pd.DataFrame({"term": names[idx], "contribution": vector.toarray()[0, idx] * coef[idx]})
    contrib = contrib.sort_values("contribution")
    word_weights = {names[i]: coef[i] for i in idx if " " not in names[i]}
    return probability, contrib, word_weights


def verdict(probability):
    if probability >= 0.7:
        return "Likely AI-generated", AI_COLOR, "🤖"
    if probability <= 0.3:
        return "Likely human-written", HUMAN_COLOR, "✍️"
    return "Uncertain (mixed signals)", "#C9A227", "⚖️"


def highlight(text, word_weights, limit=4.0):
    """Colour words in the text: red = pushes to AI, blue = pushes to human."""
    parts = re.split(r"([A-Za-z']+)", text)
    out = []
    for part in parts:
        weight = word_weights.get(part.lower().replace("'", ""), word_weights.get(part.lower()))
        safe = html.escape(part).replace("\n", "<br>")
        if weight is None or abs(weight) < 0.8:
            out.append(safe)
            continue
        alpha = min(abs(weight) / limit, 1) * 0.55 + 0.1
        rgb = "228,87,46" if weight > 0 else "46,134,171"
        out.append(f'<span style="background: rgba({rgb},{alpha:.2f}); border-radius:3px; padding:0 2px">{safe}</span>')
    return "".join(out)


# ----------------------------------------------------------------------------- database (CRUD)
def db():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            word_count INTEGER,
            ai_probability REAL,
            verdict TEXT,
            user_label TEXT DEFAULT 'Not reviewed',
            note TEXT DEFAULT ''
        )"""
    )
    return con


def create_check(title, text, words, probability, verdict_text):
    with db() as con:
        con.execute(
            "INSERT INTO checks (created_at, title, text, word_count, ai_probability, verdict) VALUES (?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), title, text, words, probability, verdict_text),
        )


def read_checks():
    with db() as con:
        return pd.read_sql("SELECT * FROM checks ORDER BY id DESC", con)


def update_check(check_id, user_label, note):
    with db() as con:
        con.execute("UPDATE checks SET user_label = ?, note = ? WHERE id = ?", (user_label, note, check_id))


def delete_check(check_id):
    with db() as con:
        con.execute("DELETE FROM checks WHERE id = ?", (check_id,))


# ----------------------------------------------------------------------------- pages
def page_check(model, stats):
    st.title("🔍 AI or Human?")
    st.markdown("Paste an English essay and the model estimates whether it was written by a **person** or by **AI**.")

    if "essay" not in st.session_state:
        st.session_state.essay = ""
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    if c1.button("Example: AI text"):
        st.session_state.essay = SAMPLE_AI
    if c2.button("Example: human text"):
        st.session_state.essay = SAMPLE_HUMAN
    if c3.button("Clear"):
        st.session_state.essay = ""

    text = st.text_area("Essay text", key="essay", height=230,
                        placeholder="Paste at least 100 words for a reliable result...")
    words = len(re.findall(r"[A-Za-z']+", text))
    st.caption(f"{words} words")

    if st.button("Analyze text", type="primary", disabled=words == 0):
        st.session_state.analyzed = text
    if not text or st.session_state.get("analyzed") != text:
        st.info("Best for **English student essays of 150+ words**. The result is a probability, not proof.")
        return

    probability, contrib, word_weights = analyse(model, text)
    label, color, icon = verdict(probability)

    st.markdown(
        f"""<div style="border-left: 8px solid {color}; padding: 14px 20px; border-radius: 8px;
        background: rgba(0,0,0,0.03); margin: 8px 0 14px 0">
        <div style="font-size: 15px; opacity: .75">Result</div>
        <div style="font-size: 30px; font-weight: 700; color: {color}">{icon} {label}</div>
        <div style="font-size: 18px">AI probability: <b>{probability * 100:.1f}%</b></div></div>""",
        unsafe_allow_html=True,
    )
    st.progress(probability, text=f"Human  ←  {probability * 100:.0f}% AI  →  AI")

    if words < 100:
        robust = {r["words"]: r["accuracy"] for r in stats["robustness"]}
        acc = robust[50] if words >= 50 else robust[25]
        st.warning(f"Short text ({words} words). With texts this short the model's accuracy drops to about "
                   f"{acc * 100:.0f}%. Use 150+ words for a reliable result.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Why? Word signals in your text")
        st.markdown(
            f'<span style="background: rgba(228,87,46,.45); padding: 0 6px; border-radius: 3px">red</span> = points to AI &nbsp; '
            f'<span style="background: rgba(46,134,171,.45); padding: 0 6px; border-radius: 3px">blue</span> = points to human',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div style="line-height: 1.9; font-size: 15px">{highlight(text, word_weights)}</div>',
                    unsafe_allow_html=True)
    with right:
        st.subheader("Strongest signals")
        top_ai = contrib[contrib["contribution"] > 0].tail(8).iloc[::-1]
        top_human = contrib[contrib["contribution"] < 0].head(8)
        a, b = st.columns(2)
        a.markdown("**🤖 AI signals**")
        a.dataframe(top_ai.assign(contribution=top_ai["contribution"].round(3)), hide_index=True)
        b.markdown("**✍️ Human signals**")
        b.dataframe(top_human.assign(contribution=top_human["contribution"].round(3)), hide_index=True)

    st.subheader("Writing style compared with the dataset")
    feats = text_features(text)
    means = stats["style_means"]
    table = pd.DataFrame({
        "feature": [f.replace("_", " ") for f in FEATURE_NAMES],
        "your text": [round(feats[f], 2) for f in FEATURE_NAMES],
        "average human essay": [means[f]["Human"] for f in FEATURE_NAMES],
        "average AI essay": [means[f]["AI"] for f in FEATURE_NAMES],
    })
    st.dataframe(table, hide_index=True)

    with st.form("save_form", clear_on_submit=True):
        st.markdown("**Save this check to the history**")
        title = st.text_input("Title", value=f"Check {datetime.now().strftime('%d.%m %H:%M')}")
        if st.form_submit_button("💾 Save to history"):
            create_check(title, text, words, probability, label)
            st.success("Saved. Open the History page to see, review or delete it.")


def page_dashboard(stats):
    st.title("📊 Dashboard")
    d, m = stats["dataset"], stats["test_metrics"]
    k = st.columns(5)
    k[0].metric("Essays (clean)", f"{d['rows_clean']:,}")
    k[1].metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
    k[2].metric("Precision", f"{m['precision'] * 100:.2f}%")
    k[3].metric("Recall", f"{m['recall'] * 100:.2f}%")
    k[4].metric("F1-score", f"{m['f1'] * 100:.2f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Model", "Limits", "My checks"])
    with tab1:
        st.markdown(f"**Source:** {d['name']}. Original rows: {d['rows_original']:,}. "
                    f"Removed {d['duplicates_removed']:,} duplicates and {d['short_removed']} texts under 50 words.")
        c1, c2 = st.columns(2)
        c1.image(os.path.join(FIG_DIR, "01_class_balance.png"), caption="Class balance")
        c2.image(os.path.join(FIG_DIR, "02_length_histogram.png"), caption="Essay length")
        st.image(os.path.join(FIG_DIR, "03_style_boxplots.png"), caption="Writing-style features")
        c3, c4 = st.columns(2)
        c3.image(os.path.join(FIG_DIR, "06_top_words.png"), caption="Most frequent words")
        c4.image(os.path.join(FIG_DIR, "04_correlation.png"), caption="Correlation matrix")
    with tab2:
        comp = pd.DataFrame(stats["model_comparison"]).set_index("model")
        st.markdown(f"**Chosen model:** {stats['best_model']} (gives probabilities and word-level explanations).")
        st.dataframe(comp.round(4))
        c1, c2, c3 = st.columns(3)
        c1.image(os.path.join(FIG_DIR, "07_model_comparison.png"), caption="Model comparison")
        c2.image(os.path.join(FIG_DIR, "08_confusion_matrix.png"), caption="Confusion matrix")
        c3.image(os.path.join(FIG_DIR, "09_roc_curve.png"), caption=f"ROC curve (AUC {m['roc_auc']:.4f})")
        cv = stats["cross_validation_f1"]
        st.markdown(f"**5-fold cross-validation F1:** {np.mean(cv):.4f} ± {np.std(cv):.4f}")
        st.image(os.path.join(FIG_DIR, "10_explainability_words.png"), caption="Words that push the decision")
    with tab3:
        c1, c2 = st.columns(2)
        c1.image(os.path.join(FIG_DIR, "11_robustness_length.png"), caption="Quality vs text length")
        c1.dataframe(pd.DataFrame(stats["robustness"]), hide_index=True)
        ext = stats["external_validation"]
        c2.markdown("#### Domain shift test")
        c2.markdown(
            f"On 2,000 **scientific abstracts** (AI-GA dataset) the model labels **{ext['predicted_as_ai_%']:.1f}%** "
            f"of texts as AI, even the human ones. Accuracy falls to **{ext['accuracy'] * 100:.1f}%**.\n\n"
            "Formal scientific language looks like AI style to a model trained on student essays. "
            "**AI detectors must never be used as the only proof against a student.**"
        )
    with tab4:
        hist = read_checks()
        if hist.empty:
            st.info("No saved checks yet. Analyze a text and save it to see statistics here.")
        else:
            c = st.columns(3)
            c[0].metric("Saved checks", len(hist))
            c[1].metric("Average AI probability", f"{hist['ai_probability'].mean() * 100:.1f}%")
            c[2].metric("Reviewed by a person", int((hist["user_label"] != "Not reviewed").sum()))
            st.bar_chart(hist["verdict"].value_counts())


def page_history():
    st.title("🗂️ History")
    st.markdown("Saved checks are stored in a small **SQLite database**. You can review, correct or delete them.")
    hist = read_checks()
    if hist.empty:
        st.info("The history is empty. Go to **Check a text**, analyze an essay and click **Save to history**.")
        return

    show = hist[["id", "created_at", "title", "word_count", "ai_probability", "verdict", "user_label", "note"]].copy()
    show["ai_probability"] = (show["ai_probability"] * 100).round(1).astype(str) + "%"
    st.dataframe(show, hide_index=True)

    st.subheader("Review or delete a check")
    options = {f"#{r.id}: {r.title}": r for r in hist.itertuples()}
    choice = st.selectbox("Choose a check", list(options))
    row = options[choice]
    with st.expander("Show the text"):
        st.write(row.text)
    labels = ["Not reviewed", "Confirmed: human", "Confirmed: AI", "Model was wrong"]
    with st.form(f"edit_{row.id}"):
        user_label = st.selectbox("Your review", labels, index=labels.index(row.user_label) if row.user_label in labels else 0)
        note = st.text_area("Note", value=row.note or "")
        c1, c2 = st.columns(2)
        save = c1.form_submit_button("✏️ Update")
        remove = c2.form_submit_button("🗑️ Delete")
    if save:
        update_check(row.id, user_label, note)
        st.success("Updated.")
        st.rerun()
    if remove:
        delete_check(row.id)
        st.success("Deleted.")
        st.rerun()

    st.download_button("⬇️ Download history (CSV)", hist.to_csv(index=False).encode(), "history.csv", "text/csv")


def page_about(stats):
    st.title("ℹ️ About the project")
    m = stats["test_metrics"]
    st.markdown(f"""
**AI or Human?** is a mini project for the course *Programming for Data Science* (Master), Riga Nordic University.
Author: **Seyran Gurbanov**.

#### Problem
Generative AI can write a full essay in seconds. Teachers need help to see when a text is probably AI-generated,
but existing detectors are often closed, paid and do not explain their decisions.

#### How it works
1. **Data:** {stats['dataset']['rows_clean']:,} essays (human and AI) after cleaning.
2. **Features:** TF-IDF of words and word pairs + 8 writing-style features.
3. **Model:** Logistic Regression, compared with Naive Bayes, Linear SVM and Random Forest.
4. **Quality on unseen essays:** accuracy {m['accuracy'] * 100:.2f}%, F1 {m['f1'] * 100:.2f}%, ROC-AUC {m['roc_auc']:.4f}.
5. **Explainable:** every result shows which words pushed the decision.

#### Limits and ethics
- Works best for **English student essays of 150+ words**.
- On other text types (e.g. scientific abstracts) the model is **not reliable**.
- The result is a **probability, not proof**. A teacher must always make the final decision and talk with the student.
- Texts are not sent to any external service. Saved checks stay in this app's own database.

#### Technology
Python, pandas, NumPy, Matplotlib, Scikit-learn, SQLite, Streamlit, GitHub, Streamlit Community Cloud.
""")


def main():
    model = load_model()
    stats = load_stats()
    with st.sidebar:
        st.markdown("## 🔍 AI or Human?")
        st.caption("AI-generated essay detector")
        page = st.radio("Menu", ["Check a text", "Dashboard", "History", "About"], label_visibility="collapsed")
        st.divider()
        st.caption(f"Model: {stats['best_model']}")
        st.caption(f"Test F1-score: {stats['test_metrics']['f1'] * 100:.2f}%")
        st.caption("Seyran Gurbanov · Programming for Data Science")
    if page == "Check a text":
        page_check(model, stats)
    elif page == "Dashboard":
        page_dashboard(stats)
    elif page == "History":
        page_history()
    else:
        page_about(stats)


main()
