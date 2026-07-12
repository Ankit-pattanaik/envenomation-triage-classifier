"""
app.py — Silent Sting: Envenomation Triage Intelligence
========================================================
A Streamlit app with three parts:
  1. Dashboard  — 12 interactive Plotly charts built from precomputed stats
  2. Predict    — score a patient with XGBoost / LightGBM / TabNet + triage advice
  3. AI Insights— Groq-powered plain-language explanations

Run:  streamlit run app.py
Needs the artifacts produced by train_models.py in ./models/.
"""

import json, os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

from data_prep import CLASS_ORDER, INT_TO_CLASS, FEATURE_ORDER, row_from_inputs

# ----------------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Silent Sting · Triage AI", page_icon="🦂",
                   layout="wide", initial_sidebar_state="expanded")

CLASS_COLORS = {
    "Harmless_Insect": "#22c55e",
    "Scorpion": "#f59e0b",
    "Viper_Snake": "#8b5cf6",
    "Black_Widow_Spider": "#ef4444",
}
CLASS_ICON = {"Harmless_Insect": "🐜", "Scorpion": "🦂",
              "Viper_Snake": "🐍", "Black_Widow_Spider": "🕷️"}
PLOT_BG = "rgba(0,0,0,0)"

st.markdown("""
<style>
  .stApp { background: radial-gradient(1200px 600px at 20% -10%, #1e293b 0%, #0f172a 45%, #020617 100%); }
  .block-container { padding-top: 1.4rem; }
  h1,h2,h3,h4 { color:#f1f5f9 !important; font-family:'Segoe UI',sans-serif; letter-spacing:-.5px;}
  p, span, label, .stMarkdown { color:#cbd5e1; }
  .hero { background:linear-gradient(120deg,#7c3aed 0%,#db2777 55%,#f59e0b 100%);
          padding:26px 30px; border-radius:20px; margin-bottom:18px;
          box-shadow:0 12px 40px rgba(124,58,237,.35);}
  .hero h1 { color:#fff !important; margin:0; font-size:2.1rem;}
  .hero p  { color:#f8fafc; margin:.3rem 0 0; font-size:1.02rem; opacity:.95;}
  .kpi { background:rgba(30,41,59,.7); border:1px solid rgba(148,163,184,.18);
         border-radius:16px; padding:18px 20px; backdrop-filter:blur(6px);}
  .kpi .v { font-size:1.9rem; font-weight:800; color:#fff; }
  .kpi .l { font-size:.82rem; text-transform:uppercase; letter-spacing:1px; color:#94a3b8;}
  .card { background:rgba(30,41,59,.55); border:1px solid rgba(148,163,184,.16);
          border-radius:18px; padding:20px 22px; margin-bottom:8px;}
  .triage-hi  {border-left:6px solid #ef4444;}
  .triage-mid {border-left:6px solid #f59e0b;}
  .triage-lo  {border-left:6px solid #22c55e;}
  .badge {display:inline-block;padding:4px 12px;border-radius:999px;font-weight:700;font-size:.8rem;}
  section[data-testid="stSidebar"] { background:#0b1120; border-right:1px solid rgba(148,163,184,.12);}
  .stButton>button { background:linear-gradient(90deg,#7c3aed,#db2777); color:#fff; border:0;
        border-radius:12px; padding:.55rem 1.1rem; font-weight:700;}
  .disc { font-size:.78rem; color:#64748b; border-top:1px dashed rgba(148,163,184,.25); padding-top:8px;}
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "models"

# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(name):
    with open(os.path.join(MODELS_DIR, name)) as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_tree(name):
    return joblib.load(os.path.join(MODELS_DIR, name))

@st.cache_resource(show_spinner=False)
def load_tabnet():
    from pytorch_tabnet.tab_model import TabNetClassifier
    clf = TabNetClassifier()
    clf.load_model(os.path.join(MODELS_DIR, "tabnet_model.zip"))
    scaler = load_json("tabnet_scaler.json")
    return clf, scaler

# ----------------------------------------------------------------------------
# Groq helper (OpenAI-compatible REST endpoint — no extra SDK needed)
# ----------------------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_chat(messages, api_key, model, temperature=0.4, max_tokens=700):
    if not api_key:
        return None, "No Groq API key provided. Add one in the sidebar to enable AI insights."
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=45,
        )
        if r.status_code != 200:
            return None, f"Groq API error {r.status_code}: {r.text[:300]}"
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, f"Request failed: {e}"

# ----------------------------------------------------------------------------
# Chart builders
# ----------------------------------------------------------------------------
def _style(fig, h=340, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      height=h, margin=dict(l=10, r=10, t=44 if title else 14, b=10),
                      title=dict(text=title or "", font=dict(size=15, color="#e2e8f0")),
                      legend=dict(font=dict(size=10)))
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    return fig

def chart_class_donut(eda):
    cc = eda["class_counts"]
    labels = [c for c in CLASS_ORDER if c in cc]
    fig = go.Figure(go.Pie(labels=[l.replace("_", " ") for l in labels],
                           values=[cc[l] for l in labels], hole=.62,
                           marker=dict(colors=[CLASS_COLORS[l] for l in labels]),
                           textinfo="percent"))
    return _style(fig, title="① Patient Distribution by Culprit")

def chart_gender_pie(eda):
    gc = eda["gender_counts"]
    fig = go.Figure(go.Pie(labels=list(gc.keys()), values=list(gc.values()), hole=.4,
                           marker=dict(colors=["#38bdf8", "#f472b6", "#a3a3a3"])))
    return _style(fig, title="② Gender Split")

def chart_model_acc(metrics):
    names = list(metrics.keys())
    acc = [metrics[m]["accuracy"] * 100 for m in names]
    f1 = [metrics[m]["macro_f1"] * 100 for m in names]
    fig = go.Figure()
    fig.add_bar(name="Accuracy", x=names, y=acc, marker_color="#7c3aed",
                text=[f"{a:.2f}%" for a in acc], textposition="outside")
    fig.add_bar(name="Macro F1", x=names, y=f1, marker_color="#db2777")
    fig.update_yaxes(range=[max(0, min(f1 + acc) - 8), 101], title="%")
    return _style(fig, title="③ Model Performance Comparison")

def chart_confusion(metrics, model):
    cm = np.array(metrics[model]["confusion_matrix"])
    labels = [c.replace("_", " ") for c in CLASS_ORDER]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Magma",
                    text=cm, texttemplate="%{text}", showscale=True))
    fig.update_yaxes(title="Actual"); fig.update_xaxes(title="Predicted")
    return _style(fig, title=f"④ Confusion Matrix — {model}")

def chart_feature_importance(imp, model):
    d = imp[model]
    order = sorted(d, key=d.get)
    fig = go.Figure(go.Bar(x=[d[k] for k in order], y=order, orientation="h",
                           marker=dict(color=[d[k] for k in order], colorscale="Viridis")))
    return _style(fig, title=f"⑤ Feature Importance — {model}")

def chart_mean_vitals(eda):
    mv = eda["mean_vitals_by_class"]
    cols = ["Heart_Rate_BPM", "Blood_Pressure_Systolic"]
    fig = go.Figure()
    for col in cols:
        fig.add_bar(name=col.replace("_", " "),
                    x=[c.replace("_", " ") for c in CLASS_ORDER],
                    y=[mv[col][c] for c in CLASS_ORDER])
    return _style(fig, title="⑥ Mean Heart Rate & Blood Pressure by Class")

def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def chart_hist_by_class(eda, col, num):
    hb = eda["hist_by_class"][col]
    fig = go.Figure()
    for c in CLASS_ORDER:
        h = hb[c]
        centers = [(h["edges"][i] + h["edges"][i + 1]) / 2 for i in range(len(h["counts"]))]
        fig.add_scatter(x=centers, y=h["counts"], mode="lines", name=c.replace("_", " "),
                        line=dict(color=CLASS_COLORS[c], width=2), fill="tozeroy",
                        fillcolor=_hex_to_rgba(CLASS_COLORS[c], 0.10))
    fig.update_xaxes(title=col.replace("_", " "))
    return _style(fig, title=f"{num} {col.replace('_',' ')} Distribution by Class")

def _stacked_by_class(eda, key, title, num, pos_label="1"):
    ct = eda[key]  # {colvalue: {class: count}}
    fig = go.Figure()
    colvals = list(ct.keys())
    for cv in colvals:
        fig.add_bar(name=str(cv),
                    x=[c.replace("_", " ") for c in CLASS_ORDER],
                    y=[ct[cv].get(c, 0) for c in CLASS_ORDER])
    fig.update_layout(barmode="stack")
    return _style(fig, title=f"{num} {title}")

def chart_corr(eda):
    corr = eda["corr"]
    cols = list(corr.keys())
    z = [[corr[a][b] for b in cols] for a in cols]
    short = [c.replace("_", " ")[:16] for c in cols]
    fig = go.Figure(go.Heatmap(z=z, x=short, y=short, colorscale="RdBu", zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in z],
                    texttemplate="%{text}", showscale=True))
    return _style(fig, h=380, title="⑪ Feature Correlation Matrix")

def chart_scatter(eda):
    s = eda["vitals_scatter_sample"]
    df = pd.DataFrame(s)
    fig = px.scatter(df, x="Heart_Rate_BPM", y="Blood_Pressure_Systolic",
                     color="Bite_Source_Target", color_discrete_map=CLASS_COLORS,
                     opacity=.6)
    fig.update_traces(marker=dict(size=5))
    return _style(fig, h=380, title="⑫ Heart Rate vs Blood Pressure (sampled)")

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🦂 Silent Sting")
    st.caption("Envenomation Triage Intelligence")
    page = st.radio("Navigate", ["📊 Dashboard", "🔬 Predict", "ℹ️ About"], label_visibility="collapsed")
    st.divider()
    st.markdown("#### 🤖 Groq AI")
    groq_key = st.text_input("API key", type="password",
                             value=os.environ.get("GROQ_API_KEY", ""),
                             help="Get one free at console.groq.com/keys")
    groq_model = st.selectbox("Model", [
        "openai/gpt-oss-20b", "openai/gpt-oss-120b",
        "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
    ], help="Groq changes model IDs often — pick one your account supports.")
    st.divider()
    st.caption("⚕️ Educational decision-support only. Not a substitute for clinical judgment.")

metrics = load_json("metrics.json")
importance = load_json("feature_importance.json")
eda = load_json("eda_stats.json")
BEST_MODEL = max(metrics, key=lambda m: metrics[m]["accuracy"])

# ============================================================================
# DASHBOARD
# ============================================================================
if page.endswith("Dashboard"):
    st.markdown(f"""<div class="hero"><h1>Envenomation Triage Dashboard</h1>
    <p>{eda['n_rows']:,} patients · 4 culprit classes · 3 trained models ·
    best accuracy {metrics[BEST_MODEL]['accuracy']*100:.2f}% ({BEST_MODEL})</p></div>""",
    unsafe_allow_html=True)

    # KPI row
    k = st.columns(4)
    kpis = [("Patients", f"{eda['n_rows']:,}"),
            ("Culprit Classes", str(len(CLASS_ORDER))),
            ("Best Model", BEST_MODEL),
            ("Best Accuracy", f"{metrics[BEST_MODEL]['accuracy']*100:.2f}%")]
    for col, (l, v) in zip(k, kpis):
        col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>',
                     unsafe_allow_html=True)
    st.write("")

    r1 = st.columns(2)
    r1[0].plotly_chart(chart_class_donut(eda), use_container_width=True)
    r1[1].plotly_chart(chart_gender_pie(eda), use_container_width=True)

    r2 = st.columns(2)
    r2[0].plotly_chart(chart_model_acc(metrics), use_container_width=True)
    r2[1].plotly_chart(chart_confusion(metrics, BEST_MODEL), use_container_width=True)

    r3 = st.columns(2)
    r3[0].plotly_chart(chart_feature_importance(importance, BEST_MODEL), use_container_width=True)
    r3[1].plotly_chart(chart_mean_vitals(eda), use_container_width=True)

    r4 = st.columns(2)
    r4[0].plotly_chart(chart_hist_by_class(eda, "Heart_Rate_BPM", "⑦"), use_container_width=True)
    r4[1].plotly_chart(chart_hist_by_class(eda, "Blood_Pressure_Systolic", "⑧"), use_container_width=True)

    r5 = st.columns(2)
    r5[0].plotly_chart(_stacked_by_class(eda, "paralysis_by_class",
                       "Muscle Paralysis by Class (0=no,1=yes)", "⑨"), use_container_width=True)
    r5[1].plotly_chart(_stacked_by_class(eda, "coag_by_class",
                       "Coagulation Failure by Class (0=no,1=yes)", "⑩"), use_container_width=True)

    r6 = st.columns(2)
    r6[0].plotly_chart(chart_corr(eda), use_container_width=True)
    r6[1].plotly_chart(chart_scatter(eda), use_container_width=True)

    st.divider()
    st.markdown("### 🤖 AI-Generated Data Insights")
    if st.button("Generate insights with Groq"):
        summary = {
            "class_counts": eda["class_counts"],
            "mean_vitals_by_class": eda["mean_vitals_by_class"],
            "paralysis_by_class": eda["paralysis_by_class"],
            "coag_by_class": eda["coag_by_class"],
            "model_accuracy": {m: metrics[m]["accuracy"] for m in metrics},
            "top_features": importance[BEST_MODEL],
        }
        with st.spinner("Groq is analyzing the dataset..."):
            txt, err = groq_chat([
                {"role": "system", "content":
                 "You are a senior clinical data scientist. Be concise, use short paragraphs "
                 "and bullet points, and reference concrete numbers from the JSON."},
                {"role": "user", "content":
                 "Given these aggregates from an envenomation triage dataset, write 5-6 crisp "
                 "insights a triage team would care about, then one caveat about data limitations.\n\n"
                 + json.dumps(summary)},
            ], groq_key, groq_model)
        if err:
            st.warning(err)
        else:
            st.markdown(f'<div class="card">{txt}</div>', unsafe_allow_html=True)

# ============================================================================
# PREDICT
# ============================================================================
elif page.endswith("Predict"):
    st.markdown("""<div class="hero"><h1>Patient Triage Prediction</h1>
    <p>Enter presenting vitals and symptoms — the model infers the likely culprit
    and suggests a triage pathway.</p></div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("#### Patient inputs")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 1, 110, 40)
        gender = c2.selectbox("Gender", ["Male", "Female", "Other"])
        time_bite = c1.number_input("Time since bite (min)", 0, 1440, 60)
        hr = c2.number_input("Heart rate (BPM)", 20, 260, 95)
        bp = c1.number_input("Systolic BP (mmHg)", 30, 260, 118)
        swelling = c2.selectbox("Local swelling", ["Mild", "Medium", "Severe", "None (not recorded)"])
        para = c1.checkbox("Muscle paralysis present")
        coag = c2.checkbox("Blood coagulation failure")
        model_name = st.selectbox("Model", list(metrics.keys()),
                                  index=list(metrics.keys()).index(BEST_MODEL))
        go_btn = st.button("🔍 Predict culprit", use_container_width=True)

    if go_btn:
        Xrow = row_from_inputs(age, time_bite, hr, bp, int(para), int(coag), gender, swelling)

        if model_name == "TabNet":
            clf, scaler = load_tabnet()
            Xs = Xrow.copy()
            mu, sd = np.array(scaler["mu"]), np.array(scaler["sd"])
            Xs.iloc[:, :4] = (Xs.iloc[:, :4].values - mu) / sd
            proba = clf.predict_proba(Xs.values)[0]
        else:
            fname = "xgb_model.joblib" if model_name == "XGBoost" else "lgbm_model.joblib"
            clf = load_tree(fname)
            proba = clf.predict_proba(Xrow)[0]

        pred_int = int(np.argmax(proba))
        pred_cls = INT_TO_CLASS[pred_int]
        conf = proba[pred_int]

        with right:
            st.markdown("#### Result")
            urg = {"Viper_Snake": ("triage-hi", "HIGH", "#ef4444"),
                   "Black_Widow_Spider": ("triage-hi", "HIGH", "#ef4444"),
                   "Scorpion": ("triage-mid", "MODERATE", "#f59e0b"),
                   "Harmless_Insect": ("triage-lo", "LOW", "#22c55e")}[pred_cls]
            st.markdown(f"""<div class="card {urg[0]}">
                <span class="badge" style="background:{urg[2]}22;color:{urg[2]}">TRIAGE: {urg[1]}</span>
                <h2 style="margin:.4rem 0">{CLASS_ICON[pred_cls]} {pred_cls.replace('_',' ')}</h2>
                <p>Model confidence: <b style="color:#fff">{conf*100:.1f}%</b> · via {model_name}</p></div>""",
                unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=[proba[i] for i, c in enumerate(CLASS_ORDER)],
                y=[c.replace("_", " ") for c in CLASS_ORDER], orientation="h",
                marker_color=[CLASS_COLORS[c] for c in CLASS_ORDER],
                text=[f"{p*100:.1f}%" for p in proba], textposition="auto"))
            _style(fig, h=240, title="Class probabilities")
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        PATHWAYS = {
            "Viper_Snake": "Hemotoxic envenomation. Urgent: type & crossmatch, coagulation panel, "
                           "consider polyvalent antivenom, monitor for hypotension and bleeding.",
            "Black_Widow_Spider": "Latrodectism. Manage muscle rigidity/pain (benzodiazepines, opioids), "
                                  "monitor blood pressure; antivenom for severe systemic cases.",
            "Scorpion": "Autonomic storm likely. Supportive care, consider prazosin, monitor cardiac "
                        "rhythm and blood pressure closely.",
            "Harmless_Insect": "Low-acuity local reaction. Clean site, antihistamine, analgesia, "
                               "safety-net advice; discharge if stable.",
        }
        st.markdown(f'<div class="card"><b>Suggested pathway.</b> {PATHWAYS[pred_cls]}</div>',
                    unsafe_allow_html=True)

        st.markdown("#### 🤖 AI clinical explanation")
        with st.spinner("Groq is explaining the prediction..."):
            feat_desc = {
                "age": age, "gender": gender, "time_since_bite_min": time_bite,
                "heart_rate_bpm": hr, "systolic_bp": bp,
                "local_swelling": swelling, "muscle_paralysis": bool(para),
                "coagulation_failure": bool(coag),
            }
            txt, err = groq_chat([
                {"role": "system", "content":
                 "You are an emergency-medicine triage assistant. Explain briefly and clearly why the "
                 "predicted culprit fits the vitals. 4-6 sentences. End with a one-line safety reminder "
                 "that this is decision-support, not diagnosis."},
                {"role": "user", "content":
                 f"Predicted culprit: {pred_cls} (confidence {conf*100:.1f}%). "
                 f"Patient features: {json.dumps(feat_desc)}. "
                 f"Class probabilities: "
                 f"{json.dumps({c: round(float(proba[i]),3) for i,c in enumerate(CLASS_ORDER)})}."},
            ], groq_key, groq_model)
        if err:
            st.info(err)
        else:
            st.markdown(f'<div class="card">{txt}</div>', unsafe_allow_html=True)

        st.markdown('<p class="disc">⚕️ This tool is an educational ML demo trained on synthetic data. '
                    'It must not be used for real clinical decisions.</p>', unsafe_allow_html=True)

# ============================================================================
# ABOUT
# ============================================================================
else:
    st.markdown("""<div class="hero"><h1>About Silent Sting</h1>
    <p>How the data was analyzed, preprocessed, and modeled.</p></div>""", unsafe_allow_html=True)
    st.markdown(f"""
<div class="card">

**The problem.** Multiclass classification of the envenomation culprit
(`{', '.join(c.replace('_',' ') for c in CLASS_ORDER)}`) from vitals and symptoms
across **{eda['n_rows']:,}** patients.

**Key preprocessing insight.** `Local_Swelling` is missing on exactly the Black-Widow
rows — latrodectism causes systemic rigidity with minimal local reaction. Rather than
dropping ~150k rows, the missingness is encoded as a feature (`Swelling_Missing`), which
alone is a near-perfect Black-Widow detector.

**Feature engineering.** Ordinal swelling severity (None→Severe), one-hot gender,
binary symptom flags, standardized numerics for the neural net. `Age` and
`Time_Since_Bite` were kept though they carry essentially no class signal.

**Models trained.**
- **XGBoost** — {metrics.get('XGBoost',{}).get('accuracy',0)*100:.2f}% accuracy
- **LightGBM** — {metrics.get('LightGBM',{}).get('accuracy',0)*100:.2f}% accuracy
- **TabNet (PyTorch)** — {metrics.get('TabNet',{}).get('accuracy',0)*100:.2f}% accuracy

**AI layer.** Groq's API turns predictions and dataset aggregates into plain-language
explanations for the triage team.

</div>
""", unsafe_allow_html=True)
