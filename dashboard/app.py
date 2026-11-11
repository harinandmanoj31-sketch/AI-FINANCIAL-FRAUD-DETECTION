"""AI Fraud Shield — Streamlit demo dashboard.

Run from the project root:
    streamlit run dashboard/app.py
Expected project files:
    models/xgb_transaction.joblib
    models/scaler.joblib
    models/thresholds.json (optional)
    data/raw/creditcard.csv
"""

import hashlib
import json
import random
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb_lib
except ImportError:
    xgb_lib = None

try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(
    page_title="AI Fraud Shield | Financial Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- DESIGN SYSTEM -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#08111f; --panel:#101d30; --panel2:#14243a; --line:#263952; --text:#edf4ff; --muted:#9fb0c8; --cyan:#4de1d2; --blue:#6ea8fe; --green:#42d392; --amber:#ffbf69; --red:#ff7185; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 10% 0%, #142b47 0%, var(--bg) 42%, #07101b 100%); color:var(--text); }
[data-testid="stHeader"] { background:rgba(8,17,31,.75); }
.block-container { max-width:1450px; padding-top:1.6rem; padding-bottom:3rem; }
section[data-testid="stSidebar"] { background:#0b1728; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] * { color:var(--text); }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; color:var(--text); }
h1 { font-size:2.35rem !important; }
h2 { font-size:1.55rem !important; }
h3 { font-size:1.18rem !important; }
p, label, .stCaption { color:var(--muted); }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#14253b,#0f1c2e); border:1px solid var(--line); padding:17px 18px; border-radius:16px; box-shadow:0 8px 25px rgba(0,0,0,.12); }
div[data-testid="stMetricLabel"] { color:#a9bbd3; }
div[data-testid="stMetricValue"] { color:#f2f7ff; font-family:'Space Grotesk',sans-serif; }
.stTabs [data-baseweb="tab-list"] { gap:8px; background:#0c192a; padding:7px; border:1px solid var(--line); border-radius:14px; }
.stTabs [data-baseweb="tab"] { border-radius:10px; color:#a9bbd3; padding:10px 15px; }
.stTabs [aria-selected="true"] { background:#1a3652 !important; color:#ffffff !important; }
.stButton button { border:1px solid #365572; background:linear-gradient(135deg,#173754,#1d4964); color:#f5fbff; border-radius:11px; min-height:42px; font-weight:600; transition:all .18s ease; }
.stButton button:hover { border-color:var(--cyan); color:#fff; transform:translateY(-1px); }
.stTextInput input, .stNumberInput input, [data-baseweb="select"] > div { background:#0d1a2b !important; color:#edf4ff !important; border-color:#2a405a !important; border-radius:10px !important; }
[data-testid="stFileUploader"] { background:#0d1a2b; border:1px dashed #3a5876; padding:12px; border-radius:14px; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stDataFrame"], .stTable { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
hr { border-color:var(--line); }
.hero { background:linear-gradient(120deg,rgba(24,54,82,.98),rgba(13,29,48,.98)); border:1px solid #2b4965; padding:26px 28px; border-radius:22px; margin:0 0 20px 0; box-shadow:0 15px 40px rgba(0,0,0,.16); }
.hero-kicker { color:var(--cyan); font-size:.78rem; letter-spacing:.15em; font-weight:700; text-transform:uppercase; }
.hero-title { color:#f2f7ff; font-family:'Space Grotesk',sans-serif; font-size:2.15rem; font-weight:700; margin:.35rem 0 .4rem; }
.hero-copy { color:#b4c6dc; font-size:1rem; margin:0; }
.pill { display:inline-block; padding:5px 10px; border-radius:99px; font-size:.75rem; font-weight:700; background:#163c3c; color:#7df0d8; border:1px solid #28645e; }
.section-card { background:linear-gradient(145deg,rgba(20,36,58,.92),rgba(13,26,43,.92)); border:1px solid var(--line); border-radius:16px; padding:18px 20px; margin-bottom:12px; }
.small-muted { color:var(--muted); font-size:.88rem; }
.status-dot { color:var(--green); }
div[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,var(--cyan),var(--blue)); }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

NAMES = ["Aarav Mehta", "Priya Nair", "Rohan Gupta", "Ananya Rao", "Vikram Singh", "Meera Iyer", "Kiran Patel", "Sneha Reddy"]
MERCHANTS = ["Grocery Mart", "Online Electronics Store", "Fuel Station", "Restaurant", "Travel Booking Portal", "Pharmacy", "Mobile Recharge"]
CITIES = ["Bengaluru", "Mysuru", "Pune", "Hyderabad", "Chennai", "Mumbai"]

# ----------------------------- MODEL LOADING -----------------------------
@st.cache_resource
def load_artifacts():
    model_path = ROOT / "models" / "xgb_transaction.joblib"
    scaler_path = ROOT / "models" / "scaler.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    cfg_path = ROOT / "models" / "thresholds.json"
    low, high = 0.02, 0.95
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        low, high = float(cfg.get("low", low)), float(cfg.get("high", high))
    if low >= high:
        raise ValueError("Threshold configuration is invalid: 'low' must be less than 'high'.")
    return model, scaler, low, high


@st.cache_data
def load_test_set():
    data_path = ROOT / "data" / "raw" / "creditcard.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}")
    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")
    X, y = df.drop("Class", axis=1), df["Class"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


# Keep the dashboard open even if a required local artifact is missing.
try:
    model, scaler, LOW, HIGH = load_artifacts()
    X_test, y_test = load_test_set()
    MODEL_READY = True
    MODEL_ERROR = ""
except Exception as exc:
    model = scaler = X_test = y_test = None
    LOW, HIGH = 0.02, 0.95
    MODEL_READY = False
    MODEL_ERROR = str(exc)


def prep(rows):
    """Apply the saved scaler only to Amount and Time, as in the training pipeline."""
    r = rows.copy()
    if scaler is not None and {"Amount", "Time"}.issubset(r.columns):
        r[["Amount", "Time"]] = scaler.transform(r[["Amount", "Time"]])
    return r


def risk_score(rows):
    if not MODEL_READY:
        raise RuntimeError("Transaction model is not ready. Check the model and dataset paths.")
    return float(model.predict_proba(prep(rows))[0, 1])


def decide(p):
    if p >= HIGH:
        return "BLOCK", "High risk: hold the payment for manual review."
    if p >= LOW:
        return "OTP", "Medium risk: request an additional customer verification step."
    return "APPROVE", "Low model score: allow the transaction in this demonstration."


def fake_details(i, row):
    r = random.Random(int(i))
    t = float(row["Time"].iloc[0])
    return {
        "Customer": r.choice(NAMES),
        "Card": f"•••• •••• •••• {r.randint(1000, 9999)}",
        "Merchant": r.choice(MERCHANTS),
        "City": r.choice(CITIES),
        "Amount": f"€{float(row['Amount'].iloc[0]):,.2f}",
        "Time": f"Dataset time {int(t):,} sec",
    }

# ----------------------------- LINK / QR SCREENING -----------------------------
OFFICIAL = {"sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com", "axisbank.com", "paytm.com", "phonepe.com", "google.com", "amazon.in", "amazon.com", "amazonaws.com", "flipkart.com", "npci.org.in", "wikipedia.org"}
BRANDS = ["sbi", "hdfc", "icici", "axis", "paytm", "phonepe", "gpay", "google", "amazon", "flipkart", "npci"]
KEYWORDS = ["kyc", "verify", "update", "login", "secure", "account", "confirm", "password", "refund", "prize", "bonus", "free", "otp", "blocked"]
SHORTENERS = ["bit.ly", "tinyurl", "t.co", "goo.gl", "cutt.ly", "is.gd", "rb.gy"]
BAD_TLDS = {"xyz", "top", "click", "loan", "work", "tk", "ml", "gq", "cf", "icu", "buzz"}


def registered_domain(host):
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "gov", "ac", "net"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def check_link(url):
    u = url.strip()
    if not u:
        return {"verdict": "SUSPICIOUS", "points": 0, "reasons": ["Enter a link to check."]}
    if "://" not in u:
        u = "http://" + u
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower().rstrip(".")
    except ValueError:
        return {"verdict": "DANGEROUS", "points": 5, "reasons": ["The link is malformed."]}
    if not host:
        return {"verdict": "SUSPICIOUS", "points": 3, "reasons": ["Could not read a website name."]}

    is_ip = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))
    reg = host if is_ip else registered_domain(host)
    # Only treat exact official registrable domains as known domains; subdomains are allowed.
    if not is_ip and any(host == d or host.endswith("." + d) for d in OFFICIAL):
        matched = next(d for d in OFFICIAL if host == d or host.endswith("." + d))
        return {"verdict": "LOOKS SAFE", "points": 0, "reasons": [f"Domain matches a listed official domain: {matched}. This is not a guarantee of safety."]}

    low = u.lower()
    tokens = [t for t in re.split(r"[.\-_]", host) if t]
    reasons, pts = [], 0
    if is_ip:
        pts += 3; reasons.append("Uses a raw IP address instead of a website name.")
    if "@" in u:
        pts += 2; reasons.append("Contains '@', which can obscure the destination.")
    if not is_ip and any(t.startswith(b) for t in tokens for b in BRANDS):
        pts += 4; reasons.append(f"Mentions a well-known brand but the domain is {reg}, not a listed official domain.")
    kws = [k for k in KEYWORDS if k in low]
    if kws:
        pts += min(len(kws), 3); reasons.append("Scam-style words found: " + ", ".join(kws) + ".")
    if host.count("-") >= 2:
        pts += 1; reasons.append("Several hyphens appear in the website name.")
    if any(s in host for s in SHORTENERS):
        pts += 2; reasons.append("Shortened link: the final destination is hidden.")
    if host.split(".")[-1] in BAD_TLDS:
        pts += 2; reasons.append("Uses a domain ending sometimes seen in suspicious links.")
    if p.scheme != "https":
        pts += 1; reasons.append("The link does not use HTTPS.")
    if p.path.lower().endswith(".apk"):
        pts += 3; reasons.append("The link appears to download an Android APK file.")
    if len(u) > 75:
        pts += 1; reasons.append("Unusually long link.")
    verdict = "DANGEROUS" if pts >= 5 else "SUSPICIOUS" if pts >= 2 else "LOOKS SAFE"
    if not reasons:
        reasons = ["No strong warning signs found. This does not prove the site is safe."]
    return {"verdict": verdict, "points": pts, "reasons": reasons}


def show_link_result(res):
    st.markdown(f"<div class='section-card'><div style='font-size:.75rem;color:#9fb0c8;text-transform:uppercase;letter-spacing:.12em'>Screening result</div><div style='font-family:Space Grotesk,sans-serif;font-size:1.55rem;font-weight:700;margin:.3rem 0'>{res['verdict']}</div><div style='color:#9fb0c8'>Warning points: {res['points']}</div></div>", unsafe_allow_html=True)
    if res["verdict"] == "DANGEROUS":
        st.error("Do not open this link or enter personal information based on this result.")
    elif res["verdict"] == "SUSPICIOUS":
        st.warning("Be cautious. Verify the domain through an official source.")
    else:
        st.success("No strong warning signs detected by these rules. This is not a safety guarantee.")
    for reason in res["reasons"]:
        st.write("• " + reason)


def decode_qr_bytes(data):
    if cv2 is None:
        return None, "OpenCV is not installed. Install opencv-python-headless to enable QR decoding."
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not read the uploaded image. Try a PNG or JPEG image."
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return (text or None), None

# ----------------------------- HEADER -----------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">Financial security • AI + risk screening</div>
  <div class="hero-title">🛡️ AI Fraud Shield</div>
  <p class="hero-copy">A demonstration dashboard for transaction-risk scoring, suspicious-link screening, and QR payment awareness.</p>
  <div style="margin-top:16px"><span class="pill">PROJECT DEMO</span> <span style="color:#9fb0c8;font-size:.82rem;margin-left:8px">Public anonymised dataset · Not connected to a bank</span></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🛡️ Fraud Shield")
    st.caption("PROJECT CONTROL CENTRE")
    st.markdown("---")
    st.markdown("**Modules**")
    st.markdown("<div class='section-card'>🧠 Transaction risk<br><span class='small-muted'>Machine-learning score</span></div><div class='section-card'>🔗 Link screening<br><span class='small-muted'>Explainable warning rules</span></div><div class='section-card'>▦ QR awareness<br><span class='small-muted'>Decode and inspect contents</span></div><div class='section-card'>⛓️ Audit fingerprint<br><span class='small-muted'>Demo hash only; not on-chain</span></div>", unsafe_allow_html=True)
    st.markdown("---")
    if MODEL_READY:
        st.success("Transaction model loaded")
    else:
        st.error("Transaction model unavailable")
        with st.expander("Show setup error"):
            st.code(MODEL_ERROR)
    st.caption("Prototype for academic demonstration. Never use this app as the sole basis for real payment decisions.")

# Quick status metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Core modules", "03", "Transactions · Links · QR")
m2.metric("Decision bands", "3", "Approve · Verify · Hold")
m3.metric("Transaction AI", "Ready" if MODEL_READY else "Offline", "Local model artifacts")
m4.metric("Blockchain", "Planned", "Hash preview only")

# ----------------------------- TABS -----------------------------
tab_how, tab_pay, tab_link, tab_res = st.tabs(["◈  Overview", "↗  Payment simulator", "⌁  Link & QR scanner", "▤  Model results"])

with tab_how:
    st.markdown("## How the system works")
    st.markdown("A transaction is scored by the saved model. The score is compared with configurable thresholds to demonstrate three possible review paths.")
    st.markdown("""
    <div class="section-card">
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
        <div style="flex:1;min-width:145px;background:#18334a;border:1px solid #315875;border-radius:12px;padding:16px;text-align:center"><b>01 · Transaction</b><div class="small-muted">Input features</div></div>
        <div style="color:#4de1d2;font-size:1.4rem">→</div>
        <div style="flex:1;min-width:145px;background:#18334a;border:1px solid #315875;border-radius:12px;padding:16px;text-align:center"><b>02 · AI model</b><div class="small-muted">Risk score 0–1</div></div>
        <div style="color:#4de1d2;font-size:1.4rem">→</div>
        <div style="flex:1;min-width:145px;background:#18334a;border:1px solid #315875;border-radius:12px;padding:16px;text-align:center"><b>03 · Decision band</b><div class="small-muted">Approve / Verify / Hold</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.success(f"LOW RISK\n\nScore < {LOW:g}\n\nDemo action: approve")
    c2.warning(f"MEDIUM RISK\n\n{LOW:g} ≤ score < {HIGH:g}\n\nDemo action: request verification")
    c3.error(f"HIGH RISK\n\nScore ≥ {HIGH:g}\n\nDemo action: hold for review")
    st.markdown("### Protection layers")
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='section-card'><h3>🧠 Transaction risk scoring</h3><p>Uses the locally saved XGBoost model and scaler to score anonymised transaction features.</p></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='section-card'><h3>🔗 Link and QR screening</h3><p>Checks links for warning indicators and decodes QR content without opening the destination.</p></div>", unsafe_allow_html=True)
    st.info("Blockchain note: this version displays a SHA-256 fingerprint as a concept demonstration. It does not write records to a blockchain.")

with tab_pay:
    st.markdown("## Payment simulator")
    st.caption("The simulator selects a row from the held-out test split. Customer and merchant details are fictional display values.")
    if not MODEL_READY:
        st.error("The payment simulator cannot run until the model, scaler, and dataset are available.")
        st.code("Expected: models/xgb_transaction.joblib\nmodels/scaler.joblib\ndata/raw/creditcard.csv")
    else:
        left, right = st.columns([1, 1])
        with left:
            if st.button("▶  Simulate a genuine transaction", use_container_width=True):
                st.session_state["pick"] = int(np.random.choice(y_test.index[y_test == 0]))
        with right:
            if st.button("⚠  Simulate a known fraud case", use_container_width=True):
                st.session_state["pick"] = int(np.random.choice(y_test.index[y_test == 1]))
        if "pick" not in st.session_state:
            st.markdown("<div class='section-card'><h3>Ready for a test transaction</h3><p>Choose either button above to run the selected row through the model.</p></div>", unsafe_allow_html=True)
        else:
            i = st.session_state["pick"]
            row = X_test.iloc[[i]]
            try:
                p = risk_score(row)
                action, message = decide(p)
                details = fake_details(i, row)
                st.markdown("### 1 · Payment request")
                cols = st.columns(3)
                for n, (key, value) in enumerate(details.items()):
                    cols[n % 3].metric(key, value)
                st.caption("Only the dataset row is used for model scoring. Displayed customer, card, merchant, and city details are randomly generated demo values.")
                st.markdown("### 2 · Model risk score")
                score_col, action_col = st.columns([2, 1])
                with score_col:
                    st.progress(float(np.clip(p, 0, 1)))
                    st.markdown(f"<div style='font-family:Space Grotesk,sans-serif;font-size:2rem;font-weight:700'>{p:.4f}<span style='font-size:.9rem;color:#9fb0c8'> / 1.0000</span></div>", unsafe_allow_html=True)
                    st.caption("Model output; not a calibrated guarantee of fraud probability.")
                with action_col:
                    if action == "BLOCK":
                        st.error("HIGH RISK · HOLD")
                    elif action == "OTP":
                        st.warning("MEDIUM RISK · VERIFY")
                    else:
                        st.success("LOW RISK · APPROVE")
                st.markdown("### 3 · Recommended demo action")
                if action == "BLOCK": st.error(message)
                elif action == "OTP": st.warning(message)
                else: st.success(message)
                                # 4. AI FRAUD EXPLANATION

                st.markdown("### 4 · AI Fraud Explanation")

                try:

                    if xgb_lib is None or not hasattr(model, "get_booster"):

                        st.warning(
                            "Detailed explanations require an XGBoost model."
                        )

                    else:

                        # Calculate XGBoost feature contributions
                        scaled = prep(row)

                        dmatrix = xgb_lib.DMatrix(scaled)

                        contributions = model.get_booster().predict(
                            dmatrix,
                            pred_contribs=True
                        )[0]

                        # Separate feature contributions and bias
                        feature_values = contributions[:-1]

                        explanation = pd.DataFrame({
                            "Feature": scaled.columns,
                            "Contribution": feature_values
                        })

                        # Positive contributions increase fraud score
                        risk_factors = explanation[
                            explanation["Contribution"] > 0
                        ].sort_values(
                            by="Contribution",
                            ascending=False
                        ).head(5)

                        # Negative contributions decrease fraud score
                        safe_factors = explanation[
                            explanation["Contribution"] < 0
                        ].sort_values(
                            by="Contribution",
                            ascending=True
                        ).head(5)

                        # FRAUD REASONS
                        st.markdown("#### 🔴 Factors increasing fraud score")

                        if not risk_factors.empty:

                            for _, item in risk_factors.iterrows():

                                feature = item["Feature"]
                                value = item["Contribution"]

                                if feature.startswith("V") and feature[1:].isdigit():
                                    description = (
                                        f"{feature} (anonymized dataset feature)"
                                    )
                                else:
                                    description = feature

                                st.error(
                                    f"**{description}** — "
                                    f"Contribution: +{value:.4f}"
                                )

                        else:

                            st.success(
                                "No individual feature had a positive "
                                "fraud contribution."
                            )

                        # SAFE FACTORS
                        st.markdown("#### 🟢 Factors reducing fraud score")

                        if not safe_factors.empty:

                            for _, item in safe_factors.iterrows():

                                feature = item["Feature"]
                                value = item["Contribution"]

                                st.success(
                                    f"**{feature}** — "
                                    f"Contribution: {value:.4f}"
                                )

                        else:

                            st.info(
                                "No individual feature reduced the fraud score."
                            )

                        # AI INTERPRETATION
                        st.markdown("#### 🧠 AI interpretation")

                        if action == "BLOCK":

                            st.error(
                                """
                                **High-risk transaction**

                                The model score crossed the configured
                                high-risk threshold. The transaction should
                                be held for manual review.
                                """
                            )

                        elif action == "OTP":

                            st.warning(
                                """
                                **Medium-risk transaction**

                                The model identified sufficient risk to
                                require additional customer verification.
                                """
                            )

                        else:

                            st.success(
                                """
                                **Low-risk transaction**

                                The model score is below the configured
                                verification threshold. No strong fraud
                                signal was identified by this model.
                                """
                            )

                        # RECOMMENDED ACTIONS
                        st.markdown("#### 🛡️ Recommended actions")

                        if action == "BLOCK":

                            st.write("""
                            1. Hold the transaction for review.
                            2. Verify the customer's identity.
                            3. Review recent transaction activity.
                            4. Confirm the payment before processing.
                            """)

                        elif action == "OTP":

                            st.write("""
                            1. Request additional authentication.
                            2. Verify the payment details.
                            3. Allow the transaction only after verification.
                            """)

                        else:

                            st.write("""
                            1. Continue normal transaction processing.
                            2. Maintain standard fraud monitoring.
                            """)

                        # FEATURE CONTRIBUTION CHART
                        st.markdown("#### 📊 Feature contribution analysis")

                        chart_data = explanation.copy()

                        chart_data["Absolute"] = (
                            chart_data["Contribution"].abs()
                        )

                        chart_data = chart_data.nlargest(
                            8, "Absolute"
                        ).set_index("Feature")

                        st.bar_chart(
                            chart_data["Contribution"]
                        )

                        st.caption(
                            "Positive contributions push the model output "
                            "towards fraud; negative contributions push it "
                            "away from fraud. These are model contributions, "
                            "not percentage-point changes in fraud probability."
                        )

                        st.info(
                            "Note: V1–V28 are anonymized PCA features from "
                            "the credit card dataset. Their real-world meanings "
                            "are not available. A model contribution does not "
                            "prove that a transaction is fraudulent."
                        )

                except Exception as exc:

                    st.warning(
                        f"Could not generate AI explanation: {exc}"
                    )
                st.markdown("### 5 · Audit fingerprint preview")
                rec = hashlib.sha256(f"{i}|{p:.4f}|{action}|xgb-v1".encode()).hexdigest()
                st.code(rec, language="text")
                st.caption("SHA-256 preview only. This fingerprint is not currently stored on a blockchain.")
                with st.expander("Reveal the dataset label (presentation demo)"):
                    st.write("Ground-truth label for this selected row: **" + ("FRAUD" if int(y_test.iloc[i]) == 1 else "GENUINE") + "**")
                    st.caption("A correct ground-truth label does not mean the model's prediction is correct. Compare the score/action with this label during testing.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.caption("Check that the saved scaler and model expect the same input columns and preprocessing used during training.")

with tab_link:
    st.markdown("## Link & QR scanner")
    st.caption("Use this as a first-pass warning tool. It does not visit links, and it cannot guarantee that a site is safe.")
    st.markdown("### Check a link")
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("Try a suspicious KYC example", use_container_width=True):
        st.session_state["link_input"] = "http://sbi-kyc-update.verify-account.xyz/login"
    if ex2.button("Try an APK example", use_container_width=True):
        st.session_state["link_input"] = "http://192.168.1.5/kyc-verify.apk"
    if ex3.button("Try Google domain", use_container_width=True):
        st.session_state["link_input"] = "https://www.google.com"
    link = st.text_input("Paste URL", key="link_input", placeholder="https://example.com/login")
    if link.strip():
        show_link_result(check_link(link))
    st.divider()
    st.markdown("### Inspect a QR code")
    st.write("Upload a PNG or JPEG image. The app decodes the QR contents locally and does not open any decoded link.")
    up = st.file_uploader("Upload QR image", type=["png", "jpg", "jpeg"], key="qr_uploader")
    if up is not None:
        decoded, err = decode_qr_bytes(up.read())
        if err:
            st.error(err)
        elif decoded is None:
            st.warning("No QR code was detected. Try a clearer, well-lit image.")
        elif decoded.lower().startswith("upi://"):
            query = parse_qs(urlparse(decoded).query)
            get_value = lambda key: query.get(key, [None])[0]
            st.error("This QR contains a UPI payment request. Confirm the recipient and amount before paying. Never share your UPI PIN; you do not need it to receive money.")
            st.write(f"**Payee name:** {get_value('pn') or 'Not specified'}")
            st.write(f"**Payee address:** {get_value('pa') or 'Not specified'}")
            if get_value("am"):
                st.write(f"**Requested amount:** {get_value('am')}")
            with st.expander("View decoded payload"):
                st.code(decoded)
        elif decoded.lower().startswith(("http://", "https://")):
            st.write("Decoded URL:")
            st.code(decoded)
            show_link_result(check_link(decoded))
        else:
            st.info("QR code decoded. Review the text before acting on it.")
            st.code(decoded[:1000])

with tab_res:
    st.markdown("## Model results")
    st.caption("The metrics below are the values currently hard-coded in the project. Verify them against a fresh evaluation before reporting them as measured results.")
    st.markdown("### Transaction model comparison")
    results = pd.DataFrame({
        "Model": ["Simple amount rule", "Logistic Regression", "Random Forest", "XGBoost (used)"],
        "Precision": [0.002, 0.061, 0.960, 0.874],
        "Recall": [0.010, 0.918, 0.735, 0.847],
        "F1 score": [0.003, 0.114, 0.832, 0.860],
        # Keep this column consistently numeric to avoid PyArrow mixed-type errors.
        "PR-AUC": [np.nan, 0.717, 0.860, 0.876],
    })
    st.dataframe(results, use_container_width=True, hide_index=True, column_config={
        "Precision": st.column_config.NumberColumn(format="%.3f"),
        "Recall": st.column_config.NumberColumn(format="%.3f"),
        "F1 score": st.column_config.NumberColumn(format="%.3f"),
        "PR-AUC": st.column_config.NumberColumn(format="%.3f", help="N/A for the simple amount rule in the original table."),
    })
    st.markdown("### Risk-band snapshot")
    if MODEL_READY:
        c1, c2, c3 = st.columns(3)
        c1.metric("Low threshold", f"{LOW:.3f}")
        c2.metric("High threshold", f"{HIGH:.3f}")
        c3.metric("Test rows", f"{len(y_test):,}")
        try:
            probs = model.predict_proba(prep(X_test))[:, 1]
            actual = np.asarray(y_test, dtype=int)
            predicted = np.where(probs >= HIGH, 2, np.where(probs >= LOW, 1, 0))
            fraud_total = int((actual == 1).sum())
            fraud_flagged = int(((actual == 1) & (predicted >= 1)).sum())
            fraud_missed = int(((actual == 1) & (predicted == 0)).sum())
            genuine_otp = int(((actual == 0) & (predicted == 1)).sum())
            genuine_blocked = int(((actual == 0) & (predicted == 2)).sum())
            a, b, c, d = st.columns(4)
            a.metric("Fraud cases", f"{fraud_total:,}")
            b.metric("Fraud flagged", f"{fraud_flagged:,}")
            c.metric("Fraud missed", f"{fraud_missed:,}")
            d.metric("Genuine blocked", f"{genuine_blocked:,}")
            st.caption(f"Genuine transactions routed to verification: {genuine_otp:,}. These counts are computed from the current local model, thresholds, and held-out split.")
        except Exception as exc:
            st.warning(f"Could not compute the risk-band snapshot: {exc}")
    else:
        st.warning("Live metrics are unavailable until the local model artifacts and dataset load successfully.")
    st.markdown("### Link model — historical values")
    link_results = pd.DataFrame({
        "Method": ["Keyword rule", "Random Forest", "XGBoost"],
        "Precision": [0.276, 0.866, 0.743],
        "Recall": [0.098, 0.831, 0.888],
        "F1 score": [0.145, 0.848, 0.809],
    })
    st.dataframe(link_results, use_container_width=True, hide_index=True, column_config={
        "Precision": st.column_config.NumberColumn(format="%.3f"),
        "Recall": st.column_config.NumberColumn(format="%.3f"),
        "F1 score": st.column_config.NumberColumn(format="%.3f"),
    })
    st.warning("Limitations: the link checker currently uses explainable rules, not the link ML model. The displayed link-model metrics are historical hard-coded values and are not connected to live predictions. This is a prototype using public anonymised data, not a production banking system.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#7186a2;font-size:.82rem;padding:8px 0'>AI FRAUD SHIELD · ACADEMIC PROJECT DEMONSTRATION · USE WITH CAUTION</div>", unsafe_allow_html=True)
