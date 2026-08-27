
import math
import re
import ipaddress
import gzip
import base64
import io
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SafeLink AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# FILE PATHS
# ============================================================

MESSAGE_MODEL_FILE = MODELS_DIR / "message_model.pkl"
MESSAGE_VECTORIZER_FILE = MODELS_DIR / "message_vectorizer.pkl"
URL_FEATURES_FILE = MODELS_DIR / "url_features.pkl"

EMBEDDED_URL_MODEL_FILE = BASE_DIR / "embedded_url_model.txt"


# ============================================================
# WELCOME PAGE
# ============================================================

if "show_scanner" not in st.session_state:
    st.session_state.show_scanner = False


if not st.session_state.show_scanner:

    st.markdown(
        """
        <style>

        .welcome-container {
            text-align: center;
            padding: 20px 15px;
        }

        .welcome-shield {
            font-size: 70px;
            margin-bottom: 10px;
        }

        .welcome-title {
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .welcome-subtitle {
            font-size: 27px;
            font-weight: 700;
            margin-bottom: 15px;
        }

        .welcome-description {
            font-size: 18px;
            line-height: 1.6;
            max-width: 700px;
            margin: auto;
        }

        .feature-card {
            padding: 22px 15px;
            border-radius: 16px;
            border: 1px solid rgba(128,128,128,0.25);
            min-height: 115px;
            margin-bottom: 20px;
            text-align: center;
        }

        .feature-icon {
            font-size: 32px;
        }

        .feature-title {
            font-size: 17px;
            font-weight: 700;
            margin-top: 8px;
        }

        div.stButton > button {
            border-radius: 12px;
            font-size: 18px;
            font-weight: 700;
            padding: 12px 35px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="welcome-container">

            <div class="welcome-shield">🛡️</div>

            <div class="welcome-title">
                SafeLink AI
            </div>

            <div class="welcome-subtitle">
                Welcome to SafeLink AI
            </div>

            <div class="welcome-subtitle">
                AI-Powered Security Scanner
            </div>

            <div class="welcome-description">
                Analyze SMS messages and URLs using machine learning
                to identify potential spam, phishing and security threats.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔗</div>
                <div class="feature-title">
                    URL Detection
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">
                    Message Detection
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">
                    AI Risk Analysis
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])

    with button_col2:

        if st.button(
            "CONTINUE",
            type="primary",
            use_container_width=True
        ):

            st.session_state.show_scanner = True
            st.rerun()

    st.stop()


# ============================================================
# SAFE WHITELIST
# ============================================================

SAFE_WHITELIST = [

    "google.com",
    "wikipedia.org",
    "stackoverflow.com",
    "apple.com",
    "github.com",
    "linkedin.com",
    "twitter.com",
    "reddit.com",
    "instagram.com",
    "facebook.com",
    "yahoo.com",
    "bing.com",
    "duckduckgo.com",
    "cloudflare.com",
    "openai.com",
    "gitlab.com",
    "amazonaws.com",
    "azure.com",
    "huggingface.com",
    "youtube.com",
    "microsoft.com"

]


# ============================================================
# WHITELIST CHECK
# ============================================================

def is_whitelisted(url):

    try:

        parsed = urlparse(url)

        domain = parsed.hostname or ""

        domain = domain.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        for safe_domain in SAFE_WHITELIST:

            if (
                domain == safe_domain
                or domain.endswith("." + safe_domain)
            ):
                return True

    except Exception:
        pass

    return False


# ============================================================
# LOAD EMBEDDED URL MODEL
# ============================================================

@st.cache_resource
def load_embedded_url_model():

    if not EMBEDDED_URL_MODEL_FILE.exists():

        raise FileNotFoundError(
            "embedded_url_model.txt is missing."
        )

    with open(
        EMBEDDED_URL_MODEL_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        encoded_data = f.read().strip()

    if not encoded_data:

        raise ValueError(
            "embedded_url_model.txt is empty."
        )

    try:

        compressed_data = base64.b64decode(
            encoded_data
        )

        model_bytes = gzip.decompress(
            compressed_data
        )

        url_model = joblib.load(
            io.BytesIO(model_bytes)
        )

        return url_model

    except Exception as e:

        raise RuntimeError(
            f"Unable to load embedded URL model: {e}"
        )


# ============================================================
# CHECK MODEL FILES
# ============================================================

def get_missing_files():

    missing = []

    required_files = [
        MESSAGE_MODEL_FILE,
        MESSAGE_VECTORIZER_FILE,
        URL_FEATURES_FILE,
        EMBEDDED_URL_MODEL_FILE
    ]

    for file in required_files:

        if not file.exists():
            missing.append(file)

    return missing


# ============================================================
# LOAD ALL MODELS
# ============================================================

@st.cache_resource
def load_models():

    missing_files = get_missing_files()

    if missing_files:

        missing_text = "\n".join(
            f"- {file}"
            for file in missing_files
        )

        raise FileNotFoundError(
            "Required SafeLink AI model files are missing:\n\n"
            + missing_text
        )

    # --------------------------------------------------------
    # MESSAGE MODEL
    # --------------------------------------------------------

    message_model = joblib.load(
        MESSAGE_MODEL_FILE
    )

    # --------------------------------------------------------
    # MESSAGE VECTORIZER
    # --------------------------------------------------------

    message_vectorizer = joblib.load(
        MESSAGE_VECTORIZER_FILE
    )

    # --------------------------------------------------------
    # URL FEATURE NAMES
    # --------------------------------------------------------

    url_feature_names = joblib.load(
        URL_FEATURES_FILE
    )

    # --------------------------------------------------------
    # EMBEDDED URL MODEL
    # --------------------------------------------------------

    url_model = load_embedded_url_model()

    return (
        message_model,
        message_vectorizer,
        url_model,
        url_feature_names
    )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    (
        message_model,
        message_vectorizer,
        url_model,
        url_feature_names
    ) = load_models()

except Exception as e:

    st.error(
        "❌ Error loading SafeLink AI models"
    )

    st.code(
        str(e)
    )

    st.warning(
        "Make sure the following structure exists in your "
        "GitHub repository:"
    )

    st.code(
        """
SafeLink-AI/
│
├── app.py
├── embedded_url_model.txt
│
└── models/
    ├── message_model.pkl
    ├── message_vectorizer.pkl
    └── url_features.pkl
        """
    )

    st.info(
        "The URL model is embedded inside "
        "embedded_url_model.txt, but the SMS model, "
        "vectorizer and URL feature file must also be "
        "uploaded to the repository."
    )

    st.stop()


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_url_features(url):

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    domain = hostname

    if domain.lower().startswith("www."):
        domain = domain[4:]

    # --------------------------------------------------------
    # BASIC FEATURES
    # --------------------------------------------------------

    url_length = len(url)

    domain_length = len(domain)

    hostname_length = len(hostname)

    path_length = len(path)

    url_depth = len(
        [
            x
            for x in path.split("/")
            if x
        ]
    )

    query_length = len(query)

    path_segments_count = len(
        [
            x
            for x in path.split("/")
            if x
        ]
    )

    # --------------------------------------------------------
    # CHARACTER FEATURES
    # --------------------------------------------------------

    num_digits = sum(
        c.isdigit()
        for c in url
    )

    num_letters = sum(
        c.isalpha()
        for c in url
    )

    num_special_chars = sum(
        not c.isalnum()
        for c in url
    )

    num_dots = url.count(".")

    num_hyphens = url.count("-")

    num_at = url.count("@")

    num_percent = url.count("%")

    num_equals = url.count("=")

    num_question = url.count("?")

    num_ampersand = url.count("&")

    num_slash = url.count("/")

    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    counts = Counter(url)

    total = len(url)

    if total > 0:

        entropy_url = -sum(
            (count / total)
            * math.log2(count / total)
            for count in counts.values()
        )

    else:

        entropy_url = 0

    # --------------------------------------------------------
    # RATIOS
    # --------------------------------------------------------

    ratio_digits = (
        num_digits / url_length
        if url_length > 0
        else 0
    )

    ratio_letters = (
        num_letters / url_length
        if url_length > 0
        else 0
    )

    # --------------------------------------------------------
    # IP ADDRESS
    # --------------------------------------------------------

    try:

        ipaddress.ip_address(
            hostname
        )

        is_ip_address = 1

    except ValueError:

        is_ip_address = 0

    # --------------------------------------------------------
    # SUSPICIOUS TLD
    # --------------------------------------------------------

    suspicious_tlds = [

        ".tk",
        ".ml",
        ".ga",
        ".cf",
        ".gq",
        ".top",
        ".xyz",
        ".click",
        ".link",
        ".work",
        ".zip",
        ".review"

    ]

    is_suspicious_tld = int(
        any(
            domain.lower().endswith(tld)
            for tld in suspicious_tlds
        )
    )

    # --------------------------------------------------------
    # HTTPS
    # --------------------------------------------------------

    uses_https = int(
        parsed.scheme.lower() == "https"
    )

    # --------------------------------------------------------
    # LOGIN / VERIFICATION
    # --------------------------------------------------------

    contains_login = int(
        any(
            keyword in url.lower()
            for keyword in [
                "login",
                "signin",
                "sign-in",
                "verify",
                "verification"
            ]
        )
    )

    # --------------------------------------------------------
    # EXACT 25 FEATURES
    # --------------------------------------------------------

    features = [

        url_length,
        domain_length,
        hostname_length,
        path_length,
        url_depth,
        query_length,
        path_segments_count,
        num_digits,
        num_letters,
        num_special_chars,
        num_dots,
        num_hyphens,
        num_at,
        num_percent,
        num_equals,
        num_question,
        num_ampersand,
        num_slash,
        entropy_url,
        ratio_digits,
        ratio_letters,
        is_ip_address,
        is_suspicious_tld,
        uses_https,
        contains_login

    ]

    # --------------------------------------------------------
    # FEATURE COUNT CHECK
    # --------------------------------------------------------

    if len(url_feature_names) != 25:

        raise ValueError(
            f"Expected 25 URL features, but "
            f"url_features.pkl contains "
            f"{len(url_feature_names)} features."
        )

    return pd.DataFrame(
        [features],
        columns=url_feature_names
    )


# ============================================================
# EXTRACT URLS FROM TEXT
# ============================================================

def extract_urls_from_text(text):

    url_pattern = r'https?://[^\s"\'<>]+'

    raw_urls = re.findall(
        url_pattern,
        text
    )

    cleaned_urls = []

    for url in raw_urls:

        url = url.rstrip(
            '".\',;:!?)]}'
        )

        if url:
            cleaned_urls.append(url)

    return cleaned_urls


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ SafeLink AI — AI-Powered Security Scanner"
)

st.markdown(
    """
Analyze incoming SMS messages and embedded URLs
for potential **Spam, Phishing, and Security Threats**.
"""
)

st.divider()


# ============================================================
# INPUT
# ============================================================

user_input = st.text_area(

    "Paste SMS / Message / URL Content:",

    height=140,

    placeholder=(
        "Example: Claim your $1000 prize now at "
        "http://free-prize.xyz/claim"
    )

)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Risk",
    type="primary",
    use_container_width=True
):

    if not user_input.strip():

        st.warning(
            "⚠️ Please enter a message to analyze."
        )

        st.stop()


    # ========================================================
    # 1. SMS TEXT ANALYSIS
    # ========================================================

    try:

        msg_tfidf = message_vectorizer.transform(
            [user_input]
        )

        msg_pred = message_model.predict(
            msg_tfidf
        )[0]

        msg_probabilities = (
            message_model.predict_proba(
                msg_tfidf
            )[0]
        )

        # ----------------------------------------------------
        # Find probability for class 1
        # ----------------------------------------------------

        if hasattr(
            message_model,
            "classes_"
        ):

            classes = list(
                message_model.classes_
            )

            if 1 in classes:

                spam_index = classes.index(1)

                msg_prob = float(
                    msg_probabilities[spam_index]
                )

            else:

                msg_prob = float(
                    max(msg_probabilities)
                )

        else:

            msg_prob = float(
                msg_probabilities[-1]
            )

    except Exception as e:

        st.error(
            f"❌ SMS model error: {e}"
        )

        st.stop()


    # ========================================================
    # 2. URL ANALYSIS
    # ========================================================

    extracted_urls = extract_urls_from_text(
        user_input
    )

    url_results = []

    max_url_prob = 0.0


    for url in extracted_urls:

        # ----------------------------------------------------
        # WHITELIST
        # ----------------------------------------------------

        if is_whitelisted(url):

            u_threat_prob = 0.0

            status_note = (
                "Verified Whitelisted Domain"
            )

        else:

            try:

                url_feats = extract_url_features(
                    url
                )

                model_probabilities = (
                    url_model.predict_proba(
                        url_feats
                    )[0]
                )

                # ------------------------------------------------
                # PhiUSIIL model:
                # index 1 = SAFE
                # threat = 1 - safe
                # ------------------------------------------------

                safe_prob = float(
                    model_probabilities[1]
                )

                u_threat_prob = (
                    1.0 - safe_prob
                )

                status_note = "ML Analyzed"

            except Exception as e:

                u_threat_prob = 1.0

                status_note = (
                    f"URL analysis error: {e}"
                )


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        url_results.append(
            {
                "url": url,
                "prob": u_threat_prob,
                "note": status_note
            }
        )


        # ----------------------------------------------------
        # MAX URL RISK
        # ----------------------------------------------------

        if u_threat_prob > max_url_prob:

            max_url_prob = u_threat_prob


    # ========================================================
    # 3. OVERALL RISK
    # ========================================================

    if extracted_urls:

        combined_risk_prob = max(
            msg_prob,
            max_url_prob
        )

    else:

        combined_risk_prob = msg_prob


    risk_score_pct = int(
        combined_risk_prob * 100
    )


    # ========================================================
    # 4. RESULTS
    # ========================================================

    st.subheader(
        "📊 Analysis Results"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "SMS Text Threat Probability",
            f"{int(msg_prob * 100)}%"
        )


    with col2:

        st.metric(
            "Detected URLs",
            len(extracted_urls)
        )


    with col3:

        st.metric(
            "Overall Risk Score",
            f"{risk_score_pct}%"
        )


    # ========================================================
    # 5. RISK BANNER
    # ========================================================

    if risk_score_pct >= 70:

        st.error(
            f"🚨 HIGH RISK THREAT DETECTED "
            f"(Risk Score: {risk_score_pct}%)"
        )

    elif risk_score_pct >= 40:

        st.warning(
            f"⚠️ MODERATE RISK DETECTED "
            f"(Risk Score: {risk_score_pct}%)"
        )

    else:

        st.success(
            f"✅ SAFE MESSAGE "
            f"(Risk Score: {risk_score_pct}%)"
        )


    st.divider()


    # ========================================================
    # 6. DETAILED BREAKDOWN
    # ========================================================

    left_col, right_col = st.columns(2)


    # ========================================================
    # TEXT ANALYSIS
    # ========================================================

    with left_col:

        st.markdown(
            "### 💬 Text Analysis"
        )

        if msg_pred == 1:

            st.write(
                "**Classification:** 🚨 Spam/Threat"
            )

            confidence = msg_prob

        else:

            st.write(
                "**Classification:** ✅ Safe"
            )

            confidence = 1 - msg_prob


        st.write(
            f"**Confidence:** {confidence:.2%}"
        )


    # ========================================================
    # URL ANALYSIS
    # ========================================================

    with right_col:

        st.markdown(
            "### 🔗 URL Analysis"
        )

        if not extracted_urls:

            st.info(
                "No URLs found in the provided text."
            )

        else:

            for result in url_results:

                if result["prob"] >= 0.5:

                    status = "🚨 THREAT"

                else:

                    status = "✅ SAFE"


                st.write(
                    f"**URL:** `{result['url']}`"
                )

                st.write(
                    f"**Status:** {status}"
                )

                st.write(
                    f"**Risk:** "
                    f"{result['prob']:.2%}"
                )

                st.caption(
                    result["note"]
                )

                st.divider()

