import math
import re
import ipaddress
import io
import gzip
import base64
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
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


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
# FIND FILE
# ============================================================

def find_file(filename):
    """
    Search for a file in:
    1. Same folder as app.py
    2. models/ folder
    """

    possible_paths = [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


# ============================================================
# WHITELIST CHECK
# ============================================================

def is_whitelisted(url: str) -> bool:
    """
    Check whether URL belongs to a trusted domain.
    """

    try:
        domain = urlparse(url).netloc.lower()

        # Remove username/password if present
        if "@" in domain:
            domain = domain.split("@")[-1]

        # Remove port
        domain = domain.split(":")[0]

        # Remove www.
        domain = domain.replace("www.", "")

        for safe_domain in SAFE_WHITELIST:
            if domain == safe_domain or domain.endswith("." + safe_domain):
                return True

    except Exception:
        pass

    return False


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # Find required files
    # --------------------------------------------------------

    message_model_file = find_file("message_model.pkl")
    message_vectorizer_file = find_file("message_vectorizer.pkl")
    url_features_file = find_file("url_features.pkl")
    embedded_url_file = find_file("embedded_url_model.txt")

    missing_files = []

    if message_model_file is None:
        missing_files.append("message_model.pkl")

    if message_vectorizer_file is None:
        missing_files.append("message_vectorizer.pkl")

    if url_features_file is None:
        missing_files.append("url_features.pkl")

    if embedded_url_file is None:
        missing_files.append("embedded_url_model.txt")

    if missing_files:

        raise FileNotFoundError(
            "Missing required model file(s): "
            + ", ".join(missing_files)
            + "\n\n"
            + "Put these files either in the same folder as app.py "
            + "or inside the models folder."
        )

    # --------------------------------------------------------
    # Load SMS model
    # --------------------------------------------------------

    message_model = joblib.load(message_model_file)

    # --------------------------------------------------------
    # Load SMS vectorizer
    # --------------------------------------------------------

    message_vectorizer = joblib.load(message_vectorizer_file)

    # --------------------------------------------------------
    # Load embedded URL model
    # --------------------------------------------------------

    with open(
        embedded_url_file,
        "r",
        encoding="utf-8"
    ) as f:
        encoded = f.read().strip()

    compressed = base64.b64decode(encoded)

    model_bytes = gzip.decompress(compressed)

    url_model = joblib.load(
        io.BytesIO(model_bytes)
    )

    # --------------------------------------------------------
    # Load URL feature names
    # --------------------------------------------------------

    url_feature_names = joblib.load(url_features_file)

    return (
        message_model,
        message_vectorizer,
        url_model,
        url_feature_names
    )


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    (
        message_model,
        message_vectorizer,
        url_model,
        url_feature_names
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        f"❌ Error loading models:\n\n{str(e)}"
    )

    st.info(
        "Make sure these files exist either beside app.py "
        "or inside the models folder:\n\n"
        "• message_model.pkl\n"
        "• message_vectorizer.pkl\n"
        "• url_features.pkl\n"
        "• embedded_url_model.txt"
    )


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_url_features(url: str) -> pd.DataFrame:
    """
    Extract the exact 25 URL features used during training.
    """

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    domain = hostname.replace("www.", "")

    # --------------------------------------------------------
    # Basic length features
    # --------------------------------------------------------

    url_length = len(url)

    domain_length = len(domain)

    hostname_length = len(hostname)

    path_length = len(path)

    url_depth = len(
        [x for x in path.split("/") if x]
    )

    query_length = len(query)

    path_segments_count = len(
        [x for x in path.split("/") if x]
    )

    # --------------------------------------------------------
    # Character counts
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
    # Shannon entropy
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
    # Ratios
    # --------------------------------------------------------

    ratio_digits = (
        num_digits / url_length
        if url_length
        else 0
    )

    ratio_letters = (
        num_letters / url_length
        if url_length
        else 0
    )

    # --------------------------------------------------------
    # IP address
    # --------------------------------------------------------

    try:

        ipaddress.ip_address(hostname)

        is_ip_address = 1

    except ValueError:

        is_ip_address = 0

    # --------------------------------------------------------
    # Suspicious TLD
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
    # Login / verification words
    # --------------------------------------------------------

    contains_login = int(
        any(
            word in url.lower()
            for word in [
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
    # Safety check
    # --------------------------------------------------------

    if len(url_feature_names) != len(features):

        raise ValueError(
            f"URL feature mismatch!\n"
            f"Model expects {len(url_feature_names)} features, "
            f"but extractor created {len(features)} features."
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    return pd.DataFrame(
        [features],
        columns=url_feature_names
    )


# ============================================================
# EXTRACT URLS FROM MESSAGE
# ============================================================

def extract_urls_from_text(text: str) -> list:
    """
    Find HTTP/HTTPS URLs inside text.
    """

    url_pattern = r'https?://[^\s"\'<>]+'

    raw_urls = re.findall(
        url_pattern,
        text
    )

    cleaned_urls = [
        url.rstrip('".\',;!?')
        for url in raw_urls
    ]

    return cleaned_urls


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🛡️ SAFELINK AI — SMS Spam Detector & Risk Analyzer"
)

st.markdown(
    "Analyze incoming SMS text and embedded URLs for "
    "**Spam**, **Phishing**, and **Security Threats**."
)

st.divider()


# ============================================================
# STOP IF MODELS ARE NOT LOADED
# ============================================================

if not models_loaded:

    st.warning(
        "⚠️ Models could not be loaded. "
        "Fix the missing model files shown above."
    )

    st.stop()


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
    "Analyze Risk",
    type="primary",
    use_container_width=True
):

    if not user_input.strip():

        st.warning(
            "Please enter a message to analyze."
        )

        st.stop()

    try:

        # ====================================================
        # 1. SMS TEXT MODEL
        # ====================================================

        msg_tfidf = message_vectorizer.transform(
            [user_input]
        )

        msg_probabilities = (
            message_model.predict_proba(msg_tfidf)[0]
        )

        msg_pred = message_model.predict(
            msg_tfidf
        )[0]

        # Assuming class 1 = Spam/Threat
        msg_prob = float(
            msg_probabilities[1]
        )

        # ====================================================
        # 2. EXTRACT URLs
        # ====================================================

        extracted_urls = extract_urls_from_text(
            user_input
        )

        url_results = []

        max_url_prob = 0.0

        # ====================================================
        # 3. ANALYZE EACH URL
        # ====================================================

        for url in extracted_urls:

            # ------------------------------------------------
            # Whitelist
            # ------------------------------------------------

            if is_whitelisted(url):

                u_threat_prob = 0.0

                status_note = (
                    "Verified Whitelisted Domain"
                )

            else:

                # --------------------------------------------
                # Extract features
                # --------------------------------------------

                url_feats = extract_url_features(
                    url
                )

                # --------------------------------------------
                # URL model prediction
                # --------------------------------------------

                url_probabilities = (
                    url_model.predict_proba(
                        url_feats
                    )[0]
                )

                # --------------------------------------------
                # Your trained model:
                #
                # Index 1 = Safe probability
                # Therefore:
                #
                # Threat = 1 - Safe
                # --------------------------------------------

                safe_prob = float(
                    url_probabilities[1]
                )

                u_threat_prob = (
                    1.0 - safe_prob
                )

                status_note = "ML Analyzed"

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            url_results.append(
                {
                    "url": url,
                    "prob": u_threat_prob,
                    "note": status_note
                }
            )

            # ------------------------------------------------
            # Maximum URL risk
            # ------------------------------------------------

            if u_threat_prob > max_url_prob:

                max_url_prob = u_threat_prob

        # ====================================================
        # 4. OVERALL RISK
        # ====================================================

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

        # ====================================================
        # 5. RESULTS
        # ====================================================

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

        # ====================================================
        # 6. RISK BANNER
        # ====================================================

        if risk_score_pct >= 70:

            st.error(
                f"🚨 **HIGH RISK THREAT DETECTED** "
                f"(Risk Score: {risk_score_pct}%)"
            )

        elif risk_score_pct >= 40:

            st.warning(
                f"⚠️ **MODERATE RISK DETECTED** "
                f"(Risk Score: {risk_score_pct}%)"
            )

        else:

            st.success(
                f"✅ **SAFE MESSAGE** "
                f"(Risk Score: {risk_score_pct}%)"
            )

        st.divider()

        # ====================================================
        # 7. DETAILED BREAKDOWN
        # ====================================================

        left_col, right_col = st.columns(2)

        # ----------------------------------------------------
        # TEXT ANALYSIS
        # ----------------------------------------------------

        with left_col:

            st.markdown(
                "### 💬 Text Analysis"
            )

            if msg_pred == 1:

                classification = (
                    "🚨 Spam/Threat"
                )

                confidence = msg_prob

            else:

                classification = (
                    "✅ Safe"
                )

                confidence = 1 - msg_prob

            st.write(
                f"**Classification:** "
                f"{classification}"
            )

            st.write(
                f"**Confidence:** "
                f"{confidence:.2%}"
            )

        # ----------------------------------------------------
        # URL ANALYSIS
        # ----------------------------------------------------

        with right_col:

            st.markdown(
                "### 🔗 URL Analysis"
            )

            if not extracted_urls:

                st.info(
                    "No URLs found in the provided text."
                )

            else:

                for res in url_results:

                    if res["prob"] > 0.5:

                        status = "🚨 THREAT"

                    else:

                        status = "✅ SAFE"

                    st.write(
                        f"**URL:** `{res['url']}`"
                    )

                    st.write(
                        f"**Status:** {status} "
                        f"({res['prob']:.2%} risk) — "
                        f"*{res['note']}*"
                    )

                    st.divider()

    except Exception as e:

        st.error(
            "❌ Error while analyzing the message."
        )

        st.code(
            str(e)
        )