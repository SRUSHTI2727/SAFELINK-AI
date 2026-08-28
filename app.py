import math
import gzip
import base64
import io
import ipaddress

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


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL FILES
# ============================================================

MESSAGE_MODEL_FILE = BASE_DIR / "message_model.pkl"

MESSAGE_VECTORIZER_FILE = (
    BASE_DIR / "message_vectorizer.pkl"
)

URL_FEATURES_FILE = (
    BASE_DIR / "url_features.pkl"
)

EMBEDDED_URL_MODEL_FILE = (
    BASE_DIR / "embedded_url_model.txt"
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "url_result" not in st.session_state:
    st.session_state.url_result = None

if "message_result" not in st.session_state:
    st.session_state.message_result = None


# ============================================================
# SECURITY LISTS
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


SUSPICIOUS_TLDS = [

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


SUSPICIOUS_KEYWORDS = [

    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "confirm",
    "confirmation",
    "password",
    "credential",
    "payment",
    "billing",
    "bank",
    "wallet",
    "claim",
    "reward",
    "prize",
    "suspended",
    "unlock"

]


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return url


# ============================================================
# GET DOMAIN
# ============================================================

def get_domain(url):

    try:

        normalized = normalize_url(url)

        parsed = urlparse(normalized)

        hostname = parsed.hostname or ""

        hostname = hostname.lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except Exception:

        return ""


# ============================================================
# WHITELIST CHECK
# ============================================================

def is_whitelisted(url):

    domain = get_domain(url)

    if not domain:
        return False

    for safe_domain in SAFE_WHITELIST:

        if (
            domain == safe_domain
            or domain.endswith("." + safe_domain)
        ):

            return True

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
    ) as file:

        encoded_data = file.read().strip()

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

        model = joblib.load(
            io.BytesIO(model_bytes)
        )

        return model

    except Exception as e:

        raise RuntimeError(
            f"Unable to load URL model: {e}"
        )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def get_missing_files():

    required_files = [

        MESSAGE_MODEL_FILE,
        MESSAGE_VECTORIZER_FILE,
        URL_FEATURES_FILE,
        EMBEDDED_URL_MODEL_FILE

    ]

    missing = []

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
            f"- {file.name}"
            for file in missing_files
        )

        raise FileNotFoundError(
            "Required SafeLink AI model files are missing:\n\n"
            + missing_text
        )

    message_model = joblib.load(
        MESSAGE_MODEL_FILE
    )

    message_vectorizer = joblib.load(
        MESSAGE_VECTORIZER_FILE
    )

    url_feature_names = joblib.load(
        URL_FEATURES_FILE
    )

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

    st.code(str(e))

    st.warning(
        "Make sure these files are present in the same "
        "folder as app.py:"
    )

    st.code(
        """
SafeLink AI/
│
├── app.py
├── embedded_url_model.txt
├── message_model.pkl
├── message_vectorizer.pkl
└── url_features.pkl
        """
    )

    st.stop()


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_url_features(url):

    original_url = url.strip()

    normalized_url = normalize_url(
        original_url
    )

    parsed = urlparse(
        normalized_url
    )

    hostname = parsed.hostname or ""

    path = parsed.path or ""

    query = parsed.query or ""

    domain = hostname

    if domain.lower().startswith("www."):

        domain = domain[4:]


    # --------------------------------------------------------
    # BASIC FEATURES
    # --------------------------------------------------------

    url_length = len(original_url)

    domain_length = len(domain)

    hostname_length = len(hostname)

    path_length = len(path)

    url_depth = len([
        item
        for item in path.split("/")
        if item
    ])

    query_length = len(query)

    path_segments_count = len([
        item
        for item in path.split("/")
        if item
    ])


    # --------------------------------------------------------
    # CHARACTER FEATURES
    # --------------------------------------------------------

    num_digits = sum(
        char.isdigit()
        for char in original_url
    )

    num_letters = sum(
        char.isalpha()
        for char in original_url
    )

    num_special_chars = sum(
        not char.isalnum()
        for char in original_url
    )

    num_dots = original_url.count(".")

    num_hyphens = original_url.count("-")

    num_at = original_url.count("@")

    num_percent = original_url.count("%")

    num_equals = original_url.count("=")

    num_question = original_url.count("?")

    num_ampersand = original_url.count("&")

    num_slash = original_url.count("/")


    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    counts = Counter(
        original_url
    )

    total = len(original_url)

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

    is_suspicious_tld = int(
        any(
            domain.lower().endswith(tld)
            for tld in SUSPICIOUS_TLDS
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
            keyword in original_url.lower()
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


    if len(url_feature_names) != 25:

        raise ValueError(
            "URL feature count mismatch. "
            "Expected 25 features, but found "
            f"{len(url_feature_names)}."
        )


    return pd.DataFrame(
        [features],
        columns=url_feature_names
    )


# ============================================================
# SECURITY INDICATOR ANALYSIS
# ============================================================

def analyze_url_security(url):

    normalized = normalize_url(url)

    parsed = urlparse(normalized)

    domain = get_domain(url)

    lower_url = url.lower()

    indicators = []

    score = 0


    # --------------------------------------------------------
    # HTTP
    # --------------------------------------------------------

    if parsed.scheme.lower() == "http":

        indicators.append(
            "Uses HTTP instead of HTTPS"
        )

        score += 15


    # --------------------------------------------------------
    # SUSPICIOUS TLD
    # --------------------------------------------------------

    if any(
        domain.endswith(tld)
        for tld in SUSPICIOUS_TLDS
    ):

        indicators.append(
            "Uses a suspicious TLD"
        )

        score += 25


    # --------------------------------------------------------
    # IP ADDRESS
    # --------------------------------------------------------

    try:

        ipaddress.ip_address(
            parsed.hostname or ""
        )

        indicators.append(
            "Uses an IP address instead of a domain name"
        )

        score += 25

    except ValueError:

        pass


    # --------------------------------------------------------
    # LOGIN / VERIFICATION
    # --------------------------------------------------------

    matched_keywords = []

    for keyword in SUSPICIOUS_KEYWORDS:

        if keyword in lower_url:

            matched_keywords.append(keyword)


    if matched_keywords:

        indicators.append(
            "Contains security-sensitive keywords: "
            + ", ".join(matched_keywords[:4])
        )

        score += min(
            25,
            len(matched_keywords) * 8
        )


    # --------------------------------------------------------
    # @ SYMBOL
    # --------------------------------------------------------

    if "@" in url:

        indicators.append(
            "Contains @ symbol"
        )

        score += 20


    # --------------------------------------------------------
    # MULTIPLE HYPHENS
    # --------------------------------------------------------

    if url.count("-") >= 3:

        indicators.append(
            "Contains multiple hyphens"
        )

        score += 10


    # --------------------------------------------------------
    # LONG URL
    # --------------------------------------------------------

    if len(url) >= 100:

        indicators.append(
            "Unusually long URL"
        )

        score += 10


    # --------------------------------------------------------
    # MANY PATH SEGMENTS
    # --------------------------------------------------------

    path_parts = [
        x
        for x in parsed.path.split("/")
        if x
    ]

    if len(path_parts) >= 4:

        indicators.append(
            "Contains many URL path segments"
        )

        score += 10


    # --------------------------------------------------------
    # ENCODED CHARACTERS
    # --------------------------------------------------------

    if url.count("%") >= 3:

        indicators.append(
            "Contains excessive encoded characters"
        )

        score += 10


    # --------------------------------------------------------
    # BRAND + SECURITY KEYWORD
    # --------------------------------------------------------

    brands = [

        "paypal",
        "google",
        "microsoft",
        "apple",
        "amazon",
        "facebook",
        "instagram",
        "linkedin",
        "bank"

    ]

    security_words = [

        "verify",
        "verification",
        "login",
        "signin",
        "secure",
        "account",
        "update",
        "confirm"

    ]

    has_brand = any(
        brand in lower_url
        for brand in brands
    )

    has_security_word = any(
        word in lower_url
        for word in security_words
    )

    if has_brand and has_security_word:

        indicators.append(
            "Contains a brand name with a security-related keyword"
        )

        score += 25


    return min(score, 100), indicators


# ============================================================
# URL ML ANALYSIS
# ============================================================

def get_url_model_signal(url):

    features = extract_url_features(url)

    probabilities = (
        url_model.predict_proba(features)[0]
    )

    classes = list(
        url_model.classes_
    )

    prediction = url_model.predict(
        features
    )[0]


    # Existing model:
    # 0 = phishing/threat
    # 1 = safe

    if 0 in classes and 1 in classes:

        threat_index = classes.index(0)

        ml_threat_probability = float(
            probabilities[threat_index]
        )

    else:

        ml_threat_probability = 0.50


    return (
        features,
        prediction,
        ml_threat_probability
    )


# ============================================================
# FINAL URL PREDICTION
# ============================================================

def predict_url(url):

    # --------------------------------------------------------
    # WHITELIST
    # --------------------------------------------------------

    if is_whitelisted(url):

        features = extract_url_features(url)

        return {

            "result": "Safe",

            "risk": 0,

            "ml_risk": 0,

            "security_score": 0,

            "indicators": [],

            "features": features,

            "note":
                "Trusted domain in SafeLink AI whitelist."

        }


    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    (
        features,
        model_prediction,
        ml_risk
    ) = get_url_model_signal(url)


    # --------------------------------------------------------
    # SECURITY RULES
    # --------------------------------------------------------

    security_score, indicators = (
        analyze_url_security(url)
    )


    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    ml_score = ml_risk * 100

    combined_score = (
        (ml_score * 0.40)
        +
        (security_score * 0.60)
    )


    # --------------------------------------------------------
    # EXTRA SAFETY RULE
    # --------------------------------------------------------

    strong_indicators = 0

    lower_url = url.lower()

    parsed = urlparse(
        normalize_url(url)
    )

    domain = get_domain(url)


    if parsed.scheme.lower() == "http":

        strong_indicators += 1


    if any(
        domain.endswith(tld)
        for tld in SUSPICIOUS_TLDS
    ):

        strong_indicators += 1


    if any(
        word in lower_url
        for word in [
            "login",
            "verify",
            "verification"
        ]
    ):

        strong_indicators += 1


    if any(
        brand in lower_url
        for brand in [
            "paypal",
            "bank",
            "account"
        ]
    ):

        strong_indicators += 1


    if strong_indicators >= 3:

        combined_score = max(
            combined_score,
            75
        )

    elif strong_indicators >= 2:

        combined_score = max(
            combined_score,
            55
        )


    combined_score = min(
        combined_score,
        100
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if combined_score >= 70:

        final_result = "Suspicious"

    elif combined_score >= 40:

        final_result = "Potential Risk"

    else:

        final_result = "Safe"


    return {

        "result": final_result,

        "risk": combined_score,

        "ml_risk": ml_score,

        "security_score": security_score,

        "indicators": indicators,

        "features": features,

        "note":
            "Hybrid analysis using machine learning "
            "and URL security indicators."

    }


# ============================================================
# CLEAR URL CALLBACK
# ============================================================

def clear_url():

    st.session_state.pop(
        "url_box",
        None
    )

    st.session_state.url_result = None


# ============================================================
# CLEAR MESSAGE CALLBACK
# ============================================================

def clear_message():

    st.session_state.pop(
        "message_box",
        None
    )

    st.session_state.message_result = None


# ============================================================
# BACK TO HOME
# ============================================================

def back_to_home():

    if st.button(
        "⬅️ Back to Home"
    ):

        st.session_state.page = "welcome"

        st.session_state.url_result = None

        st.session_state.message_result = None

        st.session_state.pop(
            "url_box",
            None
        )

        st.session_state.pop(
            "message_box",
            None
        )

        st.rerun()


# ============================================================
# WELCOME PAGE
# ============================================================

def welcome_page():

    st.title(
        "🛡️ SafeLink AI"
    )

    st.header(
        "Welcome to SafeLink AI"
    )

    st.subheader(
        "AI-Powered Security Scanner"
    )

    st.write(
        "SafeLink AI analyzes URLs and text messages "
        "to identify potential security threats, "
        "spam and phishing-related content."
    )

    st.divider()


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # URL SCANNER
    # --------------------------------------------------------

    with col1:

        st.markdown(
            "## 🔗 URL Scanner"
        )

        st.write(
            "Analyze URL structure and security "
            "indicators to identify potentially "
            "suspicious URLs."
        )

        st.write(
            "• URL characteristics\n\n"
            "• Suspicious TLD detection\n\n"
            "• HTTPS analysis\n\n"
            "• Login / verification detection\n\n"
            "• Machine learning analysis"
        )

        if st.button(
            "🔗 Open URL Scanner",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = "url"

            st.rerun()


    # --------------------------------------------------------
    # MESSAGE SCANNER
    # --------------------------------------------------------

    with col2:

        st.markdown(
            "## 💬 Message Scanner"
        )

        st.write(
            "Analyze SMS and text messages using "
            "a machine learning model to identify "
            "potential spam or harmful messages."
        )

        st.write(
            "• Text preprocessing\n\n"
            "• Machine learning classification\n\n"
            "• Spam detection\n\n"
            "• Confidence score"
        )

        if st.button(
            "💬 Open Message Scanner",
            type="primary",
            use_container_width=True
        ):

            st.session_state.page = "message"

            st.rerun()


    st.divider()

    st.info(
        "SafeLink AI provides an automated security "
        "assessment and should be treated as an "
        "assistance tool rather than a guarantee."
    )


# ============================================================
# URL SCANNER PAGE
# ============================================================

def url_scanner_page():

    back_to_home()

    st.title(
        "🔗 URL Scanner"
    )

    st.write(
        "Enter a URL to analyze its structure, "
        "security indicators and machine learning risk."
    )

    st.divider()


    # ========================================================
    # URL INPUT
    # ========================================================

    url_input = st.text_input(
        "Enter URL:",
        placeholder="Example: https://www.google.com",
        key="url_box"
    )


    # ========================================================
    # BUTTONS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        analyze_clicked = st.button(
            "🔍 Analyze URL",
            type="primary",
            use_container_width=True
        )


    with col2:

        st.button(
            "🗑️ Clear",
            use_container_width=True,
            on_click=clear_url
        )


    # ========================================================
    # ANALYZE URL
    # ========================================================

    if analyze_clicked:

        if not url_input.strip():

            st.warning(
                "⚠️ Please enter a URL."
            )

        else:

            try:

                st.session_state.url_result = (
                    predict_url(
                        url_input.strip()
                    )
                )

            except Exception as e:

                st.error(
                    f"❌ URL analysis error: {e}"
                )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if st.session_state.url_result is None:

        return


    # IMPORTANT:
    # result is defined INSIDE the function and BEFORE
    # Security Indicators, URL Characteristics, etc.

    result = st.session_state.url_result

    features = result["features"]

    risk = result["risk"]

    final_result = result["result"]


    st.divider()

    st.subheader(
        "📊 URL Analysis Result"
    )


    # ========================================================
    # RESULT METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Final Result",
            final_result
        )


    with col2:

        st.metric(
            "Risk Score",
            f"{int(risk)}%"
        )


    with col3:

        https_value = (
            "Yes"
            if features.iloc[0]["uses_https"] == 1
            else "No"
        )

        st.metric(
            "HTTPS",
            https_value
        )


    # ========================================================
    # RISK BANNER
    # ========================================================

    if risk >= 70:

        st.error(
            f"🚨 HIGH RISK URL\n\n"
            f"Risk Score: {risk:.0f}%"
        )

    elif risk >= 40:

        st.warning(
            f"⚠️ POTENTIAL RISK URL\n\n"
            f"Risk Score: {risk:.0f}%"
        )

    else:

        st.success(
            f"✅ URL APPEARS SAFE\n\n"
            f"Risk Score: {risk:.0f}%"
        )


    st.divider()


    # ========================================================
    # SECURITY INDICATORS
    # ========================================================

    st.subheader(
        "🔎 Security Indicators"
    )


    indicators = result["indicators"]


    if indicators:

        for indicator in indicators:

            st.warning(
                "⚠️ " + indicator
            )


        # ----------------------------------------------------
        # CAUTION NOTE
        # ----------------------------------------------------

        if risk >= 30:

            st.info(
                "⚠️ Caution: This URL has a risk score of "
                f"{risk:.0f}% and if it contains suspicious indicators. "
                "Even if the final result is shown, "
                "do not blindly trust the URL. Verify its source "
                "before opening or providing any information."
            )


    else:

        st.success(
            "No major suspicious indicators detected."
        )


    st.divider()


    # ========================================================
    # URL CHARACTERISTICS
    # ========================================================

    st.subheader(
        "📋 URL Characteristics"
    )


    feature_col1, feature_col2 = st.columns(2)


    with feature_col1:

        st.write(
            f"**URL Length:** "
            f"{int(features.iloc[0]['url_length'])}"
        )

        st.write(
            f"**Domain Length:** "
            f"{int(features.iloc[0]['domain_length'])}"
        )

        st.write(
            f"**URL Depth:** "
            f"{int(features.iloc[0]['url_depth'])}"
        )

        st.write(
            f"**IP Address:** "
            f"{'Yes' if features.iloc[0]['is_ip_address'] == 1 else 'No'}"
        )


    with feature_col2:

        st.write(
            f"**HTTPS:** "
            f"{'Yes' if features.iloc[0]['uses_https'] == 1 else 'No'}"
        )

        st.write(
            f"**Suspicious TLD:** "
            f"{'Yes' if features.iloc[0]['is_suspicious_tld'] == 1 else 'No'}"
        )

        st.write(
            f"**Login / Verification:** "
            f"{'Yes' if features.iloc[0]['contains_login'] == 1 else 'No'}"
        )

        st.write(
            f"**Special Characters:** "
            f"{int(features.iloc[0]['num_special_chars'])}"
        )


    st.divider()


    # ========================================================
    # ANALYSIS SUMMARY
    # ========================================================

    st.subheader(
        "🧠 Analysis Summary"
    )


    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.metric(
            "ML Model Risk Signal",
            f"{result['ml_risk']:.0f}%"
        )


    with summary_col2:

        st.metric(
            "Security Indicator Score",
            f"{result['security_score']:.0f}%"
        )


    st.caption(
        result["note"]
    )


# ============================================================
# MESSAGE SCANNER PAGE
# ============================================================

def message_scanner_page():

    back_to_home()

    st.title(
        "💬 Message Scanner"
    )

    st.write(
        "Enter an SMS or text message to identify "
        "potential spam or harmful content."
    )

    st.divider()


    # ========================================================
    # MESSAGE INPUT
    # ========================================================

    message_input = st.text_area(
        "Enter message:",
        key="message_box",
        height=180,
        placeholder=(
            "Example: Congratulations! "
            "You have won a prize. Click here to claim it."
        )
    )


    # ========================================================
    # BUTTONS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        analyze_message = st.button(
            "🔍 Analyze Message",
            type="primary",
            use_container_width=True
        )


    with col2:

        st.button(
            "🗑️ Clear",
            use_container_width=True,
            on_click=clear_message
        )


    # ========================================================
    # ANALYZE MESSAGE
    # ========================================================

    if analyze_message:

        if not message_input.strip():

            st.warning(
                "⚠️ Please enter a message."
            )

        else:

            try:

                # ------------------------------------------------
                # VECTORIZATION
                # ------------------------------------------------

                message_vector = (
                    message_vectorizer.transform(
                        [message_input]
                    )
                )


                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------

                prediction = (
                    message_model.predict(
                        message_vector
                    )[0]
                )


                # ------------------------------------------------
                # PROBABILITY
                # ------------------------------------------------

                confidence = 0.0

                if hasattr(
                    message_model,
                    "predict_proba"
                ):

                    probabilities = (
                        message_model.predict_proba(
                            message_vector
                        )[0]
                    )

                    classes = list(
                        message_model.classes_
                    )

                    if prediction in classes:

                        prediction_index = (
                            classes.index(
                                prediction
                            )
                        )

                        confidence = float(
                            probabilities[
                                prediction_index
                            ]
                        )


                # ------------------------------------------------
                # CLASSIFICATION
                # ------------------------------------------------

                # Existing project assumption:
                # 1 = Spam / Threat
                # 0 = Safe

                if prediction == 1:

                    classification = "Spam / Threat"

                else:

                    classification = "Safe"


                # ------------------------------------------------
                # SAVE RESULT
                # ------------------------------------------------

                st.session_state.message_result = {

                    "classification":
                        classification,

                    "confidence":
                        confidence,

                    "message":
                        message_input

                }


            except Exception as e:

                st.error(
                    "❌ Error while analyzing the message."
                )

                st.code(
                    str(e)
                )


    # ========================================================
    # DISPLAY MESSAGE RESULT
    # ========================================================

    if st.session_state.message_result is None:

        return


    result = st.session_state.message_result

    display_result = result["classification"]

    confidence = result["confidence"]


    st.divider()

    st.subheader(
        "📊 Message Analysis Result"
    )


    # ========================================================
    # RESULT METRICS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Classification",
            display_result
        )


    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )


    # ========================================================
    # RESULT BANNER
    # ========================================================

    if display_result == "Spam / Threat":

        st.error(
            f"🚨 SUSPICIOUS MESSAGE\n\n"
            f"Classification: {display_result}\n\n"
            f"Confidence: {confidence:.2%}"
        )

    else:

        st.success(
            f"✅ MESSAGE APPEARS SAFE\n\n"
            f"Classification: {display_result}\n\n"
            f"Confidence: {confidence:.2%}"
        )


    st.divider()


    # ========================================================
    # ANALYZED MESSAGE
    # ========================================================

    st.subheader(
        "💬 Analyzed Message"
    )

    st.info(
        result["message"]
    )


# ============================================================
# PAGE ROUTING
# ============================================================

if st.session_state.page == "welcome":

    welcome_page()


elif st.session_state.page == "url":

    url_scanner_page()


elif st.session_state.page == "message":

    message_scanner_page()