import math
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

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef4ff 100%
        );
    }

    /* ================= LOGIN ================= */

    .login-wrapper {
        min-height: 70vh;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 30px 15px;
    }

    .login-card {
        width: 100%;
        max-width: 520px;
        background: white;
        padding: 45px 40px;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 15px 45px rgba(15, 23, 42, 0.12);
        border: 1px solid #e2e8f0;
    }

    .shield {
        width: 85px;
        height: 85px;
        margin: 0 auto 20px auto;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #2563eb,
            #38bdf8
        );
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 48px;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25);
    }

    .login-title {
        font-size: 38px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .login-subtitle {
        color: #64748b;
        font-size: 16px;
        line-height: 1.7;
        margin-top: 10px;
    }

    .login-description {
        background: #f8fafc;
        border-radius: 16px;
        padding: 18px;
        margin-top: 25px;
        color: #475569;
        font-size: 14px;
        line-height: 1.6;
    }

    .login-footer {
        margin-top: 25px;
        color: #94a3b8;
        font-size: 12px;
    }

    /* ================= BUTTONS ================= */

    .stButton > button {
        border-radius: 12px;
        min-height: 46px;
        font-weight: 600;
    }

    /* ================= DASHBOARD ================= */

    .dashboard-header {
        background: white;
        padding: 25px 30px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
        margin-bottom: 20px;
    }

    .dashboard-title {
        font-size: 34px;
        font-weight: 800;
        color: #0f172a;
    }

    .dashboard-subtitle {
        color: #64748b;
        font-size: 15px;
        margin-top: 5px;
    }

    .welcome-box {
        background: linear-gradient(
            135deg,
            #eff6ff,
            #f0f9ff
        );
        border: 1px solid #bfdbfe;
        padding: 16px 20px;
        border-radius: 14px;
        margin-bottom: 20px;
        color: #1e3a8a;
    }

    .scanner-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.05);
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 25px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GOOGLE LOGIN PAGE
# ============================================================

def show_login_page():

    st.markdown(
        """
        <div class="login-wrapper">

            <div class="login-card">

                <div class="shield">
                    🛡️
                </div>

                <div class="login-title">
                    SafeLink AI
                </div>

                <div class="login-subtitle">
                    <strong>
                        AI-Powered Security Scanner
                    </strong>

                    <br><br>

                    Detect suspicious URLs and messages
                    <br>
                    using Machine Learning.
                </div>

                <div class="login-description">

                    🔗 <strong>URL Scanner</strong>
                    <br>
                    Analyze suspicious websites and links.

                    <br><br>

                    💬 <strong>Message Scanner</strong>
                    <br>
                    Detect spam and suspicious messages.

                </div>

                <div class="login-footer">
                    Secure access powered by Google
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        if st.button(
            "🔐  Continue with Google",
            use_container_width=True,
            type="primary"
        ):
            st.login()


# ============================================================
# GOOGLE AUTHENTICATION
# ============================================================

if not st.user.is_logged_in:

    show_login_page()

    st.stop()


# ============================================================
# LOGGED-IN USER INFORMATION
# ============================================================

user_name = getattr(
    st.user,
    "name",
    "User"
)

user_email = getattr(
    st.user,
    "email",
    ""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### 👤 Account")

    st.write(
        f"**{user_name}**"
    )

    if user_email:
        st.caption(user_email)

    st.divider()

    if st.button(
        "Logout",
        use_container_width=True
    ):
        st.logout()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # First check the main project folder
    root_paths = {
        "url_model": BASE_DIR / "url_model_15trees.pkl",
        "url_features": BASE_DIR / "url_features.pkl",
        "message_model": BASE_DIR / "message_model.pkl",
        "message_vectorizer": BASE_DIR / "message_vectorizer.pkl",
    }

    # Also check the models folder
    models_dir = BASE_DIR / "models"

    model_paths = {}

    for name, root_path in root_paths.items():

        if root_path.exists():

            model_paths[name] = root_path

        else:

            model_folder_path = models_dir / root_path.name

            if model_folder_path.exists():

                model_paths[name] = model_folder_path

            else:

                model_paths[name] = None

    missing_files = [
        path.name
        for path in model_paths.values()
        if path is None
    ]

    if missing_files:

        raise FileNotFoundError(
            "Missing model file(s): "
            + ", ".join(missing_files)
        )

    url_model = joblib.load(
        model_paths["url_model"]
    )

    url_features = joblib.load(
        model_paths["url_features"]
    )

    message_model = joblib.load(
        model_paths["message_model"]
    )

    message_vectorizer = joblib.load(
        model_paths["message_vectorizer"]
    )

    return (
        url_model,
        url_features,
        message_model,
        message_vectorizer
    )


try:

    (
        url_model,
        url_features,
        message_model,
        message_vectorizer
    ) = load_models()

except Exception as e:

    st.error(
        "❌ SafeLink AI model files could not be loaded."
    )

    st.code(str(e))

    st.info(
        """
Make sure these four files are available:

• url_model_15trees.pkl
• url_features.pkl
• message_model.pkl
• message_vectorizer.pkl

They can be either:
1. In the same folder as app.py
2. Inside the models folder
"""
    )

    st.stop()


# ============================================================
# URL ENTROPY
# ============================================================

def calculate_entropy(text):

    if not text:
        return 0.0

    counts = Counter(text)

    length = len(text)

    entropy = 0.0

    for count in counts.values():

        probability = count / length

        entropy -= (
            probability *
            math.log2(probability)
        )

    return entropy


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_url_features(url):

    url = url.strip()

    if not url:
        return None

    # Add HTTP if user doesn't provide a scheme
    if not url.lower().startswith(
        ("http://", "https://")
    ):

        url_to_parse = "http://" + url

    else:

        url_to_parse = url

    try:

        parsed = urlparse(
            url_to_parse
        )

        hostname = parsed.hostname or ""

        path = parsed.path or ""

        query = parsed.query or ""

        # ============================
        # BASIC FEATURES
        # ============================

        url_length = len(url)

        domain_length = len(hostname)

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

        # ============================
        # CHARACTER FEATURES
        # ============================

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

        # ============================
        # ENTROPY
        # ============================

        entropy_url = calculate_entropy(
            url
        )

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

        # ============================
        # IP ADDRESS
        # ============================

        try:

            ipaddress.ip_address(
                hostname
            )

            is_ip_address = 1

        except ValueError:

            is_ip_address = 0

        # ============================
        # SUSPICIOUS TLD
        # ============================

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
                hostname.lower().endswith(tld)
                for tld in suspicious_tlds
            )
        )

        # ============================
        # HTTPS
        # ============================

        uses_https = int(
            url.lower().startswith(
                "https://"
            )
        )

        # ============================
        # LOGIN KEYWORDS
        # ============================

        login_words = [
            "login",
            "signin",
            "sign-in",
            "verify",
            "verification"
        ]

        contains_login = int(
            any(
                word in url.lower()
                for word in login_words
            )
        )

        # ============================
        # FINAL FEATURES
        # ============================

        features = {

            "url_length":
                url_length,

            "domain_length":
                domain_length,

            "hostname_length":
                hostname_length,

            "path_length":
                path_length,

            "url_depth":
                url_depth,

            "query_length":
                query_length,

            "path_segments_count":
                path_segments_count,

            "num_digits":
                num_digits,

            "num_letters":
                num_letters,

            "num_special_chars":
                num_special_chars,

            "num_dots":
                num_dots,

            "num_hyphens":
                num_hyphens,

            "num_at":
                num_at,

            "num_percent":
                num_percent,

            "num_equals":
                num_equals,

            "num_question":
                num_question,

            "num_ampersand":
                num_ampersand,

            "num_slash":
                num_slash,

            "entropy_url":
                entropy_url,

            "ratio_digits":
                ratio_digits,

            "ratio_letters":
                ratio_letters,

            "is_ip_address":
                is_ip_address,

            "is_suspicious_tld":
                is_suspicious_tld,

            "uses_https":
                uses_https,

            "contains_login":
                contains_login
        }

        return pd.DataFrame(
            [features]
        )

    except Exception:

        return None


# ============================================================
# PREDICTION HELPER
# ============================================================

def is_dangerous_prediction(
    prediction
):

    value = str(
        prediction
    ).strip().lower()

    return value in [
        "1",
        "true",
        "spam",
        "phishing",
        "malicious",
        "dangerous",
        "suspicious"
    ]


# ============================================================
# DASHBOARD HEADER
# ============================================================

header_col1, header_col2 = st.columns(
    [5, 1]
)


with header_col1:

    st.markdown(
        """
        <div class="dashboard-header">

            <div class="dashboard-title">
                🛡️ SafeLink AI
            </div>

            <div class="dashboard-subtitle">
                AI-Powered Security Scanner
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with header_col2:

    st.write("")

    if st.button(
        "Logout",
        use_container_width=True
    ):
        st.logout()


# ============================================================
# WELCOME MESSAGE
# ============================================================

st.markdown(
    f"""
    <div class="welcome-box">

        👋 <strong>Welcome, {user_name}</strong>

        <br>

        <span style="font-size:14px;">
            Detect suspicious URLs and messages
            using machine learning.
        </span>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

url_tab, message_tab = st.tabs(
    [
        "🔗 URL Scanner",
        "💬 Message Scanner"
    ]
)


# ============================================================
# URL SCANNER
# ============================================================

with url_tab:

    st.markdown(
        """
        <div class="scanner-card">

            <h2>
                🔗 URL Threat Detection
            </h2>

            <p style="color:#64748b;">
                Enter a website URL to check whether
                it looks safe or suspicious.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    url = st.text_input(
        "Enter URL",
        key="url_input",
        placeholder="https://example.com"
    )

    col1, col2 = st.columns(2)

    with col1:

        analyze_url = st.button(
            "🔍 Analyze URL",
            key="analyze_url_button",
            use_container_width=True,
            type="primary"
        )

    with col2:

        clear_url = st.button(
            "🗑️ Clear",
            key="clear_url_button",
            use_container_width=True
        )

    if clear_url:

        st.session_state.pop(
            "url_input",
            None
        )

        st.rerun()

    if analyze_url:

        if not url.strip():

            st.warning(
                "⚠️ Please enter a URL first."
            )

        else:

            features_df = extract_url_features(
                url
            )

            if features_df is None:

                st.error(
                    "❌ Unable to analyze this URL."
                )

            else:

                try:

                    # =========================
                    # FEATURE ORDER
                    # =========================

                    feature_order = None

                    if isinstance(
                        url_features,
                        pd.DataFrame
                    ):

                        feature_order = list(
                            url_features.columns
                        )

                    elif isinstance(
                        url_features,
                        (list, tuple)
                    ):

                        feature_order = list(
                            url_features
                        )

                    # =========================
                    # MATCH MODEL FEATURES
                    # =========================

                    if feature_order:

                        missing_features = [
                            feature
                            for feature in feature_order
                            if feature not in features_df.columns
                        ]

                        if not missing_features:

                            features_df = (
                                features_df[
                                    feature_order
                                ]
                            )

                    # =========================
                    # PREDICTION
                    # =========================

                    prediction = (
                        url_model.predict(
                            features_df
                        )[0]
                    )

                    probability = None

                    # =========================
                    # PROBABILITY
                    # =========================

                    if hasattr(
                        url_model,
                        "predict_proba"
                    ):

                        probabilities = (
                            url_model.predict_proba(
                                features_df
                            )[0]
                        )

                        classes = list(
                            url_model.classes_
                        )

                        if prediction in classes:

                            prediction_index = (
                                classes.index(
                                    prediction
                                )
                            )

                            probability = (
                                probabilities[
                                    prediction_index
                                ] * 100
                            )

                    dangerous = (
                        is_dangerous_prediction(
                            prediction
                        )
                    )

                    # =========================
                    # RESULT
                    # =========================

                    st.divider()

                    st.subheader(
                        "📊 Analysis Result"
                    )

                    if dangerous:

                        if probability is not None:

                            st.error(
                                f"""
                                🚨 DANGEROUS

                                Model Confidence:
                                {probability:.2f}%
                                """
                            )

                        else:

                            st.error(
                                "🚨 DANGEROUS"
                            )

                    else:

                        if probability is not None:

                            st.success(
                                f"""
                                ✅ SAFE

                                Model Confidence:
                                {probability:.2f}%
                                """
                            )

                        else:

                            st.success(
                                "✅ SAFE"
                            )

                    # =========================
                    # URL DETAILS
                    # =========================

                    st.write(
                        "### 🔎 URL Details"
                    )

                    col1, col2, col3 = (
                        st.columns(3)
                    )

                    with col1:

                        st.metric(
                            "URL Length",
                            int(
                                features_df[
                                    "url_length"
                                ].iloc[0]
                            )
                        )

                    with col2:

                        https_value = (
                            features_df[
                                "uses_https"
                            ].iloc[0]
                        )

                        st.metric(
                            "HTTPS",
                            "Yes"
                            if https_value
                            else "No"
                        )

                    with col3:

                        ip_value = (
                            features_df[
                                "is_ip_address"
                            ].iloc[0]
                        )

                        st.metric(
                            "IP Address",
                            "Yes"
                            if ip_value
                            else "No"
                        )

                except Exception as e:

                    st.error(
                        "❌ Error while predicting URL."
                    )

                    with st.expander(
                        "Technical details"
                    ):

                        st.code(
                            str(e)
                        )


# ============================================================
# MESSAGE SCANNER
# ============================================================

with message_tab:

    st.markdown(
        """
        <div class="scanner-card">

            <h2>
                💬 Message Threat Detection
            </h2>

            <p style="color:#64748b;">
                Paste an SMS, email message or text
                to check for spam or suspicious content.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    message = st.text_area(
        "Enter message",
        key="message_input",
        height=180,
        placeholder="Paste your SMS or message here..."
    )

    col1, col2 = st.columns(2)

    with col1:

        analyze_message = st.button(
            "🔍 Analyze Message",
            key="analyze_message_button",
            use_container_width=True,
            type="primary"
        )

    with col2:

        clear_message = st.button(
            "🗑️ Clear",
            key="clear_message_button",
            use_container_width=True
        )

    if clear_message:

        st.session_state.pop(
            "message_input",
            None
        )

        st.rerun()

    if analyze_message:

        if not message.strip():

            st.warning(
                "⚠️ Please enter a message first."
            )

        else:

            try:

                # =========================
                # VECTORIZE
                # =========================

                message_vector = (
                    message_vectorizer.transform(
                        [message]
                    )
                )

                # =========================
                # PREDICTION
                # =========================

                prediction = (
                    message_model.predict(
                        message_vector
                    )[0]
                )

                probability = None

                # =========================
                # PROBABILITY
                # =========================

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

                        probability = (
                            probabilities[
                                prediction_index
                            ] * 100
                        )

                dangerous = (
                    is_dangerous_prediction(
                        prediction
                    )
                )

                # =========================
                # RESULT
                # =========================

                st.divider()

                st.subheader(
                    "📊 Analysis Result"
                )

                if dangerous:

                    st.error(
                        "🚨 SPAM / SUSPICIOUS MESSAGE"
                    )

                    if probability is not None:

                        st.metric(
                            "Model Confidence",
                            f"{probability:.2f}%"
                        )

                else:

                    st.success(
                        "✅ SAFE MESSAGE"
                    )

                    if probability is not None:

                        st.metric(
                            "Model Confidence",
                            f"{probability:.2f}%"
                        )

                # =========================
                # MESSAGE
                # =========================

                st.write(
                    "### 📝 Message Analyzed"
                )

                st.info(
                    message
                )

            except Exception as e:

                st.error(
                    "❌ Error while analyzing the message."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(e)
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">

        🛡️ <strong>SafeLink AI</strong>

        <br>

        Machine Learning based
        URL & Message Security Analysis

    </div>
    """,
    unsafe_allow_html=True
)