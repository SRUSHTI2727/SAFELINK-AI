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

    /* ---------- MAIN APP ---------- */

    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef4ff 100%);
    }

    /* ---------- LOGIN PAGE ---------- */

    .login-wrapper {
        min-height: 78vh;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 30px 15px;
    }

    .login-card {
        width: 100%;
        max-width: 520px;
        background: rgba(255, 255, 255, 0.97);
        padding: 45px 40px;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 15px 45px rgba(15, 23, 42, 0.12);
        border: 1px solid #e2e8f0;
    }

    .shield {
        width: 82px;
        height: 82px;
        margin: 0 auto 20px auto;
        border-radius: 22px;
        background: linear-gradient(135deg, #2563eb, #38bdf8);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 46px;
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
        margin-bottom: 25px;
    }

    .login-description {
        background: #f8fafc;
        border-radius: 16px;
        padding: 18px;
        margin-top: 20px;
        color: #475569;
        font-size: 14px;
        line-height: 1.6;
    }

    .login-footer {
        margin-top: 25px;
        color: #94a3b8;
        font-size: 12px;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 12px;
        min-height: 46px;
        font-weight: 600;
        border: 1px solid #dbe3ef;
    }

    /* ---------- DASHBOARD ---------- */

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
        margin-top: 4px;
    }

    .welcome-box {
        background: linear-gradient(135deg, #eff6ff, #f0f9ff);
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
# AUTHENTICATION CONFIGURATION CHECK
# ============================================================

def auth_is_configured():
    """
    Check whether Streamlit Google authentication
    has been configured correctly.
    """

    try:
        auth_config = st.secrets.get("auth")

        if not auth_config:
            return False

        # Default provider configuration
        required_keys = [
            "redirect_uri",
            "cookie_secret",
            "client_id",
            "client_secret",
            "server_metadata_url"
        ]

        if all(key in auth_config for key in required_keys):
            return True

        # Named Google provider configuration
        if "google" in auth_config:
            google_config = auth_config["google"]

            if all(
                key in google_config
                for key in [
                    "client_id",
                    "client_secret",
                    "server_metadata_url"
                ]
            ):
                return True

        return False

    except Exception:
        return False


# ============================================================
# LOGIN PAGE
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
                    <strong>AI-Powered Security Scanner</strong>
                    <br>
                    Detect suspicious URLs and messages
                    <br>
                    using Machine Learning.
                </div>

                <div class="login-description">
                    🔗 <strong>URL Scanner</strong><br>
                    Analyze suspicious websites and links.
                    <br><br>
                    💬 <strong>Message Scanner</strong><br>
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

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button(
            "🔐  Continue with Google",
            use_container_width=True,
            type="primary"
        ):
            try:

                # Use the default [auth] configuration.
                st.login()

            except Exception as e:

                st.error("Google login could not be started.")

                st.info(
                    "Your Streamlit Google authentication settings "
                    "need to be configured correctly."
                )

                with st.expander("Technical information"):
                    st.code(str(e))


# ============================================================
# CHECK GOOGLE AUTHENTICATION
# ============================================================

if not auth_is_configured():

    st.markdown(
        """
        <div style="
            max-width:750px;
            margin:80px auto;
            background:white;
            padding:40px;
            border-radius:20px;
            border:1px solid #e2e8f0;
            box-shadow:0 10px 30px rgba(0,0,0,0.08);
            text-align:center;
        ">

        <div style="font-size:55px;">⚙️</div>

        <h1 style="color:#0f172a;">
            Google Login Configuration Required
        </h1>

        <p style="color:#64748b;font-size:16px;line-height:1.7;">
            SafeLink AI is running, but Google authentication
            has not been configured correctly.
        </p>

        <p style="color:#475569;">
            Add the <b>[auth]</b> configuration to Streamlit Cloud
            Secrets and restart the application.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# LOGIN CHECK
# ============================================================

try:
    logged_in = st.user.is_logged_in
except Exception:
    logged_in = False


if not logged_in:
    show_login_page()
    st.stop()


# ============================================================
# LOAD ML MODELS
# ============================================================

@st.cache_resource
def load_models():

    url_model_path = BASE_DIR / "url_model_15trees.pkl"
    url_features_path = BASE_DIR / "url_features.pkl"
    message_model_path = BASE_DIR / "message_model.pkl"
    message_vectorizer_path = BASE_DIR / "message_vectorizer.pkl"

    required_files = [
        url_model_path,
        url_features_path,
        message_model_path,
        message_vectorizer_path
    ]

    missing_files = [
        file.name
        for file in required_files
        if not file.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model file(s): "
            + ", ".join(missing_files)
        )

    url_model = joblib.load(url_model_path)
    url_features = joblib.load(url_features_path)

    message_model = joblib.load(message_model_path)
    message_vectorizer = joblib.load(message_vectorizer_path)

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

    st.error("❌ SafeLink AI model files could not be loaded.")

    st.code(str(e))

    st.info(
        "Make sure these four files are in the same folder "
        "as app.py:\n\n"
        "• url_model_15trees.pkl\n"
        "• url_features.pkl\n"
        "• message_model.pkl\n"
        "• message_vectorizer.pkl"
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

        entropy -= probability * math.log2(probability)

    return entropy


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_url_features(url):

    url = url.strip()

    if not url:
        return None

    # Add scheme if user didn't provide one
    if not url.lower().startswith(("http://", "https://")):
        url_to_parse = "http://" + url
    else:
        url_to_parse = url

    try:

        parsed = urlparse(url_to_parse)

        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""

        url_length = len(url)
        domain_length = len(hostname)
        hostname_length = len(hostname)
        path_length = len(path)

        url_depth = len(
            [x for x in path.split("/") if x]
        )

        query_length = len(query)

        path_segments_count = len(
            [x for x in path.split("/") if x]
        )

        num_digits = sum(
            c.isdigit() for c in url
        )

        num_letters = sum(
            c.isalpha() for c in url
        )

        num_special_chars = sum(
            not c.isalnum() for c in url
        )

        num_dots = url.count(".")
        num_hyphens = url.count("-")
        num_at = url.count("@")
        num_percent = url.count("%")
        num_equals = url.count("=")
        num_question = url.count("?")
        num_ampersand = url.count("&")
        num_slash = url.count("/")

        entropy_url = calculate_entropy(url)

        ratio_digits = (
            num_digits / url_length
            if url_length else 0
        )

        ratio_letters = (
            num_letters / url_length
            if url_length else 0
        )

        # IP address detection
        try:
            ipaddress.ip_address(hostname)
            is_ip_address = 1
        except ValueError:
            is_ip_address = 0

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

        uses_https = int(
            url.lower().startswith("https://")
        )

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

        features = {
            "url_length": url_length,
            "domain_length": domain_length,
            "hostname_length": hostname_length,
            "path_length": path_length,
            "url_depth": url_depth,
            "query_length": query_length,
            "path_segments_count": path_segments_count,
            "num_digits": num_digits,
            "num_letters": num_letters,
            "num_special_chars": num_special_chars,
            "num_dots": num_dots,
            "num_hyphens": num_hyphens,
            "num_at": num_at,
            "num_percent": num_percent,
            "num_equals": num_equals,
            "num_question": num_question,
            "num_ampersand": num_ampersand,
            "num_slash": num_slash,
            "entropy_url": entropy_url,
            "ratio_digits": ratio_digits,
            "ratio_letters": ratio_letters,
            "is_ip_address": is_ip_address,
            "is_suspicious_tld": is_suspicious_tld,
            "uses_https": uses_https,
            "contains_login": contains_login
        }

        return pd.DataFrame([features])

    except Exception:

        return None


# ============================================================
# PREDICTION LABEL HELPER
# ============================================================

def is_dangerous_prediction(prediction):

    value = str(prediction).strip().lower()

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

header_col1, header_col2 = st.columns([5, 1])

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
# USER INFORMATION
# ============================================================

user_info = st.user

user_name = getattr(
    user_info,
    "name",
    None
)

user_email = getattr(
    user_info,
    "email",
    None
)

if not user_name:

    user_name = (
        user_email
        if user_email
        else "User"
    )


st.markdown(
    f"""
    <div class="welcome-box">
        👋 <strong>Welcome, {user_name}</strong>
        <br>
        <span style="font-size:14px;">
        Detect suspicious URLs and messages using machine learning.
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
            <h2>🔗 URL Threat Detection</h2>
            <p style="color:#64748b;">
                Enter a website URL to check whether it looks
                safe or suspicious.
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

        st.session_state.pop(
            "url_result",
            None
        )

        st.rerun()

    if analyze_url:

        if not url.strip():

            st.warning(
                "⚠️ Please enter a URL first."
            )

        else:

            features_df = extract_url_features(url)

            if features_df is None:

                st.error(
                    "❌ Unable to analyze this URL."
                )

            else:

                try:

                    # Match model feature order
                    if url_features is not None:

                        try:

                            feature_order = list(
                                url_features
                            )

                            if all(
                                feature in features_df.columns
                                for feature in feature_order
                            ):

                                features_df = (
                                    features_df[
                                        feature_order
                                    ]
                                )

                        except Exception:
                            pass

                    prediction = url_model.predict(
                        features_df
                    )[0]

                    probability = None

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

                            index = classes.index(
                                prediction
                            )

                            probability = (
                                probabilities[index]
                                * 100
                            )

                    dangerous = (
                        is_dangerous_prediction(
                            prediction
                        )
                    )

                    st.session_state[
                        "url_result"
                    ] = {
                        "prediction": prediction,
                        "probability": probability
                    }

                    st.divider()

                    st.subheader(
                        "📊 Analysis Result"
                    )

                    if dangerous:

                        if probability is not None:

                            st.error(
                                f"🚨 DANGEROUS\n\n"
                                f"Model Confidence: "
                                f"{probability:.2f}%"
                            )

                        else:

                            st.error(
                                "🚨 DANGEROUS"
                            )

                    else:

                        if probability is not None:

                            st.success(
                                f"✅ SAFE\n\n"
                                f"Model Confidence: "
                                f"{probability:.2f}%"
                            )

                        else:

                            st.success(
                                "✅ SAFE"
                            )

                    st.write("### 🔎 URL Details")

                    col1, col2, col3 = st.columns(3)

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

                        st.code(str(e))


# ============================================================
# MESSAGE SCANNER
# ============================================================

with message_tab:

    st.markdown(
        """
        <div class="scanner-card">
            <h2>💬 Message Threat Detection</h2>
            <p style="color:#64748b;">
                Paste an SMS, email message or text to check
                for spam or suspicious content.
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
        placeholder=(
            "Paste your SMS or message here..."
        )
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

        st.session_state.pop(
            "message_result",
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

                message_vector = (
                    message_vectorizer.transform(
                        [message]
                    )
                )

                prediction = (
                    message_model.predict(
                        message_vector
                    )[0]
                )

                probability = None

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

                        index = classes.index(
                            prediction
                        )

                        probability = (
                            probabilities[index]
                            * 100
                        )

                dangerous = (
                    is_dangerous_prediction(
                        prediction
                    )
                )

                st.session_state[
                    "message_result"
                ] = {
                    "prediction": prediction,
                    "probability": probability,
                    "message": message
                }

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

                st.write(
                    "### 📝 Message Analyzed"
                )

                st.info(message)

            except Exception as e:

                st.error(
                    "❌ Error while analyzing the message."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ SafeLink AI
        <br>
        Machine Learning based URL & Message Security Analysis
    </div>
    """,
    unsafe_allow_html=True
)