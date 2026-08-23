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
st.set_page_config(page_title="SafeLink AI", page_icon="🛡️", layout="wide")

# ============================================================
# PROJECT PATH
# ============================================================
BASE_DIR = Path(__file__).resolve().parent 

# ============================================================
# GOOGLE LOGIN PAGE
# ============================================================
def show_login_page():
    # Outer card layout using columns
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        with st.container(border=True):
            st.markdown(
                "<h1 style='text-align: center; font-size: 60px; margin-bottom: 0;'>🛡️</h1>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h2 style='text-align: center; color: #1e293b; margin-top: 0;'>SafeLink AI</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='text-align: center; color: #64748b; font-size: 16px;'>"
                "<b>AI-Powered Security Scanner</b><br><br>"
                "Detect suspicious URLs and messages<br>using machine learning."
                "</p>",
                unsafe_allow_html=True,
            )
            st.write("")

            if not hasattr(st, "login"):
                st.error("Google Login is not available in your current Streamlit version.")
                st.info("Run: python -m pip install --upgrade streamlit authlib")
                st.stop()

            if st.button(
                "🔐 Continue with Google",
                use_container_width=True,
                key="google_login_button",
            ):
                st.login()

# ============================================================
# LOGIN CHECK
# ============================================================
user_info = getattr(st, "user", None)
is_logged_in = bool(
    getattr(user_info, "is_logged_in", False)
)

if not is_logged_in:
    show_login_page()
    st.stop()

# ============================================================
# LOAD MODELS
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
            "Missing model file(s): " + ", ".join(missing_files)
        )

    url_model = joblib.load(url_model_path)
    url_features = joblib.load(url_features_path)
    message_model = joblib.load(message_model_path)
    message_vectorizer = joblib.load(message_vectorizer_path)

    return url_model, url_features, message_model, message_vectorizer

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
    st.info("Make sure the four .pkl files are in the same folder as app.py.")
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

    # Prepend http scheme if omitted so urlparse parses hostname properly
    if not url.startswith(("http://", "https://")):
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

        url_depth = len([x for x in path.split("/") if x])
        query_length = len(query)
        path_segments_count = len([x for x in path.split("/") if x])

        num_digits = sum(c.isdigit() for c in url)
        num_letters = sum(c.isalpha() for c in url)
        num_special_chars = sum(not c.isalnum() for c in url)
        num_dots = url.count(".")
        num_hyphens = url.count("-")
        num_at = url.count("@")
        num_percent = url.count("%")
        num_equals = url.count("=")
        num_question = url.count("?")
        num_ampersand = url.count("&")
        num_slash = url.count("/")

        entropy_url = calculate_entropy(url)

        ratio_digits = num_digits / url_length if url_length else 0
        ratio_letters = num_letters / url_length if url_length else 0

        try:
            ipaddress.ip_address(hostname)
            is_ip_address = 1
        except ValueError:
            is_ip_address = 0

        suspicious_tlds = [
            ".tk", ".ml", ".ga", ".cf", ".gq",
            ".top", ".xyz", ".click", ".link",
            ".work", ".zip", ".review"
        ]

        is_suspicious_tld = int(
            any(hostname.lower().endswith(tld) for tld in suspicious_tlds)
        )

        uses_https = int(url.lower().startswith("https://"))

        login_words = ["login", "signin", "sign-in", "verify", "verification"]
        contains_login = int(
            any(word in url.lower() for word in login_words)
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
# HEADER
# ============================================================
header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    st.markdown("# 🛡️ SafeLink AI")
    st.caption("AI-Powered Security Scanner")

with header_col2:
    if st.button("Logout", use_container_width=True):
        st.logout()

# ============================================================
# USER NAME
# ============================================================
user_info = getattr(st, "user", None)
user_name = getattr(user_info, "name", None)

if not user_name:
    user_name = getattr(user_info, "email", "User")

st.success(f"Welcome, {user_name} 👋")
st.write("Detect suspicious URLs and messages using machine learning.")
st.divider()

# ============================================================
# TABS
# ============================================================
url_tab, message_tab = st.tabs(["🔗 URL Scanner", "💬 Message Scanner"])

# ============================================================
# URL SCANNER
# ============================================================
with url_tab:
    st.subheader("🔗 URL Threat Detection")
    st.write("Enter a website URL to check whether it looks safe or suspicious.")

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
            use_container_width=True
        )

    with col2:
        clear_url = st.button(
            "🗑️ Clear",
            key="clear_url_button",
            use_container_width=True
        )

    if clear_url:
        st.session_state.pop("url_input", None)
        st.session_state.pop("url_result", None)
        st.rerun()

    if analyze_url:
        if not url.strip():
            st.warning("Please enter a URL first.")
        else:
            features_df = extract_url_features(url)

            if features_df is None:
                st.error("Unable to analyze this URL.")
            else:
                try:
                    if hasattr(url_features, "__iter__"):
                        feature_order = list(url_features)
                        if all(f in features_df.columns for f in feature_order):
                            features_df = features_df[feature_order]

                    prediction = url_model.predict(features_df)[0]
                    probability = None

                    if hasattr(url_model, "predict_proba"):
                        probabilities = url_model.predict_proba(features_df)[0]
                        classes = list(url_model.classes_)
                        if prediction in classes:
                            prediction_index = classes.index(prediction)
                            probability = probabilities[prediction_index] * 100

                    st.session_state["url_result"] = {
                        "prediction": prediction,
                        "probability": probability
                    }

                    st.divider()
                    st.subheader("📊 Analysis Result")

                    if prediction == 1:
                        if probability is not None:
                            st.error(
                                f"🚨 DANGEROUS\n\nModel Confidence: {probability:.2f}%"
                            )
                        else:
                            st.error("🚨 DANGEROUS")
                    else:
                        if probability is not None:
                            st.success(
                                f"✅ SAFE\n\nModel Confidence: {probability:.2f}%"
                            )
                        else:
                            st.success("✅ SAFE")

                    st.write("### URL Details")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("URL Length", int(features_df["url_length"].iloc[0]))
                    with col2:
                        https_value = features_df["uses_https"].iloc[0]
                        st.metric("HTTPS", "Yes" if https_value else "No")
                    with col3:
                        ip_value = features_df["is_ip_address"].iloc[0]
                        st.metric("IP Address", "Yes" if ip_value else "No")

                except Exception as e:
                    st.error("Error while predicting URL.")
                    st.code(str(e))

# ============================================================
# MESSAGE SCANNER
# ============================================================
with message_tab:
    st.subheader("💬 Message Threat Detection")
    st.write("Paste an SMS, email message or text to check for spam or suspicious content.")

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
            use_container_width=True
        )

    with col2:
        clear_message = st.button(
            "🗑️ Clear",
            key="clear_message_button",
            use_container_width=True
        )

    if clear_message:
        st.session_state.pop("message_input", None)
        st.session_state.pop("message_result", None)
        st.rerun()

    if analyze_message:
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            try:
                message_vector = message_vectorizer.transform([message])
                prediction = message_model.predict(message_vector)[0]
                probability = None

                if hasattr(message_model, "predict_proba"):
                    probabilities = message_model.predict_proba(message_vector)[0]
                    classes = list(message_model.classes_)
                    if prediction in classes:
                        prediction_index = classes.index(prediction)
                        probability = probabilities[prediction_index] * 100

                st.session_state["message_result"] = {
                    "prediction": prediction,
                    "probability": probability,
                    "message": message
                }

                st.divider()
                st.subheader("📊 Analysis Result")

                if prediction == 1:
                    st.error("🚨 SPAM / SUSPICIOUS MESSAGE")
                    if probability is not None:
                        st.metric("Model Confidence", f"{probability:.2f}%")
                else:
                    st.success("✅ SAFE MESSAGE")
                    if probability is not None:
                        st.metric("Model Confidence", f"{probability:.2f}%")

                st.write("### 📝 Message Analyzed")
                st.info(message)

            except Exception as e:
                st.error("Error while analyzing the message.")
                st.code(str(e))

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("SafeLink AI | Machine Learning based URL & Message Security Analysis")