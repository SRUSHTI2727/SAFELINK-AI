import math
import re
import ipaddress
import io
import gzip
import base64

from urllib.parse import urlparse
from collections import Counter

import streamlit as st
import pandas as pd
import joblib
# ============================================================ 
# PAGE CONFIGURATION 
# ============================================================ 
st.set_page_config( 
    page_title="SafeLink AI ", 
    page_icon="🛡️", 
    layout="wide" 
) 
 
# Global Safe Whitelist 
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
 
def is_whitelisted(url: str) -> bool: 
    """Check if the URL domain matches or ends with any whitelisted domain.""" 
    try: 
        domain = urlparse(url).netloc.lower().replace("www.", "") 
        for safe_domain in SAFE_WHITELIST: 
            if domain == safe_domain or domain.endswith("." + safe_domain): 
                return True 
    except Exception: 
        pass 
    return False 
 
# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource
def load_models():

    # Load SMS model
    msg_model = joblib.load("message_model.pkl")
    msg_vectorizer = joblib.load("message_vectorizer.pkl")

    # Load URL model from embedded TXT
    with open("embedded_url_model.txt", "r", encoding="utf-8") as f:
        encoded = f.read().strip()

    compressed = base64.b64decode(encoded)
    model_bytes = gzip.decompress(compressed)

    url_model = joblib.load(
        io.BytesIO(model_bytes)
        
    )

    # Load URL feature names
    url_features = joblib.load("url_features.pkl")

    return (
        msg_model,
        msg_vectorizer,
        url_model,
        url_features
    )
 
# ============================================================ 
# FEATURE EXTRACTION 
# ============================================================ 
def extract_url_features(url: str) -> pd.DataFrame: 
    """Extract 25 hand-crafted features matching the URL training pipeline.""" 
    parsed = urlparse(url) 
 
    hostname = parsed.hostname or "" 
    path = parsed.path or "" 
    query = parsed.query or "" 
 
    domain = hostname.replace("www.", "") 
 
    url_length = len(url) 
    domain_length = len(domain) 
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
 
    # Shannon Entropy 
    counts = Counter(url) 
    total = len(url) 
    entropy_url = -sum((c / total) * math.log2(c / total) for c in counts.values()) if total > 0 else 0 
 
    ratio_digits = num_digits / url_length if url_length else 0 
    ratio_letters = num_letters / url_length if url_length else 0 
 
    try: 
        ipaddress.ip_address(hostname) 
        is_ip_address = 1 
    except ValueError: 
        is_ip_address = 0 
 
    suspicious_tlds = [".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".xyz", ".click", ".link", ".work", ".zip", ".review"] 
    is_suspicious_tld = int(any(domain.lower().endswith(tld) for tld in suspicious_tlds)) 
 
    uses_https = int(parsed.scheme.lower() == "https") 
    contains_login = int(any(w in url.lower() for w in ["login", "signin", "sign-in", "verify", "verification"])) 
 
    features = [ 
        url_length, domain_length, hostname_length, path_length, url_depth, 
        query_length, path_segments_count, num_digits, num_letters, num_special_chars, 
        num_dots, num_hyphens, num_at, num_percent, num_equals, 
        num_question, num_ampersand, num_slash, entropy_url, ratio_digits, 
        ratio_letters, is_ip_address, is_suspicious_tld, uses_https, contains_login 
    ] 
 
    return pd.DataFrame([features], columns=url_feature_names) 
 
def extract_urls_from_text(text: str) -> list: 
    """Find all http/https URLs and clean trailing quotes/punctuation.""" 
    url_pattern = r'https?://[^\s"\'<>]+' 
    raw_urls = re.findall(url_pattern, text) 
    cleaned_urls = [u.rstrip('".\',;') for u in raw_urls] 
    return cleaned_urls 
 
# ============================================================ 
# STREAMLIT UI LAYOUT 
# ============================================================ 
st.title("🛡️ SAFELINK AI — Sms Spam Detector & Risk Analyzer") 
st.markdown("Analyze incoming SMS text and embedded URLs for potential **Spam**, **Phishing**, and **Security Threats**.") 
 
st.divider() 
 
user_input = st.text_area("Paste SMS / Message / URL Content:", height=140, placeholder="Example: Claim your $1000 prize now at http://free-prize.xyz/claim")

if st.button("Analyze Risk", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter a message to analyze.")
    else:
        # 1. Evaluate SMS Text Model
        msg_tfidf = message_vectorizer.transform([user_input])
        msg_prob = message_model.predict_proba(msg_tfidf)[0][1]  # Probability of Threat
        msg_pred = message_model.predict(msg_tfidf)[0]

        # 2. Extract and Evaluate URLs
        extracted_urls = extract_urls_from_text(user_input)
        url_results = []
        max_url_prob = 0.0

        for url in extracted_urls:
            # Check Whitelist First
            if is_whitelisted(url):
                u_threat_prob = 0.0  # 0% Risk
                status_note = "Verified Whitelisted Domain"
            else:
                url_feats = extract_url_features(url)
                
                # FIX: PhiUSIIL Index 1 = Safe Probability
                # Threat Probability = 1.0 - Safe Probability
                safe_prob = float(url_model.predict_proba(url_feats)[0][1])
                u_threat_prob = 1.0 - safe_prob
                status_note = "ML Analyzed"

            url_results.append({"url": url, "prob": u_threat_prob, "note": status_note})
            if u_threat_prob > max_url_prob:
                max_url_prob = u_threat_prob

        # 3. Calculate Overall Risk Score
        if extracted_urls:
            combined_risk_prob = max(msg_prob, max_url_prob)
        else:
            combined_risk_prob = msg_prob

        risk_score_pct = int(combined_risk_prob * 100)

        # 4. Display Overview Results
        st.subheader("📊 Analysis Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("SMS Text Threat Probability", f"{int(msg_prob * 100)}%")
        with col2:
            st.metric("Detected URLs", len(extracted_urls))
        with col3:
            st.metric("Overall Risk Score", f"{risk_score_pct}%")

        # Risk Banner
        if risk_score_pct >= 70:
            st.error(f"🚨 **HIGH RISK THREAT DETECTED** (Risk Score: {risk_score_pct}%)")
        elif risk_score_pct >= 40:
            st.warning(f"⚠️ **MODERATE RISK DETECTED** (Risk Score: {risk_score_pct}%)")
        else:
            st.success(f"✅ **SAFE MESSAGE** (Risk Score: {risk_score_pct}%)")

        st.divider()

        # Detailed Breakdown
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("### 💬 Text Analysis")
            st.write(f"**Classification:** {'🚨 Spam/Threat' if msg_pred == 1 else '✅ Safe'}")
            st.write(f"**Confidence:** {msg_prob if msg_pred == 1 else (1 - msg_prob):.2%}")

        with right_col:
            st.markdown("### 🔗 URL Analysis")
            if not extracted_urls:
                st.info("No URLs found in the provided text.")
            else:
                for res in url_results:
                    status = "🚨 THREAT" if res["prob"] > 0.5 else "✅ SAFE"
                    st.write(f"- **URL:** `{res['url']}`")
                    st.write(f"  - **Status:** {status} ({res['prob']:.2%} risk) — *{res['note']}*")   