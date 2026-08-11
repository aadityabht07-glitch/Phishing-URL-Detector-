"""
app.py
-------
The Streamlit UI — this is the file you run: `streamlit run app.py`

WHY THIS FILE IS THIN:
app.py only handles display logic (layout, text, colors, formatting).
It calls predict_url() from predict.py and renders whatever comes back.
This separation means a bug in the UI can't corrupt a prediction, and
a bug in the model pipeline shows up the same way whether it's called
from Streamlit, a test, or a future CLI tool.

This app performs static, string-only analysis of a submitted URL.
It never visits, fetches, or executes the URL — only the text itself
is analyzed (Section 22 of the project spec).
"""

import streamlit as st
from predict import predict_url

st.set_page_config(
    page_title="URL Phishing Detector",
    page_icon="🛡️",
    layout="centered",
)

# ---------- Header ----------
st.title("🛡️ URL Phishing Detector")
st.caption("Machine Learning–Based Phishing URL Analysis · TechSense Class Project")

st.write(
    "Enter a URL below. This tool analyzes the **text of the URL only** — "
    "it never visits, loads, or executes the link."
)

# ---------- Input ----------
url_input = st.text_input(
    "Enter URL",
    placeholder="e.g. https://www.example.com/login",
    label_visibility="collapsed",
)

analyze_clicked = st.button("Analyze URL", type="primary", use_container_width=True)

st.divider()

# ---------- Prediction ----------
if analyze_clicked:
    if not url_input or not url_input.strip():
        st.warning("Please enter a URL before analyzing.")
    else:
        try:
            with st.spinner("Analyzing..."):
                result = predict_url(url_input)
        except ValueError as e:
            st.error(f"Couldn't analyze that input: {e}")
        except FileNotFoundError as e:
            st.error(f"Model file missing: {e}")
        except Exception as e:
            st.error(f"Something went wrong while analyzing this URL: {e}")
        else:
            label = result["label"]
            confidence = result["confidence"]
            feats = result["features"]

            # ---- Verdict ----
            if label == "Legitimate":
                st.success("🟢 **LEGITIMATE**")
            else:
                st.error("🔴 **PHISHING**")

            st.metric("Confidence", f"{confidence}%")

            st.divider()

            # ---- Feature breakdown ----
            st.subheader("URL Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**URL Length:** {feats['url_length']}")
                st.write(f"**HTTPS:** {'Yes' if feats['https_flag'] else 'No'}")
                st.write(f"**IP Address as Host:** {'Yes' if feats['has_ip_address'] else 'No'}")
                st.write(f"**Subdomains:** {feats['subdomain_count']}")
            with col2:
                st.write(f"**Hyphen in Domain:** {'Yes' if feats['has_hyphen_in_domain'] else 'No'}")
                st.write(f"**Digits in URL:** {feats['number_of_digits']}")
                st.write(f"**Query Parameters:** {feats['query_param_count']}")
                st.write(f"**Suspicious File Extension:** {'Yes' if feats['suspicious_file_extension'] else 'No'}")

            st.divider()

            # ---- Explanation ----
            st.subheader("Why?")
            st.write(
                "These are the features the model relies on most heavily "
                "**overall**, based on training data — not a claim about "
                "which features drove this specific prediction:"
            )
            for name, importance in result["top_features"]:
                pretty_name = name.replace("_", " ").title()
                st.write(f"- {pretty_name} ({importance:.1%} overall importance)")

st.divider()
st.caption(
    "⚠️ This tool provides a machine-learning-based prediction and is not a "
    "guarantee that a URL is safe. It is a class project prototype, not a "
    "production security system."
)



