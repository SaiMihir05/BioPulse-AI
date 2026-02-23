import streamlit as st
import requests

st.set_page_config(
    page_title="BioPulse AI Dashboard",
    layout="centered"
)

st.title("🧬 BioPulse AI – ICU Mortality Dashboard")
st.write("Predict ICU mortality risk using real MIMIC-IV data")

# Input
stay_id = st.number_input(
    "Enter ICU Stay ID",
    min_value=1,
    step=1
)

# Button
if st.button("Predict Risk"):
    with st.spinner("Fetching prediction..."):
        try:
            response = requests.get(
                f"http://localhost:8000/predict/{stay_id}"
            )

            if response.status_code == 200:
                result = response.json()

                st.subheader(f"ICU Stay ID: {stay_id}")

                risk_prob = result["risk_probability"]
                risk_label = result["risk_label"]

                st.metric(
                    label="Mortality Risk Probability",
                    value=f"{risk_prob * 100:.2f}%"
                )

                if risk_label == 1:
                    st.error("⚠️ High Mortality Risk")
                else:
                    st.success("✅ Low Mortality Risk")

                # 🔬 Explainability Section
                st.subheader("🔬 Top Contributing Features")

                contributions = result["feature_contributions"]

                # Show top 5 contributors
                top_features = list(contributions.items())[:5]

                for feature, value in top_features:
                    if value > 0:
                        st.write(f"🔺 {feature}: +{value:.3f}")
                    else:
                        st.write(f"🔻 {feature}: {value:.3f}")

            else:
                st.warning("ICU Stay not found in database")

        except Exception:
            st.error("Backend not reachable. Is FastAPI running?")