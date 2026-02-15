import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="Loan Risk Monitoring Dashboard", layout="wide")

# ✅ Put your Render API base URL here
API_BASE_URL = "https://loan-risk-api-ecg4.onrender.com"

st.title("🏦 Loan Risk Monitoring Dashboard")
st.caption("Streamlit frontend → calls FastAPI (Render) → returns PD, decision, expected loss")

# ---------- Helpers ----------
def call_predict(payload: dict):
    url = f"{API_BASE_URL}/predict"
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def call_batch_predict(payload: dict):
    url = f"{API_BASE_URL}/batch_predict"
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def decision_badge(decision: str):
    if decision == "APPROVE":
        st.success("✅ APPROVE")
    elif decision == "MANUAL_REVIEW":
        st.warning("🟠 MANUAL REVIEW")
    else:
        st.error("⛔ REJECT")

def risk_band(pd_default: float) -> str:
    if pd_default is None or pd.isna(pd_default):
        return "Unknown"
    if pd_default < 0.30:
        return "Low"
    elif pd_default < 0.60:
        return "Medium"
    return "High"

def safe_sum(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float(s.sum())

def get_api_health():
    """Fetch /health and return dict or None"""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ---------- Sidebar: API Check + API VERSION LABEL ----------
st.sidebar.header("🔌 API Connection")
st.sidebar.write("Base URL:")
st.sidebar.code(API_BASE_URL)

health = get_api_health()
if health:
    api_ver = health.get("api_version", "unknown")
    st.sidebar.success(f"Connected ✅  API Version: {api_ver}")
else:
    st.sidebar.warning("API not reachable right now.")

if st.sidebar.button("Test /health"):
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=15).json()
        st.sidebar.success(f"Connected ✅ {res}")
    except Exception as e:
        st.sidebar.error(f"Failed ❌ {e}")

st.divider()

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📌 Single Loan Scoring", "📦 Batch Portfolio Scoring"])

# ============================
# TAB 1: Single loan
# ============================
with tab1:
    st.subheader("Single Loan Scoring")
    st.write("Fill in a few key fields (you can add more later).")

    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            application_id = st.text_input("Application ID", value="A-TEST-001")
            loan_amount = st.number_input("Loan Amount", min_value=0.0, value=300000.0, step=1000.0)
            income = st.number_input("Income (annual)", min_value=0.0, value=8000.0, step=100.0)

        with col2:
            credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=720.0, step=1.0)
            ltv = st.number_input("LTV (%)", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
            dtir1 = st.number_input("DTI Ratio (dtir1)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

        with col3:
            gender = st.selectbox("Gender", ["Male", "Female", "Joint", "Sex Not Available"])
            region = st.selectbox("Region", ["North", "south", "central", "North-East"])
            security_type = st.selectbox("Security Type", ["direct", "indirect"])
            loan_limit = st.selectbox("loan_limit", ["cf", "ncf"])
            year = st.selectbox("year", [2019])

        submitted = st.form_submit_button("Score this loan")

    if submitted:
        payload = {
            "application_id": application_id,
            "data": {
                "loan_amount": loan_amount,
                "income": income,
                "Credit_Score": credit_score,
                "LTV": ltv,
                "dtir1": dtir1,
                "Gender": gender,
                "Region": region,
                "Security_Type": security_type,
                "loan_limit": loan_limit,
                "year": year
            }
        }

        st.code(payload, language="json")

        try:
            result = call_predict(payload)

            st.subheader("Result")
            decision_badge(result.get("decision"))

            colA, colB, colC = st.columns(3)
            colA.metric("PD (Probability of Default)", f"{result.get('pd_default', 0):.4f}")
            colB.metric("Decision", result.get("decision", "N/A"))
            colC.metric(
                "Expected Loss",
                f"{result.get('expected_loss'):,.0f}" if result.get("expected_loss") is not None else "N/A"
            )

            st.caption(f"Assumptions: {result.get('assumptions')}")

        except requests.exceptions.HTTPError as e:
            st.error("API returned an error.")
            st.write(e.response.text)
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ============================
# TAB 2: Batch scoring
# ============================
with tab2:
    st.subheader("Batch Portfolio Scoring")
    st.write("Upload a CSV to score multiple loans and view portfolio-level risk metrics.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        st.write("Preview:")
        st.dataframe(df.head(10), use_container_width=True)

        if df.empty:
            st.warning("Your CSV is empty.")
            st.stop()

        # ✅ Make JSON-safe
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notna(df), None)
        items = df.to_dict(orient="records")
        payload = {"items": items}

        col_run, col_hint = st.columns([1, 2])
        with col_run:
            run = st.button("Run Batch Predict", type="primary")
        with col_hint:
            st.caption("Tip: Include `loan_amount` to get expected_loss. Include `ID` to keep traceability.")

        if run:
            try:
                result = call_batch_predict(payload)
                results_df = pd.DataFrame(result.get("results", []))

                if results_df.empty:
                    st.warning("No results returned. Check your CSV columns.")
                    st.stop()

                # ✅ Add risk band
                results_df["risk_band"] = results_df["pd_default"].apply(risk_band)

                # ✅ KPI SECTION
                st.subheader("📌 Portfolio KPIs")

                total_loans = len(results_df)
                total_expected_loss = safe_sum(results_df.get("expected_loss", pd.Series(dtype=float)))

                decision_counts = results_df["decision"].value_counts(dropna=False)
                approve = int(decision_counts.get("APPROVE", 0))
                manual = int(decision_counts.get("MANUAL_REVIEW", 0))
                reject = int(decision_counts.get("REJECT", 0))

                approval_rate = (approve / total_loans) if total_loans > 0 else 0.0
                reject_rate = (reject / total_loans) if total_loans > 0 else 0.0

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Loans Scored", f"{total_loans}")
                k2.metric("Approval Rate", f"{approval_rate*100:.1f}%")
                k3.metric("Reject Rate", f"{reject_rate*100:.1f}%")
                k4.metric("Total Expected Loss", f"{total_expected_loss:,.0f}")

                # ✅ CHARTS
                st.subheader("📊 Risk Monitoring Views")

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**Decision Distribution**")
                    st.bar_chart(results_df["decision"].value_counts())

                with c2:
                    st.markdown("**Risk Band Distribution (Low / Medium / High)**")
                    st.bar_chart(results_df["risk_band"].value_counts())

                # PD distribution (bucketed)
                st.markdown("**PD Distribution (Bucketed)**")
                pd_series = pd.to_numeric(results_df["pd_default"], errors="coerce")
                bins = pd.cut(pd_series, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], include_lowest=True)
                st.bar_chart(bins.value_counts().sort_index())

                # ✅ TABLE
                st.subheader("🧾 Scored Results Table")
                preferred_cols = ["application_id", "ID", "pd_default", "decision", "expected_loss", "risk_band"]
                final_cols = [c for c in preferred_cols if c in results_df.columns] + \
                             [c for c in results_df.columns if c not in preferred_cols]
                st.dataframe(results_df[final_cols], use_container_width=True)

                # ✅ DOWNLOAD RESULTS CSV BUTTON
                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Scored Results (CSV)",
                    data=csv_out,
                    file_name="scored_portfolio.csv",
                    mime="text/csv"
                )

                st.caption(f"Assumptions: {result.get('assumptions')}")

            except requests.exceptions.HTTPError as e:
                st.error("API returned an error.")
                st.write(e.response.text)
            except Exception as e:
                st.error(f"Unexpected error: {e}")