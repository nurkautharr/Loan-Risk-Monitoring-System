# Loan Risk Monitoring System

Production-ready Credit Risk Scoring API built using FastAPI and Logistic Regression.

This project simulates how financial institutions operationalize Probability of Default (PD) models into automated lending decision systems.

---

## 🌐 Live API

Swagger Documentation:
👉 https://loan-risk-api-ecg4.onrender.com/docs

---

## 📌 Problem

Banks must balance:

- Credit risk mitigation
- Operational efficiency
- Fairness & governance
- Capital protection

This system demonstrates how a PD model transitions from experimentation to production deployment with business rules and risk policy.

---

## 🧠 Business Logic

Model: Logistic Regression  
Metric: ROC-AUC = 0.8675  

Decision threshold policy:

- PD < 0.30 → APPROVE  
- 0.30 ≤ PD < 0.60 → MANUAL_REVIEW  
- PD ≥ 0.60 → REJECT  

Expected Loss:

Expected Loss = PD × Loan Amount × LGD

Where:
- PD = Probability of Default
- LGD = Loss Given Default (assumed 60%)

---

## 📦 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Service health check |
| POST | /predict | Single loan scoring |
| POST | /batch_predict | Portfolio scoring |

---

## 🏗 Architecture

Client  
↓  
FastAPI  
↓  
Serialized ML Pipeline (joblib)  
↓  
Business Decision Rules  
↓  
JSON Response  

---

## ⚖ Governance Considerations

- Threshold tuning based on risk appetite
- Fairness diagnostics across demographic groups
- Dependency freezing for reproducibility
- Health endpoint for monitoring