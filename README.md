<div align="center">

# FEM-NET
### Federated Early Misdiagnosis Network

**Privacy-preserving AI for early detection of underdiagnosed women's health conditions**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-1.9.0-pink?style=flat-square)](https://flower.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.5-1C3C3C?style=flat-square)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> Raw patient data **never leaves the hospital**. Only model weight updates travel across nodes.

</div>

---

# Submission Demo (Windows)

1. Install

```bash
pip install -r requirements.txt
```

2. Run

```bash
python run.py serve
```

3. Open

http://localhost:8000

4. Federated demo

- Click “Run Federated”
- Wait ~30 seconds
- Click refresh to see rounds/AUC update

Notes:
- Insights work even without an OpenAI key (automatic fallback mode)
- Use Python 3.10 or 3.11 (recommended). Python 3.13 will not work with the current Flower/NumPy requirements.

## What is FEM-NET?

FEM-NET is a federated learning platform where multiple hospitals collaboratively train early-detection AI models for women's health — without sharing a single row of patient data. Each hospital trains locally; only compressed weight updates flow to a central aggregator that uses **Byzantine-robust Krum filtering** to resist poisoned updates.

A clinician enters patient vitals into the dashboard, receives a real-time risk score from the global model, and gets an evidence-backed insight card sourced from live PubMed literature.

---

## Conditions Detected

| Condition | Source Dataset | Rows | Target AUC |
|---|---|---|---|
| PCOS | Kaggle — Prasoon Kottarathil + Shreyas Vedpathak | ~541 | > 0.80 |
| Breast Cancer | UCI Wisconsin Diagnostic | 569 | > 0.95 |
| Cervical Cancer | Ranzeet013 Risk Factors | 858 | > 0.78 |
| Thyroid Disorder | Yasser Hessein | 9,172 | > 0.85 |
| Diabetes | Pima Indians (UCI) | 768 | > 0.82 |
| Cardiovascular Disease | Sulianova | 70,000 | > 0.79 |
| Heart Disease | UCI Cleveland + multi-source | 920 | > 0.88 |

---

## System Architecture

```mermaid
graph TB
    subgraph Hospitals["Hospital Nodes (No Raw Data Leaves)"]
        H1["Hospital A\ndata/processed/pcos.csv"]
        H2["Hospital B\ndata/processed/thyroid.csv"]
        H3["Hospital C\ndata/processed/diabetes.csv"]
    end

    subgraph FL["Federated Learning Core"]
        C1["FemNetClient\nLocal Training\n5 epochs · Adam · BCELoss"]
        C2["FemNetClient\nLocal Training"]
        C3["FemNetClient\nLocal Training"]
        K["KrumStrategy\nByzantine-Robust Aggregation\nDrops outlier updates"]
        GW["Global Weights\nmodels/weights/<condition>_global.pt"]
    end

    subgraph API["FastAPI Backend"]
        P["POST /api/predict\nClinicalMLP inference"]
        I["POST /api/insights\nRAG Insight Card"]
        F["GET /api/federated/status\nPOST /api/federated/run"]
    end

    subgraph RAG["RAG Pipeline"]
        PB["PubMed Fetcher\nBioPython Entrez"]
        EM["Embedder\nall-MiniLM-L6-v2\nFAISS Index"]
        LLM["LangChain + GPT-4o-mini\nor Fallback Card"]
    end

    subgraph UI["Clinician Dashboard"]
        FORM["Patient Intake Form"]
        GAUGE["Risk Gauge SVG"]
        CARD["Insight Card\nEvidence + Next Steps"]
        FLP["FL Status Panel"]
    end

    H1 -->|weight updates only| C1
    H2 -->|weight updates only| C2
    H3 -->|weight updates only| C3

    C1 -->|gradients| K
    C2 -->|gradients| K
    C3 -->|gradients| K

    K -->|aggregated weights| GW

    GW --> P
    PB --> EM
    EM --> LLM
    LLM --> I

    FORM -->|patient JSON| P
    FORM -->|patient + score| I
    P -->|risk_score| GAUGE
    I -->|insight card| CARD
    F -->|round stats| FLP
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant C as Clinician
    participant D as Dashboard
    participant A as FastAPI
    participant M as ClinicalMLP
    participant R as RAG Pipeline
    participant PB as PubMed / FAISS

    C->>D: Fill patient form + select condition
    D->>A: POST /api/predict {age, bmi, fsh_level, ...}
    A->>M: Load global weights + normalise features
    M-->>A: risk_score (0.0 – 1.0)
    A-->>D: {risk_score, risk_level, model_version}
    D->>C: Render SVG risk gauge (green / amber / red)

    D->>A: POST /api/insights {patient, risk_score}
    A->>R: generate_insight_card()
    R->>PB: Query FAISS index (top-3 PubMed abstracts)
    PB-->>R: Relevant clinical evidence
    R->>R: LangChain prompt + GPT-4o-mini
    R-->>A: {risk_level, key_indicators, evidence_summary, next_steps}
    A-->>D: Insight card JSON
    D->>C: Render evidence card + recommended steps
```

---

## Federated Learning Flow

```mermaid
flowchart LR
    subgraph Round["One FL Round"]
        direction TB
        INIT["Global Model\nInitialised"] --> DIST["Distribute weights\nto all clients"]
        DIST --> T1["Hospital A\nLocal train 5 epochs"]
        DIST --> T2["Hospital B\nLocal train 5 epochs"]
        DIST --> T3["Hospital C\nLocal train 5 epochs"]
        T1 --> KRUM
        T2 --> KRUM
        T3 --> KRUM
        KRUM["Krum Filter\nScore L2 distances\nDrop Byzantine outlier"] --> AGG["FedAvg\nAggregate selected\nweight updates"]
        AGG --> NEW["Updated Global\nModel Weights"]
    end

    START["data/processed\n/<condition>.csv"] --> SPLIT["Non-IID Split\n3 hospital partitions\nSkewed class balance"]
    SPLIT --> Round
    NEW -->|"next round"| DIST
    NEW -->|"after N rounds"| SAVE["models/weights\n/<condition>_global.pt"]
```

---

## Project Structure

```
fem-net/
├── api/
│   ├── main.py                  FastAPI app, lifespan, CORS, static mount
│   ├── schemas.py               Pydantic models (PatientInput, InsightCard, ...)
│   └── routes/
│       ├── predict.py           POST /api/predict
│       ├── insights.py          POST /api/insights
│       └── federated.py         GET/POST federated status + trigger
│
├── federated/
│   ├── strategy.py              KrumStrategy — Byzantine-robust aggregation
│   ├── client.py                FemNetClient — Flower NumPyClient
│   └── server.py                Flower server launcher
│
├── models/
│   ├── base_model.py            ClinicalMLP (10→128→64→32→1, BatchNorm, Dropout)
│   ├── condition_models.py      get_model() factory, save_model()
│   └── weights/                 *.pt + *_norm_stats.json (gitignored, trained locally)
│
├── rag/
│   ├── pubmed_fetcher.py        BioPython Entrez + 24h cache
│   ├── embedder.py              SentenceTransformer + FAISS flat index
│   └── insight_generator.py    LangChain RAG → clinical insight card
│
├── simulation/
│   └── run_simulation.py        In-process FL sim, non-IID splits, saves global weights
│
├── data/
│   ├── preprocess_<condition>.py    Per-condition preprocessing (Suraj / Kuber)
│   └── processed/                   Standardised 10-feature CSVs (gitignored)
│
├── training/
│   └── train_model.py           Unified trainer — outputs .pt + norm_stats.json
│
└── dashboard/
    ├── index.html               Clinician UI
    └── static/
        ├── style.css
        └── app.js
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env:
# ENTREZ_EMAIL=your@email.com
# OPENAI_API_KEY=sk-...   (optional — RAG falls back gracefully without it)
```

### 3. Add processed datasets

Teammates Suraj and Kuber run preprocessing scripts. Once `data/processed/` has CSVs:

```bash
ls data/processed/
# pcos.csv  breast_cancer.csv  cervical.csv
# thyroid.csv  diabetes.csv  cardiovascular.csv  heart.csv
```

### 4. Train models

```bash
python training/train_model.py --condition pcos --epochs 50
python training/train_model.py --condition thyroid --epochs 50
# ... repeat for each condition
```

### 5. Run federated simulation

```bash
python -m simulation.run_simulation --condition pcos --rounds 5
```

```
[FEM-NET] Starting FL simulation: condition=pcos, rounds=5
[Krum] round 1: kept 2/3 clients (dropped 1 outlier)
[Krum] round 2: kept 2/3 clients (dropped 1 outlier)
...
=== Simulation complete: pcos ===
  Rounds : 5
  Last AUC: 0.8341
  Results → simulation/results_pcos.json
```

### 6. Start the API

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000` — fill the clinician form, see the risk gauge and insight card.

---

## API Reference

### `POST /api/predict`

```json
{
  "age": 28,
  "bmi": 24.5,
  "fsh_level": 7.2,
  "lh_level": 12.4,
  "amh_level": 3.1,
  "tsh_level": 2.5,
  "cycle_length_days": 30,
  "follicle_count": 10,
  "fatigue_score": 5,
  "weight_gain_kg": 3.0,
  "condition": "pcos"
}
```

Response:
```json
{
  "risk_score": 0.7821,
  "risk_level": "High",
  "model_version": "pcos-v1715234400"
}
```

### `POST /api/insights`

Request: same patient fields + `"risk_score": 0.78`

Response:
```json
{
  "risk_level": "High",
  "key_indicators": ["LH: 12.40 (elevated)", "AMH: 3.10", "..."],
  "evidence_summary": "Based on retrieved literature: ...",
  "recommended_next_steps": ["Consult specialist", "Hormone panel", "..."]
}
```

### `GET /api/federated/status?condition=pcos`

```json
{
  "condition": "pcos",
  "num_rounds_completed": 5,
  "participating_hospitals": 3,
  "last_global_auc": 0.8341
}
```

### `POST /api/federated/run?condition=pcos&rounds=5`

Triggers background FL simulation. Returns immediately:
```json
{ "status": "started", "condition": "pcos", "rounds": 5 }
```

---

## Model Architecture

```
Input (10 features)
       │
  Linear(10 → 128)
  BatchNorm1d(128)
  ReLU
  Dropout(0.3)
       │
  Linear(128 → 64)
  BatchNorm1d(64)
  ReLU
  Dropout(0.3)
       │
  Linear(64 → 32)
  ReLU
       │
  Linear(32 → 1)
  Sigmoid
       │
  risk_score ∈ [0, 1]
```

**10 input features** (standardised per condition):
`age · bmi · fsh_level · lh_level · amh_level · tsh_level · cycle_length_days · follicle_count · fatigue_score · weight_gain_kg`

---

## Byzantine-Robust Aggregation (Krum)

Standard federated averaging is vulnerable to poisoned clients. FEM-NET uses **Multi-Krum** to filter outliers before aggregation:

```mermaid
graph LR
    C1["Client 1\nweight vector w1"] --> SCORE
    C2["Client 2\nweight vector w2"] --> SCORE
    C3["Client 3\nweight vector w3"] --> SCORE
    SCORE["Krum Score\nFor each client i:\nsum of L2² distances\nto f nearest neighbours"] --> RANK["Rank by score\nlowest = most trusted"]
    RANK --> SELECT["Select top n-f clients\nn=3 clients, f=1 Byzantine\nKeep 2 best"]
    SELECT --> AVG["FedAvg on\nselected clients only"]
    AVG --> GLOBAL["Updated global model"]
```

---

## Team

| Member | Role | Owns |
|---|---|---|
| **Kruthi** | ML Lead | FL engine, API, RAG pipeline, dashboard, models |
| **Suraj** | ML Engineer | PCOS, Breast Cancer, Cervical Cancer preprocessing + training |
| **Kuber** | ML Engineer | Thyroid, Diabetes, Cardiovascular, Heart preprocessing + training |
| **Akshara** | ML + UI | Dataset research (endometriosis), dashboard UI, dataset registry |

---

## Privacy Guarantee

```
What stays at each hospital        What travels to the server
──────────────────────────         ──────────────────────────
Raw patient records          ✗     Weight gradients (numpy arrays)  ✓
Patient identifiers          ✗     Round loss / AUC metrics         ✓
Lab values                   ✗
Diagnosis history            ✗
```

No raw data, no patient identifiers, no diagnosis history ever leave a hospital node. The server sees only floating-point weight tensors, indistinguishable from random noise without the model architecture.

---

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
Built for early detection. Built for privacy. Built for women's health.
</div>
