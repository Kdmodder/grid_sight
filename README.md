# ⚡ GridSight: AI-Powered Transmission Infrastructure Risk Forecasting

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange.svg)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red.svg)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Render-brightgreen.svg)](https://gridsight-mxao.onrender.com)

**🌐 [Try Live Demo →](https://gridsight-mxao.onrender.com)**

## 📋 Project Overview

**GridSight** is an end-to-end Machine Learning Operations (MLOps) system designed to predict **cumulative delays** and **cost overruns** in power transmission infrastructure projects. Built for the energy sector, it provides phase-aware risk forecasting to enable proactive decision-making during project execution.

### 🎯 Business Problem

Power transmission projects (400kV/765kV lines & substations) face:
- **47+ days** average cumulative delays
- **₹40+ Cr** cost overruns on average
- **7 sequential phases** from Planning → Testing
- Uncertainty increases progressively through project lifecycle

**GridSight Solution:** Predict cumulative risk at each phase using only available data, simulating real-world progressive uncertainty.

---

## 🏗️ System Architecture

### Pipeline Overview
![GridSight Pipeline](Architecture_diagrams/Gridsight%20pipeline.jpg)

### Training & Prediction Strategy
![Training Strategy](Architecture_diagrams/Training%20&%20prediction%20strategy.jpg)

### Use Case Diagram
![Use Case](Architecture_diagrams/usecase-diagram.jpg)

---

## 🔬 Technical Implementation

### 1. End-to-End Data Science Pipeline

#### **Data Understanding & Engineering**
```
Raw Data (145 Projects × 7 Phases)
    ↓
Domain Analysis (Transmission Infrastructure)
    ↓
Phase-Based Masking (Temporal Integrity)
    ↓
Feature Engineering (57 Features)
    ↓
Sequential Project Snapshots (1,015 rows)
    ↓
Model-Ready Dataset
```

**Key Innovation:** **Phase-Aware Masking**
- Each row represents what we knew at that specific phase
- Future phases → NaN (simulates real-world uncertainty)
- Prevents data leakage & maintains temporal logic
- Example: At Phase 3, Phase 4-7 data is masked

#### **Dataset Structure**
- **145 unique projects** (400kV/765kV Lines & Substations)
- **1,015 snapshots** (7 phases × 145 projects)
- **63 columns**: Metadata + 7 phase-specific feature groups
- **2 targets**: Cumulative Delay (days) & Cost Overrun (₹ Cr)

```python
# Phase-Based Feature Groups
Phase 1: Engineering (Engg_Planned_Days, Engg_Actual_Days, DPR_Status...)
Phase 2: ROW (ROW_Planned_Days, Forest_Clearance, Statutory_Approvals...)
Phase 3: Supply (Supply_Planned_Days, Orders_Placed, Delivery_Status...)
Phase 4: Civil (Civil_Planned_Days, Foundations_Complete, Site_Access...)
Phase 5: Erection (Erection_Planned_Days, Towers_Erected_Pct...)
Phase 6: Stringing (Stringing_Planned_Days, Conductor_Strung_Pct...)
Phase 7: Testing (Testing_Planned_Days, Trial_Run_Status, DOCO_Date...)
```

---

### 2. Machine Learning Model Development

#### **Algorithm Selection & Comparison**

**Models Evaluated:**
| Algorithm | Delay MAE | Delay R² | Cost MAE | Cost R² | Training Time |
|-----------|-----------|----------|----------|---------|---------------|
| **XGBoost** ✅ | **3.89 days** | **0.9595** | **₹24.83 Cr** | **0.9294** | 0.33s |
| LightGBM | 3.93 days | 0.9570 | ₹32.10 Cr | 0.8134 | 1.65s |
| Random Forest | 7.38 days | 0.9051 | ₹28.23 Cr | 0.8841 | 0.38s |

**Winner:** **XGBoost** (best accuracy + fast inference)

#### **XGBoost Configuration**
```python
{
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'tree_method': 'hist'
}
```

#### **Training Strategy**
- **Group-based split** (115 train projects / 29 test projects)
- **Prevents data leakage** (same project not in train & test)
- **812 training snapshots** / **203 testing snapshots**
- **Separate models** for Delay & Cost Overrun

---

### 3. Model Performance & Evaluation

#### **Overall Performance**

**Cumulative Delay Prediction:**
- ✅ MAE: **3.89 days** (vs 47 days average actual delay)
- ✅ RMSE: **7.40 days**
- ✅ R²: **0.9595** (95.95% variance explained)

**Cumulative Cost Overrun Prediction:**
- ✅ MAE: **₹24.83 Cr** (vs ₹40+ Cr average overrun)
- ✅ RMSE: **₹38.62 Cr**
- ✅ R²: **0.9294** (92.94% variance explained)

#### **Phase-Wise Performance Analysis**

| Phase | Stage | Delay MAE | Cost MAE | Insight |
|-------|-------|-----------|----------|---------|
| 1 | Planning/Engineering | **0.31 days** | **₹2.40 Cr** | ⭐ Most accurate |
| 2 | Approvals & ROW | 2.75 days | ₹25.72 Cr | Regulatory uncertainty starts |
| 3 | Procurement/Supply | 3.34 days | ₹26.71 Cr | Supply chain variability |
| 4 | Site Preparation/Civil | 3.70 days | ₹27.78 Cr | Ground reality emerges |
| 5 | Erection/Construction | 5.38 days | ₹28.90 Cr | Execution challenges |
| 6 | Stringing/Cabling | 5.72 days | ₹29.32 Cr | Weather dependencies |
| 7 | Testing & Commissioning | 6.05 days | ₹30.25 Cr | Final uncertainties |

**Key Insight:** Error increases progressively (uncertainty accumulates through phases) - validates domain logic!

---

### 4. MLOps Infrastructure with MLflow

#### **Experiment Tracking**
```python
# Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Experiment: GridSight_PhaseAware_Cumulative_Forecasting
```

#### **What Gets Logged**

**Parameters (14):**
- Model type, hyperparameters (learning_rate, max_depth, n_estimators...)
- Masking strategy, memory logic, split strategy
- Dataset version (v1, v2, v3...)
- Training snapshots count

**Metrics (20):**
- Phase-wise MAE for Delay (7 phases)
- Phase-wise MAE for Cost (7 phases)
- Overall MAE, RMSE, R² for both targets

**Artifacts:**
- Trained XGBoost models (delay & cost)
- Feature lists (categorical, numerical, targets)
- Phase name mappings
- Error distribution tables (CSV)
- Model metadata (JSON)

**Tags:**
- Project: GridSight
- Domain: Infrastructure
- Training Type: PhaseAware
- Model Purpose: Cumulative Forecasting

#### **MLflow UI**
```bash
# Start MLflow server
mlflow server --host 127.0.0.1 --port 5000

# Access UI
http://localhost:5000
```

---

### 5. Automated Retraining Pipeline

#### **Champion/Challenger Pattern**

```
Current Champion (production)
         ↓
New Feedback Data (100+ rows)
         ↓
Merge with Original Dataset → Increment Version (v1→v2)
         ↓
Train Challenger Model
         ↓
Compare: BOTH Delay MAE AND Cost MAE must improve
         ↓
    [YES]              [NO]
     ↓                  ↓
Promote Challenger   Archive Challenger
Archive Old Champion  Keep Champion
         ↓
Updated Production Model
```

#### **Promotion Logic**
```python
# Challenger promoted ONLY if BOTH metrics improve
if (delay_improvement > 0) AND (cost_improvement > 0):
    promote_challenger()
else:
    save_challenger_to_archive()  # Keep champion
```

#### **Retraining Workflow**
```bash
# 1. Generate feedback data
python generate_feedback_data.py

# 2. Run retraining pipeline
python retrain_pipeline.py
```

**Outputs:**
- Updated `champion_model.pkl` (if promoted)
- Archived models in `best_models/archived_challengers/`
- Retraining report (`retraining_report.md`)
- Updated dataset version in `Data/dataset_versions.json`

---

### 6. Production Deployment

#### **Streamlit Web Application**

**Features:**
- ✅ **5-Step Workflow:** Metadata → Phase Selection → Data Input → Prediction → Feedback
- ✅ **Phase-Wise Tracking:** Historical predictions across phases
- ✅ **State Management:** Save/load project data
- ✅ **Autofill from CSV:** Quick testing with real data
- ✅ **Feedback Collection:** Stores actual outcomes for retraining
- ✅ **2-Decimal Precision:** Clean financial reporting

**Run Application:**
```bash
streamlit run app.py
```

**UI Workflow:**
1. **Project Metadata:** Type, voltage, contract, budget, location...
2. **Phase Selection:** Choose current phase (1-7)
3. **Phase Data Entry:** Planned vs Actual (days & cost) per phase
4. **Risk Prediction:** Get cumulative delay & cost overrun forecasts
5. **Feedback Submission:** Record actual outcomes

**Autofill Feature:**
```python
# Load random test data matching selected phase
Toggle: "Autofill data till Phase N from test_data.csv"
Button: "🎲 Load Random Data"
```

#### **🌐 Live Deployment**

The application is **deployed and accessible online**:

**🚀 [Access GridSight Live Demo](https://gridsight-mxao.onrender.com)**

Hosted on **Render** with:
- ✅ Production-ready Streamlit interface
- ✅ Real-time risk predictions
- ✅ Autofill feature with test data
- ✅ Phase-wise tracking dashboard
- ✅ Feedback collection system

**Note:** First load may take 30-60 seconds (free-tier cold start)

---

## 📊 Key Results Summary

### Model Accuracy
- **Delay Prediction:** 3.89 days MAE (92% better than baseline)
- **Cost Prediction:** ₹24.83 Cr MAE (38% better than baseline)
- **Overall R²:** 95.95% (Delay) | 92.94% (Cost)

### Business Impact
- Early warning system for project managers
- Proactive intervention before delays compound
- Budget allocation optimization
- Stakeholder transparency with phase-wise tracking

### Technical Achievements
- ✅ End-to-end ML pipeline (data → deployment)
- ✅ Phase-aware masking (temporal integrity)
- ✅ Multi-algorithm comparison (XGBoost wins)
- ✅ MLflow experiment tracking (20 metrics logged)
- ✅ Automated retraining with champion/challenger
- ✅ Production-ready Streamlit UI
- ✅ Feedback loop for continuous improvement

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd GridSight

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
source myenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

#### 1. Train Model
```bash
python gridsight_xgboost.py
```
**Output:** `models/champion_model.pkl`

#### 2. Start MLflow Server
```bash
mlflow server --host 127.0.0.1 --port 5000
```
**Access:** http://localhost:5000

#### 3. Run Streamlit App
```bash
streamlit run app.py
```
**Access:** http://localhost:8501

#### 4. Test Model
```bash
python test_model.py
```

#### 5. Compare Algorithms
```bash
python gridsight_model_comparison.py
```

#### 6. Retrain with Feedback
```bash
python generate_feedback_data.py  # Generate 120 test rows
python retrain_pipeline.py        # Retrain & evaluate
```

---

## 📁 Project Structure

```
GridSight/
│
├── Data/
│   ├── gridsight_dataset.csv           # Phase-masked training data (1,015 rows)
│   ├── test_data.csv                   # Test samples for autofill
│   ├── new_feedback_data.csv           # Feedback collection file
│   └── dataset_versions.json           # Version tracking
│
├── models/
│   ├── champion_model.pkl              # Production model (delay + cost + encoders)
│   └── best_models/
│       └── archived_challengers/       # Rejected challenger models
│
├── Architecture_diagrams/
│   ├── Gridsight pipeline.jpg          # End-to-end pipeline
│   ├── Training & prediction strategy.jpg
│   └── usecase-diagram.jpg
│
├── Guide/
│   ├── MLOPS_GUIDE.md                  # Comprehensive MLOps documentation
│   ├── QUICKSTART.md                   # 5-minute setup guide
│   ├── SIMPLE_GUIDE.md                 # Beginner-friendly explanation
│   └── IMPLEMENTATION_SUMMARY.md       # Technical implementation details
│
├── gridsight_xgboost.py                # Main training script with MLflow
├── gridsight_model_comparison.py       # Multi-algorithm comparison
├── retrain_pipeline.py                 # Automated retraining with champion/challenger
├── test_model.py                       # Model testing & evaluation
├── app.py                              # Streamlit production UI
├── generate_feedback_data.py           # Generate test feedback data
├── test_mlops.py                       # MLOps validation suite
│
├── mlruns/                             # MLflow experiment tracking
├── mlflow.db                           # MLflow metadata database
│
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [MLOPS_GUIDE.md](Guide/MLOPS_GUIDE.md) | Complete MLOps implementation guide |
| [QUICKSTART.md](Guide/QUICKSTART.md) | 5-minute setup instructions |
| [SIMPLE_GUIDE.md](Guide/SIMPLE_GUIDE.md) | Beginner-friendly explanation |
| [IMPLEMENTATION_SUMMARY.md](Guide/IMPLEMENTATION_SUMMARY.md) | Technical implementation details |

---

## 🛠️ Technologies Used

### Core ML Stack
- **Python 3.8+** - Programming language
- **XGBoost** - Gradient boosting (chosen algorithm)
- **LightGBM, Random Forest** - Comparison models
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - Preprocessing & metrics

### MLOps Tools
- **MLflow** - Experiment tracking, model registry, versioning
- **Streamlit** - Web application framework
- **Pickle** - Model serialization

### Development Tools
- **Git** - Version control
- **Jupyter Notebook** - Exploratory analysis
- **VS Code** - IDE

---

## 🎓 Learning Highlights

This project demonstrates:

1. **Domain-Specific ML:** Adapting ML to transmission infrastructure constraints
2. **Temporal Data Handling:** Phase-aware masking for sequential projects
3. **Feature Engineering:** 57 features from 7 phase groups
4. **Model Selection:** Rigorous comparison (XGBoost > LightGBM > RF)
5. **MLOps Best Practices:**
   - Experiment tracking with MLflow
   - Champion/challenger pattern
   - Automated retraining pipeline
   - Model versioning & archival
6. **Production Deployment:** Streamlit UI with state management
7. **Feedback Loop:** Continuous improvement from actual outcomes

---

## 📈 Future Enhancements

- [ ] Real-time data ingestion from project management systems
- [ ] Multi-model ensemble (XGBoost + LightGBM stacking)
- [ ] Uncertainty quantification (confidence intervals)
- [ ] Explainability (SHAP values for predictions)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure)
- [ ] Automated A/B testing of model versions
- [ ] Advanced monitoring & alerting

---

## 👤 Author

**Tejas**

- Project: GridSight - AI Infrastructure Risk Forecasting
- Domain: Power Transmission (400kV/765kV Lines & Substations)
- Focus: MLOps, Phase-Aware Forecasting, Production Deployment

---

## 📄 License

This project is part of a data science portfolio demonstrating end-to-end ML system development.

---

## 🙏 Acknowledgments

- Dataset: Inspired by real-world Indian power transmission projects
- Domain Knowledge: Power Grid Corporation of India (PGCIL) project patterns
- ML Frameworks: XGBoost, MLflow, Streamlit communities

**⚡ GridSight: Making Infrastructure Projects Predictable, One Phase at a Time ⚡**
