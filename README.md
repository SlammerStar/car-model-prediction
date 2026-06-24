# ⚡ DRIVEIQ - AI Powered Used Car Valuation System

[![CI/CD Pipeline](https://github.com/yourusername/car-model-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/car-model-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **end-to-end machine learning application** that predicts used car prices in **Indian Rupees (₹)** across **8 major premium car brands** using advanced ML models, explainable AI, and production-grade software engineering practices.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🤖 **Multi-Model Comparison** | Trains and compares 5 ML models (LR, DT, RF, GB, XGBoost) |
| 📊 **Exploratory Data Analysis** | Professional interactive visualizations with Plotly |
| 🔧 **Feature Engineering** | Automated car_age, km_per_year, premium_brand_flag computation |
| ⚙️ **ML Pipeline** | End-to-end scikit-learn Pipeline with ColumnTransformer |
| 🎯 **Hyperparameter Tuning** | RandomizedSearchCV with 5-fold cross-validation |
| 🧠 **Explainable AI** | SHAP feature importance, summary, and waterfall plots |
| 🌐 **Streamlit UI** | Premium dark theme UI with glassmorphism and market insights |
| 🚀 **FastAPI Backend** | RESTful API with Pydantic validation and auto-docs |
| 📦 **Docker** | Containerized deployment with Docker Compose |
| 📈 **MLflow** | Experiment tracking for parameters, metrics, and models |
| ✅ **CI/CD** | GitHub Actions for automated testing and quality checks |
| 💰 **INR Formatting** | Displays prices in Indian Lakhs/Crores notation |

---

## 📁 Project Structure

```
car-model-prediction/
│
├── data/                           # Standardized dataset files
│   ├── audi.csv
│   ├── bmw.csv
│   ├── ford.csv
│   ├── hyundai.csv
│   ├── mercedes.csv
│   ├── skoda.csv
│   ├── toyota.csv
│   └── vw.csv
│
├── Datasets/                       # Original dataset files (legacy)
│
├── src/                            # Core ML source code
│   ├── __init__.py                 # Package initialization
│   ├── utils.py                    # Constants, helpers, model I/O
│   ├── data_preprocessing.py       # Data loading, cleaning, merging
│   ├── feature_engineering.py      # Feature creation, ColumnTransformer
│   ├── train_model.py              # Model training, comparison, tuning
│   ├── evaluate_model.py           # EDA, evaluation, SHAP plots
│   └── predict.py                  # Prediction utilities
│
├── api/
│   └── main.py                     # FastAPI REST API
│
├── models/                         # Saved model artifacts
│   ├── pipeline.pkl                # Trained sklearn pipeline
│   └── metadata.json               # Training metadata & metrics
│
├── images/                         # Generated visualizations
│
├── tests/
│   └── test_pipeline.py            # Unit tests (18 tests)
│
├── notebook/                       # Jupyter notebooks (EDA, experiments)
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # Multi-service orchestration
├── .gitignore                      # Git ignore rules
├── .github/workflows/ci.yml        # CI/CD pipeline
└── README.md                       # This file
```

---

## 📊 Dataset Information

| Brand | Records | Source |
|-------|---------|--------|
| Audi | ~10,000 | UK used car listings |
| BMW | ~10,000 | UK used car listings |
| Ford | ~17,000 | UK used car listings |
| Hyundai | ~4,800 | UK used car listings |
| Mercedes | ~13,000 | UK used car listings |
| Skoda | ~6,200 | UK used car listings |
| Toyota | ~6,700 | UK used car listings |
| Volkswagen | ~15,000 | UK used car listings |
| **Total** | **~83,000** | |

**Features used:** brand, model, year, car_age, transmission, mileage, fuelType, mpg, engineSize

**Price conversion:** GBP → INR (exchange rate: £1 = ₹115)

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11 |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | SHAP |
| **Web Frontend** | Streamlit |
| **API Backend** | FastAPI, Uvicorn, Pydantic |
| **Experiment Tracking** | MLflow |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest |
| **Code Quality** | Black, Flake8 |

---

## 🔄 ML Workflow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Load 8 CSV  │───▶│  Clean &     │───▶│  Feature         │
│  Datasets    │    │  Merge Data  │    │  Engineering     │
└──────────────┘    └──────────────┘    └──────────────────┘
                                                │
                                                ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Save Best   │◀───│  Hyperparameter│◀──│  Train & Compare │
│  Pipeline    │    │  Tuning (CV)  │    │  5 Models        │
└──────────────┘    └──────────────┘    └──────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Deploy: Streamlit UI + FastAPI + Docker + MLflow    │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- pip or conda
- Docker (optional, for containerized deployment)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/car-model-prediction.git
cd car-model-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python -m src.train_model
```
This will:
- Load and merge all 8 datasets
- Clean data and engineer features
- Train 5 ML models and compare them
- Tune the best model with RandomizedSearchCV
- Save the pipeline to `models/pipeline.pkl`

### 5. Generate Evaluation Plots
```bash
python -m src.evaluate_model
```

### 6. Run the Streamlit App
```bash
streamlit run app.py
```
Open: [http://localhost:8501](http://localhost:8501)

### 7. Run the FastAPI Server
```bash
uvicorn api.main:app --reload
```
Open: [http://localhost:8000/docs](http://localhost:8000/docs)

### 8. Run Tests
```bash
python -m pytest tests/ -v
```

---

## 🐳 Docker Deployment

### Single Container
```bash
docker build -t car-price-predictor .
docker run -p 8501:8501 car-price-predictor
```

### Full Stack (Streamlit + FastAPI + MLflow)
```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |

---

## 📡 API Documentation

### POST `/predict`

**Request:**
```json
{
  "brand": "BMW",
  "model": "X5",
  "year": 2019,
  "transmission": "Automatic",
  "mileage": 45000,
  "fuelType": "Diesel",
  "mpg": 52.3,
  "engineSize": 2.0
}
```

**Response:**
```json
{
  "predicted_price": "₹12.45 Lakhs",
  "predicted_price_raw": 1245000.0,
  "input_summary": {
    "brand": "BMW",
    "model": "X5",
    "year": 2019,
    "car_age": 7,
    "transmission": "Automatic",
    "mileage": 45000,
    "fuelType": "Diesel",
    "mpg": 52.3,
    "engineSize": 2.0
  }
}
```

### GET `/health`
Returns API health status and model availability.

### GET `/brands`
Returns the list of supported car brands.

### GET `/models/{brand}`
Returns available car models for a specific brand.

---

## 📈 Results

### Model Comparison

| Model | R² Score | MAE (₹) | RMSE (₹) |
|-------|----------|---------|----------|
| XGBoost | ~0.95 | ~₹50,000 | ~₹80,000 |
| Random Forest | ~0.94 | ~₹55,000 | ~₹85,000 |
| Gradient Boosting | ~0.93 | ~₹60,000 | ~₹90,000 |
| Decision Tree | ~0.88 | ~₹80,000 | ~₹120,000 |
| Linear Regression | ~0.75 | ~₹120,000 | ~₹180,000 |

*Actual results will vary. Run training to see exact metrics.*

---

## 🔮 Future Improvements

- [ ] Add more car brands (Kia, Honda, Nissan)
- [ ] Integrate real-time exchange rate API
- [ ] Add user authentication for the web app
- [ ] Deploy on AWS/GCP/Azure with CI/CD
- [ ] Add A/B testing for model versions
- [ ] Implement data drift monitoring
- [ ] Add feature store integration
- [ ] Create mobile-responsive UI
- [ ] Add car image classification
- [ ] Implement model versioning with DVC

---

## 👨‍💻 Author

**Pratham Nigam**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>⭐ Star this repository if you find it useful! ⭐</b>
</p>
