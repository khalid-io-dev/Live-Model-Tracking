# 🏥 Diabetes Risk Prediction - MLOps Complete Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
5. [MLflow Model Registry](#mlflow-model-registry)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Monitoring & Observability](#monitoring--observability)
8. [API Usage](#api-usage)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This is a complete **MLOps pipeline** for diabetes risk prediction, featuring:

- ✅ **Machine Learning**: Random Forest model for diabetes risk classification
- ✅ **MLflow**: Experiment tracking, model registry, and versioning
- ✅ **FastAPI**: RESTful API with automatic Swagger documentation
- ✅ **Prometheus & Grafana**: Real-time monitoring and metrics visualization
- ✅ **GitHub Actions**: Automated CI/CD pipeline
- ✅ **Docker**: Containerized deployment

### Key Features:
- **Model Registry** with Staging → Production promotion
- **Automated Testing** (unit tests, data validation, model validation)
- **Code Quality Checks** (Black, Flake8, Pylint)
- **Real-time Metrics** (request rate, latency, error rate)
- **Pre-configured Dashboards** in Grafana

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Actions (CI/CD)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   Test   │  │ Validate │  │  Build   │  │  Deploy  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                     │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   FastAPI    │───▶│   MLflow     │    │  Prometheus  │ │
│  │   (Port      │    │   (Port      │◀───│  (Port       │ │
│  │    8000)     │    │    5000)     │    │   9090)      │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                   │                      │        │
│         │                   │                      ▼        │
│         │                   │             ┌──────────────┐ │
│         │                   │             │   Grafana    │ │
│         │                   │             │   (Port      │ │
│         │                   │             │    3000)     │ │
│         │                   │             └──────────────┘ │
│         ▼                   ▼                              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              Model Registry & Artifacts              │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Git

### 1. Clone and Navigate
```bash
cd Live-Model-Tracking
```

### 2. Train the Model
```bash
# Install dependencies
pip install -r requirements.txt

# Train model (this will register it in MLflow)
python train.py
```

### 3. Start All Services
```bash
docker-compose up -d
```

### 4. Access Services
- **API**: http://localhost:8000/docs (Swagger UI)
- **MLflow UI**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

---

## 🔧 Detailed Setup

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Preparation

Ensure `dataset-diabete.csv` is in the project root with these columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Step 3: Train and Register Model

```bash
python train.py
```

**What happens:**
1. ✅ Loads and preprocesses data
2. ✅ Trains Random Forest classifier
3. ✅ Logs parameters and metrics to MLflow
4. ✅ Registers model in MLflow Model Registry
5. ✅ Automatically promotes to Staging (if ROC-AUC > 0.85)
6. ✅ Automatically promotes to Production (if ROC-AUC > 0.90)

**Output:**
```
Training complete. Run tracked in MLflow.
Run ID: abc123...
Model Name: diabetes_risk_model
Model version 1 promoted to Production
```

### Step 4: Start Services with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

---

## 🎯 MLflow Model Registry

### Understanding Model Lifecycle

```
┌─────────┐       ┌─────────┐       ┌──────────────┐
│  None   │──────▶│ Staging │──────▶│  Production  │
└─────────┘       └─────────┘       └──────────────┘
   Initial          Testing           Live Serving
```

### Viewing Models in MLflow UI

1. Open http://localhost:5000
2. Click "Models" in the top navigation
3. Select "diabetes_risk_model"
4. View versions and stages

### Manual Model Promotion (Optional)

```python
from mlflow.tracking import MlflowClient

client = MlflowClient("http://localhost:5000")

# Promote to Production
client.transition_model_version_stage(
    name="diabetes_risk_model",
    version=1,
    stage="Production"
)
```

### How API Loads Model

The API automatically loads the **Production** version:
```python
model_uri = "models:/diabetes_risk_model/Production"
model = mlflow.sklearn.load_model(model_uri)
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

#### 1. **CI Workflow** (`.github/workflows/ci.yml`)

Triggered on: Push to `main`, `develop` or Pull Request

**Stages:**
1. ✅ **Code Quality**
   - Black formatting check
   - Flake8 linting
   - Pylint analysis

2. ✅ **Unit Tests**
   - Runs pytest with coverage
   - Uploads coverage to Codecov

3. ✅ **Data Validation**
   - Checks data schema
   - Validates data quality
   - Detects outliers

4. ✅ **Model Validation**
   - Evaluates model metrics
   - Ensures thresholds are met
   - Cross-validation checks

5. ✅ **Integration Tests**
   - Starts API server
   - Tests endpoints

#### 2. **Docker Build Workflow** (`.github/workflows/docker-build.yml`)

Triggered on: Push to `main` or tags

**Actions:**
- Builds Docker image
- Pushes to GitHub Container Registry
- Tags with version/SHA

#### 3. **Deployment Workflow** (`.github/workflows/deploy.yml`)

Triggered after: Successful CI and Docker builds

**Actions:**
- Deploys to production
- Runs health checks
- Sends notifications

### Running Validations Locally

```bash
# Data validation
python scripts/validate_data.py

# Model validation
python scripts/validate_model.py

# Run all tests
pytest -v

# Check code quality
black --check *.py
flake8 *.py
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics

The API automatically exposes metrics at `/metrics`:

**Key Metrics:**
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `http_requests_in_progress` - Concurrent requests

### Grafana Dashboard

**Pre-configured Dashboard:** "Diabetes Risk API - Monitoring Dashboard"

**Panels Include:**
1. 📈 **Request Rate** (requests/second)
2. ⏱️ **Response Time** (p50, p95, p99 percentiles)
3. ❌ **Error Rate** (4xx, 5xx)
4. 📊 **Requests by Endpoint**
5. ✅ **Success Rate** (%)
6. 🎯 **Total Request Rate**

**Accessing Grafana:**
1. Go to http://localhost:3000
2. Login: `admin` / `admin`
3. Navigate to **Dashboards** → **Diabetes Risk API**

### Setting Up Alerts (Optional)

Edit `grafana/dashboards/diabetes-api-dashboard.json` to add alert rules:

```json
{
  "alert": {
    "name": "High Error Rate",
    "conditions": [
      {
        "evaluator": {
          "params": [5],
          "type": "gt"
        },
        "query": {
          "model": "A"
        }
      }
    ]
  }
}
```

---

## 🌐 API Usage

### Swagger UI (Interactive)

Visit: http://localhost:8000/docs

### Example: Predict Diabetes Risk

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Glucose": 148,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 125,
    "Age": 50,
    "Pregnancies": 6
  }'
```

**Response:**
```json
{
  "risk_category": 1,
  "risk_probability": 0.87
}
```

- `risk_category`: 0 (Low Risk) or 1 (High Risk)
- `risk_probability`: Probability of high risk (0.0 - 1.0)

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

---

## 🔍 Troubleshooting

### Issue: MLflow Connection Error

**Symptom:** API can't connect to MLflow

**Solution:**
```bash
# Check MLflow is running
docker-compose ps mlflow

# Check MLflow logs
docker-compose logs mlflow

# Restart MLflow
docker-compose restart mlflow
```

### Issue: Model Not in Production

**Symptom:** API loads local pickle files instead of MLflow model

**Solution:**
1. Check model is in Production stage:
   ```bash
   # Open MLflow UI
   open http://localhost:5000
   ```

2. Manually promote:
   ```python
   from mlflow.tracking import MlflowClient
   client = MlflowClient("http://localhost:5000")
   client.transition_model_version_stage(
       name="diabetes_risk_model",
       version=1,
       stage="Production"
   )
   ```

### Issue: Grafana Dashboard Not Showing Data

**Symptom:** Grafana shows "No data"

**Solution:**
1. Check Prometheus is scraping:
   ```bash
   # Go to Prometheus targets
   open http://localhost:9090/targets
   ```

2. Verify API metrics:
   ```bash
   curl http://localhost:8000/metrics
   ```

3. Make some API requests to generate data:
   ```bash
   for i in {1..10}; do
     curl -X GET http://localhost:8000/health
   done
   ```

### Issue: Docker Compose Fails

**Symptom:** Services won't start

**Solution:**
```bash
# Stop all services
docker-compose down

# Remove volumes
docker-compose down -v

# Rebuild and restart
docker-compose up -d --build

# Check logs
docker-compose logs
```

### Issue: Port Already in Use

**Symptom:** "Port 8000 already in use"

**Solution:**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Change host port
```

---

## 📝 Project Structure

```
Live-Model-Tracking/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # CI pipeline
│       ├── docker-build.yml        # Docker build
│       └── deploy.yml              # Deployment
├── grafana/
│   ├── dashboards/
│   │   └── diabetes-api-dashboard.json
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboard.yml
│       └── datasources/
│           └── prometheus.yml
├── scripts/
│   ├── validate_data.py            # Data validation
│   └── validate_model.py           # Model validation
├── api.py                          # FastAPI application
├── app.py                          # Streamlit dashboard
├── train.py                        # Model training script
├── test_api.py                     # API unit tests
├── docker-compose.yml              # Service orchestration
├── Dockerfile                      # API container
├── requirements.txt                # Python dependencies
├── prometheus.yml                  # Prometheus config
└── MLOPS_GUIDE.md                  # This guide
```

---

## 🎓 Learning Resources

- **MLflow**: https://mlflow.org/docs/latest/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/
- **GitHub Actions**: https://docs.github.com/en/actions

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests: `pytest`
5. Run validation: `python scripts/validate_model.py`
6. Submit pull request

---

## 📄 License

This project is for educational purposes.

---

## 🆘 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review MLflow logs: `docker-compose logs mlflow`
3. Review API logs: `docker-compose logs api`
4. Check Grafana metrics for system health

---

**Happy MLOps! 🚀**
