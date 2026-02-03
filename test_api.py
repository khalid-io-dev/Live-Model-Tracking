
from fastapi.testclient import TestClient
from api import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_health():
    # Mock artifacts to simulate loaded state
    with patch('api.ARTIFACTS', {"model": "loaded"}):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    import numpy as np
    
    # Mock artifacts and model prediction
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    mock_imputer = MagicMock()
    mock_imputer.transform.return_value = np.array([[1, 100, 80, 20, 100, 30, 0.5, 40]])
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    artifacts = {
        "model": mock_model,
        "imputer": mock_imputer,
        "scaler": mock_scaler,
        "feature_names": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    }

    with patch.dict('api.ARTIFACTS', artifacts, clear=True):
        payload = {
            "Glucose": 100,
            "BMI": 30,
            "DiabetesPedigreeFunction": 0.5,
            "BloodPressure": 80,
            "SkinThickness": 20,
            "Insulin": 100,
            "Age": 40,
            "Pregnancies": 1
        }
        response = client.post("/predict", json=payload)
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.json()}")
        assert response.status_code == 200
        assert response.json()["risk_category"] == 1
        assert abs(response.json()["risk_probability"] - 0.8) < 0.001
