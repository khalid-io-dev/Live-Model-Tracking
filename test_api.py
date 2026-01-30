
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
    # Mock artifacts and model prediction
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    
    mock_imputer = MagicMock()
    mock_imputer.transform.return_value = [[1,2,3,4,5,6,7,8]]
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[1,2,3,4,5,6,7,8]]

    artifacts = {
        "model": mock_model,
        "imputer": mock_imputer,
        "scaler": mock_scaler,
        "feature_names": ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age', 'Pregnancies']
    }

    with patch('api.ARTIFACTS', artifacts):
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
        assert response.status_code == 200
        assert response.json()["risk_category"] == 1
        assert response.json()["risk_probability"] == 0.8
