import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the api module to patch it
import api


def test_health():
    """Test health endpoint with mocked artifacts"""
    # Mock artifacts before creating client
    mock_artifacts = {"model": "loaded"}
    
    with patch.object(api, 'ARTIFACTS', mock_artifacts):
        client = TestClient(api.app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_predict_endpoint():
    """Test predict endpoint with fully mocked artifacts"""
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

    # Patch ARTIFACTS and prevent load_artifacts from running
    with patch.object(api, 'ARTIFACTS', artifacts):
        with patch.object(api, 'load_artifacts'):
            client = TestClient(api.app)
            
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
            
            # Debug output if test fails
            if response.status_code != 200:
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.text}")
            
            assert response.status_code == 200
            result = response.json()
            assert result["risk_category"] == 1
            assert abs(result["risk_probability"] - 0.8) < 0.001
