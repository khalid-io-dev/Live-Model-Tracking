
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and artifacts
COPY api.py .
COPY final_model.pkl .
COPY imputer.pkl .
COPY scaler.pkl .
COPY feature_names.pkl .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
