# 🏥 Diabetes Prediction API

## Deployed by Sheharyar Khan
**AI-Powered Diabetes Risk Assessment System**

### Model Details
- **Algorithm:** XGBoost
- **Accuracy:** 97%
- **ROC-AUC:** 98%

### API Endpoints

#### 1. Home
```
GET /
```
Returns API information and usage instructions

#### 2. Health Check
```
GET /health
```
Check if API is running and model is loaded

#### 3. Model Info
```
GET /model-info
```
Get model details and accuracy

#### 4. Prediction
```
POST /predict
Content-Type: application/json

{
    "gender": "Female",
    "age": 45,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 28.5,
    "HbA1c_level": 6.5,
    "blood_glucose_level": 140
}
```

### Response Format
```json
{
    "prediction": "Diabetes" or "No Diabetes",
    "probability": {
        "no_diabetes": "45.3%",
        "diabetes": "54.7%"
    },
    "risk_level": "Low/Moderate/High",
    "recommendations": [...]
}
```

### Technology Stack
- **Backend:** Flask
- **ML Framework:** XGBoost, scikit-learn
- **Deployment:** Render.com
- **Language:** Python 3.10

### Features
✅ Fast predictions (2-5 seconds)
✅ Real-time risk assessment
✅ Personalized health recommendations
✅ RESTful API design
✅ CORS enabled
✅ Error handling

### Usage Example

**Python:**
```python
import requests

url = "https://your-api-url/predict"
data = {
    "gender": "Female",
    "age": 45,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 28.5,
    "HbA1c_level": 6.5,
    "blood_glucose_level": 140
}

response = requests.post(url, json=data)
print(response.json())
```

**cURL:**
```bash
curl -X POST https://your-api-url/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "age": 45,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 28.5,
    "HbA1c_level": 6.5,
    "blood_glucose_level": 140
  }'
```

### Deployment on Render

1. Push code to GitHub
2. Connect Render to repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`
5. Deploy!

### Files Included
- `app.py` - Main Flask application
- `requirements.txt` - Python dependencies
- `diabetes_model.pkl` - Trained XGBoost model
- `scaler.pkl` - Feature scaler
- `le_gender.pkl` - Gender label encoder
- `le_smoking.pkl` - Smoking history encoder
- `model_metadata.pkl` - Model information

### License
Educational/Research Project - 2024

### Contact
**Developer:** Sheharyar Khan
**Project:** Diabetes Prediction System
**Course:** Data Science with Python
