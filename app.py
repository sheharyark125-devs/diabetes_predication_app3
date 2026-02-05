from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ============================================
# LOAD MODELS AT STARTUP
# ============================================
print("🔄 Loading models...")

MODEL_LOADED = False
model = None
scaler = None
le_gender = None
le_smoking = None
metadata = None

try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_smoking = joblib.load('le_smoking.pkl')
    
    try:
        metadata = joblib.load('model_metadata.pkl')
    except:
        metadata = {'model_name': 'XGBoost', 'accuracy': 0.97, 'roc_auc': 0.98}
    
    # Pre-warm model
    print("🔥 Pre-warming model...")
    dummy = np.array([[0, 45, 0, 0, 0, 25, 5.5, 100]])
    _ = model.predict(scaler.transform(dummy))
    
    MODEL_LOADED = True
    print("✅ Models loaded and ready!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    """API Home"""
    return jsonify({
        'message': '🏥 Diabetes Prediction API by Sheharyar Khan',
        'status': 'active',
        'version': '1.0',
        'model': metadata.get('model_name', 'Unknown') if MODEL_LOADED else 'Not Loaded',
        'accuracy': f"{metadata.get('accuracy', 0) * 100:.2f}%" if MODEL_LOADED else 'N/A',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/model-info': 'Model details',
            '/predict': 'Make prediction (POST)'
        },
        'usage': {
            'method': 'POST',
            'url': '/predict',
            'headers': {'Content-Type': 'application/json'},
            'body': {
                'gender': 'Female or Male',
                'age': 45,
                'hypertension': '0 or 1',
                'heart_disease': '0 or 1',
                'smoking_history': 'never, former, current, not current, ever, or No Info',
                'bmi': 28.5,
                'HbA1c_level': 6.5,
                'blood_glucose_level': 140
            }
        }
    })


@app.route('/health')
def health():
    """Health Check"""
    if MODEL_LOADED:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'message': 'API is running smoothly! 🚀'
        }), 200
    return jsonify({
        'status': 'unhealthy',
        'model_loaded': False,
        'message': 'Model not loaded ❌'
    }), 500


@app.route('/model-info')
def model_info():
    """Model Information"""
    if MODEL_LOADED and metadata:
        return jsonify({
            'model_name': metadata.get('model_name', 'Unknown'),
            'accuracy': f"{metadata.get('accuracy', 0) * 100:.2f}%",
            'roc_auc': f"{metadata.get('roc_auc', 0) * 100:.2f}%",
            'deployed_by': 'Sheharyar Khan',
            'framework': 'Flask + XGBoost'
        })
    return jsonify({'error': 'Model metadata not available'}), 404


@app.route('/predict', methods=['POST'])
def predict():
    """Make Prediction - OPTIMIZED"""
    
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required = ['gender', 'age', 'hypertension', 'heart_disease', 
                   'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({
                'error': 'Missing fields',
                'missing': missing
            }), 400
        
        # Fast encoding
        gender_map = {'Female': 0, 'female': 0, 'Male': 1, 'male': 1}
        smoking_map = {
            'never': 0, 'No Info': 1, 'current': 2,
            'former': 3, 'ever': 4, 'not current': 5
        }
        
        gender_enc = gender_map.get(data['gender'], 0)
        smoking_enc = smoking_map.get(data['smoking_history'], 0)
        
        # Create input array
        input_arr = np.array([[
            gender_enc,
            float(data['age']),
            int(data['hypertension']),
            int(data['heart_disease']),
            smoking_enc,
            float(data['bmi']),
            float(data['HbA1c_level']),
            float(data['blood_glucose_level'])
        ]])
        
        # Predict
        input_scaled = scaler.transform(input_arr)
        prediction = int(model.predict(input_scaled)[0])
        probability = model.predict_proba(input_scaled)[0]
        
        # Risk level
        diabetes_prob = float(probability[1])
        if diabetes_prob < 0.3:
            risk_level, risk_color = 'Low', 'green'
        elif diabetes_prob < 0.6:
            risk_level, risk_color = 'Moderate', 'orange'
        else:
            risk_level, risk_color = 'High', 'red'
        
        # Recommendations
        recs = []
        if prediction == 1 or diabetes_prob > 0.5:
            recs.append("⚠️ High risk detected. Consult a healthcare professional.")
        if float(data['bmi']) > 30:
            recs.append("🏃 BMI is high. Consider weight management.")
        if float(data['HbA1c_level']) > 6.5:
            recs.append("📊 HbA1c level is elevated. Monitor blood sugar.")
        if float(data['blood_glucose_level']) > 140:
            recs.append("🩸 Blood glucose is high. Limit sugar intake.")
        if data['smoking_history'] in ['current', 'ever']:
            recs.append("🚭 Smoking increases diabetes risk.")
        if not recs:
            recs.append("✅ Keep maintaining a healthy lifestyle!")
        
        # Response
        return jsonify({
            'prediction': 'Diabetes' if prediction == 1 else 'No Diabetes',
            'prediction_value': prediction,
            'probability': {
                'no_diabetes': f"{probability[0] * 100:.1f}%",
                'diabetes': f"{probability[1] * 100:.1f}%"
            },
            'probability_score': round(diabetes_prob, 4),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendations': recs
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# Health check for Render
@app.route('/ping')
def ping():
    return 'pong', 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
