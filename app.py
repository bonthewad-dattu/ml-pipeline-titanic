# app.py - Modified version
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load your trained model
print("🚀 Loading trained ML model...")
try:
    model_artifacts = joblib.load('best_model.pkl')
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    feature_names = model_artifacts['feature_names']
    
    # Handle both 'model_type' and 'model_name'
    if 'model_type' in model_artifacts:
        model_type = model_artifacts['model_type']
    elif 'model_name' in model_artifacts:
        model_type = model_artifacts['model_name']
    else:
        model_type = "Unknown"
    
    print(f"✅ Model loaded: {model_type}")
    print(f"📋 Features: {len(feature_names)}")
    MODEL_LOADED = True
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False
    # Initialize with empty values to avoid errors
    model = None
    scaler = None
    feature_names = []
    model_type = "Error"

def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    """Preprocess input to match training features"""
    if not MODEL_LOADED:
        raise Exception("Model not loaded")
    
    # Create a DataFrame with all expected features, initialized to 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Set basic features
    input_df['Sex'] = 1 if sex == 'female' else 0
    input_df['Age'] = age
    input_df['SibSp'] = sibsp
    input_df['Parch'] = parch
    input_df['Fare'] = fare
    
    # Create engineered features
    input_df['FamilySize'] = sibsp + parch + 1
    input_df['IsAlone'] = 1 if input_df['FamilySize'].iloc[0] == 1 else 0
    
    # Set HasCabin (default to 0)
    input_df['HasCabin'] = 0
    
    # Set Title (default to 2 which is 'Mr' in encoding)
    input_df['Title'] = 2
    
    # Set one-hot encoded features
    # Embarked
    if embarked == 'C':
        input_df['Embarked_C'] = 1
    elif embarked == 'Q':
        input_df['Embarked_Q'] = 1
    else:
        input_df['Embarked_S'] = 1
    
    # Age Group
    if age <= 12:
        input_df['AgeGroup_Child'] = 1
    elif age <= 18:
        input_df['AgeGroup_Teen'] = 1
    elif age <= 35:
        input_df['AgeGroup_Young Adult'] = 1
    elif age <= 60:
        input_df['AgeGroup_Adult'] = 1
    else:
        input_df['AgeGroup_Senior'] = 1
    
    # Fare Group
    if fare <= 7.91:
        input_df['FareGroup_Low'] = 1
    elif fare <= 14.45:
        input_df['FareGroup_Medium'] = 1
    elif fare <= 31.0:
        input_df['FareGroup_High'] = 1
    else:
        input_df['FareGroup_Very High'] = 1
    
    # Pclass
    if pclass == 1:
        input_df['Pclass_1'] = 1
    elif pclass == 2:
        input_df['Pclass_2'] = 1
    else:
        input_df['Pclass_3'] = 1
    
    # Scale numerical features
    numerical_columns = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    if scaler and len(numerical_columns) > 0:
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
    
    return input_df

@app.route('/')
def home():
    model_status = "✅ TRAINED MODEL ACTIVE" if MODEL_LOADED else "⚠️ MODEL LOADING ERROR"
    accuracy = model_artifacts.get('accuracy', 0.6536) if MODEL_LOADED else 0.0
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Titanic Survival Predictor</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; }}
            .form-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }}
            .form-group {{ margin-bottom: 15px; }}
            label {{ display: block; margin-bottom: 5px; font-weight: bold; color: #34495e; }}
            input, select {{ width: 100%; padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px; font-size: 16px; }}
            button {{ background: #3498db; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 18px; width: 100%; }}
            button:hover {{ background: #2980b9; }}
            .model-info {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚢 Titanic Survival Prediction</h1>
            <p>Complete ML Pipeline - Trained Model Deployment</p>
            
            <div class="model-info">
                <h3>Model Information:</h3>
                <p><strong>Status:</strong> {model_status}</p>
                <p><strong>Algorithm:</strong> {model_type}</p>
                <p><strong>Accuracy:</strong> {accuracy:.2%}</p>
                <p><strong>Features:</strong> {len(feature_names)} engineered features</p>
            </div>
            
            <form action="/predict" method="post">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Passenger Class:</label>
                        <select name="pclass" required>
                            <option value="1">1st Class</option>
                            <option value="2">2nd Class</option>
                            <option value="3" selected>3rd Class</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Gender:</label>
                        <select name="sex" required>
                            <option value="male" selected>Male</option>
                            <option value="female">Female</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Age:</label>
                        <input type="number" name="age" value="25" min="0" max="100" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Siblings/Spouses:</label>
                        <input type="number" name="sibsp" value="1" min="0" max="10" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Parents/Children:</label>
                        <input type="number" name="parch" value="2" min="0" max="10" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Fare (£):</label>
                        <input type="number" name="fare" value="30" step="0.01" min="0" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Embarkation Port:</label>
                        <select name="embarked" required>
                            <option value="C">Cherbourg</option>
                            <option value="Q">Queenstown</option>
                            <option value="S" selected>Southampton</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit">Predict Survival</button>
            </form>
            
            <div style="text-align: center; margin-top: 20px;">
                <a href="/health" style="margin-right: 15px;">Health Check</a>
                <a href="/api/info">API Documentation</a>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not MODEL_LOADED:
            return '''
            <div style="max-width: 600px; margin: 40px auto; padding: 20px; background: #f8d7da; color: #721c24; border-radius: 5px;">
                <h1>Model Not Loaded</h1>
                <p>Please train the model first by running: <code>python train_model.py</code></p>
                <button onclick="window.history.back()">Go Back</button>
            </div>
            '''
            
        # Get form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        
        # Preprocess input
        input_features = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]
        
        # Prepare result
        result = "SURVIVED 🎉" if prediction == 1 else "DID NOT SURVIVE 😢"
        confidence = f"{probability:.2%}"
        
        status_class = "survived" if prediction == 1 else "not-survived"
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .result {{ margin: 20px 0; padding: 30px; border-radius: 10px; text-align: center; }}
                .survived {{ background: #d4edda; color: #155724; border: 3px solid #c3e6cb; }}
                .not-survived {{ background: #f8d7da; color: #721c24; border: 3px solid #f5c6cb; }}
                button {{ background: #3498db; color: white; padding: 12px 25px; border: none; border-radius: 5px; cursor: pointer; margin: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prediction Result</h1>
                <div class="result {status_class}">
                    <h2>{result}</h2>
                    <p><strong>Confidence:</strong> {confidence}</p>
                    <p><strong>Probability:</strong> {probability:.4f}</p>
                </div>
                <div style="text-align: center;">
                    <button onclick="window.location.href='/'">Make Another Prediction</button>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        return f'''
        <div style="max-width: 600px; margin: 40px auto; padding: 20px; background: #f8d7da; color: #721c24; border-radius: 5px;">
            <h1>Error</h1>
            <p><strong>Error Details:</strong> {str(e)}</p>
            <button onclick="window.history.back()">Go Back</button>
        </div>
        '''

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_type": model_type,
        "accuracy": model_artifacts.get('accuracy', 0.6536) if MODEL_LOADED else 0.0,
        "features": len(feature_names),
        "endpoints": ["GET /", "POST /predict", "GET /health", "GET /api/info"]
    })

@app.route('/api/info', methods=['GET'])
def api_info():
    return jsonify({
        "api_name": "Titanic Survival Prediction API",
        "model": model_type,
        "accuracy": model_artifacts.get('accuracy', 0.6536) if MODEL_LOADED else 0.0,
        "features": len(feature_names),
        "input_format": {
            "pclass": "int (1, 2, 3)",
            "sex": "string (male, female)",
            "age": "float",
            "sibsp": "int",
            "parch": "int",
            "fare": "float",
            "embarked": "string (C, Q, S)"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)