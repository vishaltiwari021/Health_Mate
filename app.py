from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import requests
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Google Gemini API Configuration (optional) ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}" if GEMINI_API_KEY else None

# Global variables for model and scaler
model = None
scaler = None
model_accuracy = None

def get_bmi_class(bmi):
    """Standard BMI classification"""
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal Weight'
    elif bmi < 30:
        return 'Overweight'
    elif bmi < 35:
        return 'Obese Class 1'
    elif bmi < 40:
        return 'Obese Class 2'
    else:
        return 'Obese Class 3'

def train_model():
    """Train the BMI prediction model"""
    global model, scaler, model_accuracy

    try:
        if not os.path.exists('bmi.csv'):
            msg = "Error: bmi.csv not found!"
            print(msg)
            return False

        df = pd.read_csv('bmi.csv')

        required_columns = ['Age', 'Height', 'Weight', 'BmiClass']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain columns: {required_columns}")
            return False

        # Drop rows with missing values in required columns
        df = df.dropna(subset=required_columns)

        X = df[['Age', 'Height', 'Weight']]
        y = df['BmiClass']

        # Basic sanity checks
        if X.shape[0] < 10:
            print("Warning: Very few rows to train the model. Need more data for reliable predictions.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train_scaled, y_train)

        model_accuracy = model.score(X_test_scaled, y_test)

        joblib.dump(model, 'bmi_model.joblib')
        joblib.dump(scaler, 'bmi_scaler.joblib')

        print(f"Model trained successfully! Accuracy: {model_accuracy:.4f}")
        return True

    except Exception as e:
        print("Error training model:", str(e))
        traceback.print_exc()
        return False

def load_model():
    """Load pre-trained model and scaler"""
    global model, scaler, model_accuracy

    try:
        if os.path.exists('bmi_model.joblib') and os.path.exists('bmi_scaler.joblib'):
            model = joblib.load('bmi_model.joblib')
            scaler = joblib.load('bmi_scaler.joblib')

            # Try to compute accuracy if bmi.csv exists; otherwise keep a placeholder
            try:
                if os.path.exists('bmi.csv'):
                    df = pd.read_csv('bmi.csv').dropna(subset=['Age', 'Height', 'Weight', 'BmiClass'])
                    if not df.empty:
                        X = df[['Age', 'Height', 'Weight']]
                        y = df['BmiClass']
                        X_scaled = scaler.transform(X)
                        model_accuracy = model.score(X_scaled, y)
                    else:
                        model_accuracy = None
                else:
                    model_accuracy = None
            except Exception:
                model_accuracy = None

            print("Model loaded successfully!")
            return True
        else:
            print("Pre-trained model not found. Training new model...")
            return train_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        # Try training a new model as fallback
        return train_model()

# --- Google Gemini API Function ---
def ask_gemini(user_message):
    """Query Google Gemini API (returns a string). If GEMINI_API_KEY is not set, returns helpful message."""
    if not GEMINI_API_URL or not GEMINI_API_KEY:
        return ("Gemini API not configured on server. To enable chatbot functionality, "
                "set the GEMINI_API_KEY environment variable and restart the server.")

    try:
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"You are a helpful BMI and health assistant. Provide accurate, friendly advice about BMI, nutrition, fitness, and general health topics. Keep responses concise (under 150 words).\n\nUser: {user_message}\nAssistant:"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500,
                "topP": 0.8,
                "topK": 40
            }
        }

        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)

        # handle response
        if response.status_code == 200:
            data = response.json()
            # try different known shapes
            content = None

            # old/new API sometimes returns 'candidates' or 'outputs' or 'output'
            candidates = None
            if isinstance(data, dict):
                candidates = data.get('candidates') or data.get('outputs') or data.get('output') or data.get('responses')

            if candidates and isinstance(candidates, list) and len(candidates) > 0:
                first = candidates[0]
                # nested content structure
                if isinstance(first, dict):
                    # try nested content -> parts -> text
                    content_list = first.get('content') or first.get('parts') or first.get('text')
                    if isinstance(content_list, list) and len(content_list) > 0:
                        first_part = content_list[0]
                        if isinstance(first_part, dict):
                            content = first_part.get('text') or first_part.get('content')
                        elif isinstance(first_part, str):
                            content = first_part
                    elif isinstance(content_list, str):
                        content = content_list
                    # as fallback look for 'text' or 'output'
                    if content is None:
                        content = first.get('text') or first.get('output') or first.get('response')
                elif isinstance(first, str):
                    content = first

            # final fallback: maybe data has 'candidates' with expected nested shape used originally
            if content is None:
                # attempt the original structure seen in some examples
                try:
                    content = data['candidates'][0]['content']['parts'][0]['text']
                except Exception:
                    content = None

            if content:
                return content.strip()
            else:
                return "I couldn't generate a response. Please try again."

        else:
            # non-200: parse error if possible
            try:
                error_data = response.json()
            except Exception:
                error_data = {}

            error_message = error_data.get('error', {}).get('message') if isinstance(error_data, dict) else None
            print(f"Gemini API Error: {response.status_code} - {error_message or response.text}")

            if response.status_code == 400:
                return "Invalid API request. Please check your Gemini API configuration."
            elif response.status_code == 401 or response.status_code == 403:
                return "Authentication error with Gemini API. Please verify your API key/permissions."
            elif response.status_code == 429:
                return "Rate limit exceeded. Please try again in a moment."
            else:
                return f"Sorry, I encountered an issue contacting Gemini API: {error_message or response.text}"

    except requests.exceptions.Timeout:
        return "The request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        print(f"Network error while calling Gemini API: {e}")
        return "I'm having trouble connecting to the Gemini API. Please check the server's network."
    except Exception as e:
        print("Unexpected error in ask_gemini:", str(e))
        traceback.print_exc()
        return "An unexpected error occurred while generating a reply. Please try again."

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "BMI Prediction API",
        "version": "1.0",
        "status": "active",
        "model_accuracy": round(model_accuracy * 100, 2) if model_accuracy is not None else None,
        "chatbot_enabled": GEMINI_API_KEY is not None,
        "endpoints": {
            "/predict": "POST - Predict BMI class",
            "/chatbot": "POST - Chat with AI assistant (disabled if GEMINI_API_KEY not provided)",
            "/model-info": "GET - Get model information",
            "/health": "GET - Health check",
            "/retrain": "POST - Retrain the model from bmi.csv"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_accuracy": round(model_accuracy * 100, 2) if model_accuracy is not None else None,
        "chatbot_enabled": GEMINI_API_KEY is not None
    })

@app.route('/predict', methods=['POST'])
def predict_bmi():
    """Predict BMI class based on age, height, and weight"""
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded. Please restart the server or POST to /retrain to train the model."}), 503

        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['age', 'height', 'weight']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Validate type conversion
        try:
            age = float(data['age'])
            height = float(data['height'])
            weight = float(data['weight'])
        except (ValueError, TypeError):
            return jsonify({"error": "Age, height and weight must be numbers"}), 400

        if not (0 < age <= 120):
            return jsonify({"error": "Age must be between 1 and 120"}), 400
        if not (0.5 <= height <= 2.5):
            return jsonify({"error": "Height must be between 0.5 and 2.5 meters"}), 400
        if not (10 <= weight <= 500):
            return jsonify({"error": "Weight must be between 10 and 500 kg"}), 400

        actual_bmi = weight / (height ** 2)
        actual_class = get_bmi_class(actual_bmi)

        input_data = np.array([[age, height, weight]])
        input_scaled = scaler.transform(input_data)

        predicted_class = model.predict(input_scaled)[0]

        # If the model supports predict_proba, use it; otherwise assign 100% to predicted class.
        try:
            confidence_scores = model.predict_proba(input_scaled)[0]
            class_names = model.classes_
            confidence_dict = {}
            for i, class_name in enumerate(class_names):
                confidence_dict[class_name] = float(confidence_scores[i])
        except Exception:
            # fallback
            class_names = getattr(model, "classes_", [])
            confidence_dict = {str(c): (1.0 if str(c) == str(predicted_class) else 0.0) for c in class_names}

        top_predictions = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)[:3]

        return jsonify({
            "success": True,
            "input": {"age": age, "height": height, "weight": weight},
            "calculated_bmi": round(actual_bmi, 2),
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "confidence": round(confidence_dict.get(predicted_class, 0.0) * 100, 2),
            "all_probabilities": {k: round(v * 100, 2) for k, v in confidence_dict.items()},
            "top_predictions": [(class_name, round(prob * 100, 2)) for class_name, prob in top_predictions],
            "model_accuracy": round(model_accuracy * 100, 2) if model_accuracy is not None else None
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400
    except Exception as e:
        print("Prediction error:", str(e))
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/model-info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    classes = []
    try:
        classes = model.classes_.tolist() if hasattr(model, "classes_") else []
    except Exception:
        classes = []

    return jsonify({
        "model_type": "Random Forest Classifier",
        "features": ["Age", "Height", "Weight"],
        "classes": classes,
        "accuracy": round(model_accuracy * 100, 2) if model_accuracy is not None else None,
        "training_info": {
            "algorithm": "Random Forest",
            "n_estimators": 100,
            "test_size": "20%",
            "random_state": 42
        }
    })

@app.route('/retrain', methods=['POST'])
def retrain():
    """Endpoint to retrain the model from bmi.csv (useful during development)."""
    try:
        success = train_model()
        if success:
            return jsonify({"success": True, "message": "Model retrained and saved.", "accuracy": round(model_accuracy * 100, 2) if model_accuracy is not None else None})
        else:
            return jsonify({"success": False, "message": "Retraining failed. Check server logs."}), 500
    except Exception as e:
        print("Retrain endpoint error:", str(e))
        traceback.print_exc()
        return jsonify({"error": "Retraining failed"}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Chatbot endpoint using Google Gemini API"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # Get response from Gemini (or an explanation if not configured)
        reply = ask_gemini(user_message)

        return jsonify({
            'success': True,
            'reply': reply,
            'user_message': user_message
        })

    except Exception as e:
        print("Chatbot endpoint error:", str(e))
        traceback.print_exc()
        return jsonify({'error': f'Chatbot error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    # In production you might not want to reveal details.
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Initializing BMI Prediction API...")
    print("=" * 50)

    if load_model():
        print("\n✓ Model loaded successfully!")
        if model_accuracy is not None:
            print(f"✓ Model Accuracy: {model_accuracy * 100:.2f}%")
        if GEMINI_API_KEY:
            print("✓ Gemini API configured - Chatbot enabled")
        else:
            print("⚠ Gemini API not configured - Chatbot disabled")
            print("  To enable: Set GEMINI_API_KEY in .env file")
        print("\nStarting Flask server...")
        print("API available at: http://localhost:5000")
        print("=" * 50)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load/train model.")
        print("Please ensure bmi.csv is available in the same directory.")
        print("=" * 50)