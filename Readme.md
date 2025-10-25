# HealthMarteAI - BMI Prediction System

## Purpose

This repository contains a Flask-based BMI prediction service with machine learning and AI chatbot capabilities.

## Big Picture

- **Main Application**: `app.py` (Flask) loads a pre-trained scikit-learn RandomForest model and exposes REST endpoints on port 5000
- **Data Source**: Canonical dataset is `bmi.csv` for training/retraining
- **AI Chatbot**: Optional Google Gemini-powered chatbot for health advice (requires API key)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv env

# Activate virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

# Install required packages
pip install flask flask-cors pandas numpy scikit-learn joblib requests python-dotenv
```

### 2. Configure Gemini API (Optional - for Chatbot)

1. Get your API key from: https://makersuite.google.com/app/apikey
2. Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

**Note**: If you don't set up the API key, the app will still work but the chatbot will be disabled.

### 3. Run the Application

```bash
python app.py
```

The server will start at: http://localhost:5000

### 4. Open Web Interface

Open `bmi_web_interface.html` in your browser to use the application.

## 📁 Key Files

- **`app.py`** — Main Flask application with all endpoints and ML logic
- **`bmi.csv`** — Training dataset (Age, Height, Weight, BmiClass)
- **`bmi_model.joblib`** — Trained RandomForest model (auto-generated)
- **`bmi_scaler.joblib`** — StandardScaler for feature normalization (auto-generated)
- **`bmi_web_interface.html`** — Web UI for predictions and chatbot
- **`bmi.js`** — Frontend JavaScript logic
- **`style.css`** — Styling for web interface
- **`.env`** — Environment variables (create from `.env.example`)

## 🔧 API Endpoints

### GET `/`
Returns service information and available endpoints

### GET `/health`
Health check - returns model status and chatbot availability

### POST `/predict`
Predict BMI classification

**Request Body:**
```json
{
  "age": 25,
  "height": 1.75,
  "weight": 70
}
```

**Validation Rules:**
- Age: 1-120 years
- Height: 0.5-2.5 meters
- Weight: 10-500 kg

**Response:**
```json
{
  "success": true,
  "calculated_bmi": 22.86,
  "predicted_class": "Normal Weight",
  "confidence": 98.5,
  "all_probabilities": {...},
  "top_predictions": [...]
}
```

### POST `/chatbot`
Chat with AI assistant (requires GEMINI_API_KEY)

**Request Body:**
```json
{
  "message": "What is a healthy BMI range?"
}
```

### POST `/retrain`
Retrain the model using `bmi.csv`

### GET `/model-info`
Get information about the trained model

## 🎯 Features

### BMI Prediction
- **Machine Learning**: RandomForest Classifier with 100 estimators
- **Features**: Age, Height, Weight
- **Classes**: Underweight, Normal Weight, Overweight, Obese Class 1-3
- **Accuracy**: Typically 92%+ on test data

### AI Chatbot
- Powered by Google Gemini API
- Provides health, nutrition, and fitness advice
- Context-aware responses about BMI and wellness

### Web Interface
- Real-time API status indicator
- Interactive BMI prediction form
- Confidence scores and probability distribution
- Integrated chatbot with conversation history
- Responsive design for mobile and desktop

## 🔄 Model Training

The model automatically trains on first run if joblib files don't exist. To manually retrain:

**Via API:**
```bash
curl -X POST http://localhost:5000/retrain
```

**Via Python:**
```python
# In app.py, the train_model() function is called automatically
```

## 🐛 Troubleshooting

### Chatbot Not Working
1. Verify `GEMINI_API_KEY` is set in `.env` file
2. Check API key validity at https://makersuite.google.com/
3. Restart the Flask server after adding the key
4. Check console for error messages

### Model Not Loading
1. Ensure `bmi.csv` exists in project root
2. Check CSV has required columns: Age, Height, Weight, BmiClass
3. Delete `.joblib` files and restart to retrain

### CORS Errors
- Ensure Flask server is running on http://localhost:5000
- Check that flask-cors is installed
- Update `API_BASE_URL` in `bmi.js` if using different port

## 📊 Data Format

`bmi.csv` should have these columns:
- **Age**: Integer (years)
- **Height**: Float (meters)
- **Weight**: Float (kilograms)
- **BmiClass**: String (category label)

## 🎨 Customization

### Changing Model Parameters
Edit in `app.py`:
```python
model = RandomForestClassifier(
    random_state=42,
    n_estimators=100  # Adjust this
)
```

### Updating Validation Rules
Edit in `app.py` `/predict` endpoint:
```python
if not (0 < age <= 120):  # Modify these
    return jsonify({"error": "Age must be between 1 and 120"}), 400
```

### Styling Changes
Modify `style.css` for colors, fonts, and layout

## 🔒 Security Notes

- Never commit `.env` file to version control
- Use environment variables for sensitive data
- In production, set `debug=False` in `app.run()`
- Consider adding rate limiting for API endpoints
- Implement proper authentication for production use

## 📝 Development Workflow

1. Make changes to code
2. Test endpoints with Postman or curl
3. Test UI in browser
4. If changing features, update training code and retrain
5. Keep client/server validation aligned

## 🤝 Contributing

When making changes:
1. Update `app.py` only where necessary
2. Keep error response format consistent: `{'error': 'message'}`
3. Update README if adding new features
4. Test all endpoints after changes

## 📄 License

[Add your license here]

## 👥 Authors

[Add your information here]

## 🙏 Acknowledgments

- scikit-learn for ML framework
- Google Gemini for AI capabilities
- Flask for web framework