from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load all 10 models and scalers into dictionaries
models = {
    f'stock_{i}': joblib.load(f'models/stock_{i}_model.pkl') for i in range(1, 11)
}
scalers = {
    f'stock_{i}': joblib.load(f'models/stock_{i}_scaler.pkl') for i in range(1, 11)
}

# Route to serve the index.html page
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receive input data in JSON format
    
    # Ensure stock_name is provided in the request
    stock_name = data.get('stock_name')
    if not stock_name:
        return jsonify({'error': 'Please provide a stock_name in the request data'}), 400
    
    if stock_name not in models:
        return jsonify({'error': f'Invalid stock_name: {stock_name}. Valid names are stock_1 to stock_10'}), 400
    
    # Extract features for prediction (make sure the data format matches your features)
    df = pd.DataFrame(data['features'])
    
    # Check feature columns
    expected_features = ['Open', 'High', 'Low', 'Volume', 'lag_1', 'lag_2', 'lag_3',
                         'MA_10', 'MA_50', 'RSI', 'Volatility', 'MACD', 'MACD_Signal',
                         'Bollinger_Upper', 'Bollinger_Lower']
    
    if not all(feature in df.columns for feature in expected_features):
        return jsonify({'error': 'Feature names do not match expected training features'}), 400
    
    # Load appropriate model and scaler
    model = models[stock_name]
    scaler = scalers[stock_name]
    
    # Preprocess the data using the scaler
    X_scaled = scaler.transform(df[expected_features])  # Make sure to only pass the expected features
    
    # Make prediction
    predictions = model.predict(X_scaled)
    
    # Return the result as JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
