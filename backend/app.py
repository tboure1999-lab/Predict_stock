from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from data_handler import StockDataHandler
from model import StockPredictor
import os

app = Flask(__name__)
CORS(app)

data_handler = StockDataHandler()
predictor = StockPredictor(sequence_length=60)

# Charger le modèle si disponible
if os.path.exists('stock_model.h5'):
    loaded = predictor.load_model()
    if loaded:
        print("✓ Modèle chargé avec succès")
    else:
        print("⚠️ Le fichier 'stock_model.h5' est présent mais n'a pas pu être chargé (fichier corrompu ou format incompatible).")
        print("   Lance 'train_model.py' pour régénérer le modèle ou remplace le fichier par un modèle valide.")
else:
    print("⚠️ Aucun modèle trouvé. Lance train_model.py pour l'entraîner")

@app.route('/')
def home():
    # If a frontend index exists in the parent `frontend` folder, serve it.
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
    index_path = os.path.join(frontend_dir, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(frontend_dir, 'index.html')

    return jsonify({
        'message': 'API de Prédiction Boursière',
        'version': '1.0'
    })

@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock_info(ticker):
    try:
        ticker = ticker.upper()
        info = data_handler.get_stock_info(ticker)

        if info['current_price'] is None:
            return jsonify({'error': f'Action {ticker} non trouvée'}), 404

        return jsonify({'ticker': ticker, 'info': info})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'ticker' not in data:
            return jsonify({'error': 'Veuillez fournir un ticker'}), 400

        ticker = data['ticker'].upper()

        if predictor.model is None:
            return jsonify({'error': 'Modèle non disponible'}), 503

        df = data_handler.get_stock_data(ticker, period='1y')

        if df is None or len(df) < 100:
            return jsonify({'error': f'Données insuffisantes pour {ticker}'}), 404

        df = data_handler.calculate_technical_indicators(df)
        X = data_handler.get_latest_sequence(df, 60)
        prediction = predictor.predict_with_confidence(X)
        current_price = data_handler.get_current_price(ticker)

        return jsonify({
            'ticker': ticker,
            'current_price': current_price,
            'prediction': prediction,
            'timestamp': df.index[-1].strftime('%Y-%m-%d')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("\n🚀 Serveur API lancé → http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
