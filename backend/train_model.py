import numpy as np
import pandas as pd
from data_handler import StockDataHandler
from model import StockPredictor
import yfinance as yf
from sklearn.model_selection import train_test_split

try:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except Exception:
    from keras.callbacks import EarlyStopping, ModelCheckpoint

def prepare_training_data():
    """Prépare les données d'entraînement"""
    handler = StockDataHandler()
    
    # Liste d'actions pour l'entraînement
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    
    X_data = []
    y_data = []
    
    for ticker in tickers:
        print(f"Récupération des données pour {ticker}...")
        df = handler.get_stock_data(ticker, period='5y')
        
        if df is None or len(df) < 200:
            continue
            
        df = handler.calculate_technical_indicators(df)
        
        if len(df) < 100:
            continue
            
        features = [
            'Close', 'Volume', 'RSI', 'MA_20', 'MA_50',
            'MACD', 'Signal', 'Volatility', 'Price_Change'
        ]
        
        data = df[features].values
        scaled = handler.scaler.fit_transform(data)
        
        # Créer des séquences
        for i in range(60, len(scaled) - 1):
            X_data.append(scaled[i-60:i])
            # Cible: 1 si le prix augmente le jour suivant, 0 sinon
            target = 1 if df['Close'].iloc[i+1] > df['Close'].iloc[i] else 0
            y_data.append(target)
    
    return np.array(X_data), np.array(y_data)

def train_model():
    """Entraîne le modèle"""
    print("🔍 Préparation des données d'entraînement...")
    X, y = prepare_training_data()
    
    if len(X) == 0:
        print("❌ Aucune donnée d'entraînement disponible")
        return False
    
    print(f"📊 Données d'entraînement: {X.shape[0]} échantillons")
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Construction du modèle
    predictor = StockPredictor(sequence_length=60)
    model = predictor.build_model(X_train.shape[2])
    
    print("🚀 Entraînement du modèle...")
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('stock_model.h5', save_best_only=True)
    ]
    
    # Entraînement
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluation
    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    
    print(f"✅ Entraînement terminé!")
    print(f"📈 Accuracy - Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    return True

if __name__ == "__main__":
    train_model()