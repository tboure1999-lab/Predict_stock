try:
    # Try TensorFlow 2.x style import
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except ImportError:
    # Fall back to direct Keras import
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib
import os

class StockPredictor:
    def __init__(self, model_name='stock_predictor.h5', scaler_name='scaler.save'):
        self.model_name = model_name
        self.scaler_name = scaler_name
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
class StockPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model_path = "stock_model.h5"
        self.model = None

    def build_model(self, n_features):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            Dropout(0.3),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(50),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')  # Classification binaire
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

    def load_model(self, path=None):
        """Charge le modèle pré-entraîné"""
        if path is None:
            path = self.model_path
        try:
            self.model = load_model(path)
            print(f"✅ Modèle chargé depuis {path}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            return False

    def predict(self, X):
        """Fait une prédiction"""
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        pred = self.model.predict(X, verbose=0)
        return float(pred[0][0])

    def predict_with_confidence(self, X):
        """Prédit avec niveau de confiance"""
        prob = self.predict(X)
        
        return {
            'direction': "HAUSSE" if prob > 0.5 else "BAISSE",
            'confidence': round(max(prob, 1-prob) * 100, 2),
            'probability_up': round(prob * 100, 2),
            'probability_down': round((1 - prob) * 100, 2),
        }