# Predict_stock

Description complète

`Predict_stock` est un projet de prototype visant à explorer une chaîne complète de prédiction boursière basée sur des réseaux de neurones récurrents (LSTM). Il rassemble des composants pour la collecte des données, la préparation, l'entraînement d'un modèle, l'inférence via une API et une interface web simple pour tester les prédictions. Ce dépôt a vocation pédagogique et expérimentale — il n'est pas prêt pour un usage en production sans validation, tests rigoureux et contrôle des risques.

Contenu et objectif
- Collecte de données : récupération des séries temporelles (prix historiques) via `yfinance` et stockage local minimal (ex. `data/stock_data.db`).
- Prétraitement : normalisation (MinMax), construction de fenêtres temporelles pour l'entrée LSTM.
- Modélisation : architecture LSTM (Keras/TensorFlow) pour prédire la direction ou la probabilité d'un mouvement (exemple binaire). Le code d'entraînement est dans `backend/train_model.py`.
- Serveur d'inférence : petite API Flask dans `backend/app.py` qui charge le modèle sauvegardé (`backend/stock_model.h5`) et propose des endpoints pour obtenir des prédictions.
- Frontend : interface minimale (`frontend/index.html`, `frontend/script.js`) pour envoyer une requête au backend et afficher les résultats.

Architecture et flux de données
- `train_model.py` : lit/assemble les données, prépare jeux d'entraînement/validation, entraîne le modèle avec callbacks (EarlyStopping, ModelCheckpoint) et sauvegarde le modèle en `backend/stock_model.h5`.
- `backend/model.py` : wrapper utilitaire pour construire, charger et effectuer des prédictions depuis le modèle Keras. Les imports TensorFlow sont faits de façon paresseuse pour éviter les erreurs d'importation lors des opérations qui n'ont pas besoin de TF immédiatement.
- `backend/app.py` : expose des routes REST (par exemple `/api/predict`) ; charge le modèle au démarrage (si disponible) et retourne JSON avec prédiction et confiance.

Installation et exécution (Windows, PowerShell)
1. Créer et activer un environnement virtuel (64-bit) :
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
2. Installer les dépendances :
```powershell
pip install -r backend/requirements.txt
```
3. (Optionnel) Entraîner :
```powershell
python backend/train_model.py
```
4. Lancer l'API Flask :
```powershell
python backend/app.py
```
Accéder ensuite à `http://127.0.0.1:5000` pour tester le frontend.

Dépendances et prérequis importants
- Python 3.8+ (64-bit recommandé pour TensorFlow)
- TensorFlow (version compatible : voir `backend/requirements.txt`)
- Microsoft Visual C++ Redistributable (x64) sur Windows si import TensorFlow échoue (erreurs concernant `msvcp140_1.dll`).

Sécurité et limites
- Ne jamais exposer un modèle non vérifié en production ; les prédictions financières peuvent être risquées.
- Les performances et l'utilité d'un modèle LSTM sur les séries financières dépendent fortement des choix de features, de l'échantillonnage, du labeling et de la robustesse du dataset (éviter overfitting, fuite de données temporales).

Conseils d'usage
- Utilisez des jeux de tests hors-temps pour valider la généralisation.
- Ajoutez des métriques robustes (AUC, precision-recall, profit simulé) et des routines de backtesting séparées.
- En production, externalisez le stockage des modèles (S3, artifact registry) et utilisez des containers pour l'isolation.

Historique Git
- `main` : branche nettoyée contenant le projet sans le dossier `venv`.
- `main-backup` : sauvegarde de l'ancien historique (conserve les gros fichiers). Si vous avez besoin d'inspecter l'ancien contenu, ce backup est disponible.

Fichiers clés
- `backend/app.py` — serveur Flask.
- `backend/model.py` — wrapper modèle (build/load/predict).
- `backend/train_model.py` — script d'entraînement.
- `backend/requirements.txt` — dépendances.
- `frontend/` — UI minimale pour tester l'API.

Prochaines améliorations suggérées
- Ajouter gestionnaire de retries et logs pour `yfinance`.
- Nettoyer davantage le pipeline de données et ajouter tests unitaires.
- Ajouter CI (tests + linting) et un script de déploiement en container.

Disclaimer
Les résultats produits par ce dépôt sont expérimentaux. Ne pas utiliser ce code pour prendre des décisions financières réelles sans évaluation indépendante et mesures de mitigation des risques.

## Repository structure

- `backend/` — Flask API, training and model code (Python). Contains `model.py`, `train_model.py`, `app.py` and `requirements.txt`.
- `frontend/` — static UI to test the backend (`index.html`, `script.js`, `style.css`).
- `data/` — sqlite or data artifacts used during development (do not commit large datasets).

## Quickstart (Windows / PowerShell)

1. Create and activate a virtual environment (64-bit Python):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r backend/requirements.txt
```

3. (Optional) Train model locally:

```powershell
python backend/train_model.py
```

This saves a model to `backend/stock_model.h5` when complete.

4. Run the Flask server (development):

```powershell
# Option A: run the app directly
python backend/app.py

# Option B: use flask CLI
$env:FLASK_APP = "backend.app"
flask run --host=0.0.0.0 --port=5000
```

Then open `http://127.0.0.1:5000` to view the frontend.

## Notes & tips

- Do NOT commit the `venv/` directory. This repo includes a `.gitignore` that ignores `venv/` and common OS/IDE files.
- TensorFlow on Windows requires the Microsoft Visual C++ Redistributable (x64). If you see errors about `msvcp140_1.dll`, install the 2015-2022 redistributable from Microsoft.
- The repo originally contained a large `venv/` that caused the push to fail; I created a clean branch and updated `main` to remove those files. A backup of the original remote `main` was saved as `main-backup`.

## Contributing / Git

- Branches:
  - `main` — cleaned project (no `venv`).
  - `main-backup` — backup of the previous remote `main` (contains previous history and large files).

- If you need to restore or inspect the old history, check `main-backup` on GitHub.

## License

Add your preferred license here.
# Predict_stock