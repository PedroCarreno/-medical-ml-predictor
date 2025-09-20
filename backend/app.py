from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permitir requests desde React

# Variables globales para datos y modelo
dataset = None
model = None
scaler = None

# Configuración
DATASET_PATH = 'data/dataset.csv'
MODEL_PATH = 'models/trained_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

@app.route('/')
def home():
    return jsonify({
        'message': 'Medical ML Predictor API',
        'version': '1.0.0',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'dataset_loaded': dataset is not None,
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }

    if dataset is not None:
        status['dataset_shape'] = dataset.shape

    return jsonify(status)

@app.route('/api/load-dataset', methods=['POST'])
def load_dataset():
    """Cargar dataset desde CSV"""
    global dataset

    try:
        if not os.path.exists(DATASET_PATH):
            return jsonify({
                'error': f'Dataset no encontrado en {DATASET_PATH}',
                'status': 'error'
            }), 404

        logger.info(f"Cargando dataset desde {DATASET_PATH}")
        dataset = pd.read_csv(DATASET_PATH)

        # Información básica del dataset
        info = {
            'status': 'success',
            'message': 'Dataset cargado exitosamente',
            'shape': dataset.shape,
            'columns': len(dataset.columns),
            'missing_values': dataset.isnull().sum().sum(),
            'target_distribution': dataset['hospital_death'].value_counts().to_dict() if 'hospital_death' in dataset.columns else 'No target column found'
        }

        logger.info(f"Dataset cargado: {info}")
        return jsonify(info)

    except Exception as e:
        logger.error(f"Error cargando dataset: {str(e)}")
        return jsonify({
            'error': f'Error cargando dataset: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/dataset-info')
def dataset_info():
    """Información del dataset cargado"""
    if dataset is None:
        return jsonify({
            'error': 'Dataset no cargado. Usar /api/load-dataset primero',
            'status': 'error'
        }), 400

    try:
        info = {
            'shape': dataset.shape,
            'columns': list(dataset.columns),
            'dtypes': dataset.dtypes.astype(str).to_dict(),
            'missing_values': dataset.isnull().sum().to_dict(),
            'sample_data': dataset.head().to_dict('records')
        }

        # Información específica médica
        if 'hospital_death' in dataset.columns:
            info['target_distribution'] = dataset['hospital_death'].value_counts().to_dict()
            info['mortality_rate'] = dataset['hospital_death'].mean()

        return jsonify(info)

    except Exception as e:
        return jsonify({
            'error': f'Error obteniendo info del dataset: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Realizar predicción médica - LAS 3 SALIDAS"""
    from services.ml_service import ml_service

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No se enviaron datos para predicción',
                'status': 'error'
            }), 400

        # Cargar modelo si no está cargado
        if not ml_service.is_model_loaded():
            model_loaded = ml_service.load_model()
            if not model_loaded:
                return jsonify({
                    'error': 'Modelo no disponible',
                    'message': 'Entrenar modelo primero usando /api/train',
                    'status': 'error'
                }), 400

        # Realizar predicción con las 3 salidas
        prediction_result = ml_service.predict_patient_outcome(data)

        return jsonify(prediction_result)

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({
            'error': f'Error en predicción: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Entrenar modelo ML"""
    try:
        # Importar función de entrenamiento
        import sys
        import os

        # En Docker, agregar el directorio raíz del contenedor al path
        app_root = '/app' if os.path.exists('/app/ml_service') else os.path.dirname(os.path.dirname(__file__))
        sys.path.append(app_root)

        from ml_service.train_model import train_medical_model

        logger.info("🏥 Iniciando entrenamiento del modelo...")

        # Entrenar modelo
        predictor = train_medical_model()

        logger.info("✅ Modelo entrenado exitosamente")

        return jsonify({
            'status': 'success',
            'message': 'Modelo entrenado exitosamente',
            'modelo_info': {
                'algoritmo_seleccionado': predictor.best_model_name,
                'features_utilizadas': len(predictor.feature_columns),
                'modelos_evaluados': list(predictor.models.keys())
            },
            'siguiente_paso': 'Ya puedes hacer predicciones usando /api/predict'
        })

    except Exception as e:
        logger.error(f"Error entrenando modelo: {str(e)}")
        return jsonify({
            'error': f'Error entrenando modelo: {str(e)}',
            'status': 'error',
            'solucion': 'Verificar que el dataset esté en data/dataset.csv'
        }), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Realizar predicciones para múltiples pacientes"""
    from services.ml_service import ml_service

    try:
        data = request.get_json()

        if not data or 'patients' not in data:
            return jsonify({
                'error': 'Formato incorrecto. Enviar {"patients": [...]}'
            }), 400

        patients_data = data['patients']

        if not isinstance(patients_data, list):
            return jsonify({
                'error': 'Campo "patients" debe ser una lista'
            }), 400

        # Cargar modelo si es necesario
        if not ml_service.is_model_loaded():
            ml_service.load_model()

        # Realizar predicciones en lote
        results = ml_service.predict_batch(patients_data)

        return jsonify({
            'status': 'success',
            'total_patients': len(patients_data),
            'predictions': results
        })

    except Exception as e:
        logger.error(f"Error en predicción por lotes: {str(e)}")
        return jsonify({
            'error': f'Error en predicción por lotes: {str(e)}'
        }), 500

@app.route('/api/model-info')
def model_info():
    """Información del modelo cargado"""
    from services.ml_service import ml_service

    try:
        info = ml_service.get_model_info()
        return jsonify(info)

    except Exception as e:
        return jsonify({
            'error': f'Error obteniendo información del modelo: {str(e)}'
        }), 500

@app.route('/api/predict-explain', methods=['POST'])
def predict_with_explanation():
    """Predicción con explicación detallada"""
    from services.ml_service import ml_service

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No se enviaron datos para predicción'
            }), 400

        # Cargar modelo si es necesario
        if not ml_service.is_model_loaded():
            ml_service.load_model()

        # Obtener explicación detallada
        explanation = ml_service.get_prediction_explanation(data)

        return jsonify(explanation)

    except Exception as e:
        logger.error(f"Error en explicación: {str(e)}")
        return jsonify({
            'error': f'Error en explicación: {str(e)}'
        }), 500

if __name__ == '__main__':
    logger.info("Iniciando Medical ML Predictor API")
    logger.info(f"Dataset path: {DATASET_PATH}")
    logger.info(f"Model path: {MODEL_PATH}")

    app.run(host='0.0.0.0', port=8000, debug=True)