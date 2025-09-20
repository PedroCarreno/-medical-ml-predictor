import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

# Agregar el directorio padre al path para importar
app_root = '/app' if os.path.exists('/app/ml_service') else os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(app_root)

try:
    from ml_service.train_model import MedicalMLPredictor
except ImportError:
    # Si no se puede importar, crear una clase stub
    class MedicalMLPredictor:
        def __init__(self):
            self.trained = False

        @classmethod
        def load_model(cls, model_dir):
            return cls()

        def predict_single_patient(self, data):
            return {"error": "Modelo no entrenado"}

logger = logging.getLogger(__name__)

class MLService:
    """Servicio de Machine Learning para predicciones médicas"""

    def __init__(self):
        self.predictor = None
        self.model_loaded = False
        self.model_dir = 'models'

    def load_model(self):
        """Cargar modelo entrenado"""
        try:
            model_path = os.path.join(self.model_dir, 'best_model.pkl')

            if not os.path.exists(model_path):
                logger.warning(f"Modelo no encontrado en {model_path}")
                return False

            self.predictor = MedicalMLPredictor.load_model(self.model_dir)
            self.model_loaded = True
            logger.info("✅ Modelo ML cargado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            self.model_loaded = False
            return False

    def is_model_loaded(self) -> bool:
        """Verificar si el modelo está cargado"""
        return self.model_loaded and self.predictor is not None

    def normalize_numeric_value(self, value):
        """Normalizar valores numéricos, convirtiendo comas en puntos"""
        if value is None or value == '':
            return None

        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            # Reemplazar coma decimal por punto
            normalized = value.replace(',', '.')
            try:
                return float(normalized)
            except ValueError:
                return value  # Devolver original si no se puede convertir

        return value

    def normalize_patient_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizar todos los valores numéricos en los datos del paciente"""
        normalized_data = {}

        # Campos que deben ser numéricos decimales
        numeric_fields = [
            'age', 'height', 'weight', 'bmi', 'pre_icu_los_days',
            'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache',
            'heart_rate_apache', 'map_apache', 'resprate_apache', 'temp_apache',
            'd1_diasbp_max', 'd1_diasbp_min', 'd1_sysbp_max', 'd1_sysbp_min',
            'd1_heartrate_max', 'd1_heartrate_min', 'd1_resprate_max', 'd1_resprate_min',
            'd1_spo2_max', 'd1_spo2_min', 'd1_temp_max', 'd1_temp_min',
            'd1_glucose_max', 'd1_glucose_min', 'd1_potassium_max', 'd1_potassium_min',
            'apache_2_diagnosis', 'apache_3j_diagnosis'
        ]

        # Campos categóricos binarios que deben ser enteros 0/1
        binary_fields = [
            'elective_surgery', 'apache_post_operative', 'arf_apache', 'gcs_unable_apache',
            'intubated_apache', 'ventilated_apache', 'aids', 'cirrhosis', 'diabetes_mellitus',
            'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma',
            'solid_tumor_with_metastasis'
        ]

        for key, value in data.items():
            if key in numeric_fields:
                # Normalizar valores numéricos (convertir comas en puntos)
                normalized_data[key] = self.normalize_numeric_value(value)
            elif key in binary_fields:
                # Convertir strings '0'/'1' a enteros 0/1 para XGBoost
                if value in ['0', '1']:
                    normalized_data[key] = int(value)
                elif value in [0, 1]:
                    normalized_data[key] = value
                else:
                    normalized_data[key] = 0  # Valor por defecto
            else:
                # Dejar todos los demás campos como están
                # El modelo se encargará de aplicar LabelEncoders si es necesario
                normalized_data[key] = value

        return normalized_data

    def validate_patient_data(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Validar datos del paciente"""
        required_fields = ['age']  # Campos mínimos requeridos

        for field in required_fields:
            if field not in data:
                return False, f"Campo requerido faltante: {field}"

        # Validaciones específicas
        age = data.get('age')
        if age is None:
            return False, "Edad es requerida"

        # Normalizar edad si viene como string
        if isinstance(age, str):
            age = self.normalize_numeric_value(age)

        if not isinstance(age, (int, float)) or age < 16 or age > 120:
            return False, "Edad debe ser un número entre 16 y 120"

        return True, "Datos válidos"

    def predict_patient_outcome(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realizar predicción médica con las 3 salidas requeridas

        Args:
            patient_data: Diccionario con datos del paciente

        Returns:
            Diccionario con las 3 predicciones
        """
        if not self.is_model_loaded():
            return {
                'error': 'Modelo no cargado',
                'message': 'El modelo ML no está disponible. Entrenar modelo primero.',
                'status': 'error'
            }

        # Normalizar datos numéricos (convertir comas en puntos)
        normalized_data = self.normalize_patient_data(patient_data)

        # Validar datos de entrada
        is_valid, message = self.validate_patient_data(normalized_data)
        if not is_valid:
            return {
                'error': 'Datos inválidos',
                'message': message,
                'status': 'error'
            }

        try:
            # Hacer predicción usando el modelo entrenado con datos normalizados
            prediction_result = self.predictor.predict_single_patient(normalized_data)

            # Formatear resultado según especificaciones
            return {
                'status': 'success',
                'patient_id': normalized_data.get('patient_id', 'unknown'),
                'timestamp': pd.Timestamp.now().isoformat(),

                # SALIDA 1: Clasificación Binaria Simple
                'resultado_binario': prediction_result['salida_1_binaria'],

                # SALIDA 2: Probabilidades Detalladas
                'probabilidades': prediction_result['salida_2_probabilidades'],

                # SALIDA 3: Niveles de Riesgo con Recomendaciones
                'evaluacion_riesgo': prediction_result['salida_3_riesgo'],

                # Información adicional
                'modelo_info': {
                    'algoritmo_usado': prediction_result['modelo_usado'],
                    'variables_mas_importantes': prediction_result['variables_importantes']
                }
            }

        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return {
                'error': 'Error en predicción',
                'message': f'Error interno: {str(e)}',
                'status': 'error'
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo cargado"""
        # Intentar cargar modelo si no está cargado pero existe
        if not self.is_model_loaded():
            self.load_model()

        if not self.is_model_loaded():
            return {
                'model_loaded': False,
                'message': 'Modelo no cargado'
            }

        try:
            # Información básica del modelo
            info = {
                'model_loaded': True,
                'algorithm': self.predictor.best_model_name,
                'features_count': len(self.predictor.feature_columns),
                'feature_names': self.predictor.feature_columns[:20],  # Primeras 20
                'model_dir': self.model_dir
            }

            # Importancia de características
            if hasattr(self.predictor, 'get_feature_importance'):
                importance = self.predictor.get_feature_importance()[:10]
                info['top_features'] = [
                    {'feature': str(feat), 'importance': float(imp)}
                    for feat, imp in importance
                ]

            return info

        except Exception as e:
            logger.error(f"Error obteniendo info del modelo: {str(e)}")
            return {
                'model_loaded': True,
                'error': f'Error obteniendo información: {str(e)}'
            }

    def predict_batch(self, patients_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Realizar predicciones para múltiples pacientes"""
        if not self.is_model_loaded():
            return [{
                'error': 'Modelo no cargado',
                'status': 'error'
            }] * len(patients_data)

        results = []
        for i, patient_data in enumerate(patients_data):
            try:
                patient_data['batch_index'] = i
                result = self.predict_patient_outcome(patient_data)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': f'Error en paciente {i}',
                    'message': str(e),
                    'status': 'error',
                    'batch_index': i
                })

        return results

    def get_prediction_explanation(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Obtener explicación detallada de la predicción"""
        if not self.is_model_loaded():
            return {'error': 'Modelo no cargado'}

        try:
            # Normalizar datos primero
            normalized_data = self.normalize_patient_data(patient_data)

            # Hacer predicción básica
            prediction = self.predict_patient_outcome(patient_data)

            if 'error' in prediction:
                return prediction

            # Agregar explicaciones médicas usando datos normalizados
            explanation = {
                'prediccion_base': prediction,
                'factores_riesgo': self._analyze_risk_factors(normalized_data),
                'comparacion_poblacional': self._compare_with_population(normalized_data),
                'recomendaciones_detalladas': self._get_detailed_recommendations(prediction)
            }

            return explanation

        except Exception as e:
            logger.error(f"Error en explicación: {str(e)}")
            return {
                'error': 'Error generando explicación',
                'message': str(e)
            }

    def _analyze_risk_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Analizar factores de riesgo específicos del paciente"""
        risk_factors = []

        # Edad
        age = patient_data.get('age', 0)
        if age > 70:
            risk_factors.append(f"Edad avanzada ({age} años) - Mayor riesgo")
        elif age > 60:
            risk_factors.append(f"Edad moderada ({age} años) - Riesgo moderado")

        # Comorbilidades críticas
        critical_conditions = {
            'aids': 'SIDA/VIH avanzado - Riesgo extremo',
            'cirrhosis': 'Cirrosis hepática - Riesgo alto',
            'hepatic_failure': 'Falla hepática - Riesgo extremo',
            'solid_tumor_with_metastasis': 'Cáncer metastásico - Riesgo extremo',
            'leukemia': 'Leucemia - Riesgo extremo'
        }

        for condition, description in critical_conditions.items():
            if patient_data.get(condition, 0) == 1:
                risk_factors.append(description)

        # Estado neurológico (Glasgow)
        glasgow_total = (
            patient_data.get('gcs_eyes_apache', 4) +
            patient_data.get('gcs_motor_apache', 6) +
            patient_data.get('gcs_verbal_apache', 5)
        )

        if glasgow_total < 8:
            risk_factors.append(f"Estado neurológico crítico (Glasgow {glasgow_total}) - Riesgo muy alto")
        elif glasgow_total < 12:
            risk_factors.append(f"Estado neurológico alterado (Glasgow {glasgow_total}) - Riesgo moderado")

        # Soporte vital
        if patient_data.get('intubated_apache', 0) == 1:
            risk_factors.append("Intubación endotraqueal - Requiere soporte respiratorio")

        if patient_data.get('ventilated_apache', 0) == 1:
            risk_factors.append("Ventilación mecánica - Estado crítico")

        return risk_factors

    def _compare_with_population(self, patient_data: Dict[str, Any]) -> Dict[str, str]:
        """Comparar paciente con población general"""
        age = patient_data.get('age', 50)

        comparisons = {
            'grupo_edad': f"Paciente de {age} años",
            'comorbilidades': "Análisis de comorbilidades vs población general",
            'estado_general': "Comparación con casos similares en UCI"
        }

        return comparisons

    def _get_detailed_recommendations(self, prediction: Dict[str, Any]) -> List[str]:
        """Obtener recomendaciones médicas detalladas"""
        if 'evaluacion_riesgo' not in prediction:
            return ["No se pueden generar recomendaciones detalladas"]

        risk_level = prediction['evaluacion_riesgo']['nivel_riesgo']
        base_recommendations = prediction['evaluacion_riesgo']['recomendaciones']

        # Agregar recomendaciones adicionales según nivel de riesgo
        additional_recommendations = []

        if "CRÍTICO" in risk_level:
            additional_recommendations.extend([
                "Revisar DNR (Do Not Resuscitate) con familia",
                "Considerar consulta de cuidados paliativos",
                "Evaluación por comité de ética médica si es necesario"
            ])
        elif "ALTO" in risk_level:
            additional_recommendations.extend([
                "Monitoreo hemodinámico invasivo",
                "Laboratorios cada 6 horas",
                "Considerar traslado a UCI de mayor complejidad"
            ])

        return base_recommendations + additional_recommendations

# Instancia global del servicio
ml_service = MLService()