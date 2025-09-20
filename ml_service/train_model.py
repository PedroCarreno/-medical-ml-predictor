import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalMLPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None

    def load_and_preprocess_data(self, dataset_path='data/dataset.csv'):
        """Cargar y preprocesar dataset médico"""
        logger.info(f"Cargando dataset desde {dataset_path}")

        # Cargar datos
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset original: {df.shape}")

        # Eliminar columnas ID (según el PDF)
        id_columns = ['encounter_id', 'patient_id', 'hospital_id']
        df = df.drop(columns=[col for col in id_columns if col in df.columns])

        # IMPORTANTE: Eliminar también las probabilidades Apache porque son OUTPUTS de otro modelo
        # No deben ser usadas como predictores (sería circular)
        apache_output_columns = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
        df = df.drop(columns=[col for col in apache_output_columns if col in df.columns])
        logger.info(f"Variables Apache eliminadas (son outputs): {apache_output_columns}")

        # Eliminar columnas vacías o innecesarias
        df = df.dropna(axis=1, how='all')  # Eliminar columnas completamente vacías

        # Variable objetivo
        if 'hospital_death' not in df.columns:
            raise ValueError("Columna 'hospital_death' no encontrada en el dataset")

        y = df['hospital_death']
        X = df.drop('hospital_death', axis=1)

        # Tratar valores faltantes
        logger.info("Tratando valores faltantes...")
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # Para variables numéricas, usar mediana
                X[col] = X[col].fillna(X[col].median())
            else:
                # Para variables categóricas, usar moda
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown')

        # Codificar variables categóricas
        logger.info("Codificando variables categóricas...")
        categorical_columns = X.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        self.feature_columns = list(X.columns)
        logger.info(f"Features finales: {len(self.feature_columns)} columnas")
        logger.info(f"Distribución objetivo - Sobrevive: {(y==0).sum()}, Muere: {(y==1).sum()}")

        return X, y

    def train_models(self, X, y):
        """Entrenar múltiples modelos ML"""
        logger.info("Dividiendo datos en entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Escalar datos
        logger.info("Escalando características...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 1. Random Forest (recomendado para datos médicos)
        logger.info("Entrenando Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model

        # 2. XGBoost (alta precisión)
        logger.info("Entrenando XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            enable_categorical=True
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model

        # 3. Regresión Logística (interpretable)
        logger.info("Entrenando Regresión Logística...")
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        lr_model.fit(X_train_scaled, y_train)
        self.models['logistic_regression'] = lr_model

        # Evaluar modelos
        self.evaluate_models(X_test, X_test_scaled, y_test)

        return X_test, y_test

    def evaluate_models(self, X_test, X_test_scaled, y_test):
        """Evaluar rendimiento de los modelos"""
        logger.info("\n=== EVALUACIÓN DE MODELOS ===")

        results = {}

        # Random Forest
        rf_pred = self.models['random_forest'].predict(X_test)
        rf_proba = self.models['random_forest'].predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_proba)
        results['random_forest'] = {'auc': rf_auc, 'predictions': rf_pred}
        logger.info(f"Random Forest AUC: {rf_auc:.4f}")

        # XGBoost
        xgb_pred = self.models['xgboost'].predict(X_test)
        xgb_proba = self.models['xgboost'].predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_proba)
        results['xgboost'] = {'auc': xgb_auc, 'predictions': xgb_pred}
        logger.info(f"XGBoost AUC: {xgb_auc:.4f}")

        # Regresión Logística
        lr_pred = self.models['logistic_regression'].predict(X_test_scaled)
        lr_proba = self.models['logistic_regression'].predict_proba(X_test_scaled)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_proba)
        results['logistic_regression'] = {'auc': lr_auc, 'predictions': lr_pred}
        logger.info(f"Regresión Logística AUC: {lr_auc:.4f}")

        # Seleccionar mejor modelo
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        self.best_model_name = best_model_name
        logger.info(f"\n🏆 MEJOR MODELO: {best_model_name.upper()} (AUC: {results[best_model_name]['auc']:.4f})")

        return results

    def predict_single_patient(self, patient_data):
        """Hacer predicción para un paciente individual - LAS 3 SALIDAS"""
        if self.best_model_name not in self.models:
            raise ValueError("Modelo no entrenado")

        best_model = self.models[self.best_model_name]

        # Preprocesar datos del paciente
        patient_df = pd.DataFrame([patient_data])

        # Aplicar mismas transformaciones que en entrenamiento
        for col in patient_df.columns:
            if col in self.label_encoders:
                # Manejar valores nuevos no vistos en entrenamiento
                le = self.label_encoders[col]
                try:
                    patient_df[col] = le.transform(patient_df[col].astype(str))
                except ValueError:
                    # Si es un valor nuevo, asignar la clase más común
                    patient_df[col] = 0

        # Asegurar que tenga todas las columnas necesarias
        for col in self.feature_columns:
            if col not in patient_df.columns:
                patient_df[col] = 0  # Valor por defecto

        patient_df = patient_df[self.feature_columns]

        # SALIDA 1: Clasificación Binaria Simple
        prediction = best_model.predict(patient_df)[0]
        binary_result = "PACIENTE MORIRÁ" if prediction == 1 else "PACIENTE SOBREVIVIRÁ"

        # SALIDA 2: Probabilidades Detalladas
        probabilities = best_model.predict_proba(patient_df)[0]
        prob_survive = probabilities[0] * 100
        prob_death = probabilities[1] * 100
        confidence = max(prob_survive, prob_death)

        # SALIDA 3: Clasificación por Niveles de Riesgo
        if prob_death <= 25:
            risk_level = "RIESGO BAJO"
            recommendations = [
                "Monitoreo estándar",
                "Seguimiento rutinario de signos vitales",
                "Continuar tratamiento actual"
            ]
        elif prob_death <= 50:
            risk_level = "RIESGO MODERADO"
            recommendations = [
                "Atención reforzada",
                "Monitoreo cada 4 horas",
                "Evaluar necesidad de intervenciones adicionales"
            ]
        elif prob_death <= 75:
            risk_level = "RIESGO ALTO"
            recommendations = [
                "Cuidados intensivos inmediatos",
                "Monitoreo continuo",
                "Considerar traslado a UCI especializada",
                "Informar a familia sobre situación"
            ]
        else:
            risk_level = "RIESGO CRÍTICO"
            recommendations = [
                "Atención médica inmediata y urgente",
                "Considerar medidas extraordinarias",
                "Informar a familia sobre pronóstico grave",
                "Evaluar cuidados paliativos si corresponde",
                "Activar protocolo de emergencia"
            ]

        return {
            'salida_1_binaria': {
                'prediction': int(prediction),
                'result_text': binary_result
            },
            'salida_2_probabilidades': {
                'prob_muerte': round(float(prob_death), 2),
                'prob_supervivencia': round(float(prob_survive), 2),
                'confianza': round(float(confidence), 2)
            },
            'salida_3_riesgo': {
                'nivel_riesgo': risk_level,
                'probabilidad_muerte': round(float(prob_death), 2),
                'recomendaciones': recommendations
            },
            'modelo_usado': self.best_model_name,
            'variables_importantes': [(str(feat), float(imp)) for feat, imp in self.get_feature_importance()[:10]]
        }

    def get_feature_importance(self):
        """Obtener importancia de características"""
        if self.best_model_name == 'random_forest':
            importance = self.models['random_forest'].feature_importances_
        elif self.best_model_name == 'xgboost':
            importance = self.models['xgboost'].feature_importances_
        else:
            # Para regresión logística, usar coeficientes absolutos
            importance = np.abs(self.models['logistic_regression'].coef_[0])

        feature_importance = list(zip(self.feature_columns, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        return feature_importance

    def save_model(self, model_dir='models'):
        """Guardar modelo entrenado"""
        os.makedirs(model_dir, exist_ok=True)

        # Guardar el mejor modelo
        model_path = os.path.join(model_dir, 'best_model.pkl')
        joblib.dump(self.models[self.best_model_name], model_path)

        # Guardar scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        # Guardar encoders
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        joblib.dump(self.label_encoders, encoders_path)

        # Guardar información del modelo
        model_info = {
            'best_model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'model_performance': {}
        }

        info_path = os.path.join(model_dir, 'model_info.pkl')
        joblib.dump(model_info, info_path)

        logger.info(f"✅ Modelo guardado en {model_dir}")
        logger.info(f"   - Mejor modelo: {self.best_model_name}")
        logger.info(f"   - Features: {len(self.feature_columns)}")

    @classmethod
    def load_model(cls, model_dir='models'):
        """Cargar modelo entrenado"""
        instance = cls()

        # Cargar información del modelo
        info_path = os.path.join(model_dir, 'model_info.pkl')
        model_info = joblib.load(info_path)

        instance.best_model_name = model_info['best_model_name']
        instance.feature_columns = model_info['feature_columns']

        # Cargar modelo
        model_path = os.path.join(model_dir, 'best_model.pkl')
        instance.models[instance.best_model_name] = joblib.load(model_path)

        # Cargar scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        instance.scaler = joblib.load(scaler_path)

        # Cargar encoders
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        instance.label_encoders = joblib.load(encoders_path)

        logger.info(f"✅ Modelo cargado desde {model_dir}")
        return instance

def train_medical_model():
    """Función principal para entrenar el modelo médico"""
    logger.info("🏥 Iniciando entrenamiento del modelo médico...")

    predictor = MedicalMLPredictor()

    try:
        # Cargar y preprocesar datos
        X, y = predictor.load_and_preprocess_data()

        # Entrenar modelos
        X_test, y_test = predictor.train_models(X, y)

        # Guardar modelo
        predictor.save_model()

        logger.info("🎉 ¡Entrenamiento completado exitosamente!")
        logger.info("El modelo está listo para hacer predicciones médicas")

        return predictor

    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    train_medical_model()