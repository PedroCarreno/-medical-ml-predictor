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

    def load_and_preprocess_data(self, dataset_path='dataset_clean_final.csv'):
        """Cargar dataset LIMPIO (ya preprocesado por clean_dataset_complete.py)"""
        logger.info(f"Cargando dataset LIMPIO desde {dataset_path}")

        # Cargar datos LIMPIOS (sin nulls, con encoding)
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset limpio cargado: {df.shape}")
        logger.info(f"Nulls en dataset: {df.isnull().sum().sum()} (debería ser 0)")

        # Variable objetivo
        if 'hospital_death' not in df.columns:
            raise ValueError("Columna 'hospital_death' no encontrada en el dataset")

        y = df['hospital_death']
        X = df.drop('hospital_death', axis=1)

        # Eliminar columnas ORIGINALES categóricas (mantener solo _encoded)
        logger.info("Eliminando columnas categóricas originales (manteniendo solo _encoded)...")
        categorical_originals = ['ethnicity', 'gender', 'icu_admit_source', 'icu_stay_type',
                                'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']

        for col in categorical_originals:
            if col in X.columns:
                # Guardar el encoder para predicciones futuras
                if col + '_encoded' in X.columns:
                    # Crear LabelEncoder desde los valores originales para usar en predict
                    le = LabelEncoder()
                    le.fit(df[col].astype(str))
                    self.label_encoders[col] = le
                    logger.info(f"✅ {col} -> usando {col}_encoded (guardado encoder)")
                    X = X.drop(columns=[col])
                else:
                    logger.warning(f"⚠️ {col} no tiene versión _encoded")

        # Eliminar probabilidades Apache (son outputs de otro modelo)
        apache_output_columns = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
        X = X.drop(columns=[col for col in apache_output_columns if col in X.columns], errors='ignore')

        self.feature_columns = list(X.columns)
        logger.info(f"✅ Features finales: {len(self.feature_columns)} columnas (todas numéricas)")
        logger.info(f"📊 Distribución objetivo - Sobrevive: {(y==0).sum()}, Muere: {(y==1).sum()}")

        return X, y

    def train_models(self, X, y, models_to_train=None, test_size=0.2):
        """
        Entrenar modelos ML específicos o todos

        Args:
            X: Features
            y: Target
            models_to_train: Lista de modelos a entrenar ['random_forest', 'xgboost'] o None (todos)
            test_size: Proporción de datos para test (0.2 = 20% test, 80% train)
        """
        if models_to_train is None:
            models_to_train = ['random_forest', 'xgboost']

        logger.info("Dividiendo datos en entrenamiento y prueba...")
        logger.info(f"   - Training set: {int((1-test_size)*100)}%")
        logger.info(f"   - Test set: {int(test_size*100)}%")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Guardar info del split para mostrar en frontend
        self.training_info = {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_percentage': int((1-test_size)*100),
            'test_percentage': int(test_size*100),
            'train_deaths': int((y_train == 1).sum()),
            'train_survivors': int((y_train == 0).sum()),
            'test_deaths': int((y_test == 1).sum()),
            'test_survivors': int((y_test == 0).sum()),
            'features_count': len(self.feature_columns),
            'features_used': self.feature_columns[:20]  # Primeras 20 para mostrar
        }

        # Escalar datos
        logger.info("Escalando características...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # MODELO 1: Random Forest (Ensemble de árboles)
        if 'random_forest' in models_to_train:
            logger.info("Entrenando Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=200,        # Más árboles = mejor generalización
                max_depth=15,            # Profundidad moderada para evitar overfitting
                min_samples_split=10,    # Mínimo 10 muestras para dividir nodo (evita overfitting)
                min_samples_leaf=5,      # Mínimo 5 muestras en hoja (mejor generalización)
                max_features='sqrt',     # Sqrt(n_features) para cada split (reduce correlación entre árboles)
                class_weight='balanced', # Balancear clases (importante: 91% sobrevive, 9% muere)
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            self.models['random_forest'] = rf_model
            logger.info("✅ Random Forest entrenado")

        # MODELO 2: XGBoost (Gradient Boosting optimizado)
        if 'xgboost' in models_to_train:
            logger.info("Entrenando XGBoost...")
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # Balancear clases
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,                  # 200 boosting rounds
                max_depth=6,                       # Profundidad controlada (evita overfitting)
                learning_rate=0.05,                # Learning rate bajo = aprendizaje gradual
                subsample=0.8,                     # 80% de muestras por árbol (reduce overfitting)
                colsample_bytree=0.8,              # 80% de features por árbol (reduce overfitting)
                gamma=1,                           # Regularización mínima de ganancia
                reg_alpha=0.1,                     # Regularización L1
                reg_lambda=1,                      # Regularización L2
                scale_pos_weight=scale_pos_weight, # Balancear clases automáticamente
                random_state=42,
                n_jobs=-1,
                eval_metric='auc'
            )
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model
            logger.info("✅ XGBoost entrenado")

        # Evaluar modelos
        self.evaluate_models(X_test, X_test_scaled, y_test)

        return X_test, y_test

    def evaluate_models(self, X_test, X_test_scaled, y_test):
        """Evaluar rendimiento de los modelos con métricas completas"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        logger.info("\n" + "="*80)
        logger.info("EVALUACIÓN COMPLETA DE MODELOS - ENTREGA 3")
        logger.info("="*80)

        results = {}

        # MODELO 1: Random Forest
        logger.info("\n📊 MODELO 1: RANDOM FOREST")
        logger.info("-" * 60)
        rf_pred = self.models['random_forest'].predict(X_test)
        rf_proba = self.models['random_forest'].predict_proba(X_test)[:, 1]

        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1_score': f1_score(y_test, rf_pred),
            'auc_roc': roc_auc_score(y_test, rf_proba)
        }

        logger.info(f"   ✅ Accuracy (Exactitud):  {rf_metrics['accuracy']:.4f} ({rf_metrics['accuracy']*100:.2f}%)")
        logger.info(f"   ✅ Precision (Precisión): {rf_metrics['precision']:.4f} ({rf_metrics['precision']*100:.2f}%)")
        logger.info(f"   ✅ Recall (Sensibilidad): {rf_metrics['recall']:.4f} ({rf_metrics['recall']*100:.2f}%)")
        logger.info(f"   ✅ F1-Score:              {rf_metrics['f1_score']:.4f}")
        logger.info(f"   ✅ AUC-ROC:               {rf_metrics['auc_roc']:.4f}")

        results['random_forest'] = {
            'metrics': rf_metrics,
            'predictions': rf_pred,
            'probabilities': rf_proba
        }

        # MODELO 2: XGBoost
        logger.info("\n📊 MODELO 2: XGBOOST")
        logger.info("-" * 60)
        xgb_pred = self.models['xgboost'].predict(X_test)
        xgb_proba = self.models['xgboost'].predict_proba(X_test)[:, 1]

        xgb_metrics = {
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred),
            'recall': recall_score(y_test, xgb_pred),
            'f1_score': f1_score(y_test, xgb_pred),
            'auc_roc': roc_auc_score(y_test, xgb_proba)
        }

        logger.info(f"   ✅ Accuracy (Exactitud):  {xgb_metrics['accuracy']:.4f} ({xgb_metrics['accuracy']*100:.2f}%)")
        logger.info(f"   ✅ Precision (Precisión): {xgb_metrics['precision']:.4f} ({xgb_metrics['precision']*100:.2f}%)")
        logger.info(f"   ✅ Recall (Sensibilidad): {xgb_metrics['recall']:.4f} ({xgb_metrics['recall']*100:.2f}%)")
        logger.info(f"   ✅ F1-Score:              {xgb_metrics['f1_score']:.4f}")
        logger.info(f"   ✅ AUC-ROC:               {xgb_metrics['auc_roc']:.4f}")

        results['xgboost'] = {
            'metrics': xgb_metrics,
            'predictions': xgb_pred,
            'probabilities': xgb_proba
        }

        # Comparación y selección del mejor modelo
        logger.info("\n" + "="*80)
        logger.info("COMPARACIÓN DE MODELOS")
        logger.info("="*80)

        # Usar AUC-ROC como métrica principal (mejor para datos desbalanceados)
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['auc_roc'])
        self.best_model_name = best_model_name

        logger.info(f"\n🏆 MEJOR MODELO SELECCIONADO: {best_model_name.upper()}")
        logger.info(f"   Criterio: AUC-ROC = {results[best_model_name]['metrics']['auc_roc']:.4f}")
        logger.info("\n💡 Justificación de selección:")
        logger.info("   - AUC-ROC es la métrica más apropiada para datos médicos desbalanceados")
        logger.info("   - Mide la capacidad del modelo de discriminar entre clases")
        logger.info("   - Valores cercanos a 1.0 indican excelente rendimiento")

        # Guardar resultados de evaluación
        self.evaluation_results = results

        return results

    def predict_single_patient(self, patient_data, model_name=None):
        """
        Hacer predicción para un paciente individual - LAS 3 SALIDAS

        Args:
            patient_data: dict con datos del paciente (valores originales como 'M', 'F', etc.)
            model_name: 'random_forest', 'xgboost' o None (usa best_model)
        """
        # Seleccionar modelo a usar
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no disponible. Opciones: {list(self.models.keys())}")

        selected_model = self.models[model_name]

        # Preprocesar datos del paciente
        patient_df = pd.DataFrame([patient_data])

        # Convertir valores categóricos a _encoded
        for col in self.label_encoders.keys():
            if col in patient_df.columns:
                le = self.label_encoders[col]
                try:
                    # Convertir valor categórico (ej: 'M') a número (ej: 0)
                    encoded_value = le.transform(patient_df[col].astype(str))[0]
                    # Crear columna _encoded
                    patient_df[col + '_encoded'] = encoded_value
                except ValueError:
                    # Si es un valor nuevo no visto en entrenamiento, usar valor más común (0)
                    patient_df[col + '_encoded'] = 0
                    logger.warning(f"Valor no visto para {col}: {patient_df[col].iloc[0]}, usando 0")

        # Asegurar que tenga todas las columnas necesarias
        for col in self.feature_columns:
            if col not in patient_df.columns:
                patient_df[col] = 0  # Valor por defecto

        patient_df = patient_df[self.feature_columns]

        # SALIDA 1: Clasificación Binaria Simple
        prediction = selected_model.predict(patient_df)[0]
        binary_result = "PACIENTE MORIRÁ" if prediction == 1 else "PACIENTE SOBREVIVIRÁ"

        # SALIDA 2: Probabilidades Detalladas
        probabilities = selected_model.predict_proba(patient_df)[0]
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
            'modelo_usado': model_name,
            'variables_importantes': [(str(feat), float(imp)) for feat, imp in self.get_feature_importance(model_name)[:10]]
        }

    def get_feature_importance(self, model_name=None):
        """Obtener importancia de características para un modelo específico"""
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            return []

        if model_name in ['random_forest', 'xgboost']:
            importance = self.models[model_name].feature_importances_
        else:
            # Para otros modelos, usar coeficientes absolutos si existen
            try:
                importance = np.abs(self.models[model_name].coef_[0])
            except:
                return []

        feature_importance = list(zip(self.feature_columns, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        return feature_importance

    def get_available_models(self):
        """Retorna lista de modelos disponibles"""
        return list(self.models.keys())

    def save_model(self, model_dir='models'):
        """Guardar TODOS los modelos entrenados (no solo el mejor)"""
        os.makedirs(model_dir, exist_ok=True)

        # Guardar TODOS los modelos entrenados
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{model_name}.pkl')
            joblib.dump(model, model_path)
            logger.info(f"   - Guardado: {model_name}.pkl")

        # Guardar el mejor modelo también con nombre genérico para compatibilidad
        best_model_path = os.path.join(model_dir, 'best_model.pkl')
        joblib.dump(self.models[self.best_model_name], best_model_path)

        # Guardar scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        # Guardar encoders
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        joblib.dump(self.label_encoders, encoders_path)

        # Guardar información COMPLETA del modelo (incluyendo resultados de evaluación)
        model_info = {
            'best_model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'available_models': list(self.models.keys()),
            'evaluation_results': self.evaluation_results if hasattr(self, 'evaluation_results') else {},
            'training_info': self.training_info if hasattr(self, 'training_info') else {}
        }

        info_path = os.path.join(model_dir, 'model_info.pkl')
        joblib.dump(model_info, info_path)

        logger.info(f"✅ Modelo guardado en {model_dir}")
        logger.info(f"   - Mejor modelo: {self.best_model_name}")
        logger.info(f"   - Modelos guardados: {list(self.models.keys())}")
        logger.info(f"   - Features: {len(self.feature_columns)}")

    @classmethod
    def load_model(cls, model_dir='models'):
        """Cargar TODOS los modelos entrenados"""
        instance = cls()

        # Cargar información del modelo
        info_path = os.path.join(model_dir, 'model_info.pkl')
        model_info = joblib.load(info_path)

        instance.best_model_name = model_info['best_model_name']
        instance.feature_columns = model_info['feature_columns']

        # Cargar resultados de evaluación si existen
        if 'evaluation_results' in model_info:
            instance.evaluation_results = model_info['evaluation_results']

        # Cargar training info si existe
        if 'training_info' in model_info:
            instance.training_info = model_info['training_info']

        # Cargar TODOS los modelos disponibles
        available_models = model_info.get('available_models', [instance.best_model_name])
        for model_name in available_models:
            model_path = os.path.join(model_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                instance.models[model_name] = joblib.load(model_path)
                logger.info(f"   - Cargado: {model_name}")
            else:
                logger.warning(f"   - No encontrado: {model_name}.pkl")

        # Si no se cargó ningún modelo, intentar cargar best_model.pkl
        if not instance.models:
            best_model_path = os.path.join(model_dir, 'best_model.pkl')
            instance.models[instance.best_model_name] = joblib.load(best_model_path)

        # Cargar scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        instance.scaler = joblib.load(scaler_path)

        # Cargar encoders
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        instance.label_encoders = joblib.load(encoders_path)

        logger.info(f"✅ Modelos cargados desde {model_dir}")
        logger.info(f"   - Mejor modelo: {instance.best_model_name}")
        logger.info(f"   - Modelos disponibles: {list(instance.models.keys())}")

        return instance

def train_medical_model(models_to_train=None, test_size=0.2):
    """
    Función principal para entrenar el modelo médico

    Args:
        models_to_train: Lista de modelos a entrenar ['random_forest', 'xgboost'] o None (todos)
        test_size: Proporción de datos para test (default 0.2 = 20%)
    """
    logger.info("🏥 Iniciando entrenamiento del modelo médico...")

    predictor = MedicalMLPredictor()

    try:
        # Cargar y preprocesar datos
        X, y = predictor.load_and_preprocess_data()

        # Entrenar modelos
        X_test, y_test = predictor.train_models(X, y, models_to_train=models_to_train, test_size=test_size)

        # Guardar modelo
        predictor.save_model()

        logger.info("🎉 ¡Entrenamiento completado exitosamente!")
        logger.info("El modelo está listo para hacer predicciones médicas")

        return predictor

    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {str(e)}")
        raise

def get_model_parameters_info():
    """Retorna información detallada sobre los parámetros de cada modelo"""
    return {
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Ensemble de árboles de decisión - Robusto y versátil',
            'parameters': {
                'n_estimators': {
                    'value': 200,
                    'description': 'Número de árboles en el bosque',
                    'justification': 'Más árboles mejoran la generalización y reducen overfitting'
                },
                'max_depth': {
                    'value': 15,
                    'description': 'Profundidad máxima de cada árbol',
                    'justification': 'Profundidad moderada evita overfitting mientras captura patrones complejos'
                },
                'min_samples_split': {
                    'value': 10,
                    'description': 'Mínimo de muestras para dividir un nodo',
                    'justification': 'Evita divisiones con pocas muestras, reduciendo overfitting'
                },
                'min_samples_leaf': {
                    'value': 5,
                    'description': 'Mínimo de muestras en cada hoja',
                    'justification': 'Asegura que cada decisión final tenga suficiente evidencia'
                },
                'max_features': {
                    'value': 'sqrt',
                    'description': 'Número de features consideradas en cada split',
                    'justification': 'Usar sqrt(n_features) reduce correlación entre árboles'
                },
                'class_weight': {
                    'value': 'balanced',
                    'description': 'Ponderación de clases desbalanceadas',
                    'justification': 'CRÍTICO: dataset tiene 91% supervivencia, 9% muerte'
                }
            },
            'advantages': [
                'Robusto ante outliers y datos faltantes',
                'No requiere normalización de datos',
                'Captura relaciones no lineales automáticamente',
                'Provee feature importance interpretable'
            ]
        },
        'xgboost': {
            'name': 'XGBoost',
            'description': 'Gradient Boosting optimizado - Estado del arte en ML',
            'parameters': {
                'n_estimators': {
                    'value': 200,
                    'description': 'Número de boosting rounds',
                    'justification': '200 iteraciones balancean precisión y tiempo de entrenamiento'
                },
                'max_depth': {
                    'value': 6,
                    'description': 'Profundidad máxima de árboles',
                    'justification': 'Profundidad controlada evita overfitting en boosting'
                },
                'learning_rate': {
                    'value': 0.05,
                    'description': 'Tasa de aprendizaje',
                    'justification': 'Learning rate bajo permite aprendizaje más gradual y preciso'
                },
                'subsample': {
                    'value': 0.8,
                    'description': 'Proporción de muestras por árbol',
                    'justification': '80% de muestras introduce variabilidad y reduce overfitting'
                },
                'colsample_bytree': {
                    'value': 0.8,
                    'description': 'Proporción de features por árbol',
                    'justification': '80% de features reduce correlación entre árboles'
                },
                'gamma': {
                    'value': 1,
                    'description': 'Regularización de ganancia mínima',
                    'justification': 'Penaliza splits poco informativos'
                },
                'reg_alpha': {
                    'value': 0.1,
                    'description': 'Regularización L1',
                    'justification': 'Promueve sparsity en los pesos'
                },
                'reg_lambda': {
                    'value': 1,
                    'description': 'Regularización L2',
                    'justification': 'Penaliza pesos grandes, mejora generalización'
                },
                'scale_pos_weight': {
                    'value': 'auto (ratio de clases)',
                    'description': 'Balanceo automático de clases',
                    'justification': 'Calcula ratio supervivientes/muertes automáticamente'
                }
            },
            'advantages': [
                'Mejor rendimiento en AUC-ROC (métrica clave médica)',
                'Maneja desbalance de clases nativamente',
                'Regularización incorporada evita overfitting',
                'Optimizado para velocidad y precisión'
            ]
        }
    }

if __name__ == "__main__":
    train_medical_model()