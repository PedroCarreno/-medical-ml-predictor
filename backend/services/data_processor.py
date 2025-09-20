import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class MedicalDataProcessor:
    """Procesador de datos médicos para el dataset de supervivencia hospitalaria"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'hospital_death'

    def load_and_preprocess(self, csv_path):
        """
        Cargar y preprocesar el dataset médico

        Returns:
            X, y: Features y target procesados
        """
        logger.info(f"Cargando dataset desde {csv_path}")

        # Cargar datos
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset cargado: {df.shape}")

        # Preprocesar
        X, y = self.preprocess_medical_data(df)

        return X, y

    def preprocess_medical_data(self, df):
        """
        Preprocesar datos médicos específicos
        """
        logger.info("Iniciando preprocesamiento de datos médicos")

        # 1. Eliminar columnas de identificación (no predictoras)
        id_columns = ['encounter_id', 'patient_id', 'hospital_id']
        df_clean = df.drop(columns=[col for col in id_columns if col in df.columns])

        # 2. Eliminar columnas vacías o irrelevantes
        # La columna 82 parece estar vacía según la documentación
        empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
        if empty_cols:
            logger.info(f"Eliminando columnas vacías: {empty_cols}")
            df_clean = df_clean.drop(columns=empty_cols)

        # 3. Separar target
        if self.target_column not in df_clean.columns:
            raise ValueError(f"Columna objetivo '{self.target_column}' no encontrada")

        y = df_clean[self.target_column].copy()
        X = df_clean.drop(columns=[self.target_column])

        # 4. Procesar features
        X = self._process_features(X)

        logger.info(f"Preprocesamiento completado: {X.shape} features, {len(y)} samples")
        logger.info(f"Distribución target: {y.value_counts().to_dict()}")

        return X, y

    def _process_features(self, X):
        """Procesar features específicas del dataset médico"""

        # Variables categóricas importantes
        categorical_vars = ['gender', 'ethnicity', 'icu_admit_source', 'icu_stay_type',
                          'icu_type', 'apache_2_bodysystem', 'apache_3j_bodysystem']

        # Variables numéricas que necesitan scaling
        numeric_vars = ['age', 'height', 'weight', 'bmi', 'pre_icu_los_days']

        # Procesar cada tipo de variable
        for col in X.columns:

            # 1. Manejar valores faltantes
            if X[col].dtype == 'object':  # Categórica
                # Rellenar con 'Unknown' para categóricas
                X[col] = X[col].fillna('Unknown')

                # Codificar categóricas
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    # Para nuevos datos, usar encoder existente
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))

            else:  # Numérica
                # Para numéricas, usar mediana por variable específica
                if col in ['age']:
                    X[col] = X[col].fillna(X[col].median())
                elif 'apache' in col.lower():
                    # Variables APACHE: usar 0 para faltantes
                    X[col] = X[col].fillna(0)
                elif any(vital in col.lower() for vital in ['heart', 'temp', 'resprate', 'bp']):
                    # Signos vitales: usar mediana
                    X[col] = X[col].fillna(X[col].median())
                else:
                    # Otros: usar mediana
                    X[col] = X[col].fillna(X[col].median())

        # Guardar nombres de columnas
        self.feature_columns = X.columns.tolist()

        return X

    def get_feature_importance_info(self):
        """
        Información sobre las variables más importantes según la documentación médica
        """
        important_features = {
            'critical': [
                'age',  # Factor crítico según documentación
                'apache_4a_hospital_death_prob',  # Probabilidad APACHE
                'apache_4a_icu_death_prob',
                'gcs_eyes_apache',  # Escala Glasgow
                'gcs_motor_apache',
                'gcs_verbal_apache',
                'aids',  # Comorbilidades críticas
                'hepatic_failure',
                'solid_tumor_with_metastasis',
                'intubated_apache',  # Soporte vital
                'ventilated_apache'
            ],
            'important': [
                'bmi',
                'diabetes_mellitus',
                'cirrhosis',
                'heart_rate_apache',
                'map_apache',
                'temp_apache',
                'elective_surgery'
            ],
            'vitals': [
                'd1_heartrate_max', 'd1_heartrate_min',
                'd1_mbp_max', 'd1_mbp_min',
                'd1_temp_max', 'd1_temp_min',
                'd1_glucose_max', 'd1_glucose_min'
            ]
        }

        return important_features

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Dividir datos en entrenamiento y prueba"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def scale_features(self, X_train, X_test=None):
        """Escalar features numéricas"""
        logger.info("Escalando features")

        # Identificar columnas numéricas (excluyendo categóricas codificadas)
        numeric_columns = []
        for col in X_train.columns:
            if X_train[col].dtype in ['int64', 'float64']:
                # Verificar si no es una variable categórica codificada
                unique_vals = X_train[col].nunique()
                if unique_vals > 10:  # Probablemente numérica real
                    numeric_columns.append(col)

        if numeric_columns:
            logger.info(f"Escalando {len(numeric_columns)} columnas numéricas")
            X_train_scaled = X_train.copy()
            X_train_scaled[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])

            if X_test is not None:
                X_test_scaled = X_test.copy()
                X_test_scaled[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
                return X_train_scaled, X_test_scaled

            return X_train_scaled

        return X_train if X_test is None else (X_train, X_test)

    def get_dataset_summary(self, df):
        """Generar resumen del dataset"""
        summary = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }

        if self.target_column in df.columns:
            target_dist = df[self.target_column].value_counts()
            summary['target_distribution'] = target_dist.to_dict()
            summary['mortality_rate'] = df[self.target_column].mean()

        return summary