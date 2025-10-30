# 🏥 GUÍA COMPLETA DE USO - Medical ML Predictor

## 📋 TABLA DE CONTENIDOS
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Características Implementadas](#características-implementadas)
3. [Requisitos del Sistema](#requisitos-del-sistema)
4. [Instalación y Configuración](#instalación-y-configuración)
5. [Uso de la Aplicación](#uso-de-la-aplicación)
6. [Estructura del Proyecto](#estructura-del-proyecto)
7. [API Endpoints](#api-endpoints)
8. [Modelos de Machine Learning](#modelos-de-machine-learning)

---

## 🎯 DESCRIPCIÓN DEL PROYECTO

**Medical ML Predictor** es un sistema completo de predicción de supervivencia hospitalaria en UCI que utiliza **Machine Learning** para analizar datos médicos de pacientes y predecir su probabilidad de supervivencia.

### Características Principales:
- ✅ **87 columnas** del dataset completo (todas las variables predictoras)
- ✅ **2 modelos ML**: Random Forest y XGBoost
- ✅ **Selección de modelo** en tiempo real
- ✅ **3 tipos de predicción**: Binaria, Probabilidades y Niveles de Riesgo
- ✅ **Comparación de modelos** con métricas y ROC
- ✅ **Frontend React** con formulario completo
- ✅ **Backend Flask** con API RESTful
- ✅ **Docker** para despliegue fácil

---

## 🚀 CARACTERÍSTICAS IMPLEMENTADAS

### ✅ FRONTEND (React)
- **87 campos organizados en 13 grupos**:
  1. Datos Demográficos
  2. Comorbilidades CRÍTICAS
  3. Estado Neurológico (Glasgow)
  4. Soporte Vital
  5. Signos Vitales Apache
  6. Hospitalización
  7. Presión Arterial - Día 1 (todas las variantes: invasiva/no invasiva)
  8. Otros Signos Vitales - Día 1
  9. Presión Arterial - Hora 1 (todas las variantes)
  10. Otros Signos Vitales - Hora 1
  11. Laboratorios (glucosa, potasio)
  12. Scores Apache (probabilidades)
  13. Diagnósticos Apache

- **Selector de Modelo ML**:
  - Random Forest (mayor precisión general)
  - XGBoost (mejor AUC-ROC - recomendado)

- **Casos Predefinidos** para pruebas rápidas:
  - 👤 Paciente Estable
  - 🚨 Paciente Crítico
  - 👨‍💼 Paciente Joven
  - 👴 Paciente Anciano

- **Encoding automático** de variables categóricas:
  - `gender`: M → 1, F → 0
  - `ethnicity`: Caucasian, African American, etc. → numérico
  - `icu_admit_source`, `icu_stay_type`, `icu_type` → numérico
  - `apache_3j_bodysystem`, `apache_2_bodysystem` → numérico

### ✅ BACKEND (Flask)
- **API RESTful completa**:
  - `POST /api/predict` - Predicción con modelo seleccionado
  - `POST /api/train` - Entrenar modelos
  - `GET /api/model-info` - Info de modelos cargados
  - `GET /api/model-comparison` - Comparación de ambos modelos
  - `POST /api/compare-predictions` - Comparar predicciones de ambos modelos para un paciente
  - `POST /api/predict-explain` - Predicción con explicación detallada

- **Soporte para ambos modelos**:
  - Carga automática de Random Forest y XGBoost
  - Selección dinámica del modelo a usar
  - Comparación en tiempo real

### ✅ MACHINE LEARNING
- **2 Modelos Entrenados**:
  1. **Random Forest**:
     - 200 árboles
     - Accuracy: ~91%
     - Precision: ~49%
     - Recall: ~45%
     - AUC-ROC: **0.877**

  2. **XGBoost** (RECOMENDADO):
     - 200 boosting rounds
     - Accuracy: ~84%
     - Precision: ~31%
     - Recall: **75%** (mejor detección de casos críticos)
     - AUC-ROC: **0.888** (mejor discriminación)

- **Características**:
  - Balanceo de clases automático
  - Regularización para evitar overfitting
  - Importancia de variables
  - 3 tipos de salidas:
    1. Clasificación Binaria (sobrevive/muere)
    2. Probabilidades detalladas (%)
    3. Niveles de Riesgo (BAJO/MODERADO/ALTO/CRÍTICO) con recomendaciones

---

## 💻 REQUISITOS DEL SISTEMA

### Opción 1: Con Docker (RECOMENDADO)
- Docker Desktop instalado
- 4GB RAM mínimo
- 2GB espacio en disco

### Opción 2: Sin Docker
- Python 3.8+
- Node.js 14+
- npm o yarn

---

## 🔧 INSTALACIÓN Y CONFIGURACIÓN

### OPCIÓN A: Con Docker (Más Fácil)

1. **Asegurarse de tener los datos**:
```bash
# Verificar que existe el dataset limpio
ls PRESENTACION/dataset_clean_final.csv
```

2. **Entrenar los modelos** (IMPORTANTE - hacer ANTES de Docker):
```bash
# Instalar dependencias Python localmente solo para entrenar
pip install -r requirements.txt

# Entrenar modelos (esto crea models/ con ambos modelos)
python ml_service/train_model.py
```

3. **Construir y levantar con Docker**:
```bash
# Construir imágenes
docker-compose build

# Levantar servicios
docker-compose up
```

4. **Acceder a la aplicación**:
```
Frontend: http://localhost:3000
Backend: http://localhost:8000
```

### OPCIÓN B: Sin Docker (Manual)

1. **Backend**:
```bash
cd backend
pip install -r requirements.txt

# Entrenar modelos primero
cd ..
python ml_service/train_model.py

# Luego iniciar backend
cd backend
python app.py
```

2. **Frontend**:
```bash
cd frontend
npm install
npm start
```

3. **Acceder**:
```
Frontend: http://localhost:3000
Backend API: http://localhost:8000
```

---

## 📱 USO DE LA APLICACIÓN

### 1. ENTRENAR MODELOS (Primer Uso)

**En el frontend**:
1. Ir a la pestaña "Modelo"
2. Click en "🚀 Entrenar Modelo"
3. Esperar ~2-5 minutos (entrena ambos modelos)
4. Ver métricas de evaluación

**Desde línea de comandos** (más rápido):
```bash
python ml_service/train_model.py
```

### 2. REALIZAR PREDICCIONES

1. **Ir a pestaña "Predicción"**

2. **Seleccionar modelo**:
   - Random Forest (mayor exactitud)
   - XGBoost (mejor recall - recomendado)

3. **Opción A: Usar caso predefinido**:
   - Click en "👤 Paciente Estable" / "🚨 Paciente Crítico" / etc.
   - Se llenan automáticamente todos los campos

4. **Opción B: Ingresar datos manualmente**:
   - Completar los 87 campos organizados en grupos
   - Campos marcados con * son requeridos
   - Campos con ⚠️ son comorbilidades críticas

5. **Click en "🔍 Realizar Predicción"**

6. **Ver resultados**:
   - ✅ SALIDA 1: Clasificación Binaria (Sobrevive/Muere)
   - 📊 SALIDA 2: Probabilidades (% de muerte, % de supervivencia)
   - ⚠️ SALIDA 3: Nivel de Riesgo + Recomendaciones médicas

### 3. COMPARAR MODELOS

**Endpoint en desarrollo** - Próximamente en el frontend:
```bash
# Obtener comparación general
curl http://localhost:8000/api/model-comparison

# Comparar predicciones para un paciente
curl -X POST http://localhost:8000/api/compare-predictions \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "gender": "M", ...}'
```

---

## 📁 ESTRUCTURA DEL PROYECTO

```
medical-ml-predictor/
│
├── PRESENTACION/                    # Dataset y código de limpieza
│   ├── dataset.csv                  # Dataset original
│   ├── dataset_clean_final.csv      # Dataset limpio (87 columnas)
│   └── clean_dataset_complete.py    # Script de limpieza
│
├── notebooks/                       # Jupyter Notebooks
│   ├── ENTREGA_3_MODELADO_Y_EVALUACION.ipynb
│   └── presentacion_limpieza_dataset.ipynb
│
├── ml_service/                      # Código de ML
│   └── train_model.py               # Entrenamiento de modelos
│
├── backend/                         # Backend Flask
│   ├── app.py                       # API principal
│   ├── services/
│   │   ├── ml_service.py            # Servicio ML
│   │   └── data_processor.py
│   └── requirements.txt
│
├── frontend/                        # Frontend React
│   ├── src/
│   │   ├── App.js                   # Componente principal (87 campos)
│   │   └── App.css                  # Estilos
│   ├── public/
│   └── package.json
│
├── models/                          # Modelos entrenados (generado)
│   ├── random_forest.pkl            # Modelo Random Forest
│   ├── xgboost.pkl                  # Modelo XGBoost
│   ├── best_model.pkl               # Mejor modelo
│   ├── scaler.pkl                   # Escalador
│   ├── label_encoders.pkl           # Encoders categóricos
│   └── model_info.pkl               # Metadata
│
├── docker-compose.yml               # Orquestación Docker
├── GUIA_COMPLETA_USO.md            # Esta guía
└── README.md
```

---

## 🔌 API ENDPOINTS

### 1. Predicción

**POST /api/predict**
```json
{
  "model_name": "xgboost",  // o "random_forest"
  "age": 45,
  "gender": "M",
  "ethnicity": "Caucasian",
  "height": 175,
  "weight": 75,
  "bmi": 24.5,
  // ... resto de las 87 columnas
}
```

**Respuesta**:
```json
{
  "status": "success",
  "resultado_binario": {
    "prediction": 0,
    "result_text": "PACIENTE SOBREVIVIRÁ"
  },
  "probabilidades": {
    "prob_muerte": 12.5,
    "prob_supervivencia": 87.5,
    "confianza": 87.5
  },
  "evaluacion_riesgo": {
    "nivel_riesgo": "RIESGO BAJO",
    "probabilidad_muerte": 12.5,
    "recomendaciones": [...]
  },
  "modelo_info": {
    "algoritmo_usado": "xgboost",
    "variables_mas_importantes": [...]
  }
}
```

### 2. Comparación de Modelos

**GET /api/model-comparison**
```json
{
  "status": "success",
  "best_model": "xgboost",
  "available_models": ["random_forest", "xgboost"],
  "models_metrics": {
    "random_forest": {
      "accuracy": 0.9121,
      "precision": 0.4896,
      "recall": 0.4473,
      "f1_score": 0.4675,
      "auc_roc": 0.8767
    },
    "xgboost": {
      "accuracy": 0.8352,
      "precision": 0.3107,
      "recall": 0.7467,
      "f1_score": 0.4388,
      "auc_roc": 0.8875
    }
  },
  "feature_importance": {...}
}
```

---

## 🤖 MODELOS DE MACHINE LEARNING

### Encoding de Variables

El sistema maneja automáticamente el encoding de variables categóricas:

| Variable Original | Tipo | Encoding |
|------------------|------|----------|
| `gender` | M / F | M=1, F=0 |
| `ethnicity` | Caucasian, African American, etc. | LabelEncoder (0-4) |
| `icu_admit_source` | Floor, Emergency, etc. | LabelEncoder (0-4) |
| `icu_stay_type` | admit / transfer | admit=0, transfer=1 |
| `icu_type` | CTICU, Med-Surg ICU, etc. | LabelEncoder (0-5) |
| `apache_3j_bodysystem` | Cardiovascular, etc. | LabelEncoder (0-9) |
| `apache_2_bodysystem` | Cardiovascular, etc. | LabelEncoder (0-9) |

**IMPORTANTE**: En el frontend, puedes ingresar valores como "M", "F", "Caucasian", etc.
El backend automáticamente los convierte a números usando los LabelEncoders entrenados.

### Parámetros de los Modelos

**Random Forest**:
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
```

**XGBoost**:
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1,
    scale_pos_weight=10.6,
    random_state=42
)
```

---

## ✅ VERIFICACIÓN Y TROUBLESHOOTING

### Verificar que todo funciona:

1. **Modelos entrenados**:
```bash
ls models/
# Debe mostrar:
# - random_forest.pkl
# - xgboost.pkl
# - best_model.pkl
# - scaler.pkl
# - label_encoders.pkl
# - model_info.pkl
```

2. **Dataset limpio**:
```bash
python -c "import pandas as pd; df=pd.read_csv('PRESENTACION/dataset_clean_final.csv'); print(f'Shape: {df.shape}'); print(f'Nulls: {df.isnull().sum().sum()}')"
# Debe mostrar: Shape: (91713, 87), Nulls: 0
```

3. **Test de predicción**:
```bash
python test_prediction.py
# Debe hacer una predicción exitosa con ambos modelos
```

### Problemas Comunes:

**Error: "Modelo no disponible"**
- Solución: Entrenar modelos con `python ml_service/train_model.py`

**Error: "Dataset no encontrado"**
- Solución: Verificar que existe `PRESENTACION/dataset_clean_final.csv`

**Error: "Port already in use"**
- Solución: Cambiar puertos en docker-compose.yml o matar proceso

**Frontend no carga**:
- Verificar que backend está corriendo en http://localhost:8000
- Revisar consola del navegador para errores de CORS

---

## 📊 MÉTRICAS Y RESULTADOS

### Comparación de Modelos:

| Métrica | Random Forest | XGBoost | Ganador |
|---------|--------------|---------|---------|
| **Accuracy** | 91.21% | 83.52% | RF |
| **Precision** | 48.96% | 31.07% | RF |
| **Recall** | 44.73% | **74.67%** | XGB |
| **F1-Score** | 46.75% | 43.88% | RF |
| **AUC-ROC** | 0.8767 | **0.8875** | XGB |

**Recomendación**: Usar **XGBoost** porque:
- Mejor AUC-ROC (métrica clave para datos desbalanceados)
- Mayor Recall (detecta más casos críticos)
- Menos falsos negativos (crítico en medicina)

---

## 🎓 AUTORES Y CRÉDITOS

**Estudiante**: [Tu Nombre]
**Materia**: Ciencia de Datos
**Universidad**: UTN
**Año**: 2025

**Dataset**: WiDS Datathon 2020 (Kaggle)
- 91,713 pacientes
- 87 variables predictoras
- Variable objetivo: hospital_death

---

## 📝 NOTAS FINALES

1. **Código de Limpieza**: Ver `PRESENTACION/clean_dataset_complete.py`
2. **Notebook de Limpieza**: Ver `PRESENTACION/presentacion_limpieza_dataset.ipynb`
3. **Notebook de Modelado**: Ver `notebooks/ENTREGA_3_MODELADO_Y_EVALUACION.ipynb`
4. **Documentación API**: Ver `docs/` (si existe)

**¡Listo para usar! 🎉**

Para cualquier duda, revisar el código o consultar los notebooks.
