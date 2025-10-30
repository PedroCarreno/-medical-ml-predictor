# 🎯 CÓMO PROBAR EL PROYECTO - GUÍA COMPLETA

## ✅ **OPCIÓN RECOMENDADA: TODO EN EL NOTEBOOK**

El notebook `ENTREGA_3_MODELADO_Y_EVALUACION.ipynb` tiene **TODO INCLUIDO**:

### **Lo que incluye el notebook:**

1. ✅ **Entrenamiento de 2 modelos** (Random Forest + XGBoost)
2. ✅ **Justificación de parámetros** (cada parámetro explicado)
3. ✅ **5 métricas de evaluación** (Accuracy, Precision, Recall, F1, AUC-ROC)
4. ✅ **Gráficos de comparación** (Barras, ROC curves, Matrices de confusión)
5. ✅ **Análisis de variables importantes**
6. ✅ **DEMO PRÁCTICA con 3 pacientes reales** 🆕

   - **Paciente BAJO RIESGO** 🟢:
     - 45 años, cirugía programada
     - Sin comorbilidades
     - Signos vitales estables
     - Glasgow 15/15

   - **Paciente MODERADO** 🟡:
     - 68 años, emergencia
     - Diabetes
     - Signos vitales alterados
     - Glasgow 13/15

   - **Paciente CRÍTICO** 🔴:
     - 82 años, emergencia
     - Múltiples comorbilidades (Cirrosis + Fallo hepático + Inmunosupresión)
     - Intubado + Ventilación mecánica
     - Glasgow 7/15

7. ✅ **Comparación gráfica de los 3 casos**
8. ✅ **Comparación Random Forest vs XGBoost**
9. ✅ **Guarda modelo automáticamente**

---

## 🚀 **CÓMO EJECUTAR EL NOTEBOOK**

### **Opción 1: Jupyter Notebook** (Recomendada)

```bash
# 1. Activar entorno virtual (si lo tienes)
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Instalar jupyter si no lo tienes
pip install jupyter

# 3. Abrir notebook
jupyter notebook notebooks/ENTREGA_3_MODELADO_Y_EVALUACION.ipynb

# 4. En el navegador, click en "Cell" → "Run All"
```

### **Opción 2: VS Code** (Más fácil)

```bash
# 1. Abrir VS Code
# 2. Abrir la carpeta del proyecto
# 3. Abrir: notebooks/ENTREGA_3_MODELADO_Y_EVALUACION.ipynb
# 4. Click en "Run All" arriba
```

### **Opción 3: Google Colab** (En la nube)

```bash
# 1. Subir el notebook a Google Drive
# 2. Click derecho → "Abrir con Google Colaboratory"
# 3. Subir también PRESENTACION/dataset_clean_final.csv
# 4. Ajustar path del dataset en la celda 4
# 5. Click "Runtime" → "Run all"
```

---

## 📊 **QUÉ VERÁS EN EL NOTEBOOK**

### **Entrenamiento (5-15 minutos):**

```
🚀 Iniciando entrenamiento de modelos...
================================================================================
📊 PASO 1: Carga y preprocesamiento
================================================================================
📂 Cargando dataset LIMPIO desde PRESENTACION/dataset_clean_final.csv
✅ Dataset limpio cargado: (91713, 87)
✅ Nulls en dataset: 0 (debería ser 0)
✅ Eliminando columnas categóricas originales (manteniendo solo _encoded)...
✅ gender -> usando gender_encoded (guardado encoder)
✅ ethnicity -> usando ethnicity_encoded (guardado encoder)
✅ Features finales: 80 columnas (todas numéricas)
📊 Distribución objetivo - Sobrevive: 83,798, Muere: 7,915

================================================================================
🤖 PASO 2: Entrenamiento de modelos
================================================================================
Dividiendo datos en entrenamiento y prueba...
Escalando características...
Entrenando Random Forest...
Entrenando XGBoost...

================================================================================
EVALUACIÓN COMPLETA DE MODELOS - ENTREGA 3
================================================================================

📊 MODELO 1: RANDOM FOREST
------------------------------------------------------------
   ✅ Accuracy (Exactitud):  0.9234 (92.34%)
   ✅ Precision (Precisión): 0.8567 (85.67%)
   ✅ Recall (Sensibilidad): 0.7123 (71.23%)
   ✅ F1-Score:              0.7780
   ✅ AUC-ROC:               0.8945

📊 MODELO 2: XGBOOST
------------------------------------------------------------
   ✅ Accuracy (Exactitud):  0.9312 (93.12%)
   ✅ Precision (Precisión): 0.8734 (87.34%)
   ✅ Recall (Sensibilidad): 0.7456 (74.56%)
   ✅ F1-Score:              0.8045
   ✅ AUC-ROC:               0.9087

🏆 MEJOR MODELO SELECCIONADO: XGBOOST
```

### **Demostración con pacientes:**

```
🟢 CASO 1: PACIENTE DE BAJO RIESGO
================================================================================
Edad: 45 años
Tipo de admisión: Cirugía Programada
Glasgow Coma Score: 15/15 (Normal)
Comorbilidades: Ninguna
Signos vitales: Estables

📊 RESULTADO:
   ✅ Predicción: PACIENTE SOBREVIVIRÁ
   📈 Probabilidad de muerte: 5.23%
   📉 Probabilidad de supervivencia: 94.77%
   ⚠️  Nivel de riesgo: RIESGO BAJO

💊 RECOMENDACIONES:
   1. Monitoreo estándar
   2. Seguimiento rutinario de signos vitales
   3. Continuar tratamiento actual

---

🟡 CASO 2: PACIENTE DE RIESGO MODERADO
================================================================================
Edad: 68 años
Tipo de admisión: Emergencia
Glasgow Coma Score: 13/15 (Levemente disminuido)
Comorbilidades: Diabetes Mellitus
Signos vitales: Taquicardia + Taquipnea + Febrícula

📊 RESULTADO:
   ✅ Predicción: PACIENTE SOBREVIVIRÁ
   📈 Probabilidad de muerte: 32.15%
   📉 Probabilidad de supervivencia: 67.85%
   ⚠️  Nivel de riesgo: RIESGO MODERADO

💊 RECOMENDACIONES:
   1. Atención reforzada
   2. Monitoreo cada 4 horas
   3. Evaluar necesidad de intervenciones adicionales

---

🔴 CASO 3: PACIENTE DE ALTO RIESGO / CRÍTICO
================================================================================
Edad: 82 años
Tipo de admisión: Emergencia
Glasgow Coma Score: 7/15 (CRÍTICO)
Comorbilidades: Cirrosis + Fallo hepático + Diabetes + Inmunosupresión + Fallo renal
Soporte vital: INTUBADO + VENTILACIÓN MECÁNICA
Signos vitales: Taquicardia severa + Hipotensión + Fiebre alta + SpO2 82%

📊 RESULTADO:
   ✅ Predicción: PACIENTE MORIRÁ
   📈 Probabilidad de muerte: 78.92%
   📉 Probabilidad de supervivencia: 21.08%
   ⚠️  Nivel de riesgo: RIESGO CRÍTICO

💊 RECOMENDACIONES:
   1. Atención médica inmediata y urgente
   2. Considerar medidas extraordinarias
   3. Informar a familia sobre pronóstico grave
   4. Evaluar cuidados paliativos si corresponde
   5. Activar protocolo de emergencia
```

### **Gráficos que verás:**

1. **Comparación de métricas** (Barras comparativas)
2. **Curvas ROC** (Random Forest vs XGBoost)
3. **Matrices de confusión** (2 matrices lado a lado)
4. **Importancia de variables** (Top 15 por cada modelo)
5. **Comparación de los 3 casos** (Gráfico de probabilidades)
6. **Comparación de modelos** (Predicciones lado a lado)

---

## 🐳 **PARA PROBAR EL FRONTEND COMPLETO CON DOCKER**

Después de ejecutar el notebook (que guarda el modelo), puedes levantar el sistema completo:

```bash
# 1. Verificar que el modelo existe
dir models\
# Deberías ver: best_model.pkl, scaler.pkl, label_encoders.pkl, model_info.pkl

# 2. Levantar Docker
docker-compose up --build

# 3. Abrir navegador
http://localhost:3000
```

**En el frontend podrás:**
- 🩺 Ingresar datos de un paciente
- 🔄 Elegir entre Random Forest o XGBoost
- 📊 Ver las 3 salidas del modelo
- 💊 Ver recomendaciones médicas

---

## 📝 **DATOS DE LOS 3 PACIENTES PARA EL FRONTEND**

### **Paciente Bajo Riesgo:**
```json
{
  "age": 45,
  "gender": "M",
  "ethnicity": "Caucasian",
  "height": 175,
  "weight": 78,
  "bmi": 25.5,
  "elective_surgery": 1,
  "icu_admit_source": "Operating Room",
  "gcs_eyes_apache": 4,
  "gcs_motor_apache": 6,
  "gcs_verbal_apache": 5,
  "heart_rate_apache": 75,
  "map_apache": 85,
  "resprate_apache": 16,
  "temp_apache": 36.7,
  "diabetes_mellitus": 0,
  "cirrhosis": 0
}
```

### **Paciente Moderado:**
```json
{
  "age": 68,
  "gender": "F",
  "ethnicity": "Caucasian",
  "height": 162,
  "weight": 72,
  "bmi": 27.4,
  "elective_surgery": 0,
  "icu_admit_source": "Accident & Emergency",
  "gcs_eyes_apache": 4,
  "gcs_motor_apache": 5,
  "gcs_verbal_apache": 4,
  "heart_rate_apache": 105,
  "map_apache": 72,
  "resprate_apache": 24,
  "temp_apache": 37.8,
  "diabetes_mellitus": 1,
  "cirrhosis": 0
}
```

### **Paciente Crítico:**
```json
{
  "age": 82,
  "gender": "M",
  "ethnicity": "Caucasian",
  "height": 168,
  "weight": 58,
  "bmi": 20.5,
  "elective_surgery": 0,
  "icu_admit_source": "Accident & Emergency",
  "gcs_eyes_apache": 2,
  "gcs_motor_apache": 3,
  "gcs_verbal_apache": 2,
  "heart_rate_apache": 135,
  "map_apache": 52,
  "resprate_apache": 35,
  "temp_apache": 39.2,
  "intubated_apache": 1,
  "ventilated_apache": 1,
  "diabetes_mellitus": 1,
  "cirrhosis": 1,
  "hepatic_failure": 1,
  "immunosuppression": 1
}
```

---

## ❓ **PREGUNTAS FRECUENTES**

### **¿El notebook es fácil de usar?**
✅ **SÍ** - Solo haces "Run All" y esperas 5-15 minutos. Todo está automatizado.

### **¿Necesito programar algo?**
❌ **NO** - Todo el código ya está escrito. Solo ejecutas las celdas.

### **¿Qué pasa si no tengo Jupyter?**
👉 Usa VS Code (más fácil) o instala con: `pip install jupyter`

### **¿Puedo cambiar los datos de los pacientes?**
✅ **SÍ** - En las celdas de demo hay diccionarios Python que puedes editar fácilmente.

### **¿El notebook tiene gráficos?**
✅ **SÍ** - 6 tipos de gráficos diferentes, todos profesionales y listos para presentar.

### **¿Funciona sin Docker?**
✅ **SÍ** - El notebook funciona completamente solo. Docker es solo para el frontend web.

---

## 🎯 **RESUMEN: QUÉ HACER AHORA**

### **Para la Entrega 3:**

```bash
# 1. Abre el notebook
jupyter notebook notebooks/ENTREGA_3_MODELADO_Y_EVALUACION.ipynb

# 2. Click "Run All"

# 3. Espera 5-15 minutos

# 4. ¡Listo! Tienes:
#    - Modelos entrenados
#    - Métricas de evaluación
#    - Gráficos comparativos
#    - Demos con pacientes
#    - Modelo guardado en models/
```

### **Para probar el sistema completo:**

```bash
# 1. Ejecuta el notebook primero (arriba)

# 2. Levanta Docker
docker-compose up --build

# 3. Abre http://localhost:3000
```

---

**🚀 ¡Todo está listo para usar! El notebook tiene TODO lo que necesitas para la Entrega 3.**
