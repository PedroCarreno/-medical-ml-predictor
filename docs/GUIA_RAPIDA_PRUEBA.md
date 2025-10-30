# 🚀 GUÍA RÁPIDA - CÓMO PROBAR EL PROYECTO

## ⚠️ IMPORTANTE: HAY 2 FORMAS DE PROBAR

### **OPCIÓN 1: PRIMERO ENTRENAR EL MODELO (RECOMENDADO PARA ENTREGA 3)** ⭐

Esta opción te permite ver todo el proceso de entrenamiento, métricas y comparación de modelos.

### **OPCIÓN 2: USAR DOCKER (PARA DEMO RÁPIDA)**

Esta opción levanta el sistema completo pero necesitas tener el modelo ya entrenado.

---

## 🎓 OPCIÓN 1: ENTRENAR Y EVALUAR MODELOS (ENTREGA 3)

### **Paso 1: Abrir el Notebook**

```bash
# Desde la raíz del proyecto
jupyter notebook notebooks/ENTREGA_3_MODELADO_Y_EVALUACION.ipynb
```

O si usas VS Code:
- Abre el archivo `notebooks/ENTREGA_3_MODELADO_Y_EVALUACION.ipynb`
- Click en "Run All" arriba

### **Paso 2: Ejecutar todas las celdas**

El notebook hará automáticamente:

1. ✅ Cargar el dataset limpio (91,713 pacientes)
2. ✅ Entrenar Random Forest (con 200 árboles)
3. ✅ Entrenar XGBoost (con 200 boosting rounds)
4. ✅ Evaluar ambos modelos con 5 métricas
5. ✅ Comparar modelos visualmente
6. ✅ Mostrar variables más importantes
7. ✅ Guardar el mejor modelo en `models/`

### **Paso 3: Ver los resultados**

El notebook mostrará:

```
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

**Tiempo estimado:** 5-15 minutos (dependiendo de tu PC)

---

## 🐳 OPCIÓN 2: PROBAR CON DOCKER (DEMO COMPLETA)

### **Requisitos previos:**
- Docker Desktop instalado
- Tener el modelo entrenado (ejecuta Opción 1 primero)

### **Paso 1: Verificar que tienes el modelo**

```bash
# Deberías ver estos archivos:
dir models\
```

Deberías ver:
```
best_model.pkl
scaler.pkl
label_encoders.pkl
model_info.pkl
```

⚠️ **Si NO los tienes:** Ejecuta primero la Opción 1

### **Paso 2: Levantar Docker**

```bash
# Desde la raíz del proyecto
docker-compose up --build
```

Verás algo como:
```
✅ Backend running on http://localhost:5000
✅ Frontend running on http://localhost:3000
✅ ML Service ready
```

### **Paso 3: Abrir el navegador**

Abre tu navegador en: **http://localhost:3000**

Verás la interfaz web del sistema.

---

## 🖥️ OPCIÓN 3: PRUEBA RÁPIDA SIN DOCKER (SOLO BACKEND)

Esta es la forma MÁS RÁPIDA para probar el modelo localmente.

### **Paso 1: Instalar dependencias**

```bash
# Crear entorno virtual (si no lo tienes)
python -m venv venv

# Activar entorno
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### **Paso 2: Entrenar el modelo (si no lo hiciste)**

```bash
cd ml_service
python train_model.py
```

Verás:
```
🏥 Iniciando entrenamiento del modelo médico...
📊 Cargando dataset LIMPIO desde PRESENTACION/dataset_clean_final.csv
✅ Dataset limpio cargado: (91713, 87)
✅ Nulls en dataset: 0 (debería ser 0)
...
🎉 ¡Entrenamiento completado exitosamente!
```

### **Paso 3: Probar el modelo con un paciente de prueba**

Crea un archivo `test_prediction.py`:

```python
from ml_service.train_model import MedicalMLPredictor

# Cargar modelo entrenado
predictor = MedicalMLPredictor.load_model('models')

# Datos de prueba de un paciente
patient_data = {
    'age': 68,
    'gender': 'M',
    'ethnicity': 'Caucasian',
    'height': 180,
    'weight': 85,
    'bmi': 26.2,
    'elective_surgery': 0,
    'icu_admit_source': 'Accident & Emergency',
    'icu_stay_type': 'admit',
    'icu_type': 'Med-Surg ICU',
    'pre_icu_los_days': 0.5,
    # ... más variables (ver ejemplo completo abajo)
}

# Hacer predicción con el mejor modelo
result = predictor.predict_single_patient(patient_data)

print("=" * 80)
print("RESULTADO DE LA PREDICCIÓN")
print("=" * 80)
print(f"\n🎯 SALIDA 1 - Clasificación Binaria:")
print(f"   Predicción: {result['salida_1_binaria']['result_text']}")

print(f"\n📊 SALIDA 2 - Probabilidades:")
print(f"   Probabilidad de muerte: {result['salida_2_probabilidades']['prob_muerte']:.2f}%")
print(f"   Probabilidad de supervivencia: {result['salida_2_probabilidades']['prob_supervivencia']:.2f}%")

print(f"\n⚠️ SALIDA 3 - Nivel de Riesgo:")
print(f"   Nivel: {result['salida_3_riesgo']['nivel_riesgo']}")
print(f"   Recomendaciones:")
for rec in result['salida_3_riesgo']['recomendaciones']:
    print(f"      • {rec}")

print(f"\n🤖 Modelo usado: {result['modelo_usado']}")
```

Ejecutar:
```bash
python test_prediction.py
```

### **Paso 4: Probar con diferentes modelos**

```python
# Probar con Random Forest
result_rf = predictor.predict_single_patient(patient_data, model_name='random_forest')

# Probar con XGBoost
result_xgb = predictor.predict_single_patient(patient_data, model_name='xgboost')

# Comparar resultados
print(f"Random Forest predice: {result_rf['salida_2_probabilidades']['prob_muerte']:.2f}% muerte")
print(f"XGBoost predice: {result_xgb['salida_2_probabilidades']['prob_muerte']:.2f}% muerte")
```

---

## 📝 EJEMPLO COMPLETO DE DATOS DE PACIENTE

```python
patient_data_completo = {
    # Demográficas
    'age': 68,
    'gender': 'M',
    'ethnicity': 'Caucasian',
    'height': 180.3,
    'weight': 85.2,
    'bmi': 26.2,

    # Admisión
    'elective_surgery': 0,
    'icu_admit_source': 'Accident & Emergency',
    'icu_stay_type': 'admit',
    'icu_type': 'Med-Surg ICU',
    'pre_icu_los_days': 0.5,

    # Scores APACHE
    'apache_2_diagnosis': 113.0,
    'apache_3j_diagnosis': 502.01,
    'apache_post_operative': 0,
    'arf_apache': 0,

    # Glasgow Coma Scale
    'gcs_eyes_apache': 4,
    'gcs_motor_apache': 6,
    'gcs_unable_apache': 0,
    'gcs_verbal_apache': 5,

    # Signos vitales APACHE
    'heart_rate_apache': 88,
    'intubated_apache': 0,
    'map_apache': 70,
    'resprate_apache': 18,
    'temp_apache': 36.8,
    'ventilated_apache': 0,

    # Presión arterial día 1
    'd1_diasbp_max': 82,
    'd1_diasbp_min': 48,
    'd1_sysbp_max': 138,
    'd1_sysbp_min': 92,

    # Signos vitales día 1
    'd1_heartrate_max': 95,
    'd1_heartrate_min': 65,
    'd1_resprate_max': 22,
    'd1_resprate_min': 12,
    'd1_spo2_max': 99,
    'd1_spo2_min': 94,
    'd1_temp_max': 37.2,
    'd1_temp_min': 36.5,

    # Labs día 1
    'd1_glucose_max': 142,
    'd1_glucose_min': 98,
    'd1_potassium_max': 4.1,
    'd1_potassium_min': 3.8,

    # Comorbilidades
    'aids': 0,
    'cirrhosis': 0,
    'diabetes_mellitus': 1,
    'hepatic_failure': 0,
    'immunosuppression': 0,
    'leukemia': 0,
    'lymphoma': 0,
    'solid_tumor_with_metastasis': 0,

    # Sistemas corporales
    'apache_3j_bodysystem': 'Cardiovascular',
    'apache_2_bodysystem': 'Cardiovascular'
}
```

---

## ❓ PREGUNTAS FRECUENTES

### **¿Cuál opción debo usar para la Entrega 3?**
👉 **Opción 1 (Notebook)** - Te da todos los gráficos, métricas y análisis que pide la profesora.

### **¿El notebook es fácil de usar?**
👉 **SÍ** - Solo haces click en "Run All" y se ejecuta todo solo. Está súper documentado.

### **¿Cuánto tarda en entrenar?**
👉 **5-15 minutos** dependiendo de tu PC (dataset de 91,713 pacientes, 2 modelos)

### **¿Necesito saber programar para usarlo?**
👉 **NO** - El notebook tiene TODO explicado paso a paso. Solo ejecutas las celdas.

### **¿Puedo cambiar los parámetros de los modelos?**
👉 **SÍ** - Están en `ml_service/train_model.py` con comentarios explicando cada uno.

### **¿Cómo sé si funcionó bien?**
👉 Verás:
- ✅ AUC-ROC > 0.85 (excelente)
- ✅ Gráficos de comparación
- ✅ Archivos en carpeta `models/`

---

## 🎯 RESUMEN: ¿QUÉ HAGO AHORA?

### **Para la Entrega 3 (Recomendado):**

```bash
# 1. Abrir notebook
jupyter notebook notebooks/ENTREGA_3_MODELADO_Y_EVALUACION.ipynb

# 2. Click en "Run All"

# 3. Esperar 5-15 minutos

# 4. Listo! Tienes todos los gráficos y análisis
```

### **Para probar predicciones rápido:**

```bash
# 1. Entrenar modelo
cd ml_service
python train_model.py

# 2. Crear test_prediction.py (código arriba)

# 3. Ejecutar
python test_prediction.py
```

### **Para demo completa con interfaz web:**

```bash
# 1. Entrenar modelo primero (opción 1)

# 2. Levantar Docker
docker-compose up --build

# 3. Abrir http://localhost:3000
```

---

## 📞 ¿NECESITAS AYUDA?

Si algo no funciona:

1. **Verifica que el dataset limpio existe:**
   ```bash
   dir PRESENTACION\dataset_clean_final.csv
   ```

2. **Verifica dependencias instaladas:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Mira los logs del notebook** - Cualquier error se muestra claramente

---

**¡Todo está listo para probar! 🚀**
