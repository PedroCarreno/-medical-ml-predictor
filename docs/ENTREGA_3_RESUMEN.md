# 📊 ENTREGA 3: Modelado y Análisis de Resultados

## ✅ Objetivos Cumplidos

### 1. Seleccionar al menos 2 algoritmos de modelado
✅ **Implementados:**
- **Random Forest** (Ensemble de árboles de decisión)
- **XGBoost** (Gradient Boosting optimizado)

### 2. Definir y justificar parámetros utilizados
✅ **Totalmente documentado en la app** - Pestaña "Modelo" muestra:

#### Random Forest
| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `n_estimators` | 200 | Más árboles mejoran la generalización y reducen overfitting |
| `max_depth` | 15 | Profundidad moderada evita overfitting mientras captura patrones complejos |
| `min_samples_split` | 10 | Evita divisiones con pocas muestras, reduciendo overfitting |
| `min_samples_leaf` | 5 | Asegura que cada decisión final tenga suficiente evidencia |
| `max_features` | sqrt | Usar sqrt(n_features) reduce correlación entre árboles |
| `class_weight` | balanced | **CRÍTICO**: dataset tiene 91% supervivencia, 9% muerte |

#### XGBoost
| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `n_estimators` | 200 | 200 iteraciones balancean precisión y tiempo de entrenamiento |
| `max_depth` | 6 | Profundidad controlada evita overfitting en boosting |
| `learning_rate` | 0.05 | Learning rate bajo permite aprendizaje más gradual y preciso |
| `subsample` | 0.8 | 80% de muestras introduce variabilidad y reduce overfitting |
| `colsample_bytree` | 0.8 | 80% de features reduce correlación entre árboles |
| `gamma` | 1 | Penaliza splits poco informativos |
| `reg_alpha` | 0.1 | Regularización L1 promueve sparsity |
| `reg_lambda` | 1 | Regularización L2 penaliza pesos grandes |
| `scale_pos_weight` | auto | Calcula ratio supervivientes/muertes automáticamente |

### 3. Evaluar el desempeño con métricas apropiadas
✅ **Métricas implementadas:**

#### Random Forest
- **Accuracy**: 91.21%
- **Precision**: 48.96%
- **Recall**: 44.73%
- **F1-Score**: 46.75%
- **AUC-ROC**: 87.67%

#### XGBoost
- **Accuracy**: 83.52%
- **Precision**: 31.07%
- **Recall**: 74.67%
- **F1-Score**: 43.88%
- **AUC-ROC**: 88.75% ⭐ **MEJOR**

✅ **Interpretación de métricas** (visible en la app):
- **Accuracy**: Porcentaje de predicciones correctas (⚠️ puede ser engañoso con datos desbalanceados)
- **Precision**: De los que predijo "muerte", cuántos realmente murieron
- **Recall**: De los que murieron, cuántos detectó el modelo (MUY IMPORTANTE en medicina)
- **F1-Score**: Media armónica de Precision y Recall
- **AUC-ROC**: Capacidad de discriminación del modelo (0.5 = azar, 1.0 = perfecto)

### 4. Identificar patrones o tendencias relevantes
✅ **Variables más importantes identificadas:**
1. `ventilated_apache` (27.06%) - Ventilación mecánica es el predictor más fuerte
2. `gcs_motor_apache` (5.74%) - Estado neurológico crítico
3. `elective_surgery` (4.22%) - Tipo de cirugía importa
4. `d1_sysbp_noninvasive_min` (3.56%) - Presión arterial mínima
5. `gcs_verbal_apache` (2.88%) - Comunicación verbal del paciente

✅ **Patrones identificados:**
- **Soporte vital**: Pacientes con ventilación mecánica tienen mucho mayor riesgo
- **Estado neurológico**: Glasgow bajo (< 8) indica riesgo muy alto
- **Comorbilidades críticas**: SIDA, cirrosis, cáncer metastásico son predictores fuertes
- **Edad**: > 70 años aumenta significativamente el riesgo

### 5. Interpretación del conocimiento obtenido
✅ **Interpretación clínica implementada:**

El modelo identifica que los **factores de soporte vital** son los predictores más importantes:
- Ventilación mecánica indica que el paciente no puede respirar por sí mismo
- Estado neurológico (Glasgow) refleja la función cerebral
- Presión arterial baja indica shock o falla circulatoria

**Insight clave**: El modelo confirma que los pacientes que requieren soporte vital intensivo (ventilación, presión arterial inestable) tienen mayor mortalidad, lo cual es consistente con la literatura médica.

## 🎯 Características Implementadas en la Aplicación

### Pestaña "Predicción"
1. ✅ Selector de modelo (Random Forest o XGBoost)
2. ✅ Formulario completo con 77 features médicas
3. ✅ 3 tipos de salida:
   - Clasificación binaria (Sobrevive/Muere)
   - Probabilidades detalladas (%)
   - Nivel de riesgo con recomendaciones

### Pestaña "Modelo" (NUEVA - Entrega 3)
1. ✅ **Configuración de entrenamiento:**
   - Seleccionar qué modelo(s) entrenar (RF, XGBoost, o ambos)
   - Configurar train/test split (80/20, 75/25, 70/30)

2. ✅ **Información del Dataset:**
   - Total de muestras y distribución train/test
   - Distribución de muertes vs supervivientes
   - Lista completa de 77 features utilizadas
   - Features eliminadas y por qué (categóricas, apache probs)

3. ✅ **Parámetros de los Modelos:**
   - Tabla completa de parámetros para cada modelo
   - Justificación técnica de cada parámetro
   - Ventajas de cada algoritmo

4. ✅ **Comparación de Modelos:**
   - Gráfico de barras comparativo
   - Tabla con todas las métricas
   - Identificación del mejor modelo (XGBoost por AUC-ROC)
   - Explicación de por qué se eligió cada métrica

5. ✅ **Feature Importance:**
   - Top 10 variables más importantes del mejor modelo
   - Porcentaje de importancia de cada feature

## 📁 Dataset y Features

### Dataset Utilizado
- **Fuente**: `PRESENTACION/dataset_clean_final.csv`
- **Total de registros**: ~91,000 pacientes UCI
- **Distribución de clases**:
  - Sobreviven: ~91%
  - Mueren: ~9% (clase desbalanceada)

### Train/Test Split
- **Default**: 80% Training / 20% Test
- **Configurable** en la app (70/30, 75/25, 80/20)
- **Estratificado**: Mantiene la proporción de clases
- **Random State**: 42 (reproducibilidad)

### Features Utilizadas (77 total)
✅ **Features numéricas directas:**
- Demográficas: age, height, weight, bmi
- Signos vitales Apache: heart_rate, map, resprate, temp
- Signos vitales día 1: d1_heartrate_max/min, d1_sysbp_max/min, d1_spo2_max/min, etc.
- Signos vitales hora 1: h1_heartrate_max/min, h1_sysbp_max/min, etc.
- Laboratorios: d1_glucose_max/min, d1_potassium_max/min
- Diagnósticos: apache_2_diagnosis, apache_3j_diagnosis

✅ **Features categóricas (codificadas numéricamente):**
- gender_encoded (0/1)
- ethnicity_encoded (0-4)
- icu_admit_source_encoded (0-4)
- icu_stay_type_encoded (0-1)
- icu_type_encoded (0-5)
- apache_3j_bodysystem_encoded (0-9)
- apache_2_bodysystem_encoded (0-9)

✅ **Features binarias:**
- Soporte vital: intubated_apache, ventilated_apache, arf_apache
- Comorbilidades: aids, cirrhosis, hepatic_failure, leukemia, lymphoma, solid_tumor_with_metastasis, diabetes_mellitus, immunosuppression
- Cirugía: elective_surgery, apache_post_operative
- Glasgow: gcs_unable_apache

### Features Eliminadas (justificación)
❌ **Variables categóricas originales** (se usan las versiones _encoded):
- ethnicity, gender, icu_admit_source, icu_stay_type, icu_type, apache_3j_bodysystem, apache_2_bodysystem

❌ **Probabilidades Apache** (son outputs de otro modelo - evita data leakage):
- apache_4a_hospital_death_prob
- apache_4a_icu_death_prob

## 🏆 Selección del Mejor Modelo

### Criterio de Selección: AUC-ROC
**Justificación:**
1. **Datos desbalanceados** (91% sobrevive, 9% muere)
2. En medicina, necesitamos balance entre detectar todos los casos de muerte (Recall) y no alarmar falsamente (Precision)
3. AUC-ROC mide la capacidad del modelo de discriminar entre clases independientemente del threshold

### Resultado: XGBoost Ganador
- **AUC-ROC**: 88.75% (mejor que RF: 87.67%)
- **Recall**: 74.67% (mucho mejor que RF: 44.73%)
- **Ventaja**: Detecta casi 3 de cada 4 muertes (vs solo 1 de cada 2 en RF)

⚠️ **Trade-off aceptado:**
- XGBoost tiene menor Accuracy (83% vs 91%), pero esto no importa tanto porque el 91% de RF viene simplemente de predecir "sobrevive" para casi todos
- En medicina, es preferible tener más falsos positivos (alarmas innecesarias) que falsos negativos (muertes no detectadas)

## 🚀 Cómo Probar la Aplicación

### 1. Levantar Docker
```bash
docker-compose up --build
```

### 2. Acceder a la aplicación
```
http://localhost:3000
```

### 3. Ir a pestaña "Modelo"
- Seleccionar modelos a entrenar (ambos por defecto)
- Elegir split 80/20 (recomendado)
- Hacer clic en "Entrenar"
- Esperar ~2-3 minutos
- Ver comparación completa de métricas

### 4. Ir a pestaña "Predicción"
- Seleccionar modelo (RF o XGBoost)
- Usar casos predefinidos (Paciente Crítico, Estable, etc.)
- Obtener 3 tipos de predicción

## 📊 Endpoints API Nuevos

### POST /api/train
Entrenar modelos con configuración personalizada
```json
{
  "models": ["random_forest", "xgboost"],
  "test_size": 0.2
}
```

### GET /api/model-parameters
Obtener parámetros y justificación de todos los modelos

### GET /api/training-info
Obtener información del último entrenamiento (split, features, distribución)

### GET /api/model-comparison
Obtener comparación completa de modelos con métricas

## 📝 Conclusiones para la Entrega

### Modelo Recomendado: XGBoost
1. **Mejor métrica clave**: AUC-ROC 88.75%
2. **Mejor Recall**: 74.67% (detecta más muertes)
3. **Ideal para medicina**: Prefiere detectar casos críticos aunque genere falsas alarmas

### Aprendizajes del Proyecto
1. **Desbalance de clases** es crítico en medicina
2. **Feature importance** identifica que soporte vital es lo más importante
3. **Regularización** en XGBoost ayuda a no hacer overfitting
4. **Train/test split 80/20** es suficiente con 91K registros

### Próximos Pasos (fuera del alcance actual)
- Implementar SHAP values para explicabilidad
- Calibración de probabilidades
- Cross-validation con K-Folds
- Ensemble voting de RF + XGBoost
