# 🎓 CÓMO FUNCIONA EL PROYECTO - GUÍA COMPLETA

## 📊 FLUJO DE TRABAJO COMPLETO

### 1️⃣ LIMPIEZA DE DATOS (Entrega 2)

**Objetivo:** Limpiar dataset médico con valores faltantes

```
PRESENTACION/dataset.csv (original)
         ↓
    [Ejecutar script de limpieza]
         ↓
PRESENTACION/dataset_clean_final.csv (limpio)
```

**Script usado:**
- `PRESENTACION/clean_dataset_complete.py`
- **Método:** Mediana para numéricos, Moda para categóricos
- **Resultado:** 0 valores faltantes (100% completo)

**Cómo ejecutar:**
```bash
cd PRESENTACION
python clean_dataset_complete.py
```

---

### 2️⃣ MODELADO Y ENTRENAMIENTO (Entrega 3)

**Objetivo:** Entrenar Random Forest y XGBoost

```
PRESENTACION/dataset_clean_final.csv
         ↓
    [Entrenar modelos]
         ↓
models/best_model.pkl (XGBoost)
models/scaler.pkl
models/label_encoders.pkl
```

**Script usado:**
- `ml_service/train_model.py`
- **Modelos:** Random Forest + XGBoost
- **Mejor:** XGBoost (AUC-ROC: 88.75%)

**Cómo ejecutar:**
```bash
# Opción 1: Ejecutar notebook
jupyter notebook notebooks/ENTREGA_3_MODELADO_Y_EVALUACION.ipynb

# Opción 2: Script directo
python ml_service/train_model.py
```

---

### 3️⃣ BACKEND (API REST)

**Objetivo:** Servir predicciones vía API

```
Usuario → Frontend → Backend/app.py
                        ↓
              ml_service.py (carga modelo)
                        ↓
              models/best_model.pkl
                        ↓
              Predicción → Usuario
```

**Archivos clave:**
- `backend/app.py` - API Flask
- `backend/services/ml_service.py` - Servicio ML
- `backend/services/data_processor.py` - Validación datos

**Cómo ejecutar:**
```bash
cd backend
python app.py
# Backend en http://localhost:5000
```

---

### 4️⃣ FRONTEND (React)

**Objetivo:** Interfaz web para usuarios

```
Usuario ingresa datos → React Form
                           ↓
                   POST /api/predict
                           ↓
                   Backend procesa
                           ↓
                   Muestra resultado
```

**Cómo ejecutar:**
```bash
cd frontend
npm install
npm start
# Frontend en http://localhost:3000
```

---

## 📁 ESTRUCTURA DE ARCHIVOS

```
medical-ml-predictor/
│
├── PRESENTACION/                     ← ENTREGA 2
│   ├── dataset.csv                   (original)
│   ├── dataset_clean_final.csv       (limpio) ⭐
│   ├── clean_dataset_complete.py     (script limpieza) ⭐
│   └── presentacion_limpieza_dataset.ipynb
│
├── notebooks/                        ← ENTREGAS
│   ├── ENTREGA_3_MODELADO_Y_EVALUACION.ipynb ⭐ ENTREGA 3
│   ├── exploratory_data_analysis.ipynb
│   └── presentacion_limpieza_dataset.ipynb
│
├── ml_service/                       ← ENTRENAMIENTO
│   └── train_model.py                (entrena RF + XGBoost) ⭐
│
├── models/                           ← MODELOS ENTRENADOS
│   ├── best_model.pkl                (XGBoost) ⭐
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── model_info.pkl
│
├── backend/                          ← API REST
│   ├── app.py                        (Flask API) ⭐
│   └── services/
│       ├── ml_service.py             (Servicio ML) ⭐
│       └── data_processor.py
│
├── frontend/                         ← INTERFAZ WEB
│   └── src/                          (React app)
│
└── docker-compose.yml                (Despliegue completo)
```

---

## ⚠️ IMPORTANTE: ¿Qué hace cada cosa?

### 🐍 Scripts de Python:

| Script | Función | Cuándo usar |
|--------|---------|-------------|
| `PRESENTACION/clean_dataset_complete.py` | Limpia datos | Solo una vez (ya ejecutado) |
| `ml_service/train_model.py` | Entrena modelos | Cuando cambias parámetros |
| `backend/app.py` | Servidor API | Para usar en producción |

### 📓 Notebooks:

| Notebook | Función | Para quién |
|----------|---------|-----------|
| `ENTREGA_3_MODELADO_Y_EVALUACION.ipynb` | Modelado completo | **Profesora** ⭐ |
| `presentacion_limpieza_dataset.ipynb` | Limpieza datos | Profesora (Entrega 2) |
| `exploratory_data_analysis.ipynb` | Análisis exploratorio | Referencia |

### 📊 Datasets:

| Archivo | Descripción | Ubicación |
|---------|-------------|-----------|
| `dataset.csv` | Original (con nulls) | PRESENTACION/ |
| `dataset_clean_final.csv` | Limpio (sin nulls) | PRESENTACION/ ⭐ |

---

## 🚀 CÓMO USAR ESTE PROYECTO

### Para la profesora (Entrega 3):

1. Abrir `notebooks/ENTREGA_3_MODELADO_Y_EVALUACION.ipynb`
2. Ejecutar todas las celdas (Cell → Run All)
3. Ver resultados, gráficos, métricas

### Para desarrollo:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar modelos (opcional, ya están entrenados)
python ml_service/train_model.py

# 3. Iniciar backend
cd backend
python app.py

# 4. Iniciar frontend (en otra terminal)
cd frontend
npm install
npm start

# 5. Probar en navegador
# Frontend: http://localhost:3000
# Backend API: http://localhost:5000
```

### Con Docker:

```bash
docker-compose up
```

---

## ❓ PREGUNTAS FRECUENTES

**Q: ¿El backend limpia los datos?**
A: NO. El backend solo hace predicciones con datos ya limpios.

**Q: ¿Dónde está el script de limpieza?**
A: `PRESENTACION/clean_dataset_complete.py` (único y oficial)

**Q: ¿Por qué había duplicados?**
A: Archivos temporales de desarrollo. Ya eliminados.

**Q: ¿Cómo entreno nuevamente los modelos?**
A: Ejecuta `python ml_service/train_model.py` o el notebook de Entrega 3.

**Q: ¿Qué entrego a la profesora?**
A: El notebook `ENTREGA_3_MODELADO_Y_EVALUACION.ipynb` completo.

---

## ✅ CHECKLIST DE VERIFICACIÓN

- [ ] Dataset limpio existe: `PRESENTACION/dataset_clean_final.csv`
- [ ] Modelos entrenados existen: `models/best_model.pkl`
- [ ] Notebook Entrega 3 ejecuta sin errores
- [ ] Backend funciona: `python backend/app.py`
- [ ] Frontend funciona: `npm start` en frontend/
- [ ] No hay duplicados de clean_dataset.py en raíz

---

📧 **Contacto:** [Tu nombre]
🎓 **Universidad:** UTN - Ciencia de Datos
📅 **Fecha:** 2025
