# 🎓 GUÍA PARA PRESENTAR LA ENTREGA 3 A LA PROFESORA

## ✅ **RESPUESTA RÁPIDA:**

**SÍ, con mostrar el notebook es suficiente.**

El notebook `ENTREGA_3_MODELADO_Y_EVALUACION.ipynb` tiene TODO lo que necesitas:
- ✅ Comparación de modelos
- ✅ Justificación científica
- ✅ Gráficos comparativos
- ✅ Criterios de selección
- ✅ Explicación médica

---

## 📊 **¿CÓMO FUNDAMENTAMOS QUE XGBOOST ES MEJOR?**

### 1️⃣ **Tabla Comparativa de Métricas** (Celda 11)

```
📊 TABLA COMPARATIVA DE MÉTRICAS
================================================================================
                accuracy  precision     recall   f1_score    auc_roc
Random Forest  91.206455  48.962656  44.725205  46.748102  87.671410
XGBoost        83.519599  31.072555  74.668351  43.883423  88.752849

🏆 GANADOR POR MÉTRICA
================================================================================
   accuracy            : Random Forest   ( 91.21%)
   precision           : Random Forest   ( 48.96%)
   recall              : XGBoost         ( 74.67%) ✅
   f1_score            : Random Forest   ( 46.75%)
   auc_roc             : XGBoost         ( 88.75%) ✅
```

**📌 Punto clave para la profesora:**
- XGBoost gana en las **2 métricas más importantes** para datos desbalanceados
- Recall (74.67%) = Detecta 3 de cada 4 pacientes que morirán
- AUC-ROC (88.75%) = Mejor capacidad de discriminación

---

### 2️⃣ **Criterio de Selección** (Código automático)

**Script que selecciona automáticamente el mejor modelo:**

El código en `ml_service/train_model.py` selecciona automáticamente usando:

```python
# Seleccionar mejor modelo basado en AUC-ROC
best_model_name = max(
    self.evaluation_results.items(),
    key=lambda x: x[1]['metrics']['auc_roc']
)[0]
```

**📌 Fundamento científico:**
- **AUC-ROC** es la métrica estándar para datos desbalanceados
- En medicina es crucial detectar casos de riesgo (Recall alto)
- Accuracy engaña con datos 91% vs 9%

---

### 3️⃣ **Gráficos Comparativos** (Celda 13)

**El notebook incluye 2 gráficos importantes:**

#### **A) Gráfico de Barras - Comparación de Métricas**
- Azul = Random Forest
- Rojo = XGBoost

**Qué muestra:**
- Visualización clara de dónde gana cada modelo
- XGBoost destaca en Recall (barra más alta)
- RF gana en Accuracy/Precision (pero menos importante)

#### **B) Curvas ROC - Capacidad de Discriminación**
- Línea azul = Random Forest (AUC=0.8767)
- Línea roja = XGBoost (AUC=0.8875)
- Línea negra discontinua = Azar (AUC=0.5)

**Qué muestra:**
- XGBoost tiene curva más alejada de la diagonal (mejor)
- Área bajo la curva mayor = mejor discriminación
- Interpretación visual clara

---

### 4️⃣ **Matrices de Confusión** (Celda 14)

```
Random Forest:
[[15235   550]     ← 550 falsos positivos
 [  877   681]]    ← 877 falsos negativos (CRÍTICO)

XGBoost:
[[14047  1738]     ← 1738 falsos positivos
 [  395  1163]]    ← 395 falsos negativos (MEJOR)
```

**📌 Interpretación médica crucial:**

**Falsos Negativos (esquina inferior izquierda):**
- **Random Forest:** 877 pacientes → Predice "sobrevive" pero MUEREN ⚠️
- **XGBoost:** 395 pacientes → 482 pacientes MENOS en riesgo no detectado ✅

**Conclusión:** En medicina es MÁS GRAVE no detectar un caso crítico que dar una falsa alarma.

---

### 5️⃣ **Justificación Científica** (Celdas 10 y 18)

**El notebook explica por qué cada métrica importa:**

Para datos médicos DESBALANCEADOS (91% sobrevive, 9% muere):

1. **Accuracy** - ⚠️ NO es suficiente con datos desbalanceados
2. **Precision** - Evita alarmas falsas
3. **Recall** - ⭐ MUY IMPORTANTE: No perder casos críticos
4. **F1-Score** - Balance entre Precision y Recall
5. **AUC-ROC** - ⭐ LA MÁS IMPORTANTE para discriminación

**📌 Argumento para la profesora:**

"Seleccionamos XGBoost como mejor modelo usando **AUC-ROC como criterio principal** porque:

1. **Datos desbalanceados (91% vs 9%):** Accuracy engaña
2. **Mejor AUC-ROC (88.75% vs 87.67%):** Discrimina mejor entre clases
3. **Mejor Recall (74.67% vs 44.73%):** Detecta más casos críticos
4. **Contexto médico:** Preferimos falsos positivos a falsos negativos
5. **Impacto clínico:** Salva más vidas al identificar pacientes de riesgo"

---

## 🎯 **LO QUE LA PROFESORA VERÁ EN EL NOTEBOOK:**

### **Sección 5: Evaluación con Métricas Apropiadas**
✅ Tabla con 5 métricas comparadas
✅ Explicación de por qué cada métrica importa
✅ Identificación del ganador por métrica

### **Sección 6: Comparación Visual de Modelos**
✅ Gráfico de barras comparativo
✅ Curvas ROC superpuestas
✅ Interpretación de resultados

### **Sección 7: Matrices de Confusión (Celda 14)**
✅ Heatmaps lado a lado
✅ Interpretación médica (falsos negativos críticos)

### **Sección 8: Interpretación del Conocimiento**
✅ Conclusión científica
✅ Justificación de selección de XGBoost
✅ Aplicación práctica en medicina

### **Sección 9: Conclusiones Finales**
✅ Resumen de por qué XGBoost es superior
✅ Aprendizajes clave sobre datos desbalanceados

---

## 🗣️ **SCRIPT PARA PRESENTAR A LA PROFESORA:**

### **Inicio (1 minuto):**

"Profesora, para la Entrega 3 entrené dos modelos:
- Random Forest
- XGBoost

Ambos con datos médicos desbalanceados (91% sobrevive, 9% muere)."

### **Justificación del mejor modelo (2 minutos):**

"Seleccioné **XGBoost** como mejor modelo usando estos criterios:

1. **AUC-ROC = 88.75%** (vs 87.67% de RF)
   - Métrica más apropiada para datos desbalanceados
   - Mide capacidad de discriminación entre clases
   - [Mostrar curva ROC en notebook]

2. **Recall = 74.67%** (vs 44.73% de RF)
   - Detecta 3 de cada 4 pacientes que morirán
   - En medicina es crítico no perder casos de riesgo
   - [Mostrar matriz de confusión]

3. **Impacto clínico:**
   - XGBoost tiene 395 falsos negativos
   - Random Forest tiene 877 falsos negativos
   - **482 pacientes más detectados** con XGBoost ✅"

### **Mostrar evidencia (2 minutos):**

"En el notebook puede ver:

1. **Tabla comparativa** [Celda 11]
   - 5 métricas calculadas
   - XGBoost gana en las más importantes

2. **Gráficos** [Celdas 13-14]
   - Curvas ROC: XGBoost más alejada de diagonal
   - Matrices de confusión: Menos falsos negativos

3. **Código automático** [train_model.py]
   - Selección basada en AUC-ROC
   - Proceso reproducible y científico"

### **Cierre (1 minuto):**

"En conclusión:
- Para datos desbalanceados, AUC-ROC > Accuracy
- XGBoost es superior en detección de casos críticos
- El modelo está desplegado en backend funcional
- Todo el proceso está documentado en el notebook"

---

## 📋 **CHECKLIST ANTES DE PRESENTAR:**

- [ ] Notebook ejecutado completo (Cell → Run All)
- [ ] Todas las celdas muestran output
- [ ] Gráficos se visualizan correctamente
- [ ] Tabla comparativa muestra XGBoost ganador
- [ ] AUC-ROC = 88.75% visible
- [ ] Matrices de confusión generadas
- [ ] Conclusiones al final completas

---

## ❓ **PREGUNTAS QUE LA PROFESORA PODRÍA HACER:**

**Q1: ¿Por qué no usaste Accuracy?**
A: Con datos 91% vs 9%, un modelo que siempre prediga "sobrevive" tendría 91% accuracy pero sería inútil. AUC-ROC mide la verdadera capacidad de discriminación.

**Q2: ¿Cómo seleccionaste el mejor modelo?**
A: Automáticamente usando AUC-ROC como criterio (código en train_model.py). Es la métrica estándar para datos desbalanceados en medicina.

**Q3: ¿Random Forest gana en 3 métricas, por qué elegiste XGBoost?**
A: Porque las métricas donde gana XGBoost (AUC-ROC y Recall) son las críticas para datos desbalanceados. Accuracy y Precision engañan con clases desproporcionadas.

**Q4: ¿Qué pasa con los falsos positivos de XGBoost?**
A: XGBoost tiene más falsos positivos (1738 vs 550), pero en medicina es preferible una falsa alarma (monitoreo extra) que no detectar un caso crítico (muerte no prevista).

**Q5: ¿Dónde está el código de evaluación?**
A: En `ml_service/train_model.py` (método `evaluate_models`). El notebook ejecuta ese código y muestra los resultados visualmente.

---

## ✅ **RESUMEN EJECUTIVO:**

### **¿Es suficiente mostrar el notebook?**
✅ **SÍ, absolutamente.**

### **¿Qué tiene el notebook?**
- ✅ Todas las métricas calculadas
- ✅ Comparación visual (gráficos)
- ✅ Justificación científica escrita
- ✅ Criterio de selección explicado
- ✅ Interpretación médica

### **¿Qué código selecciona el modelo?**
- `ml_service/train_model.py`
- Criterio: Modelo con mayor AUC-ROC

### **¿Qué gráficos fundamentan?**
1. Curvas ROC (discriminación)
2. Gráfico de barras (métricas)
3. Matrices de confusión (errores)

### **¿Cuál es el criterio?**
- **Principal:** AUC-ROC (88.75%)
- **Secundario:** Recall (74.67%)
- **Contexto:** Medicina + datos desbalanceados

---

🎓 **¡Éxito en tu presentación!** El notebook tiene TODO lo necesario.
