# 🔬 Estrategia de Limpieza del Dataset

## 📊 RESUMEN EJECUTIVO

### ✅ Lo que HICIMOS:
- **Rellenar** valores faltantes con MEDIANA (numéricos) y MODA (categóricos)
- **Eliminar** solo columnas vacías e IDs innecesarios
- **Conservar** todos los 91,713 pacientes

### ❌ Lo que NO hicimos:
- ~~Eliminar filas con datos faltantes~~ (perderíamos el 100% del dataset)
- ~~Usar promedios~~ (se distorsionan por valores extremos)

---

## 1️⃣ ¿Por qué MEDIANA y no PROMEDIO?

### 🎯 MEDIANA = Valor del medio (más robusto)
### 📉 PROMEDIO = Suma/Cantidad (sensible a extremos)

---

### 📝 Cómo se calcula la MEDIANA

**Paso 1:** Ordenar valores de menor a mayor
**Paso 2:**
- Si hay cantidad **IMPAR** de valores → tomar el del **MEDIO**
- Si hay cantidad **PAR** de valores → **PROMEDIO de los 2 del medio**

---

## 🧮 EJEMPLO: Edad de 5 pacientes UCI

```
Edades originales:  18,  72,  65,  95,  70
Edades ordenadas:   18,  65,  70,  72,  95
                            ↑
                       MEDIANA = 70
                    (valor del medio)

PROMEDIO = (18+65+70+72+95) ÷ 5 = 64
```

### 🤔 ¿Por qué 70 > 64?

| Método | Valor | Problema |
|--------|-------|----------|
| **PROMEDIO** | 64 | El paciente de 18 años "tira para abajo" |
| **MEDIANA** | 70 | Ignora valores extremos ✅ |

---

## 📊 DATOS REALES del Dataset

### Variable: **EDAD** (`age`)

```
📦 Total pacientes:     87,485
❓ Valores faltantes:    4,228  (4.8%)

📊 MEDIANA:   65.0 años  ← usamos esto ✅
📊 PROMEDIO:  62.3 años  ← NO usamos (distorsionado)

📈 Distribución:
   25% tiene ≤ 52 años
   50% tiene ≤ 65 años (MEDIANA)
   75% tiene ≤ 75 años
```

### 🎯 Decisión:
**Si falta edad → Rellenar con 65 años**

---

## 🏥 Otras Variables Numéricas Procesadas

| Variable | Nulls | Mediana | Media | ¿Por qué mediana? |
|----------|-------|---------|-------|-------------------|
| `age` | 4,228 | 65.0 | 62.3 | Edades extremas (16-89) |
| `bmi` | 3,429 | 27.7 | 29.2 | Obesidad extrema distorsiona |
| `weight` | 2,720 | 80.3 | 84.0 | Pesos extremos |
| `temp_apache` | 4,108 | 36.5 | 36.4 | Valores muy similares |
| `d1_glucose_max` | 5,807 | 150.0 | 174.6 | Hiperglucemia extrema |
| `d1_sysbp_max` | 159 | 146.0 | 148.3 | Hipertensión extrema |
| `heart_rate_apache` | 878 | 104.0 | 99.7 | Taquicardias extremas |

### 🔍 Observación:
En **glucosa** la diferencia es ENORME:
- Mediana: 150 mg/dL (rango normal-alto)
- Media: 174.6 mg/dL (distorsionado por diabéticos con 400-500 mg/dL)

---

## 2️⃣ Variables Categóricas: MODA

### 🎯 MODA = Valor más frecuente

| Variable | Nulls | Moda | Significado |
|----------|-------|------|-------------|
| `ethnicity` | 1,395 | Caucasian | Etnia más común |
| `gender` | 25 | M | Género más frecuente |
| `icu_admit_source` | 112 | Accident & Emergency | Fuente más común |
| `apache_2_bodysystem` | 1,662 | Cardiovascular | Sistema más afectado |

---

## 3️⃣ ¿Por qué NO eliminamos filas?

### ❌ Si eliminamos pacientes con ≥1 dato faltante:

```
📦 Dataset original:     91,713 pacientes
🗑️  Pacientes eliminados: 91,713 (100%)
✅ Pacientes restantes:       0 (0%)
```

**IMPOSIBLE** entrenar modelo con 0 datos!

---

### ✅ Al rellenar con mediana/moda:

```
📦 Dataset original:     91,713 pacientes
✅ Pacientes conservados: 91,713 (100%)
📊 Nulls eliminados:     ~200,000 valores
🎯 Dataset listo:        100% completo
```

---

## 🎯 DECISIONES FINALES

### ✅ Relleno (Imputation)
- **Numéricas:** MEDIANA (robusta a extremos)
- **Categóricas:** MODA (valor más frecuente)

### ❌ Eliminación
- Solo columnas 100% vacías (`Unnamed: 83`)
- Solo IDs innecesarios (`patient_id`, `encounter_id`, etc.)
- Variable objetivo con null → eliminar esa fila específica

### 🔢 Encoding
- Convertir categóricas a números (LabelEncoder)
- Ejemplo: `gender` → `M=0, F=1`

---

## 📈 RESULTADO

```
✅ 91,713 pacientes conservados
✅ 0 valores faltantes
✅ 79 variables numéricas listas
✅ Dataset 100% completo para ML
```

---

## 💡 CONCLUSIÓN

**MEDIANA > PROMEDIO** porque:
1. 🛡️ No se distorsiona por valores extremos
2. 🎯 Representa al "paciente típico"
3. 📊 Estadísticamente más robusta en datos médicos
4. ✅ Utilizada en papers científicos para imputación

**CONSERVAR > ELIMINAR** porque:
1. 📦 Mantenemos el 100% de datos
2. 🎯 Más información = mejor modelo
3. 📊 Método estándar en ciencia de datos médicos
