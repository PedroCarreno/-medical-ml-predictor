#!/usr/bin/env python3
"""
CLEAN_DATASET.PY
Script de limpieza para dataset médico de supervivencia hospitalaria
Dataset: 91,713 pacientes UCI con 85 variables (84 predictoras + 1 objetivo)
Objetivo: Limpiar datos para predicción de mortalidad hospitalaria
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_dataset():
    """Carga y analiza el dataset original"""
    print("="*60)
    print("LIMPIEZA DE DATASET MEDICO - SUPERVIVENCIA HOSPITALARIA")
    print("="*60)

    df = pd.read_csv('data/dataset.csv')
    print(f"📊 Dataset original: {df.shape[0]:,} filas, {df.shape[1]} columnas")

    # Análisis inicial de missing values
    missing_summary = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing_summary / len(df) * 100).round(2)

    print(f"\n📈 Estadísticas iniciales:")
    print(f"   • Total valores faltantes: {df.isnull().sum().sum():,}")
    print(f"   • Variable objetivo 'hospital_death': {df['hospital_death'].value_counts()[0]:,} supervivientes, {df['hospital_death'].value_counts()[1]:,} muertes")
    print(f"   • Tasa de mortalidad: {(df['hospital_death'].mean() * 100):.1f}%")

    return df

def remove_unnecessary_columns(df):
    """Elimina columnas innecesarias según Dataset-Info.pdf"""
    print("\n" + "="*60)
    print("PASO 1: ELIMINANDO COLUMNAS INNECESARIAS")
    print("="*60)

    columns_to_remove = []

    # 1. Eliminar columna completamente vacía
    if 'Unnamed: 83' in df.columns:
        columns_to_remove.append('Unnamed: 83')

    # 2. Variables de identificación (SE ELIMINAN PARA ML según PDF)
    id_vars = ['encounter_id', 'patient_id', 'hospital_id']
    for var in id_vars:
        if var in df.columns:
            columns_to_remove.append(var)

    # 3. icu_id también es ID según análisis
    if 'icu_id' in df.columns:
        columns_to_remove.append('icu_id')

    # Eliminar columnas identificadas
    df_clean = df.drop(columns=columns_to_remove, errors='ignore')

    print(f"✅ Eliminadas {len(columns_to_remove)} columnas:")
    for col in columns_to_remove:
        print(f"   • {col}")

    print(f"📊 Dataset después de eliminar IDs: {df_clean.shape[0]:,} filas, {df_clean.shape[1]} columnas")

    return df_clean

def analyze_missing_patterns(df):
    """Analiza patrones de valores faltantes"""
    print("\n" + "="*60)
    print("PASO 2: ANALIZANDO PATRONES DE VALORES FALTANTES")
    print("="*60)

    missing_analysis = pd.DataFrame({
        'columna': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_percent': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('missing_percent', ascending=False)

    # Categorizar columnas por porcentaje de missing
    high_missing = missing_analysis[missing_analysis['missing_percent'] > 30]
    medium_missing = missing_analysis[(missing_analysis['missing_percent'] > 5) & (missing_analysis['missing_percent'] <= 30)]
    low_missing = missing_analysis[(missing_analysis['missing_percent'] > 0) & (missing_analysis['missing_percent'] <= 5)]

    print(f"📋 Categorías por valores faltantes:")
    print(f"   • Alto missing (>30%): {len(high_missing)} columnas")
    print(f"   • Medio missing (5-30%): {len(medium_missing)} columnas")
    print(f"   • Bajo missing (0-5%): {len(low_missing)} columnas")
    print(f"   • Sin missing: {len(missing_analysis[missing_analysis['missing_percent'] == 0])} columnas")

    if len(high_missing) > 0:
        print(f"\n⚠️  Columnas con ALTO missing (>30%):")
        for _, row in high_missing.head(10).iterrows():
            print(f"   • {row['columna']}: {row['missing_percent']}%")

    if len(medium_missing) > 0:
        print(f"\n📊 Columnas con MEDIO missing (5-30%):")
        for _, row in medium_missing.head(10).iterrows():
            print(f"   • {row['columna']}: {row['missing_percent']}%")

    return missing_analysis

def clean_critical_variables(df):
    """Limpia variables críticas identificadas en Dataset-Info.pdf"""
    print("\n" + "="*60)
    print("PASO 3: LIMPIANDO VARIABLES CRÍTICAS")
    print("="*60)

    df_clean = df.copy()

    # Variables críticas según Dataset-Info.pdf
    critical_vars = {
        'age': 'Edad - PREDICTOR CRÍTICO',
        'bmi': 'Índice de masa corporal',
        'apache_4a_hospital_death_prob': 'Probabilidad muerte hospitalaria APACHE',
        'apache_4a_icu_death_prob': 'Probabilidad muerte UCI APACHE',
        'gcs_eyes_apache': 'Escala Glasgow - Apertura ocular',
        'gcs_motor_apache': 'Escala Glasgow - Respuesta motora',
        'gcs_verbal_apache': 'Escala Glasgow - Respuesta verbal',
        'heart_rate_apache': 'Frecuencia cardíaca',
        'map_apache': 'Presión arterial media',
        'temp_apache': 'Temperatura corporal',
        'd1_spo2_min': 'Saturación oxígeno mínima día 1'
    }

    # Limpiar cada variable crítica
    for var, description in critical_vars.items():
        if var in df_clean.columns:
            missing_count = df_clean[var].isnull().sum()
            missing_pct = (missing_count / len(df_clean)) * 100

            if missing_count > 0:
                if missing_pct < 15:  # Imputar si missing < 15%
                    if df_clean[var].dtype in ['float64', 'int64']:
                        # Usar mediana para variables numéricas
                        median_val = df_clean[var].median()
                        df_clean[var] = df_clean[var].fillna(median_val)
                        print(f"✅ {var}: Imputados {missing_count} valores con mediana ({median_val:.2f})")
                    else:
                        # Usar moda para variables categóricas
                        mode_val = df_clean[var].mode()[0] if not df_clean[var].mode().empty else 'Unknown'
                        df_clean[var] = df_clean[var].fillna(mode_val)
                        print(f"✅ {var}: Imputados {missing_count} valores con moda ({mode_val})")
                else:
                    print(f"⚠️  {var}: {missing_pct:.1f}% missing - Demasiado alto para imputar")
            else:
                print(f"✓ {var}: Sin valores faltantes")

    return df_clean

def handle_comorbidities(df):
    """Maneja las 8 comorbilidades críticas identificadas en el PDF"""
    print("\n" + "="*60)
    print("PASO 4: PROCESANDO COMORBILIDADES CRÍTICAS")
    print("="*60)

    # Comorbilidades según Dataset-Info.pdf
    comorbidities = {
        'aids': 'SIDA/VIH - RIESGO EXTREMO',
        'cirrhosis': 'Cirrosis hepática - RIESGO ALTO',
        'diabetes_mellitus': 'Diabetes mellitus - RIESGO MODERADO',
        'hepatic_failure': 'Falla hepática - RIESGO EXTREMO',
        'immunosuppression': 'Inmunosupresión - RIESGO ALTO',
        'leukemia': 'Leucemia - RIESGO EXTREMO',
        'lymphoma': 'Linfoma - RIESGO ALTO',
        'solid_tumor_with_metastasis': 'Cáncer metastásico - RIESGO EXTREMO'
    }

    df_clean = df.copy()

    for comorbidity, description in comorbidities.items():
        if comorbidity in df_clean.columns:
            missing_count = df_clean[comorbidity].isnull().sum()
            if missing_count > 0:
                # Para comorbilidades, asumir que missing = NO (0)
                df_clean[comorbidity] = df_clean[comorbidity].fillna(0)
                print(f"✅ {comorbidity}: {missing_count} valores faltantes → 0 (No presente)")

            # Verificar distribución
            if df_clean[comorbidity].nunique() == 2:
                positive_cases = df_clean[comorbidity].sum()
                positive_pct = (positive_cases / len(df_clean)) * 100
                print(f"   📊 {comorbidity}: {positive_cases:,} casos positivos ({positive_pct:.2f}%)")

    return df_clean

def handle_categorical_variables(df):
    """Procesa variables categóricas según Dataset-Info.pdf"""
    print("\n" + "="*60)
    print("PASO 5: PROCESANDO VARIABLES CATEGÓRICAS")
    print("="*60)

    df_clean = df.copy()

    # Variables categóricas identificadas en el PDF
    categorical_vars = {
        'gender': 'Sexo (M=54%, F=46%)',
        'ethnicity': 'Etnia (Caucásico 77%, Afroamericano 10%)',
        'icu_admit_source': 'Origen (Emergencia 59%, Quirófano 20%)',
        'icu_stay_type': 'Tipo de estancia en UCI',
        'icu_type': 'Tipo de UCI especializada',
        'apache_3j_bodysystem': 'Sistema corporal afectado APACHE-3J',
        'apache_2_bodysystem': 'Sistema corporal afectado APACHE-2'
    }

    for var, description in categorical_vars.items():
        if var in df_clean.columns:
            missing_count = df_clean[var].isnull().sum()

            if missing_count > 0:
                # Usar moda para imputar
                mode_val = df_clean[var].mode()[0] if not df_clean[var].mode().empty else 'Unknown'
                df_clean[var] = df_clean[var].fillna(mode_val)
                print(f"✅ {var}: {missing_count} valores → '{mode_val}' (moda)")

            # Mostrar distribución
            unique_values = df_clean[var].nunique()
            print(f"   📊 {var}: {unique_values} categorías únicas")

            # Label encoding para variables categóricas
            if df_clean[var].dtype == 'object':
                le = LabelEncoder()
                df_clean[var + '_encoded'] = le.fit_transform(df_clean[var].astype(str))
                print(f"   🔢 Creada columna: {var}_encoded")

    return df_clean

def handle_remaining_missing_values(df):
    """Estrategia final para valores faltantes restantes"""
    print("\n" + "="*60)
    print("PASO 6: MANEJO FINAL DE VALORES FALTANTES")
    print("="*60)

    df_clean = df.copy()

    # Analizar qué queda
    remaining_missing = df_clean.isnull().sum()
    cols_with_missing = remaining_missing[remaining_missing > 0].sort_values(ascending=False)

    print(f"📋 Columnas con valores faltantes restantes: {len(cols_with_missing)}")

    for col, missing_count in cols_with_missing.head(15).items():
        missing_pct = (missing_count / len(df_clean)) * 100

        if missing_pct > 50:
            # Eliminar columnas con >50% missing
            df_clean = df_clean.drop(col, axis=1)
            print(f"🗑️  {col}: Eliminada ({missing_pct:.1f}% missing)")

        elif missing_pct > 15:
            # Marcar como columna de alta missingness pero conservar
            print(f"⚠️  {col}: Conservada con {missing_pct:.1f}% missing")
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna('Missing')

        else:
            # Imputar normalmente
            if df_clean[col].dtype in ['float64', 'int64']:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"✅ {col}: Imputado con mediana ({median_val:.2f})")
            else:
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_val)
                print(f"✅ {col}: Imputado con moda ({mode_val})")

    return df_clean

def final_cleanup(df):
    """Limpieza final del dataset"""
    print("\n" + "="*60)
    print("PASO 7: LIMPIEZA FINAL")
    print("="*60)

    df_clean = df.copy()

    # 1. Eliminar filas donde la variable objetivo es nula
    before_target_clean = len(df_clean)
    df_clean = df_clean.dropna(subset=['hospital_death'])
    after_target_clean = len(df_clean)
    target_removed = before_target_clean - after_target_clean
    print(f"✅ Eliminadas {target_removed:,} filas sin variable objetivo 'hospital_death'")

    # 2. Eliminar duplicados
    before_duplicates = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after_duplicates = len(df_clean)
    duplicates_removed = before_duplicates - after_duplicates
    print(f"✅ Eliminados {duplicates_removed:,} registros duplicados")

    # 3. Verificar tipos de datos
    print(f"\n📊 Tipos de datos finales:")
    data_types = df_clean.dtypes.value_counts()
    for dtype, count in data_types.items():
        print(f"   • {dtype}: {count} columnas")

    return df_clean

def generate_final_report(df_original, df_clean):
    """Genera reporte final de la limpieza"""
    print("\n" + "="*60)
    print("REPORTE FINAL DE LIMPIEZA")
    print("="*60)

    # Estadísticas generales
    print(f"📊 COMPARACIÓN ANTES/DESPUÉS:")
    print(f"   • Filas originales: {len(df_original):,}")
    print(f"   • Filas finales: {len(df_clean):,}")
    print(f"   • Reducción: {len(df_original) - len(df_clean):,} filas ({((len(df_original) - len(df_clean)) / len(df_original) * 100):.1f}%)")
    print(f"   • Columnas originales: {df_original.shape[1]}")
    print(f"   • Columnas finales: {df_clean.shape[1]}")

    # Missing values finales
    final_missing = df_clean.isnull().sum().sum()
    total_cells = df_clean.shape[0] * df_clean.shape[1]
    missing_pct = (final_missing / total_cells) * 100

    print(f"\n📈 CALIDAD DE DATOS:")
    print(f"   • Valores faltantes restantes: {final_missing:,} ({missing_pct:.2f}%)")
    print(f"   • Celdas completas: {total_cells - final_missing:,} ({100 - missing_pct:.2f}%)")

    # Distribución variable objetivo
    print(f"\n🎯 VARIABLE OBJETIVO 'hospital_death':")
    death_counts = df_clean['hospital_death'].value_counts()
    death_pcts = df_clean['hospital_death'].value_counts(normalize=True) * 100
    print(f"   • Supervivientes (0): {death_counts[0]:,} ({death_pcts[0]:.1f}%)")
    print(f"   • Muertes (1): {death_counts[1]:,} ({death_pcts[1]:.1f}%)")

    # Top variables más importantes (según PDF)
    print(f"\n⭐ VARIABLES CRÍTICAS PRESERVADAS:")
    critical_preserved = [
        'age', 'bmi', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob',
        'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache',
        'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure',
        'heart_rate_apache', 'map_apache', 'temp_apache'
    ]

    for var in critical_preserved[:10]:  # Mostrar solo top 10
        if var in df_clean.columns:
            missing = df_clean[var].isnull().sum()
            print(f"   ✓ {var}: {missing} valores faltantes")

    return df_clean

def save_cleaned_dataset(df):
    """Guarda el dataset limpio"""
    print("\n" + "="*60)
    print("GUARDANDO DATASETS LIMPIOS")
    print("="*60)

    # Dataset completo limpio
    df.to_csv('data/dataset_clean.csv', index=False)
    print(f"✅ Guardado: 'data/dataset_clean.csv' ({df.shape[0]:,} filas, {df.shape[1]} columnas)")

    # Dataset solo numéricas para ML rápido
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df.to_csv('data/dataset_numeric.csv', index=False)
    print(f"✅ Guardado: 'data/dataset_numeric.csv' ({numeric_df.shape[0]:,} filas, {numeric_df.shape[1]} columnas)")

    # Dataset para entrenamiento (sin variables objetivo)
    if 'hospital_death' in df.columns:
        training_df = df.drop('hospital_death', axis=1)
        training_df.to_csv('data/dataset_features.csv', index=False)
        print(f"✅ Guardado: 'data/dataset_features.csv' (solo características)")

        # Variable objetivo separada
        target_df = df[['hospital_death']]
        target_df.to_csv('data/dataset_target.csv', index=False)
        print(f"✅ Guardado: 'data/dataset_target.csv' (solo variable objetivo)")

    print(f"\n🎉 LIMPIEZA COMPLETADA EXITOSAMENTE!")
    print(f"📁 Archivos generados en carpeta 'data/'")

def main():
    """Función principal de limpieza"""
    try:
        # Cargar y analizar dataset original
        df_original = load_and_analyze_dataset()

        # Paso 1: Eliminar columnas innecesarias
        df = remove_unnecessary_columns(df_original)

        # Paso 2: Analizar patrones de missing values
        missing_analysis = analyze_missing_patterns(df)

        # Paso 3: Limpiar variables críticas
        df = clean_critical_variables(df)

        # Paso 4: Procesar comorbilidades
        df = handle_comorbidities(df)

        # Paso 5: Procesar variables categóricas
        df = handle_categorical_variables(df)

        # Paso 6: Manejo final de missing values
        df = handle_remaining_missing_values(df)

        # Paso 7: Limpieza final
        df_clean = final_cleanup(df)

        # Reporte final
        df_final = generate_final_report(df_original, df_clean)

        # Guardar datasets limpios
        save_cleaned_dataset(df_final)

    except Exception as e:
        print(f"\n❌ ERROR durante la limpieza: {str(e)}")
        print("Verifique que existe el archivo 'data/dataset.csv'")
        return False

    return True

if __name__ == "__main__":
    success = main()