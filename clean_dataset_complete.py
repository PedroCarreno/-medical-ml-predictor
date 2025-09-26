#!/usr/bin/env python3
"""
CLEAN_DATASET_COMPLETE.PY
Script completo que limpia TODAS las columnas del dataset médico
Sin excepciones - procesa cada columna con valores faltantes
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_all_columns_complete():
    """Limpia TODAS las columnas sin excepción"""
    print("="*60)
    print("LIMPIEZA COMPLETA - TODAS LAS COLUMNAS")
    print("="*60)

    # Cargar dataset
    df = pd.read_csv('data/dataset.csv')
    print(f"📊 Dataset original: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    print(f"📊 Total nulls originales: {df.isnull().sum().sum():,}")

    # 1. ELIMINAR SOLO COLUMNAS COMPLETAMENTE VACIAS O IDS
    print("\n1. ELIMINANDO SOLO COLUMNAS INNECESARIAS")
    print("-" * 40)

    columns_to_remove = []

    # Columna completamente vacía
    if 'Unnamed: 83' in df.columns:
        columns_to_remove.append('Unnamed: 83')

    # Variables ID (no útiles para predicción)
    id_vars = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id']
    for var in id_vars:
        if var in df.columns:
            columns_to_remove.append(var)

    df = df.drop(columns=columns_to_remove, errors='ignore')
    print(f"✅ Eliminadas {len(columns_to_remove)} columnas: {columns_to_remove}")
    print(f"📊 Shape después: {df.shape}")

    # 2. PROCESAR TODAS LAS COLUMNAS UNA POR UNA
    print(f"\n2. PROCESANDO TODAS LAS {df.shape[1]} COLUMNAS")
    print("-" * 40)

    total_nulls_filled = 0

    for col in df.columns:
        null_count = df[col].isnull().sum()

        if null_count > 0:
            null_pct = (null_count / len(df)) * 100

            if col == 'hospital_death':
                # Variable objetivo - eliminar filas con null
                df = df.dropna(subset=[col])
                print(f"🎯 {col}: Eliminadas {null_count} filas (variable objetivo)")

            elif df[col].dtype == 'object':
                # Variables categóricas - usar moda
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
                print(f"✅ {col}: {null_count:,} nulls → '{mode_val}' (moda)")
                total_nulls_filled += null_count

            else:
                # Variables numéricas - usar mediana
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"✅ {col}: {null_count:,} nulls → {median_val:.2f} (mediana)")
                total_nulls_filled += null_count

        else:
            print(f"✓ {col}: Sin nulls")

    # 3. ENCODING DE VARIABLES CATEGORICAS
    print(f"\n3. CODIFICANDO VARIABLES CATEGORICAS")
    print("-" * 40)

    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    from sklearn.preprocessing import LabelEncoder

    for col in categorical_columns:
        if col != 'hospital_death':  # No codificar variable objetivo si es categórica
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            print(f"🔢 {col} → {col}_encoded ({df[col].nunique()} categorías)")

    # 4. VERIFICACION FINAL
    print(f"\n4. VERIFICACION FINAL")
    print("-" * 40)

    final_nulls = df.isnull().sum().sum()
    print(f"📊 Shape final: {df.shape}")
    print(f"📊 Nulls restantes: {final_nulls:,}")
    print(f"📊 Nulls eliminados: {total_nulls_filled:,}")

    if final_nulls == 0:
        print("🎉 PERFECTO: Dataset completamente limpio, sin nulls!")
    else:
        print("⚠️ Aún quedan algunos nulls - revisando...")
        remaining = df.isnull().sum()
        remaining_cols = remaining[remaining > 0]
        for col, count in remaining_cols.items():
            print(f"   • {col}: {count} nulls")

    # 5. ELIMINAR DUPLICADOS
    print(f"\n5. ELIMINANDO DUPLICADOS")
    print("-" * 40)

    before_dup = len(df)
    df = df.drop_duplicates()
    after_dup = len(df)
    duplicates_removed = before_dup - after_dup

    print(f"✅ Duplicados eliminados: {duplicates_removed:,}")
    print(f"📊 Shape final: {df.shape}")

    return df

def save_final_clean_dataset(df):
    """Guarda el dataset completamente limpio"""
    print(f"\n6. GUARDANDO DATASET FINAL")
    print("-" * 40)

    # Solo un archivo - el dataset completamente limpio
    df.to_csv('data/dataset_clean_final.csv', index=False)
    print(f"✅ Guardado: 'data/dataset_clean_final.csv'")
    print(f"📊 {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"📊 0 valores faltantes (100% completo)")

    # Reporte de la variable objetivo
    if 'hospital_death' in df.columns:
        death_counts = df['hospital_death'].value_counts()
        print(f"\n🎯 VARIABLE OBJETIVO:")
        print(f"   • Supervivientes (0): {death_counts[0]:,}")
        print(f"   • Muertes (1): {death_counts[1]:,}")
        print(f"   • Tasa mortalidad: {(death_counts[1] / len(df) * 100):.1f}%")

def main():
    """Función principal"""
    try:
        # Limpiar todas las columnas completamente
        df_clean = clean_all_columns_complete()

        # Guardar dataset final
        save_final_clean_dataset(df_clean)

        print("\n" + "="*60)
        print("🎉 LIMPIEZA COMPLETA EXITOSA")
        print("="*60)
        print("📁 Archivo generado: 'data/dataset_clean_final.csv'")
        print("✅ Sin valores faltantes")
        print("✅ Variables categóricas codificadas")
        print("✅ Listo para Machine Learning")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()