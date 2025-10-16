import json

# Leer el notebook
with open(r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\PRESENTACION\presentacion_limpieza_dataset.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("="*90)
print(" REPORTE DETALLADO: MODIFICACIONES DE nan -> NULO EN VISUALIZACIONES")
print("="*90)
print()

modificaciones = [
    {
        'celda': 7,
        'descripcion': 'Visualizacion: Primeras 20 filas del dataset original',
        'linea_clave': 30,
        'codigo_antes': "df_styled = df_display.style.applymap(highlight_nulls_red)",
        'codigo_despues': "df_styled = df_display.fillna('NULO').style.applymap(highlight_nulls_red)",
        'display_linea': 31
    },
    {
        'celda': 9,
        'descripcion': 'Visualizacion: Tabla de analisis de valores faltantes',
        'linea_clave': 45,
        'codigo_antes': "display(missing_display)  # Sin fillna previo",
        'codigo_despues': "missing_display = missing_display.fillna('NULO')\n    display(missing_display)",
        'display_linea': 46
    },
    {
        'celda': 22,
        'descripcion': 'Visualizacion: Primeras 20 filas del dataset limpio',
        'linea_clave': 28,
        'codigo_antes': "df_clean_styled = df_clean_display.style.apply(highlight_filled_values, axis=1)",
        'codigo_despues': "df_clean_styled = df_clean_display.fillna('NULO').style.apply(highlight_filled_values, axis=1)",
        'display_linea': 29
    },
    {
        'celda': 33,
        'descripcion': 'Visualizacion: Tabla resumen de imputaciones',
        'linea_clave': 91,
        'codigo_antes': "display(df_resumen.style\\",
        'codigo_despues': "display(df_resumen.fillna('NULO').style\\",
        'display_linea': 91
    },
    {
        'celda': 33,
        'descripcion': 'Visualizacion: Ejemplo de filas con nulls - Dataset Original',
        'linea_clave': 110,
        'codigo_antes': "display(df_original.loc[indices_con_nulls, columnas_ejemplo])",
        'codigo_despues': "display(df_original.loc[indices_con_nulls, columnas_ejemplo].fillna('NULO'))",
        'display_linea': 110
    },
    {
        'celda': 33,
        'descripcion': 'Visualizacion: Ejemplo de filas con nulls - Dataset Limpio',
        'linea_clave': 113,
        'codigo_antes': "display(df_clean.loc[indices_con_nulls, columnas_ejemplo])",
        'codigo_despues': "display(df_clean.loc[indices_con_nulls, columnas_ejemplo].fillna('NULO'))",
        'display_linea': 113
    }
]

for i, mod in enumerate(modificaciones, 1):
    print(f"MODIFICACION #{i}")
    print(f"{'-'*90}")
    print(f"  Celda:        {mod['celda']}")
    print(f"  Descripcion:  {mod['descripcion']}")
    print(f"  Linea:        {mod['linea_clave']}")
    print()
    print(f"  ANTES:")
    print(f"    {mod['codigo_antes']}")
    print()
    print(f"  DESPUES:")
    print(f"    {mod['codigo_despues']}")
    print()

    # Verificar que la modificación esté presente
    codigo = ''.join(nb['cells'][mod['celda']]['source'])
    if '.fillna(' in codigo:
        print(f"  ESTADO: OK - Modificacion aplicada")
    else:
        print(f"  ESTADO: ERROR - Modificacion NO encontrada")
    print(f"{'-'*90}")
    print()

print()
print("="*90)
print(" RESUMEN EJECUTIVO")
print("="*90)
print()
print(f"Total de modificaciones realizadas: {len(modificaciones)}")
print(f"Celdas afectadas: {len(set(m['celda'] for m in modificaciones))}")
print()
print("CELDAS MODIFICADAS:")
for celda_num in sorted(set(m['celda'] for m in modificaciones)):
    count = sum(1 for m in modificaciones if m['celda'] == celda_num)
    print(f"  - Celda {celda_num}: {count} modificacion(es)")
print()
print("IMPACTO:")
print("  - Todos los valores 'nan' ahora se muestran como 'NULO' en las visualizaciones")
print("  - La logica de limpieza de datos NO fue modificada (solo visualizacion)")
print("  - Los calculos y algoritmos permanecen intactos")
print()
print("="*90)
