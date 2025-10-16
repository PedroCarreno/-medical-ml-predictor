import json

# Leer el notebook
with open(r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\PRESENTACION\presentacion_limpieza_dataset.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("VERIFICACION DE fillna('NULO') EN DISPLAYS DE DATAFRAMES")
print("=" * 80)

celdas_clave = {
    7: "df_styled (primeras 20 filas dataset original)",
    9: "missing_display (tabla de valores faltantes)",
    22: "df_clean_styled (primeras 20 filas dataset limpio)",
    33: "df_resumen, df_original.loc, df_clean.loc (comparacion)"
}

for celda_num, descripcion in celdas_clave.items():
    print(f"\n{'='*80}")
    print(f"CELDA {celda_num}: {descripcion}")
    print(f"{'='*80}")

    codigo = ''.join(nb['cells'][celda_num]['source'])
    lineas = codigo.split('\n')

    displays_encontrados = []

    for i, linea in enumerate(lineas):
        # Buscar líneas con display( que no sean HTML
        if 'display(' in linea and 'HTML' not in linea:
            displays_encontrados.append((i, linea.strip()))

    if displays_encontrados:
        for num_linea, linea in displays_encontrados:
            tiene_fillna = '.fillna(' in linea or "fillna('NULO')" in linea

            # Buscar en líneas anteriores si el DataFrame se preparó con fillna
            if not tiene_fillna:
                # Buscar definición del DataFrame en líneas anteriores
                for j in range(num_linea-1, max(num_linea-5, -1), -1):
                    if '.fillna(' in lineas[j]:
                        tiene_fillna = True
                        break

            estado = "OK" if tiene_fillna else "FALTA"
            print(f"  [{estado}] Linea {num_linea}: {linea[:90]}")

            if not tiene_fillna:
                print(f"        ** NECESITA fillna('NULO') **")
    else:
        print("  No se encontraron displays de DataFrames (solo HTML)")

print(f"\n{'='*80}")
print("RESUMEN FINAL")
print(f"{'='*80}")

# Contar cuántos displays de DataFrames hay y cuántos tienen fillna
total_displays = 0
displays_con_fillna = 0

for celda_num in celdas_clave.keys():
    codigo = ''.join(nb['cells'][celda_num]['source'])
    lineas = codigo.split('\n')

    for i, linea in enumerate(lineas):
        if 'display(' in linea and 'HTML' not in linea:
            total_displays += 1

            # Verificar si tiene fillna en la misma línea o en preparación previa
            tiene_fillna = False
            if '.fillna(' in linea:
                tiene_fillna = True
            else:
                # Buscar en líneas anteriores
                for j in range(i-1, max(i-5, -1), -1):
                    if any(var in lineas[j] for var in ['df_styled', 'df_clean_styled', 'missing_display', 'df_resumen']) and '.fillna(' in lineas[j]:
                        tiene_fillna = True
                        break

            if tiene_fillna:
                displays_con_fillna += 1

print(f"Total de displays de DataFrames encontrados: {total_displays}")
print(f"Displays con fillna('NULO') aplicado: {displays_con_fillna}")

if total_displays == displays_con_fillna:
    print("\nESTADO: TODAS las visualizaciones tienen fillna aplicado correctamente")
else:
    print(f"\nESTADO: FALTAN {total_displays - displays_con_fillna} visualizaciones por modificar")
