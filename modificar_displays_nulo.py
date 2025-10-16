import json
import re

# Leer el notebook
with open(r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\PRESENTACION\presentacion_limpieza_dataset.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

modificaciones = []

# Función para modificar el código de las celdas
def agregar_fillna_a_displays(codigo):
    cambios_locales = []
    lineas = codigo.split('\n')
    nuevas_lineas = []

    for i, linea in enumerate(lineas):
        nueva_linea = linea

        # Patrón 1: display(df_styled) - modificar la línea donde se crea df_styled
        if 'df_styled = df_display.style.applymap' in linea:
            # Agregar .fillna('NULO') antes del .style
            if '.fillna(' not in linea:
                nueva_linea = linea.replace('df_display.style', "df_display.fillna('NULO').style")
                if nueva_linea != linea:
                    cambios_locales.append('  Modificado: df_styled con fillna(NULO)')

        # Patrón 2: display(missing_display) - agregar fillna antes
        elif 'display(missing_display)' in linea:
            if i > 0 and '.fillna(' not in lineas[i-1]:
                # Agregar línea antes del display
                indent = len(linea) - len(linea.lstrip())
                nuevas_lineas.append(' ' * indent + "missing_display = missing_display.fillna('NULO')")
                cambios_locales.append('  Agregado fillna antes de display(missing_display)')

        # Patrón 3: df_clean_styled - modificar donde se crea
        elif 'df_clean_styled = df_clean_display.style.apply' in linea:
            if '.fillna(' not in linea:
                nueva_linea = linea.replace('df_clean_display.style', "df_clean_display.fillna('NULO').style")
                if nueva_linea != linea:
                    cambios_locales.append('  Modificado: df_clean_styled con fillna(NULO)')

        # Patrón 4: display(df_original.loc[...])
        elif 'display(df_original.loc[' in linea:
            if '.fillna(' not in linea:
                # Insertar .fillna('NULO') antes del cierre del paréntesis
                nueva_linea = linea.replace('])', "].fillna('NULO'))")
                if nueva_linea != linea:
                    cambios_locales.append('  Modificado: display(df_original.loc[...]) con fillna')

        # Patrón 5: display(df_clean.loc[...])
        elif 'display(df_clean.loc[' in linea:
            if '.fillna(' not in linea:
                nueva_linea = linea.replace('])', "].fillna('NULO'))")
                if nueva_linea != linea:
                    cambios_locales.append('  Modificado: display(df_clean.loc[...]) con fillna')

        # Patrón 6: display(df_resumen.style...)
        elif 'display(df_resumen.style' in linea:
            if '.fillna(' not in linea:
                nueva_linea = linea.replace('df_resumen.style', "df_resumen.fillna('NULO').style")
                if nueva_linea != linea:
                    cambios_locales.append('  Modificado: df_resumen con fillna')

        nuevas_lineas.append(nueva_linea)

    return '\n'.join(nuevas_lineas), cambios_locales

# Procesar cada celda
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        codigo_original = ''.join(cell['source'])

        # Solo procesar si tiene display( y referencias a dataframes
        if 'display(' in codigo_original and ('df_' in codigo_original or 'missing_display' in codigo_original):
            codigo_nuevo, cambios = agregar_fillna_a_displays(codigo_original)

            if cambios:
                # Actualizar la celda - mantener formato de lista con \n
                lineas_nuevas = codigo_nuevo.split('\n')
                # Agregar \n a todas las líneas excepto la última
                nb['cells'][i]['source'] = [l + '\n' if idx < len(lineas_nuevas) - 1 else l
                                            for idx, l in enumerate(lineas_nuevas)]

                modificaciones.append({
                    'celda': i,
                    'cambios': cambios
                })

# Guardar el notebook modificado
with open(r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\PRESENTACION\presentacion_limpieza_dataset.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Mostrar resumen
print('MODIFICACIONES COMPLETADAS')
print('=' * 60)
print(f'Total de celdas modificadas: {len(modificaciones)}')
print()

for mod in modificaciones:
    print(f"Celda {mod['celda']}:")
    for cambio in mod['cambios']:
        print(cambio)
    print()

print('=' * 60)
print('Archivo guardado exitosamente')
print()
print('RESUMEN:')
print(f'  - Celdas modificadas: {len(modificaciones)}')
print(f'  - Total de cambios: {sum(len(m["cambios"]) for m in modificaciones)}')
