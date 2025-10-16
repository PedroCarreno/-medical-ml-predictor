import json
import sys

# Leer el notebook
with open(r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\PRESENTACION\presentacion_limpieza_dataset.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("ANALISIS DE CELDAS CON display()")
print("=" * 80)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        codigo = ''.join(cell['source'])

        if 'display(' in codigo:
            print(f"\n{'='*80}")
            print(f"CELDA {i}")
            print(f"{'='*80}")

            lineas = codigo.split('\n')
            for j, linea in enumerate(lineas):
                if 'display(' in linea or 'df_styled' in linea or 'df_clean_styled' in linea or 'missing_display' in linea or 'df_resumen' in linea:
                    tiene_fillna = '.fillna(' in linea
                    marca = '[YA OK]' if tiene_fillna else '[FALTA]'
                    print(f"{marca} Linea {j}: {linea[:100]}")

            # Buscar patrones específicos
            if 'df_styled = ' in codigo:
                print("\n  >> Tiene df_styled")
            if 'df_clean_styled = ' in codigo:
                print("\n  >> Tiene df_clean_styled")
            if 'display(missing_display)' in codigo:
                print("\n  >> Tiene display(missing_display)")
            if 'df_resumen.style' in codigo:
                print("\n  >> Tiene df_resumen.style")
            if 'display(df_original.loc[' in codigo:
                print("\n  >> Tiene display(df_original.loc[...])")
            if 'display(df_clean.loc[' in codigo:
                print("\n  >> Tiene display(df_clean.loc[...])")
