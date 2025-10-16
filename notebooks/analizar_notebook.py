import nbformat
import sys
import io

# Configurar salida UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Leer notebook
nb = nbformat.read(r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\notebooks\presentacion_limpieza_dataset.ipynb', as_version=4)

print(f'Total celdas: {len(nb.cells)}\n')
print('='*80)
print('ESTRUCTURA DEL NOTEBOOK')
print('='*80)

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown':
        # Extraer título si existe
        lines = cell.source.split('\n')
        first_line = lines[0][:100] if lines else ''

        print(f"\nCelda {i} [MARKDOWN]:")
        print(f"  Título: {first_line}")
        print(f"  Caracteres: {len(cell.source)}")

        # Buscar palabras clave
        keywords = ['mediana', 'promedio', 'VALOR DEL MEDIO', 'eliminamos', 'encoding', 'moda', 'categóricas']
        found_keywords = [kw for kw in keywords if kw.lower() in cell.source.lower()]
        if found_keywords:
            print(f"  >> Contiene: {', '.join(found_keywords)}")

    else:  # code
        print(f"\nCelda {i} [CODE]: {len(cell.source)} caracteres")

print('\n' + '='*80)
print('CELDAS CON EXPLICACIONES CLAVE:')
print('='*80)

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown':
        if 'VALOR DEL MEDIO' in cell.source or ('mediana' in cell.source.lower() and len(cell.source) > 200):
            print(f"\n{'='*80}")
            print(f">>> CELDA {i} - Sección sobre MEDIANA/PROMEDIO:")
            print('='*80)
            print(cell.source[:800])
            print('\n[... continúa ...]\n')
