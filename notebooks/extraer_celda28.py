import nbformat
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

nb = nbformat.read(r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\notebooks\presentacion_limpieza_dataset.ipynb', as_version=4)

print('='*80)
print('CELDA 28 - JUSTIFICACIÓN DE METODOLOGÍAS (CÓDIGO)')
print('='*80)
print(nb.cells[28].source)

print('\n\n')
print('='*80)
print('CELDA 22 - ENCODING DE VARIABLES CATEGÓRICAS (MARKDOWN)')
print('='*80)
print(nb.cells[22].source)
