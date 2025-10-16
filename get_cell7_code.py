import json

notebook_path = r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\notebooks\presentacion_limpieza_dataset.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][7]

print("="*70)
print("CELL 7 - COMPLETE CODE")
print("="*70)
print(f"Cell Type: {cell['cell_type']}")
print(f"Cell ID: {cell.get('id', 'N/A')}")
print("="*70)

source = ''.join(cell['source'])
# Print with line numbers, ASCII-safe
lines = source.split('\n')
for i, line in enumerate(lines, 1):
    clean_line = line.encode('ascii', 'ignore').decode('ascii')
    print(f"{i:3d}: {clean_line}")

print("\n" + "="*70)
