import json

notebook_path = r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\notebooks\presentacion_limpieza_dataset.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Check cells 21, 22, 23
for i in range(20, 25):
    if i < len(cells):
        cell = cells[i]
        print(f'\n{"="*70}')
        print(f'CELL INDEX: {i}')
        print(f'Cell Type: {cell["cell_type"]}')
        print(f'Cell ID: {cell.get("id", "N/A")}')
        print(f'{"="*70}')

        source = ''.join(cell['source'])

        # Show first 500 chars
        clean_source = source[:500].encode('ascii', 'ignore').decode('ascii')
        print(clean_source)

        if len(source) > 500:
            print(f'\n... ({len(source) - 500} more characters)')

print('\nDone!')
