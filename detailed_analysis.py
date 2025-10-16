import json

notebook_path = r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\notebooks\presentacion_limpieza_dataset.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

print("="*70)
print("DETAILED ANALYSIS: CELLS AROUND 'VISUALIZACION 20 FILAS LIMPIO'")
print("="*70)

# Find the markdown cell about 20 filas del dataset limpio
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source']).lower()
        if '20 filas' in source and 'limpio' in source:
            print(f'\nFound markdown at cell {i}')
            clean_content = source[:200].encode('ascii', 'ignore').decode('ascii')
            print(f'Content: {clean_content}')

            # Check the next 3 cells
            for j in range(i+1, min(i+4, len(cells))):
                next_cell = cells[j]
                print(f'\n--- Cell {j} ({next_cell["cell_type"]}) ---')
                next_source = ''.join(next_cell['source'])

                if next_cell['cell_type'] == 'code':
                    # Show full code if it's short
                    if len(next_source) < 500:
                        print('FULL CODE:')
                        clean_code = next_source.encode('ascii', 'ignore').decode('ascii')
                        print(clean_code)
                    else:
                        print('FIRST 500 CHARS:')
                        clean_code = next_source[:500].encode('ascii', 'ignore').decode('ascii')
                        print(clean_code)
                        print(f'\n... ({len(next_source) - 500} more chars)')
                else:
                    clean_md = next_source[:300].encode('ascii', 'ignore').decode('ascii')
                    print(clean_md)

print('\nDone!')
