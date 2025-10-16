import json

notebook_path = r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\notebooks\presentacion_limpieza_dataset.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find markdown cells mentioning "20 filas" and clean dataset
print("=" * 70)
print("MARKDOWN CELLS MENTIONING '20 FILAS' AND 'LIMPIO/CLEAN'")
print("=" * 70)

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])

        if '20 filas' in source.lower() or '20 rows' in source.lower():
            if 'limpio' in source.lower() or 'clean' in source.lower():
                print(f'\nCell {i} (markdown):')
                clean_text = source.encode('ascii', 'ignore').decode('ascii')
                print(clean_text[:200])
                print('...')

# Now find the code cells immediately after those markdown cells
print("\n" + "=" * 70)
print("CODE CELLS THAT MIGHT DISPLAY 20 ROWS OF CLEAN DATA")
print("=" * 70)

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Check if this is near a markdown about clean dataset
        # and contains display commands
        if i > 0 and i < len(cells) - 1:
            prev_cell_source = ''.join(cells[i-1]['source']) if cells[i-1]['cell_type'] == 'markdown' else ''

            # If previous cell mentions "20 filas" and "limpio"
            if ('20 filas' in prev_cell_source.lower() or '20 rows' in prev_cell_source.lower()) and \
               ('limpio' in prev_cell_source.lower() or 'clean' in prev_cell_source.lower()):

                if 'display' in source or 'head' in source:
                    print(f'\n{"="*70}')
                    print(f'CELL INDEX: {i}')
                    print(f'Cell ID: {cell.get("id", "N/A")}')
                    print(f'{"="*70}')

                    # Show all lines
                    lines = source.split('\n')
                    for j, line in enumerate(lines[:30]):  # First 30 lines
                        clean_line = line.encode('ascii', 'ignore').decode('ascii')
                        print(f'{j}: {clean_line}')

                    if len(lines) > 30:
                        print(f'... ({len(lines) - 30} more lines)')

print('\nDone!')
