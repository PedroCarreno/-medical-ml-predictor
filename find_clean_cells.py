import json

notebook_path = r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\notebooks\presentacion_limpieza_dataset.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find cells that display cleaned dataset
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Look for df_clean with display patterns
        if ('df_clean' in source or 'df_limpio' in source) and ('display' in source or '.head' in source):
            print(f'\n{"="*70}')
            print(f'CELL INDEX: {i}')
            print(f'Cell ID: {cell.get("id", "N/A")}')
            print(f'{"="*70}')

            # Show lines with df_clean, head, or display
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ['df_clean', 'df_limpio', 'head', 'display', 'primeras', 'filas']):
                    clean_line = line.encode('ascii', 'ignore').decode('ascii')
                    print(f'Line {j}: {clean_line[:120]}')

            # Check for 20 rows
            if 'head(20)' in source or 'head (20)' in source:
                print('\n*** DISPLAYS 20 ROWS ***')

            print(f'\n{"="*70}\n')

print('\nDone!')
