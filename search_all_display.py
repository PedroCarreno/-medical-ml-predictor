import json
import re

notebook_path = r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\notebooks\presentacion_limpieza_dataset.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

print("="*70)
print("ALL CELLS WITH display() OR .head() SHOWING DATAFRAMES")
print("="*70)

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Look for display patterns with df_original or df_clean
        if re.search(r'display\s*\(\s*df_', source) or re.search(r'df_\w+\.head\(\d+\)', source):
            print(f'\n{"="*70}')
            print(f'CELL INDEX: {i}')
            print(f'Cell ID: {cell.get("id", "N/A")}')
            print(f'{"="*70}')

            # Extract relevant lines
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if 'display' in line or '.head(' in line or 'df_original' in line or 'df_clean' in line:
                    clean_line = line.encode('ascii', 'ignore').decode('ascii')
                    print(f'{j}: {clean_line[:120]}')

            # Highlight if it shows 20 rows
            if '.head(20)' in source:
                print('\n*** SHOWS 20 ROWS ***')

                # Determine which dataset
                if 'df_original' in source and 'df_clean' not in source:
                    print('*** DATASET: ORIGINAL ***')
                elif 'df_clean' in source and 'df_original' not in source:
                    print('*** DATASET: CLEANED ***')
                elif 'df_clean' in source and 'df_original' in source:
                    print('*** DATASET: BOTH (check carefully) ***')

print('\nDone!')
