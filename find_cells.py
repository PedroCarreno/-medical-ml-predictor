import json

notebook_path = r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\notebooks\presentacion_limpieza_dataset.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find cells with head(20) and display
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Look for head(20) patterns
        if 'head(20)' in source:
            print(f'\n{"="*70}')
            print(f'CELL INDEX: {i}')
            print(f'Cell ID: {cell.get("id", "N/A")}')
            print(f'{"="*70}')

            # Show the relevant parts of the source
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if 'head(20)' in line or 'display' in line or 'print' in line:
                    # Clean line to avoid emoji encoding issues
                    clean_line = line.encode('ascii', 'ignore').decode('ascii')
                    print(f'Line {j}: {clean_line}')

            # Determine type
            if 'df_original' in source:
                print('\nTYPE: ORIGINAL DATASET')
            elif 'df_clean' in source or 'df_limpio' in source:
                print('\nTYPE: CLEANED DATASET')
            else:
                print('\nTYPE: UNKNOWN - Check source')

            print(f'\n{"="*70}\n')

print('\nDone!')
