import json

notebook_path = r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\notebooks\presentacion_limpieza_dataset.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

print("="*70)
print("SUMMARY: CELLS DISPLAYING 20 ROWS OF DATASETS")
print("="*70)

# Cell displaying 20 rows of ORIGINAL dataset
print("\n1. ORIGINAL DATASET (20 rows):")
print("-" * 70)
cell = cells[7]
print(f"Cell Index: 7")
print(f"Cell ID: {cell.get('id', 'N/A')}")
source = ''.join(cell['source'])
key_lines = [line for line in source.split('\n') if 'head(20)' in line or 'display(' in line or 'PRIMERAS 20' in line]
for line in key_lines[:5]:
    clean_line = line.encode('ascii', 'ignore').decode('ascii')
    print(f"  {clean_line}")

# Check if there's a cell displaying 20 rows of CLEANED dataset
print("\n2. CLEANED DATASET (20 rows):")
print("-" * 70)

found_clean_20 = False
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        # Look specifically for df_clean with head(20)
        if 'df_clean' in source and '.head(20)' in source:
            found_clean_20 = True
            print(f"Cell Index: {i}")
            print(f"Cell ID: {cell.get('id', 'N/A')}")
            key_lines = [line for line in source.split('\n') if 'head(20)' in line or 'display' in line]
            for line in key_lines[:5]:
                clean_line = line.encode('ascii', 'ignore').decode('ascii')
                print(f"  {clean_line}")

if not found_clean_20:
    print("NOT FOUND!")
    print("There is a markdown cell (index 21) that mentions:")
    print("  'Visualizacion de las primeras 20 filas del dataset limpio'")
    print("But NO corresponding code cell that actually displays df_clean.head(20)")
    print("\nThis cell appears to be MISSING from the notebook.")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nONLY ONE cell found that displays 20 rows:")
print("  - Cell 7: Shows df_original.head(20) - ORIGINAL DATASET")
print("\nThe cell that should show df_clean.head(20) appears to be MISSING")
print("  - Cell 21 (markdown) mentions it, but cell 22 is a different markdown")
print("  - Expected code cell after cell 21 is missing")

print('\nDone!')
