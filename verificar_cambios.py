import json

# Leer el notebook modificado
notebook_path = r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\PRESENTACION\presentacion_limpieza_dataset.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print("="*80)
print("VERIFICACION DE CAMBIOS EN EL NOTEBOOK")
print("="*80)

# Buscar las celdas modificadas
celda_codigo = None
celda_markdown = None

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'ENFOQUE PROFUNDO: ¿Por qué MEDIANA' in source:
            celda_codigo = i
            print(f"\n[OK] Celda de codigo mejorada encontrada en posicion: {i}")
            print(f"     ID: {cell.get('id', 'N/A')}")

    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '¿Qué son los OUTLIERS?' in source:
            celda_markdown = i
            print(f"[OK] Celda markdown explicativa encontrada en posicion: {i}")
            print(f"     ID: {cell.get('id', 'N/A')}")

print("\n" + "="*80)
print("CONTENIDO DE LA CELDA DE CODIGO (primeras 20 lineas)")
print("="*80)

if celda_codigo:
    source_lines = notebook['cells'][celda_codigo]['source'][:20]
    for i, line in enumerate(source_lines, 1):
        print(f"{i:3d}: {line[:75]}" + ("..." if len(line) > 75 else ""))

print("\n" + "="*80)
print("CONTENIDO DE LA CELDA MARKDOWN")
print("="*80)

if celda_markdown:
    source_full = ''.join(notebook['cells'][celda_markdown]['source'])
    lines = source_full.split('\n')[:30]
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: {line}")

print("\n" + "="*80)
print("ELEMENTOS CLAVE AGREGADOS")
print("="*80)

elementos_clave = [
    "ENFOQUE PROFUNDO",
    "DEMOSTRACION VISUAL: OUTLIERS Y SU IMPACTO",
    "EJEMPLO PRACTICO: Frecuencia Cardiaca",
    "datos_normales",
    "datos_con_outliers",
    "fig, axes = plt.subplots(2, 2",
    "IMPACTO DE OUTLIERS: Media vs Mediana",
    "Boxplot",
    "¿Qué son los OUTLIERS?",
    "Analogía Simple"
]

if celda_codigo:
    source = ''.join(notebook['cells'][celda_codigo]['source'])
    for elemento in elementos_clave[:7]:
        if elemento in source:
            print(f"[OK] {elemento}")
        else:
            print(f"[X]  {elemento} - NO ENCONTRADO")

if celda_markdown:
    source = ''.join(notebook['cells'][celda_markdown]['source'])
    for elemento in elementos_clave[7:]:
        if elemento in source:
            print(f"[OK] {elemento}")
        else:
            print(f"[X]  {elemento} - NO ENCONTRADO")

print("\n" + "="*80)
print("RESUMEN DE MEJORAS")
print("="*80)
print("\n1. VISUALIZACIONES AGREGADAS:")
print("   - Grafico de dispersion (datos sin outliers)")
print("   - Grafico de dispersion (datos con outliers)")
print("   - Boxplot comparativo")
print("   - Grafico de barras (media vs mediana)")
print("\n2. EJEMPLOS NUMERICOS:")
print("   - Tabla comparativa con calculos reales")
print("   - Demostracion cuantitativa del impacto")
print("\n3. EXPLICACIONES DIDACTICAS:")
print("   - Definicion clara de outliers")
print("   - Interpretacion de graficos")
print("   - Analogia del CEO (ejemplo simple)")
print("\n4. APLICACION AL PROYECTO:")
print("   - Justificacion cientifica")
print("   - Relacion con el dataset medico")
print("\n" + "="*80)
print("VERIFICACION COMPLETADA")
print("="*80)
