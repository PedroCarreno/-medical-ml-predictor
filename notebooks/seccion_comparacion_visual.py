"""
VISUALIZACIONES PARA EXPLICAR MEDIANA vs PROMEDIO
Para presentación de limpieza de dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Cargar dataset
df = pd.read_csv(r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\data\dataset.csv')

# ==============================================================================
# FIGURA 1: EJEMPLO SIMPLE - 5 PACIENTES
# ==============================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Datos del ejemplo
edades_ejemplo = [18, 65, 70, 72, 95]
edades_ordenadas = sorted(edades_ejemplo)

# Calcular estadísticas
mediana = np.median(edades_ordenadas)
promedio = np.mean(edades_ordenadas)

# Visualizar
x_pos = np.arange(len(edades_ordenadas))
bars = ax.bar(x_pos, edades_ordenadas, color=['red', 'lightblue', 'green', 'lightblue', 'lightblue'],
               edgecolor='black', linewidth=2, alpha=0.7)

# Resaltar mediana
bars[2].set_color('green')
bars[2].set_alpha(1)

# Líneas de referencia
ax.axhline(y=mediana, color='green', linestyle='--', linewidth=3, label=f'MEDIANA = {mediana:.0f} años')
ax.axhline(y=promedio, color='orange', linestyle='--', linewidth=3, label=f'PROMEDIO = {promedio:.0f} años')

# Anotaciones
ax.annotate('Valor extremo\n(distorsiona promedio)',
            xy=(0, 18), xytext=(0.5, 30),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', weight='bold')

ax.annotate('MEDIANA\n(valor del medio)',
            xy=(2, 70), xytext=(2.5, 85),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=12, color='green', weight='bold')

# Etiquetas
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{edad}\naños' for edad in edades_ordenadas], fontsize=11)
ax.set_ylabel('Edad (años)', fontsize=13, weight='bold')
ax.set_title('🧮 EJEMPLO: 5 Pacientes UCI\nMediana vs Promedio', fontsize=16, weight='bold', pad=20)
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Texto explicativo
textstr = f'''
📊 Valores ordenados: {edades_ordenadas}

✅ MEDIANA = {mediana:.0f} (valor central - posición 3)
❌ PROMEDIO = {promedio:.0f} (distorsionado por el 18)

💡 La mediana representa mejor al paciente típico UCI
'''
ax.text(0.98, 0.55, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('img_ejemplo_mediana_5pacientes.png', dpi=300, bbox_inches='tight')
print("✅ Guardado: img_ejemplo_mediana_5pacientes.png")
plt.close()

# El resto del código sigue igual...
if False:
    from IPython.display import display, HTML
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # ============================================================================
    # 1. TÍTULO PRINCIPAL CON ESTILO
    # ============================================================================
    display(HTML("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px; border-radius: 15px; margin: 20px 0; text-align: center;'>
        <h1 style='color: white; font-size: 42px; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
            📊 ANÁLISIS COMPARATIVO
        </h1>
        <h2 style='color: #f0f0f0; font-size: 28px; margin: 10px 0 0 0; font-weight: 300;'>
            Dataset Original vs Dataset Limpio
        </h2>
    </div>
    """))

    # ============================================================================
    # 2. MÉTRICAS PRINCIPALES EN TARJETAS
    # ============================================================================
    filas_orig = df_original.shape[0]
    filas_clean = df_clean.shape[0]
    cols_orig = df_original.shape[1]
    cols_clean = df_clean.shape[1]
    nulls_orig = df_original.isnull().sum().sum()
    nulls_clean = df_clean.isnull().sum().sum()
    completitud_orig = ((df_original.size - nulls_orig) / df_original.size) * 100
    completitud_clean = ((df_clean.size - nulls_clean) / df_clean.size) * 100

    display(HTML(f"""
    <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 30px 0;'>

        <!-- FILAS -->
        <div style='background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                    padding: 25px; border-radius: 12px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <div style='font-size: 16px; opacity: 0.9; margin-bottom: 8px;'>📝 FILAS</div>
            <div style='font-size: 42px; font-weight: bold; margin: 10px 0;'>{filas_orig:,} → {filas_clean:,}</div>
            <div style='font-size: 18px; opacity: 0.95;'>
                Cambio: <span style='background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 5px;'>
                {filas_clean - filas_orig:+,}
                </span>
            </div>
        </div>

        <!-- COLUMNAS -->
        <div style='background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
                    padding: 25px; border-radius: 12px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <div style='font-size: 16px; opacity: 0.9; margin-bottom: 8px;'>📋 COLUMNAS</div>
            <div style='font-size: 42px; font-weight: bold; margin: 10px 0;'>{cols_orig} → {cols_clean}</div>
            <div style='font-size: 18px; opacity: 0.95;'>
                Cambio: <span style='background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 5px;'>
                {cols_clean - cols_orig:+}
                </span>
            </div>
        </div>

        <!-- VALORES FALTANTES -->
        <div style='background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                    padding: 25px; border-radius: 12px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <div style='font-size: 16px; opacity: 0.9; margin-bottom: 8px;'>❌ VALORES FALTANTES</div>
            <div style='font-size: 42px; font-weight: bold; margin: 10px 0;'>{nulls_orig:,} → {nulls_clean:,}</div>
            <div style='font-size: 18px; opacity: 0.95;'>
                Eliminados: <span style='background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 5px;'>
                {nulls_orig - nulls_clean:,}
                </span>
            </div>
        </div>

        <!-- COMPLETITUD -->
        <div style='background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
                    padding: 25px; border-radius: 12px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <div style='font-size: 16px; opacity: 0.9; margin-bottom: 8px;'>✅ COMPLETITUD</div>
            <div style='font-size: 42px; font-weight: bold; margin: 10px 0;'>{completitud_orig:.1f}% → {completitud_clean:.1f}%</div>
            <div style='font-size: 18px; opacity: 0.95;'>
                Mejora: <span style='background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 5px;'>
                +{completitud_clean - completitud_orig:.1f}%
                </span>
            </div>
        </div>
    </div>
    """))

    # ============================================================================
    # 3. GRÁFICOS COMPARATIVOS
    # ============================================================================
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Comparación Visual: Antes vs Después', fontsize=26, fontweight='bold', y=0.98)

    # Gráfico 1: Comparación de filas y columnas
    ax1 = plt.subplot(2, 3, 1)
    categorias = ['Filas', 'Columnas']
    original = [filas_orig, cols_orig]
    limpio = [filas_clean, cols_clean]
    x = np.arange(len(categorias))
    width = 0.35
    bars1 = ax1.bar(x - width/2, original, width, label='Original', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, limpio, width, label='Limpio', color='#27ae60', alpha=0.8)
    ax1.set_ylabel('Cantidad', fontsize=14, fontweight='bold')
    ax1.set_title('Dimensiones del Dataset', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categorias, fontsize=13)
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Agregar valores sobre las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Gráfico 2: Valores faltantes
    ax2 = plt.subplot(2, 3, 2)
    valores = [nulls_orig, nulls_clean]
    colores = ['#e74c3c', '#27ae60']
    bars = ax2.bar(['Original', 'Limpio'], valores, color=colores, alpha=0.8, width=0.6)
    ax2.set_ylabel('Cantidad de Nulls', fontsize=14, fontweight='bold')
    ax2.set_title('Valores Faltantes', fontsize=16, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Gráfico 3: Completitud
    ax3 = plt.subplot(2, 3, 3)
    completitud = [completitud_orig, completitud_clean]
    bars = ax3.bar(['Original', 'Limpio'], completitud, color=['#3498db', '#27ae60'], alpha=0.8, width=0.6)
    ax3.set_ylabel('Porcentaje (%)', fontsize=14, fontweight='bold')
    ax3.set_title('Completitud de Datos', fontsize=16, fontweight='bold', pad=15)
    ax3.set_ylim([0, 105])
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Gráfico 4: Memoria utilizada
    ax4 = plt.subplot(2, 3, 4)
    mem_orig = df_original.memory_usage(deep=True).sum() / 1024**2
    mem_clean = df_clean.memory_usage(deep=True).sum() / 1024**2
    memoria = [mem_orig, mem_clean]
    bars = ax4.bar(['Original', 'Limpio'], memoria, color=['#9b59b6', '#27ae60'], alpha=0.8, width=0.6)
    ax4.set_ylabel('Memoria (MB)', fontsize=14, fontweight='bold')
    ax4.set_title('Uso de Memoria', fontsize=16, fontweight='bold', pad=15)
    ax4.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} MB', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Gráfico 5: Distribución de nulls por columna (Top 10 originales)
    ax5 = plt.subplot(2, 3, 5)
    null_counts_orig = df_original.isnull().sum().sort_values(ascending=False).head(10)
    if len(null_counts_orig) > 0 and null_counts_orig.iloc[0] > 0:
        bars = ax5.barh(range(len(null_counts_orig)), null_counts_orig.values, color='#e74c3c', alpha=0.7)
        ax5.set_yticks(range(len(null_counts_orig)))
        ax5.set_yticklabels(null_counts_orig.index, fontsize=10)
        ax5.set_xlabel('Cantidad de Nulls', fontsize=12, fontweight='bold')
        ax5.set_title('Top 10 Columnas con Más Nulls (Original)', fontsize=14, fontweight='bold', pad=15)
        ax5.grid(axis='x', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, null_counts_orig.values)):
            ax5.text(val, i, f' {int(val):,}', va='center', fontsize=9, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'Sin valores nulos', ha='center', va='center', fontsize=14)
        ax5.set_xlim([0, 1])
        ax5.set_ylim([0, 1])

    # Gráfico 6: Cambios en columnas
    ax6 = plt.subplot(2, 3, 6)
    cols_orig_set = set(df_original.columns)
    cols_clean_set = set(df_clean.columns)
    eliminadas = len(cols_orig_set - cols_clean_set)
    agregadas = len(cols_clean_set - cols_orig_set)
    mantenidas = len(cols_orig_set & cols_clean_set)

    sizes = [mantenidas, eliminadas, agregadas]
    labels = [f'Mantenidas\n({mantenidas})', f'Eliminadas\n({eliminadas})', f'Agregadas\n({agregadas})']
    colors = ['#27ae60', '#e74c3c', '#3498db']
    explode = (0.05, 0.1, 0.05)

    wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, explode=explode, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax6.set_title('Cambios en Columnas', fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.show()

    # ============================================================================
    # 4. DETALLES DE COLUMNAS ELIMINADAS Y AGREGADAS
    # ============================================================================
    eliminadas_set = cols_orig_set - cols_clean_set
    agregadas_set = cols_clean_set - cols_orig_set

    display(HTML(f"""
    <div style='background: #f8f9fa; padding: 25px; border-radius: 12px; margin: 30px 0;
                border-left: 5px solid #e74c3c;'>
        <h2 style='color: #e74c3c; margin-top: 0; font-size: 24px;'>
            🗑️ COLUMNAS ELIMINADAS ({len(eliminadas_set)})
        </h2>
    """))

    if eliminadas_set:
        html_eliminadas = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px;'>"
        for col in sorted(eliminadas_set):
            if col in df_original.columns:
                nulls_col = df_original[col].isnull().sum()
                pct_null = (nulls_col / len(df_original)) * 100
                html_eliminadas += f"""
                <div style='background: white; padding: 15px; border-radius: 8px;
                            border-left: 3px solid #e74c3c; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='font-weight: bold; color: #2c3e50; margin-bottom: 8px; font-size: 14px;'>
                        {col}
                    </div>
                    <div style='color: #7f8c8d; font-size: 13px;'>
                        ❌ {nulls_col:,} nulls ({pct_null:.1f}%)
                    </div>
                </div>
                """
        html_eliminadas += "</div>"
        display(HTML(html_eliminadas))
    else:
        display(HTML("<p style='color: #7f8c8d; font-style: italic;'>• Ninguna columna eliminada</p>"))

    display(HTML("</div>"))

    display(HTML(f"""
    <div style='background: #f8f9fa; padding: 25px; border-radius: 12px; margin: 30px 0;
                border-left: 5px solid #27ae60;'>
        <h2 style='color: #27ae60; margin-top: 0; font-size: 24px;'>
            ➕ COLUMNAS AGREGADAS ({len(agregadas_set)})
        </h2>
    """))

    if agregadas_set:
        html_agregadas = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 15px;'>"
        for col in sorted(agregadas_set):
            if col.endswith('_encoded'):
                original = col.replace('_encoded', '')
                if original in df_original.columns:
                    unique_count = df_clean[col].nunique()
                    html_agregadas += f"""
                    <div style='background: white; padding: 15px; border-radius: 8px;
                                border-left: 3px solid #27ae60; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <div style='font-weight: bold; color: #2c3e50; margin-bottom: 8px; font-size: 14px;'>
                            {col}
                        </div>
                        <div style='color: #7f8c8d; font-size: 13px;'>
                            🔄 Codificación de '{original}'<br>
                            📊 {unique_count} valores únicos
                        </div>
                    </div>
                    """
            else:
                html_agregadas += f"""
                <div style='background: white; padding: 15px; border-radius: 8px;
                            border-left: 3px solid #27ae60; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='font-weight: bold; color: #2c3e50; font-size: 14px;'>
                        {col}
                    </div>
                </div>
                """
        html_agregadas += "</div>"
        display(HTML(html_agregadas))
    else:
        display(HTML("<p style='color: #7f8c8d; font-style: italic;'>• Ninguna columna agregada</p>"))

    display(HTML("</div>"))

    # ============================================================================
    # 5. RESUMEN FINAL
    # ============================================================================
    display(HTML(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px; border-radius: 15px; margin: 30px 0; color: white; text-align: center;'>
        <h2 style='margin: 0 0 20px 0; font-size: 28px;'>✨ RESUMEN DEL PROCESO</h2>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;'>
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;'>
                <div style='font-size: 36px; font-weight: bold;'>{abs(nulls_clean - nulls_orig):,}</div>
                <div style='font-size: 16px; margin-top: 8px; opacity: 0.9;'>Nulls Eliminados</div>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;'>
                <div style='font-size: 36px; font-weight: bold;'>+{completitud_clean - completitud_orig:.1f}%</div>
                <div style='font-size: 16px; margin-top: 8px; opacity: 0.9;'>Mejora en Completitud</div>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;'>
                <div style='font-size: 36px; font-weight: bold;'>{len(agregadas_set)}</div>
                <div style='font-size: 16px; margin-top: 8px; opacity: 0.9;'>Columnas Codificadas</div>
            </div>
        </div>
    </div>
    """))

else:
    display(HTML("""
    <div style='background: #e74c3c; color: white; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2 style='margin: 0;'>❌ No se puede hacer la comparación sin el dataset limpio</h2>
    </div>
    """))
