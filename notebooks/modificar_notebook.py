import nbformat
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Leer notebook
nb = nbformat.read('presentacion_limpieza_dataset.ipynb', as_version=4)

print("Modificando celda 28 del notebook...")

# NUEVO CÓDIGO CORREGIDO PARA CELDA 28
nuevo_codigo_celda_28 = """from IPython.display import HTML, display
import pandas as pd
import numpy as np

# Calcular estadísticas reales del dataset
age_data = df_original['age'].dropna()
mean_age = age_data.mean()
median_age = age_data.median()

# Para el ejemplo visual, usamos valores que coincidan con la mediana real (65)
ejemplo_edades = [18, 60, 65, 70, 95]  # 5 valores - mediana será el del medio = 65
ejemplo_median = np.median(ejemplo_edades)  # = 65
ejemplo_mean = np.mean(ejemplo_edades)      # = 61.6

# Ejemplo con cantidad PAR para explicar el cálculo
ejemplo_par = [50, 60, 70, 80]  # 4 valores - mediana = (60+70)/2 = 65
ejemplo_par_median = np.median(ejemplo_par)

# Calcular impacto de eliminar filas
filas_con_missing = df_original.isnull().any(axis=1).sum()
pct_filas_missing = (filas_con_missing / len(df_original)) * 100

# Analizar variables categóricas reales
cat_cols = df_original.select_dtypes(include=['object']).columns
cat_data = []
for col in ['ethnicity', 'gender', 'icu_admit_source', 'apache_3j_bodysystem', 'apache_2_bodysystem']:
    if col in df_original.columns:
        nulls = df_original[col].isnull().sum()
        if nulls > 0 and len(df_original[col].mode()) > 0:
            moda = df_original[col].mode()[0]
            freq = (df_original[col] == moda).sum()
            pct = (freq / len(df_original)) * 100
            cat_data.append({
                'columna': col,
                'nulls': nulls,
                'moda': moda,
                'frecuencia': freq,
                'porcentaje': pct
            })

# Analizar algunas variables numéricas para mostrar
num_vars_ejemplos = [
    ('age', 'Edad (años)'),
    ('bmi', 'IMC'),
    ('weight', 'Peso (kg)'),
    ('d1_glucose_max', 'Glucosa máx (mg/dL)'),
    ('heart_rate_apache', 'Frecuencia cardíaca'),
]

num_data = []
for col, nombre in num_vars_ejemplos:
    if col in df_original.columns:
        nulls = df_original[col].isnull().sum()
        if nulls > 0:
            median_val = df_original[col].median()
            mean_val = df_original[col].mean()
            diff = abs(median_val - mean_val)
            num_data.append({
                'columna': nombre,
                'nulls': nulls,
                'mediana': median_val,
                'media': mean_val,
                'diferencia': diff
            })

# Crear HTML con diseño profesional y lenguaje simple
html_content = f\"\"\"
<style>
    .justification-container {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1400px;
        margin: 20px auto;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }}
    .justification-inner {{
        background: white;
        padding: 40px;
        border-radius: 13px;
    }}
    .justification-title {{
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    .justification-subtitle {{
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 40px;
        font-style: italic;
    }}
    .section {{
        margin: 40px 0;
        padding: 30px;
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        border-left: 6px solid #667eea;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }}
    .section-title {{
        font-size: 26px;
        font-weight: bold;
        color: #333;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 15px;
    }}
    .section-number {{
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        width: 45px;
        height: 45px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: bold;
        flex-shrink: 0;
    }}
    .example-box {{
        background: #fff3cd;
        border: 3px solid #ffc107;
        border-radius: 12px;
        padding: 25px;
        margin: 25px 0;
    }}
    .example-title {{
        font-size: 20px;
        font-weight: bold;
        color: #856404;
        margin-bottom: 20px;
    }}
    .example-scenario {{
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        font-size: 17px;
        line-height: 2;
    }}
    .comparison-visual {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 25px;
        margin: 25px 0;
    }}
    .comparison-card {{
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        text-align: center;
        transition: transform 0.3s;
    }}
    .comparison-card:hover {{
        transform: translateY(-5px);
    }}
    .comparison-bad {{
        border: 4px solid #dc3545;
    }}
    .comparison-good {{
        border: 4px solid #28a745;
    }}
    .comparison-label {{
        font-size: 16px;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 15px;
        letter-spacing: 1px;
    }}
    .comparison-bad .comparison-label {{
        color: #dc3545;
    }}
    .comparison-good .comparison-label {{
        color: #28a745;
    }}
    .comparison-value {{
        font-size: 56px;
        font-weight: bold;
        margin: 20px 0;
        line-height: 1;
    }}
    .comparison-bad .comparison-value {{
        color: #dc3545;
    }}
    .comparison-good .comparison-value {{
        color: #28a745;
    }}
    .comparison-desc {{
        font-size: 15px;
        color: #555;
        line-height: 1.7;
    }}
    .highlight-box {{
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border: 3px solid #667eea;
        border-radius: 12px;
        padding: 25px;
        margin: 25px 0;
    }}
    .highlight-title {{
        font-size: 20px;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 15px;
    }}
    .method-badge {{
        display: inline-block;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
        margin: 5px;
    }}
    .badge-mediana {{
        background: #667eea;
        color: white;
    }}
    .badge-moda {{
        background: #f093fb;
        color: white;
    }}
    .calc-box {{
        background: #e8f5e9;
        border: 3px dashed #4caf50;
        border-radius: 12px;
        padding: 25px;
        margin: 25px 0;
    }}
    .calc-title {{
        color: #2e7d32;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 20px;
    }}
    .calc-step {{
        margin: 15px 0;
        font-size: 17px;
        line-height: 2;
        padding: 10px;
        background: white;
        border-radius: 8px;
    }}
    .visual-example {{
        background: #e3f2fd;
        border: 3px solid #2196f3;
        border-radius: 12px;
        padding: 25px;
        margin: 25px 0;
    }}
    .visual-patients {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
        flex-wrap: wrap;
        font-size: 18px;
    }}
    .patient-icon {{
        font-size: 40px;
    }}
    .data-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }}
    .data-table th {{
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 18px;
        text-align: left;
        font-weight: bold;
        font-size: 16px;
    }}
    .data-table td {{
        padding: 15px 18px;
        border-bottom: 2px solid #e0e0e0;
        font-size: 16px;
    }}
    .data-table tr:nth-child(even) {{
        background: #f8f9fa;
    }}
    .data-table tr:hover {{
        background: #e3f2fd;
    }}
    .reason-list {{
        margin: 20px 0;
    }}
    .reason-item {{
        display: flex;
        align-items: start;
        margin: 15px 0;
        padding: 18px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    }}
    .reason-icon {{
        font-size: 28px;
        margin-right: 18px;
        flex-shrink: 0;
    }}
    .reason-text {{
        font-size: 17px;
        color: #444;
        line-height: 1.8;
    }}
    .summary-box {{
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 35px;
        border-radius: 15px;
        margin-top: 40px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }}
    .summary-title {{
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }}
    .summary-text {{
        font-size: 18px;
        line-height: 2;
    }}
</style>

<div class="justification-container">
    <div class="justification-inner">
        <div class="justification-title">🔬 ¿Cómo Rellenamos los Datos Faltantes?</div>
        <div class="justification-subtitle">Análisis científico de las decisiones tomadas</div>

        <!-- SECCIÓN 1: MEDIANA vs PROMEDIO -->
        <div class="section">
            <div class="section-title">
                <div class="section-number">1</div>
                ¿Por qué usamos MEDIANA en lugar de PROMEDIO?
            </div>

            <div class="highlight-box">
                <div class="highlight-title">📊 Decisión tomada:</div>
                <p style="font-size: 18px; color: #444; margin: 15px 0;">
                    <span class="method-badge badge-mediana">MEDIANA</span>
                    Para variables numéricas (edad, peso, presión, glucosa, etc.) → Rellenar con la <strong>MEDIANA</strong>
                </p>
            </div>

            <div class="calc-box">
                <div class="calc-title">📝 ¿Cómo se calcula la MEDIANA?</div>
                <div class="calc-step">
                    <strong>Paso 1:</strong> Ordenar todos los valores de menor a mayor
                </div>
                <div class="calc-step">
                    <strong>Paso 2:</strong><br>
                    • Si hay cantidad <strong>IMPAR</strong> de valores → tomar el valor del <strong>MEDIO</strong><br>
                    • Si hay cantidad <strong>PAR</strong> de valores → <strong>PROMEDIO de los 2 valores centrales</strong>
                </div>
            </div>

            <div class="example-box">
                <div class="example-title">💡 EJEMPLO: 5 pacientes en UCI (cantidad IMPAR)</div>
                <div class="visual-example">
                    <p style="font-size: 16px; margin-bottom: 15px;"><strong>Edades desordenadas:</strong></p>
                    <div class="visual-patients">
                        <span class="patient-icon">🧑</span> <strong>70 años</strong>
                        <span class="patient-icon">👨</span> <strong>18 años</strong>
                        <span class="patient-icon">👵</span> <strong>95 años</strong>
                        <span class="patient-icon">👴</span> <strong>60 años</strong>
                        <span class="patient-icon">🧓</span> <strong>65 años</strong>
                    </div>
                    <p style="font-size: 16px; margin: 20px 0;"><strong>Paso 1: Ordenamos →</strong></p>
                    <div style="text-align: center; font-size: 22px; font-weight: bold; margin: 20px 0;">
                        18, 60, <span style="color: #28a745; font-size: 32px;">65</span>, 70, 95
                    </div>
                    <p style="text-align: center; font-size: 18px; color: #28a745; font-weight: bold;">
                        ↑ Este es el VALOR DEL MEDIO (posición 3 de 5)
                    </p>
                </div>

                <div class="comparison-visual">
                    <div class="comparison-card comparison-bad">
                        <div class="comparison-label">❌ PROMEDIO</div>
                        <div class="comparison-value">{ejemplo_mean:.0f}</div>
                        <div class="comparison-desc">
                            (18+60+65+70+95) ÷ 5 = {ejemplo_mean:.1f}<br><br>
                            <strong>PROBLEMA:</strong> El paciente de 18 años "tira para abajo" el resultado
                        </div>
                    </div>

                    <div class="comparison-card comparison-good">
                        <div class="comparison-label">✅ MEDIANA</div>
                        <div class="comparison-value">{ejemplo_median:.0f}</div>
                        <div class="comparison-desc">
                            Valor del medio = 65<br><br>
                            <strong>VENTAJA:</strong> No se distorsiona por valores extremos
                        </div>
                    </div>
                </div>
            </div>

            <div class="example-box">
                <div class="example-title">💡 EJEMPLO: 4 pacientes (cantidad PAR)</div>
                <div class="visual-example">
                    <div style="text-align: center; font-size: 22px; font-weight: bold; margin: 20px 0;">
                        Ordenados: 50, <span style="color: #f39c12;">60</span>, <span style="color: #f39c12;">70</span>, 80
                    </div>
                    <p style="text-align: center; font-size: 18px; color: #f39c12; font-weight: bold;">
                        ↑ Los 2 del MEDIO
                    </p>
                    <div style="text-align: center; font-size: 20px; margin: 20px 0;">
                        <strong>MEDIANA = (60 + 70) ÷ 2 = {ejemplo_par_median:.0f}</strong>
                    </div>
                </div>
            </div>

            <div class="example-scenario">
                <strong style="font-size: 20px; color: #667eea;">🎯 DATOS REALES de nuestro dataset:</strong><br><br>
                • Total pacientes con edad: <strong>{len(age_data):,}</strong><br>
                • <span style="color: #28a745; font-weight: bold;">MEDIANA edad: {median_age:.1f} años</span> ✅ (usamos esta)<br>
                • <span style="color: #dc3545; font-weight: bold;">PROMEDIO edad: {mean_age:.1f} años</span> ❌ (NO usamos esta)<br>
                • Diferencia: <strong>{abs(median_age - mean_age):.1f} años</strong><br><br>

                <strong>¿Por qué {median_age:.0f} y no {mean_age:.0f}?</strong><br>
                Porque hay pacientes muy jóvenes (mínimo 16 años) y muy ancianos (máximo 89 años) que distorsionan el promedio.
                La mediana ignora estos extremos y representa mejor al "paciente típico de UCI".
            </div>

            <div class="highlight-box">
                <div class="highlight-title">📊 Variables numéricas procesadas con MEDIANA:</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Nulls</th>
                            <th>Mediana</th>
                            <th>Media</th>
                            <th>Diferencia</th>
                        </tr>
                    </thead>
                    <tbody>
"""

for item in num_data:
    html_content += f"""
                        <tr>
                            <td><strong>{item['columna']}</strong></td>
                            <td style="color: #dc3545; font-weight: bold;">{item['nulls']:,}</td>
                            <td style="color: #28a745; font-weight: bold;">{item['mediana']:.1f}</td>
                            <td style="color: #666;">{item['media']:.1f}</td>
                            <td style="color: #667eea; font-weight: bold;">{item['diferencia']:.1f}</td>
                        </tr>
"""

html_content += f"""
                    </tbody>
                </table>
                <p style="font-size: 15px; color: #666; margin-top: 15px; font-style: italic;">
                    💡 Nota: La diferencia entre mediana y media muestra cuánto distorsionan los valores extremos
                </p>
            </div>

            <div class="reason-list">
                <div class="reason-item">
                    <div class="reason-icon">✅</div>
                    <div class="reason-text">
                        <strong>Robusta a valores extremos:</strong> Si un paciente tiene 120 años o glucosa de 800 mg/dL, el promedio se altera mucho. La mediana NO se altera.
                    </div>
                </div>
                <div class="reason-item">
                    <div class="reason-icon">✅</div>
                    <div class="reason-text">
                        <strong>Representa al paciente típico:</strong> {median_age:.0f} años es la edad central en nuestro dataset UCI. Tiene sentido médico y estadístico.
                    </div>
                </div>
                <div class="reason-item">
                    <div class="reason-icon">✅</div>
                    <div class="reason-text">
                        <strong>Aplicado consistentemente:</strong> Usamos mediana para TODAS las {len(num_data)} variables numéricas mostradas (y muchas más).
                    </div>
                </div>
            </div>
        </div>

        <!-- SECCIÓN 2: NO ELIMINAR FILAS -->
        <div class="section">
            <div class="section-title">
                <div class="section-number">2</div>
                ¿Por qué NO eliminamos pacientes con datos faltantes?
            </div>

            <div class="example-box">
                <div class="example-title">💡 Análisis del impacto:</div>
                <div class="visual-example">
                    <p style="font-size: 18px; margin: 20px 0;">
                        Dataset original: <strong>{len(df_original):,} pacientes</strong><br><br>
                        Pacientes con <strong>al menos 1 dato faltante</strong>: <strong>{filas_con_missing:,}</strong> ({pct_filas_missing:.1f}%)
                    </p>

                    <div class="comparison-visual">
                        <div class="comparison-card comparison-bad">
                            <div class="comparison-label">❌ Si ELIMINAMOS filas con nulls</div>
                            <div class="comparison-value">{len(df_original) - filas_con_missing:,}</div>
                            <div class="comparison-desc">
                                Pacientes restantes<br><br>
                                <strong>SOLO {100 - pct_filas_missing:.1f}% de los datos!</strong><br>
                                Perdemos {filas_con_missing:,} historias clínicas valiosas
                            </div>
                        </div>

                        <div class="comparison-card comparison-good">
                            <div class="comparison-label">✅ Si RELLENAMOS con mediana/moda</div>
                            <div class="comparison-value">{len(df_original):,}</div>
                            <div class="comparison-desc">
                                Pacientes conservados<br><br>
                                <strong>100% de los datos disponibles!</strong><br>
                                Aprovechamos toda la información
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="reason-list">
                <div class="reason-item">
                    <div class="reason-icon">❌</div>
                    <div class="reason-text">
                        <strong>Pérdida masiva de información:</strong> Eliminar {filas_con_missing:,} pacientes significa desperdiciar {pct_filas_missing:.1f}% del dataset. Imposible entrenar modelos robustos con tan pocos datos.
                    </div>
                </div>
                <div class="reason-item">
                    <div class="reason-icon">❌</div>
                    <div class="reason-text">
                        <strong>Sesgo de selección:</strong> Los pacientes sin datos faltantes pueden ser diferentes (ej: estadías más largas en UCI = más mediciones). Eliminarlos introduce SESGO.
                    </div>
                </div>
                <div class="reason-item">
                    <div class="reason-icon">✅</div>
                    <div class="reason-text">
                        <strong>Mejor práctica en ciencia de datos médicos:</strong> La imputación (relleno) con mediana/moda es el estándar de la industria. Conserva información y evita sesgos.
                    </div>
                </div>
            </div>
        </div>

        <!-- SECCIÓN 3: VARIABLES CATEGÓRICAS CON MODA -->
        <div class="section">
            <div class="section-title">
                <div class="section-number">3</div>
                ¿Cómo rellenamos variables CATEGÓRICAS (texto)?
            </div>

            <div class="highlight-box">
                <div class="highlight-title">📊 Decisión tomada:</div>
                <p style="font-size: 18px; color: #444; margin: 15px 0;">
                    <span class="method-badge badge-moda">MODA (valor más frecuente)</span>
                    Para variables de texto (género, etnia, tipo UCI, etc.) → Rellenar con el <strong>valor MÁS COMÚN</strong>
                </p>
            </div>

            <div class="example-box">
                <div class="example-title">💡 EJEMPLO: Variable "gender" (género)</div>
                <div class="example-scenario">
                    <strong>Supongamos 100 pacientes:</strong><br>
                    👨👨👨👨👨👨 <strong>54 son Masculino (M)</strong><br>
                    👩👩👩👩 <strong>46 son Femenino (F)</strong><br><br>

                    <strong>🎯 Si falta el género de 1 paciente:</strong><br>
                    Le asignamos <strong>"M"</strong> (la MODA, el valor más frecuente)<br><br>

                    <strong>¿Por qué tiene sentido?</strong><br>
                    Estadísticamente, hay 54% de probabilidad de que sea masculino vs 46% femenino.
                    Es la "mejor adivinanza" basada en los datos que SÍ tenemos.
                </div>
            </div>

            <div class="highlight-box">
                <div class="highlight-title">📊 Variables categóricas reales procesadas con MODA:</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Nulls</th>
                            <th>Moda (Valor + Frecuente)</th>
                            <th>Frecuencia</th>
                        </tr>
                    </thead>
                    <tbody>
"""

for item in cat_data:
    html_content += f"""
                        <tr>
                            <td><strong>{item['columna']}</strong></td>
                            <td style="color: #dc3545; font-weight: bold;">{item['nulls']:,}</td>
                            <td style="color: #f093fb; font-weight: bold;">{item['moda']}</td>
                            <td style="color: #667eea; font-weight: bold;">{item['porcentaje']:.1f}%</td>
                        </tr>
"""

html_content += f"""
                    </tbody>
                </table>
            </div>

            <div class="reason-list">
                <div class="reason-item">
                    <div class="reason-icon">✅</div>
                    <div class="reason-text">
                        <strong>Máxima probabilidad:</strong> Si el 77% de pacientes son "Caucasian", es más probable que un paciente con etnia faltante también lo sea.
                    </div>
                </div>
                <div class="reason-item">
                    <div class="reason-icon">✅</div>
                    <div class="reason-text">
                        <strong>Consistencia con distribución real:</strong> Al usar la moda, mantenemos las proporciones originales del dataset.
                    </div>
                </div>
            </div>
        </div>

        <!-- SECCIÓN 4: ENCODING -->
        <div class="section">
            <div class="section-title">
                <div class="section-number">4</div>
                ¿Por qué CODIFICAMOS las variables categóricas?
            </div>

            <div class="highlight-box">
                <div class="highlight-title">🤖 Problema fundamental:</div>
                <p style="font-size: 18px; color: #444; margin: 15px 0;">
                    Los algoritmos de Machine Learning <strong>SOLO entienden números</strong>. No pueden procesar texto directamente.
                </p>
            </div>

            <div class="example-box">
                <div class="example-title">💡 Proceso completo paso a paso:</div>
                <div class="calc-box" style="background: white; border-color: #667eea;">
                    <div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                        <strong style="font-size: 18px;">PASO 1: Rellenar nulls con MODA</strong><br><br>
                        <code style="font-size: 16px;">gender: [M, F, <span style="color: #dc3545; font-weight: bold;">null</span>, M, F, <span style="color: #dc3545; font-weight: bold;">null</span>]</code><br>
                        ↓ Aplicamos moda (M es más frecuente)<br>
                        <code style="font-size: 16px;">gender: [M, F, <span style="color: #28a745; font-weight: bold;">M</span>, M, F, <span style="color: #28a745; font-weight: bold;">M</span>]</code>
                    </div>

                    <div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                        <strong style="font-size: 18px;">PASO 2: Encoding (texto → números)</strong><br><br>
                        <code style="font-size: 16px;">gender: [M, F, M, M, F, M]</code><br>
                        ↓ Aplicamos Label Encoding<br>
                        <code style="font-size: 16px;">gender_encoded: [<span style="color: #667eea; font-weight: bold;">0, 1, 0, 0, 1, 0</span>]</code>
                    </div>

                    <div style="text-align: center; font-size: 20px; color: #28a745; font-weight: bold; margin-top: 20px;">
                        🎯 ¡Ahora el modelo puede procesar la variable!
                    </div>
                </div>
            </div>

            <div class="reason-list">
                <div class="reason-item">
                    <div class="reason-icon">1️⃣</div>
                    <div class="reason-text">
                        <strong>Primero rellenamos (MODA):</strong> Completamos los nulls de la variable ORIGINAL con el valor más frecuente
                    </div>
                </div>
                <div class="reason-item">
                    <div class="reason-icon">2️⃣</div>
                    <div class="reason-text">
                        <strong>Después codificamos (Label Encoding):</strong> Convertimos texto → números (M→0, F→1, etc.)
                    </div>
                </div>
                <div class="reason-item">
                    <div class="reason-icon">✅</div>
                    <div class="reason-text">
                        <strong>Resultado:</strong> Las variables `_encoded` NO tienen nulls y el modelo ML puede trabajar con ellas
                    </div>
                </div>
                <div class="reason-item">
                    <div class="reason-icon">⚠️</div>
                    <div class="reason-text">
                        <strong>Orden importante:</strong> Si codificáramos ANTES de rellenar, no sabríamos qué número asignar a los nulls
                    </div>
                </div>
            </div>
        </div>

        <!-- RESUMEN FINAL -->
        <div class="summary-box">
            <div class="summary-title">✨ RESUMEN DE DECISIONES CIENTÍFICAS</div>
            <div class="summary-text">
                <strong>📊 Variables Numéricas:</strong> MEDIANA (robusta a extremos, representa al paciente típico)<br>
                <strong>📝 Variables Categóricas:</strong> MODA (valor más frecuente, máxima probabilidad)<br>
                <strong>🔄 Codificación:</strong> Label Encoding DESPUÉS de rellenar (texto → números)<br>
                <strong>💾 Conservación de datos:</strong> {len(df_original):,} pacientes ({100 - pct_filas_missing:.1f}% más que si elimináramos)<br>
                <strong>🎯 Resultado:</strong> Dataset completo, sin nulls, listo para Machine Learning ✅
            </div>
        </div>
    </div>
</div>
\"\"\"

display(HTML(html_content))"""

# Reemplazar la celda 28
nb.cells[28].source = nuevo_codigo_celda_28

# Guardar notebook modificado
nbformat.write(nb, 'presentacion_limpieza_dataset.ipynb')

print("✅ Celda 28 modificada exitosamente!")
print("\nCambios aplicados:")
print("1. ✅ Ejemplo corregido: [18, 60, 65, 70, 95] → mediana = 65")
print("2. ✅ Explicación de cálculo de mediana (impar y par)")
print("3. ✅ Tabla con variables numéricas reales del dataset")
print("4. ✅ Tabla con variables categóricas reales")
print("5. ✅ Sección completa sobre encoding con orden claro")
print("6. ✅ Datos reales: mediana edad = 65.0, promedio = 62.3")
print("\n📁 Archivo guardado: presentacion_limpieza_dataset.ipynb")
