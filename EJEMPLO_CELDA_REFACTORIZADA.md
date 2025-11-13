# EJEMPLO DE REFACTORIZACIÓN CORRECTA

## ANTES (INCORRECTO) ❌

### Celda con HTML crudo:
```python
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
    <h1>Hallazgo #4: Glasgow</h1>
</div>
```

**Problema**: El HTML se muestra como texto crudo, no renderiza la UI.

---

## DESPUÉS (CORRECTO) ✅

### Celda con display(HTML()):
```python
from IPython.display import HTML, display

display(HTML("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 15px; margin: 50px 0; text-align: center;'>
    <h1 style='color: white; font-size: 48px; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>HALLAZGO #4</h1>
    <h2 style='color: white; font-size: 36px; margin: 15px 0 0 0; font-weight: 300;'>Escala de Glasgow (Escala de Coma de Glasgow)</h2>
</div>
"""))

display(HTML("""
<div style='background: #e8f5e9; padding: 30px; border-left: 6px solid #4caf50; border-radius: 10px; margin: 30px 0;'>
    <h3 style='color: #2e7d32; margin-top: 0; font-size: 26px;'>¿Qué es la Escala de Glasgow?</h3>
    <p style='font-size: 18px; line-height: 1.8; color: #1b5e20; margin: 0;'>
        La <strong>Escala de Coma de Glasgow</strong> es una escala médica que evalúa el <strong>nivel de conciencia</strong> del paciente.
        Se mide en un rango de <strong>3 a 15 puntos</strong>, evaluando tres componentes:
    </p>
    <ul style='font-size: 18px; line-height: 1.8; color: #1b5e20;'>
        <li><strong>Apertura Ocular</strong> (1-4 puntos): Respuesta al estímulo visual</li>
        <li><strong>Respuesta Verbal</strong> (1-5 puntos): Capacidad de comunicación</li>
        <li><strong>Respuesta Motora</strong> (1-6 puntos): Movimiento en respuesta a estímulos</li>
    </ul>
    <p style='font-size: 18px; line-height: 1.8; color: #1b5e20; margin: 20px 0 0 0;'>
        <strong>Interpretación:</strong><br>
        - <strong>13-15 puntos:</strong> Lesión cerebral leve<br>
        - <strong>9-12 puntos:</strong> Lesión cerebral moderada<br>
        - <strong>3-8 puntos:</strong> Lesión cerebral severa (coma)
    </p>
</div>
"""))
```

**Resultado**: El HTML se renderiza correctamente con toda la UI/UX visual.

---

## EJEMPLO COMPLETO DE ANÁLISIS CON DATOS REALES

```python
# Calcular Escala de Glasgow Total
df['gcs_total'] = df['gcs_eyes_apache'] + df['gcs_verbal_apache'] + df['gcs_motor_apache']

# Clasificar por rangos
df['gcs_rango'] = pd.cut(df['gcs_total'],
                          bins=[0, 8, 12, 15],
                          labels=['Severo (3-8)', 'Moderado (9-12)', 'Leve (13-15)'])

# Calcular estadísticas por rango
glasgow_stats = df.groupby('gcs_rango', observed=True).agg({
    'hospital_death': ['count', 'sum', 'mean']
}).round(4)

glasgow_stats.columns = ['Total Pacientes', 'Muertes', 'Tasa Mortalidad']
glasgow_stats['Sobreviven'] = glasgow_stats['Total Pacientes'] - glasgow_stats['Muertes']
glasgow_stats['% Mortalidad'] = (glasgow_stats['Tasa Mortalidad'] * 100).round(2)

print('\nESTADÍSTICAS POR RANGO DE GLASGOW:\n')
print(glasgow_stats[['Total Pacientes', 'Sobreviven', 'Muertes', '% Mortalidad']])
```

**Salida esperada**:
```
ESTADÍSTICAS POR RANGO DE GLASGOW:

                 Total Pacientes  Sobreviven  Muertes  % Mortalidad
gcs_rango
Severo (3-8)               12237        9018     3219         26.31
Moderado (9-12)            10419        9294     1125         10.80
Leve (13-15)               69057       65486     3571          5.17
```

---

## VISUALIZACIÓN CON GRÁFICOS

```python
# Visualización de Glasgow vs Mortalidad
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Tasa de Mortalidad por Rango
rangos = glasgow_stats.index
mortalidad = glasgow_stats['% Mortalidad'].values
colores = ['#e74c3c', '#f39c12', '#27ae60']

bars = axes[0].bar(rangos, mortalidad, color=colores, alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Tasa de Mortalidad (%)', fontsize=14, fontweight='bold')
axes[0].set_title('Mortalidad por Escala de Glasgow', fontsize=16, fontweight='bold')
axes[0].set_ylim([0, max(mortalidad) * 1.2])

# Agregar valores en las barras
for bar, val in zip(bars, mortalidad):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.2f}%', ha='center', fontweight='bold', fontsize=13)

# Gráfico 2: Distribución de Pacientes
totales = glasgow_stats['Total Pacientes'].values
axes[1].bar(rangos, totales, color=colores, alpha=0.8, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Número de Pacientes', fontsize=14, fontweight='bold')
axes[1].set_title('Distribución de Pacientes por Rango', fontsize=16, fontweight='bold')

for i, (rango, total) in enumerate(zip(rangos, totales)):
    axes[1].text(i, total + 1000, f'{total:,}', ha='center', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('hallazgo_4_glasgow.png', dpi=300, bbox_inches='tight')
plt.show()

print('\nGráfico guardado: hallazgo_4_glasgow.png')
```

---

## CONCLUSIONES CON HTML DINÁMICO

```python
from IPython.display import HTML, display

# Extraer datos para mostrar
severo_mort = glasgow_stats.loc['Severo (3-8)', '% Mortalidad']
moderado_mort = glasgow_stats.loc['Moderado (9-12)', '% Mortalidad']
leve_mort = glasgow_stats.loc['Leve (13-15)', '% Mortalidad']

severo_count = int(glasgow_stats.loc['Severo (3-8)', 'Total Pacientes'])
moderado_count = int(glasgow_stats.loc['Moderado (9-12)', 'Total Pacientes'])
leve_count = int(glasgow_stats.loc['Leve (13-15)', 'Total Pacientes'])

display(HTML(f"""
<div style='background: white; padding: 35px; border-radius: 15px; margin: 30px 0; box-shadow: 0 5px 20px rgba(0,0,0,0.1);'>
    <h3 style='color: #c0392b; font-size: 28px; margin-top: 0;'>Conclusiones Clave - Escala de Glasgow</h3>

    <div style='background: #fee; padding: 25px; border-left: 6px solid #e74c3c; border-radius: 8px; margin: 20px 0;'>
        <h4 style='color: #c0392b; margin-top: 0; font-size: 22px;'>SEVERO (3-8 puntos)</h4>
        <p style='font-size: 18px; line-height: 1.8; margin: 0;'>
            <strong>{severo_count:,} pacientes</strong> con lesión cerebral severa<br>
            <strong style='font-size: 24px; color: #c0392b;'>{severo_mort:.2f}%</strong> de mortalidad<br>
            <span style='color: #c0392b; font-weight: bold;'>RIESGO MUY ALTO</span> - Coma profundo
        </p>
    </div>

    <div style='background: #fff3cd; padding: 25px; border-left: 6px solid #f39c12; border-radius: 8px; margin: 20px 0;'>
        <h4 style='color: #d68910; margin-top: 0; font-size: 22px;'>MODERADO (9-12 puntos)</h4>
        <p style='font-size: 18px; line-height: 1.8; margin: 0;'>
            <strong>{moderado_count:,} pacientes</strong> con lesión cerebral moderada<br>
            <strong style='font-size: 24px; color: #d68910;'>{moderado_mort:.2f}%</strong> de mortalidad<br>
            <span style='color: #d68910; font-weight: bold;'>RIESGO MODERADO</span> - Requiere monitoreo constante
        </p>
    </div>

    <div style='background: #d4edda; padding: 25px; border-left: 6px solid #27ae60; border-radius: 8px; margin: 20px 0;'>
        <h4 style='color: #1e7e34; margin-top: 0; font-size: 22px;'>LEVE (13-15 puntos)</h4>
        <p style='font-size: 18px; line-height: 1.8; margin: 0;'>
            <strong>{leve_count:,} pacientes</strong> con lesión cerebral leve o sin lesión<br>
            <strong style='font-size: 24px; color: #1e7e34;'>{leve_mort:.2f}%</strong> de mortalidad<br>
            <span style='color: #1e7e34; font-weight: bold;'>RIESGO BAJO</span> - Conciencia preservada
        </p>
    </div>

    <div style='background: #e8f4f8; padding: 25px; border-radius: 8px; margin: 30px 0;'>
        <h4 style='color: #1565c0; margin-top: 0; font-size: 22px;'>Hallazgo Principal</h4>
        <p style='font-size: 19px; line-height: 1.8; color: #0d47a1; margin: 0; font-weight: 500;'>
            La <strong>Escala de Glasgow</strong> es un <strong>predictor crítico de mortalidad</strong>.
            Los pacientes con puntuaciones bajas (3-8) tienen <strong>5 veces más riesgo</strong> de fallecer
            comparado con pacientes con puntuaciones altas (13-15).
        </p>
    </div>
</div>
"""))
```

**Resultado**: Se renderiza un panel visual con cajas de colores según la severidad, mostrando los datos reales del dataset.

---

## PUNTOS CLAVE PARA REFACTORIZACIÓN

1. ✅ **Siempre usar** `display(HTML("""..."""))` para HTML
2. ✅ **Traducir todo** al español
3. ✅ **Usar datos reales** del archivo DATOS_REALES_PARA_REFACTORIZACION.md
4. ✅ **Explicar términos médicos** (Glasgow, APACHE, etc.)
5. ✅ **Calcular desde el dataset**, no inventar números
6. ✅ **Formato visual atractivo** con colores por severidad
7. ✅ **F-strings** para insertar datos calculados en el HTML
8. ✅ **Gráficos con matplotlib/seaborn** guardados como PNG
9. ✅ **Conclusiones claras** con hallazgos principales destacados

---

## ESTRUCTURA DE CADA HALLAZGO

```
1. Título con gradiente (display(HTML()))
2. Explicación de qué es el concepto (display(HTML()))
3. Análisis con código Python (cálculos reales)
4. Visualización con gráficos (matplotlib)
5. Conclusiones con HTML dinámico (display(HTML(f"...")))
```

---

## VERIFICACIÓN FINAL

Antes de dar por terminado el notebook:

- [ ] TODO el HTML usa `display(HTML())`
- [ ] TODOS los términos están en español
- [ ] TODOS los números provienen del dataset real
- [ ] TODOS los conceptos médicos están explicados
- [ ] TODOS los gráficos se guardan como PNG
- [ ] Cada hallazgo tiene su estructura completa
- [ ] Las conclusiones finales resumen todo
- [ ] El notebook se ejecuta sin errores
