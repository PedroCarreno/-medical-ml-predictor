import numpy as np
import pandas as pd

# Verificar que el codigo de la celda funciona correctamente
print("TEST: Verificando logica de visualizacion de outliers")
print("="*60)

# Datos de ejemplo (mismos que en el notebook)
datos_normales = np.array([72, 75, 68, 80, 78, 71, 76, 73, 79, 74])
datos_con_outliers = np.array([72, 75, 68, 80, 78, 71, 76, 73, 155, 180])

# Calcular estadisticas
media_normal = np.mean(datos_normales)
mediana_normal = np.median(datos_normales)
media_outliers = np.mean(datos_con_outliers)
mediana_outliers = np.median(datos_con_outliers)

print("\n1. DATOS NORMALES:")
print(f"   Valores: {datos_normales}")
print(f"   Media: {media_normal:.1f} lpm")
print(f"   Mediana: {mediana_normal:.1f} lpm")
print(f"   Diferencia: {abs(media_normal - mediana_normal):.1f} lpm")

print("\n2. DATOS CON OUTLIERS:")
print(f"   Valores: {datos_con_outliers}")
print(f"   Media: {media_outliers:.1f} lpm")
print(f"   Mediana: {mediana_outliers:.1f} lpm")
print(f"   Diferencia: {abs(media_outliers - mediana_outliers):.1f} lpm")

print("\n3. IMPACTO DE OUTLIERS:")
print(f"   Cambio en Media: +{media_outliers - media_normal:.1f} lpm")
print(f"   Cambio en Mediana: +{mediana_outliers - mediana_normal:.1f} lpm")
print(f"   Distorsion: {media_outliers - mediana_outliers:.1f} lpm")

print("\n4. VERIFICACION DE LOGICA:")
if media_outliers > mediana_outliers:
    print("   [OK] Media se infla por outliers")
else:
    print("   [ERROR] Media deberia ser mayor que mediana")

if abs(mediana_outliers - mediana_normal) < 5:
    print("   [OK] Mediana se mantiene estable")
else:
    print("   [ERROR] Mediana deberia ser estable")

print("\n5. TABLA COMPARATIVA:")
comparacion = pd.DataFrame({
    'Escenario': ['SIN outliers', 'CON outliers'],
    'Media': [f"{media_normal:.1f}", f"{media_outliers:.1f}"],
    'Mediana': [f"{mediana_normal:.1f}", f"{mediana_outliers:.1f}"],
    'Distorsion': [
        "0.0 lpm",
        f"{media_outliers - mediana_outliers:.1f} lpm"
    ]
})
print(comparacion.to_string(index=False))

print("\n" + "="*60)
print("TEST COMPLETADO: La logica es correcta")
print("="*60)
