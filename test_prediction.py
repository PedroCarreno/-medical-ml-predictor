#!/usr/bin/env python3
"""
Script de prueba rápida para el modelo de predicción médica

USO:
    python test_prediction.py
"""

import sys
sys.path.append('.')

from ml_service.train_model import MedicalMLPredictor

def test_prediction():
    """Probar predicción con un paciente de ejemplo"""

    print("=" * 80)
    print("🏥 PRUEBA DE PREDICCIÓN MÉDICA")
    print("=" * 80)

    # Cargar modelo entrenado
    print("\n📂 Cargando modelo entrenado...")
    try:
        predictor = MedicalMLPredictor.load_model('models')
        print("✅ Modelo cargado exitosamente")
        print(f"   • Mejor modelo: {predictor.best_model_name}")
        print(f"   • Modelos disponibles: {predictor.get_available_models()}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        print("\n⚠️  SOLUCIÓN: Primero entrena el modelo:")
        print("   cd ml_service")
        print("   python train_model.py")
        return

    # Datos de prueba de un paciente
    print("\n👤 PACIENTE DE PRUEBA:")
    patient_data = {
        # Demográficas
        'age': 68,
        'gender': 'M',
        'ethnicity': 'Caucasian',
        'height': 180.3,
        'weight': 85.2,
        'bmi': 26.2,

        # Admisión
        'elective_surgery': 0,
        'icu_admit_source': 'Accident & Emergency',
        'icu_stay_type': 'admit',
        'icu_type': 'Med-Surg ICU',
        'pre_icu_los_days': 0.5,

        # Scores APACHE
        'apache_2_diagnosis': 113.0,
        'apache_3j_diagnosis': 502.01,
        'apache_post_operative': 0,
        'arf_apache': 0,

        # Glasgow Coma Scale
        'gcs_eyes_apache': 4,
        'gcs_motor_apache': 6,
        'gcs_unable_apache': 0,
        'gcs_verbal_apache': 5,

        # Signos vitales APACHE
        'heart_rate_apache': 88,
        'intubated_apache': 0,
        'map_apache': 70,
        'resprate_apache': 18,
        'temp_apache': 36.8,
        'ventilated_apache': 0,

        # Presión arterial día 1
        'd1_diasbp_max': 82,
        'd1_diasbp_min': 48,
        'd1_sysbp_max': 138,
        'd1_sysbp_min': 92,
        'd1_mbp_max': 100,
        'd1_mbp_min': 62,

        # Signos vitales día 1
        'd1_heartrate_max': 95,
        'd1_heartrate_min': 65,
        'd1_resprate_max': 22,
        'd1_resprate_min': 12,
        'd1_spo2_max': 99,
        'd1_spo2_min': 94,
        'd1_temp_max': 37.2,
        'd1_temp_min': 36.5,

        # Labs día 1
        'd1_glucose_max': 142,
        'd1_glucose_min': 98,
        'd1_potassium_max': 4.1,
        'd1_potassium_min': 3.8,

        # Comorbilidades
        'aids': 0,
        'cirrhosis': 0,
        'diabetes_mellitus': 1,
        'hepatic_failure': 0,
        'immunosuppression': 0,
        'leukemia': 0,
        'lymphoma': 0,
        'solid_tumor_with_metastasis': 0,

        # Sistemas corporales
        'apache_3j_bodysystem': 'Cardiovascular',
        'apache_2_bodysystem': 'Cardiovascular'
    }

    print(f"   • Edad: {patient_data['age']} años")
    print(f"   • Género: {patient_data['gender']}")
    print(f"   • Diabetes: {'Sí' if patient_data['diabetes_mellitus'] else 'No'}")
    print(f"   • Glasgow: {patient_data['gcs_eyes_apache'] + patient_data['gcs_motor_apache'] + patient_data['gcs_verbal_apache']}/15")

    # Hacer predicción con el mejor modelo
    print(f"\n🤖 Realizando predicción con {predictor.best_model_name.upper()}...")
    result = predictor.predict_single_patient(patient_data)

    # Mostrar resultados
    print("\n" + "=" * 80)
    print("📊 RESULTADOS DE LA PREDICCIÓN")
    print("=" * 80)

    print(f"\n🎯 SALIDA 1 - Clasificación Binaria:")
    print(f"   Predicción: {result['salida_1_binaria']['result_text']}")
    print(f"   Valor numérico: {result['salida_1_binaria']['prediction']} (0=Sobrevive, 1=Muere)")

    print(f"\n📊 SALIDA 2 - Probabilidades Detalladas:")
    print(f"   Probabilidad de muerte: {result['salida_2_probabilidades']['prob_muerte']:.2f}%")
    print(f"   Probabilidad de supervivencia: {result['salida_2_probabilidades']['prob_supervivencia']:.2f}%")
    print(f"   Confianza de la predicción: {result['salida_2_probabilidades']['confianza']:.2f}%")

    print(f"\n⚠️  SALIDA 3 - Evaluación de Riesgo:")
    print(f"   Nivel de riesgo: {result['salida_3_riesgo']['nivel_riesgo']}")
    print(f"   Probabilidad de muerte: {result['salida_3_riesgo']['probabilidad_muerte']:.2f}%")
    print(f"   Recomendaciones médicas:")
    for i, rec in enumerate(result['salida_3_riesgo']['recomendaciones'], 1):
        print(f"      {i}. {rec}")

    print(f"\n🔬 INFORMACIÓN DEL MODELO:")
    print(f"   Modelo usado: {result['modelo_usado']}")
    print(f"   Top 5 variables más importantes:")
    for i, (feat, imp) in enumerate(result['variables_importantes'][:5], 1):
        print(f"      {i}. {feat}: {imp:.4f}")

    # Comparar con otros modelos disponibles
    print("\n" + "=" * 80)
    print("🔄 COMPARACIÓN CON OTROS MODELOS")
    print("=" * 80)

    for model_name in predictor.get_available_models():
        result_model = predictor.predict_single_patient(patient_data, model_name=model_name)
        prob_muerte = result_model['salida_2_probabilidades']['prob_muerte']
        nivel = result_model['salida_3_riesgo']['nivel_riesgo']

        print(f"\n{model_name.upper()}:")
        print(f"   • Probabilidad de muerte: {prob_muerte:.2f}%")
        print(f"   • Nivel de riesgo: {nivel}")

    print("\n" + "=" * 80)
    print("✅ PRUEBA COMPLETADA EXITOSAMENTE")
    print("=" * 80)


def test_high_risk_patient():
    """Probar con un paciente de alto riesgo"""

    print("\n\n" + "=" * 80)
    print("🚨 PRUEBA CON PACIENTE DE ALTO RIESGO")
    print("=" * 80)

    predictor = MedicalMLPredictor.load_model('models')

    # Paciente crítico
    critical_patient = {
        'age': 85,  # Edad muy avanzada
        'gender': 'F',
        'ethnicity': 'Caucasian',
        'height': 165,
        'weight': 55,
        'bmi': 20.2,

        'elective_surgery': 0,
        'icu_admit_source': 'Accident & Emergency',
        'icu_stay_type': 'admit',
        'icu_type': 'Med-Surg ICU',
        'pre_icu_los_days': 2.5,

        'apache_2_diagnosis': 113.0,
        'apache_3j_diagnosis': 502.01,
        'apache_post_operative': 0,
        'arf_apache': 1,  # Fallo renal

        # Glasgow bajo (estado neurológico comprometido)
        'gcs_eyes_apache': 2,
        'gcs_motor_apache': 3,
        'gcs_unable_apache': 0,
        'gcs_verbal_apache': 2,

        'heart_rate_apache': 125,  # Taquicardia
        'intubated_apache': 1,  # Intubado
        'map_apache': 55,  # Presión baja
        'resprate_apache': 32,  # Taquipnea
        'temp_apache': 38.5,  # Fiebre
        'ventilated_apache': 1,  # Ventilación mecánica

        'd1_diasbp_max': 95,
        'd1_diasbp_min': 40,
        'd1_sysbp_max': 160,
        'd1_sysbp_min': 75,
        'd1_mbp_max': 110,
        'd1_mbp_min': 50,

        'd1_heartrate_max': 135,
        'd1_heartrate_min': 95,
        'd1_resprate_max': 38,
        'd1_resprate_min': 22,
        'd1_spo2_max': 92,  # Saturación baja
        'd1_spo2_min': 85,
        'd1_temp_max': 39.1,
        'd1_temp_min': 37.8,

        'd1_glucose_max': 220,  # Hiperglicemia
        'd1_glucose_min': 85,
        'd1_potassium_max': 5.5,
        'd1_potassium_min': 3.2,

        # Múltiples comorbilidades
        'aids': 0,
        'cirrhosis': 1,  # Cirrosis
        'diabetes_mellitus': 1,  # Diabetes
        'hepatic_failure': 1,  # Fallo hepático
        'immunosuppression': 1,  # Inmunosupresión
        'leukemia': 0,
        'lymphoma': 0,
        'solid_tumor_with_metastasis': 0,

        'apache_3j_bodysystem': 'Cardiovascular',
        'apache_2_bodysystem': 'Cardiovascular'
    }

    print("\n👤 PACIENTE CRÍTICO:")
    print(f"   • Edad: {critical_patient['age']} años")
    print(f"   • Glasgow: {critical_patient['gcs_eyes_apache'] + critical_patient['gcs_motor_apache'] + critical_patient['gcs_verbal_apache']}/15 (CRÍTICO)")
    print(f"   • Ventilación mecánica: Sí")
    print(f"   • Comorbilidades: Cirrosis, Diabetes, Fallo hepático, Inmunosupresión")

    result = predictor.predict_single_patient(critical_patient)

    print(f"\n📊 RESULTADO:")
    print(f"   • Predicción: {result['salida_1_binaria']['result_text']}")
    print(f"   • Probabilidad de muerte: {result['salida_2_probabilidades']['prob_muerte']:.2f}%")
    print(f"   • Nivel de riesgo: {result['salida_3_riesgo']['nivel_riesgo']}")
    print(f"   • Recomendaciones:")
    for rec in result['salida_3_riesgo']['recomendaciones']:
        print(f"      • {rec}")


if __name__ == "__main__":
    try:
        # Prueba con paciente normal
        test_prediction()

        # Prueba con paciente crítico
        test_high_risk_patient()

        print("\n✅ Todas las pruebas completadas exitosamente!")

    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
