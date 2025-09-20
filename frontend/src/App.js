import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  // Valores por defecto realistas para pruebas rápidas (paciente estable de referencia)
  const defaultValues = {
    // Demográficos - Paciente adulto promedio
    age: 45,
    gender: 'M',
    ethnicity: 'Caucasian',
    height: 175,
    weight: 75,
    bmi: 24.5,

    // Hospitalización - Paciente estable
    elective_surgery: '0',
    apache_post_operative: '0',
    icu_admit_source: 'Floor',
    icu_stay_type: 'admit',
    icu_type: 'Med-Surg ICU',
    pre_icu_los_days: 1,

    // Glasgow - Paciente consciente y alerta
    gcs_eyes_apache: 4,
    gcs_motor_apache: 6,
    gcs_verbal_apache: 5,
    gcs_unable_apache: '0',

    // Soporte vital - Paciente estable sin soporte
    intubated_apache: '0',
    ventilated_apache: '0',
    arf_apache: '0',

    // Signos vitales Apache - Valores normales
    heart_rate_apache: 80,
    map_apache: 85,
    resprate_apache: 18,
    temp_apache: 37.0,

    // Comorbilidades - Paciente sin comorbilidades críticas
    aids: '0',
    cirrhosis: '0',
    hepatic_failure: '0',
    leukemia: '0',
    lymphoma: '0',
    solid_tumor_with_metastasis: '0',
    diabetes_mellitus: '0',
    immunosuppression: '0',

    // Día 1 - Signos vitales estables
    d1_diasbp_max: 85,
    d1_diasbp_min: 65,
    d1_sysbp_max: 140,
    d1_sysbp_min: 110,
    d1_heartrate_max: 95,
    d1_heartrate_min: 65,
    d1_resprate_max: 22,
    d1_resprate_min: 14,
    d1_spo2_max: 99,
    d1_spo2_min: 95,
    d1_temp_max: 37.5,
    d1_temp_min: 36.5,

    // Laboratorios - Valores normales
    d1_glucose_max: 140,
    d1_glucose_min: 90,
    d1_potassium_max: 4.5,
    d1_potassium_min: 3.8,

    // Diagnósticos - Códigos comunes
    apache_2_diagnosis: 113,
    apache_3j_diagnosis: 502.01,
    apache_3j_bodysystem: 'Cardiovascular',
    apache_2_bodysystem: 'Cardiovascular'
  };

  const [formData, setFormData] = useState(defaultValues);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('predict');

  // Casos predefinidos para pruebas rápidas
  const predefinedCases = {
    paciente_estable: {
      nombre: "👤 Paciente Estable",
      descripcion: "Paciente adulto sin comorbilidades críticas",
      datos: {
        // Ya están los valores por defecto que son estables
      }
    },
    paciente_critico: {
      nombre: "🚨 Paciente Crítico",
      descripcion: "Paciente con múltiples comorbilidades y soporte vital",
      datos: {
        age: 78,
        gender: 'M',
        ethnicity: 'Caucasian',
        height: 175,
        weight: 65,
        bmi: 21.2,

        // Comorbilidades críticas
        aids: '1',
        cirrhosis: '1',
        solid_tumor_with_metastasis: '1',

        // Estado neurológico crítico
        gcs_eyes_apache: 1,
        gcs_motor_apache: 2,
        gcs_verbal_apache: 1,

        // Soporte vital máximo
        intubated_apache: '1',
        ventilated_apache: '1',
        arf_apache: '1',

        // Signos vitales inestables
        heart_rate_apache: 150,
        map_apache: 45,
        resprate_apache: 35,
        temp_apache: 39.8,

        // Día 1 - valores críticos
        d1_sysbp_max: 200,
        d1_sysbp_min: 70,
        d1_heartrate_max: 160,
        d1_heartrate_min: 45,
        d1_spo2_max: 88,
        d1_spo2_min: 70,
        d1_temp_max: 40.2,
        d1_temp_min: 35.1,

        // Laboratorios alterados
        d1_glucose_max: 350,
        d1_glucose_min: 45,
        d1_potassium_max: 6.2,
        d1_potassium_min: 2.8
      }
    },
    paciente_joven: {
      nombre: "👨‍💼 Paciente Joven",
      descripcion: "Paciente joven post-quirúrgico",
      datos: {
        age: 28,
        gender: 'M',
        ethnicity: 'Hispanic',
        height: 180,
        weight: 80,
        bmi: 24.7,

        // Post-operatorio
        elective_surgery: '1',
        apache_post_operative: '1',
        icu_admit_source: 'Operating Room / Recovery',
        icu_type: 'CTICU',

        // Sin comorbilidades
        aids: '0',
        cirrhosis: '0',
        diabetes_mellitus: '0',

        // Estado neurológico normal
        gcs_eyes_apache: 4,
        gcs_motor_apache: 6,
        gcs_verbal_apache: 5,

        // Soporte mínimo
        intubated_apache: '0',
        ventilated_apache: '0',
        arf_apache: '0',

        // Signos vitales buenos
        heart_rate_apache: 72,
        map_apache: 90,
        resprate_apache: 16,
        temp_apache: 36.8
      }
    },
    paciente_anciano: {
      nombre: "👴 Paciente Anciano",
      descripcion: "Paciente de edad avanzada con diabetes",
      datos: {
        age: 85,
        gender: 'F',
        ethnicity: 'Caucasian',
        height: 160,
        weight: 55,
        bmi: 21.5,

        // Comorbilidades relacionadas con edad
        diabetes_mellitus: '1',
        cirrhosis: '0',
        aids: '0',

        // Estado neurológico ligeramente alterado
        gcs_eyes_apache: 3,
        gcs_motor_apache: 5,
        gcs_verbal_apache: 4,

        // Sin soporte vital mayor
        intubated_apache: '0',
        ventilated_apache: '0',
        arf_apache: '0',

        // Signos vitales estables pero limítrofes
        heart_rate_apache: 95,
        map_apache: 75,
        resprate_apache: 22,
        temp_apache: 37.3,

        // Laboratorios con glucosa elevada
        d1_glucose_max: 220,
        d1_glucose_min: 140,
        d1_potassium_max: 4.8,
        d1_potassium_min: 3.5
      }
    }
  };

  // Campos principales del formulario médico (84 columnas)
  const medicalFields = [
    // Demográficos
    { name: 'age', label: 'Edad', type: 'number', required: true, min: 16, max: 120 },
    { name: 'gender', label: 'Sexo', type: 'select', options: ['M', 'F'], required: true },
    { name: 'ethnicity', label: 'Etnia', type: 'select',
      options: ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'], required: false },
    { name: 'height', label: 'Altura (cm)', type: 'number', min: 100, max: 250 },
    { name: 'weight', label: 'Peso (kg)', type: 'number', min: 30, max: 300 },
    { name: 'bmi', label: 'BMI', type: 'number', min: 10, max: 60 },

    // Tipo de cirugía y admisión
    { name: 'elective_surgery', label: 'Cirugía Electiva', type: 'select', options: ['0', '1'] },
    { name: 'apache_post_operative', label: 'Post-operatorio', type: 'select', options: ['0', '1'] },
    { name: 'icu_admit_source', label: 'Fuente de Admisión UCI', type: 'select',
      options: ['Floor', 'Accident & Emergency', 'Operating Room / Recovery', 'Other Hospital', 'Other ICU'] },
    { name: 'icu_stay_type', label: 'Tipo de Estancia UCI', type: 'select', options: ['admit', 'transfer'] },
    { name: 'icu_type', label: 'Tipo de UCI', type: 'select',
      options: ['CTICU', 'Med-Surg ICU', 'MICU', 'Neuro ICU', 'SICU', 'Cardiac ICU'] },
    { name: 'pre_icu_los_days', label: 'Días Pre-UCI', type: 'number', min: 0, max: 100 },

    // Escalas Apache y Glasgow (CRÍTICAS)
    { name: 'gcs_eyes_apache', label: 'Glasgow - Ojos (1-4)', type: 'number', min: 1, max: 4, required: true },
    { name: 'gcs_motor_apache', label: 'Glasgow - Motor (1-6)', type: 'number', min: 1, max: 6, required: true },
    { name: 'gcs_verbal_apache', label: 'Glasgow - Verbal (1-5)', type: 'number', min: 1, max: 5, required: true },
    { name: 'gcs_unable_apache', label: 'Glasgow - No Evaluable', type: 'select', options: ['0', '1'] },

    // Soporte vital (MUY IMPORTANTE)
    { name: 'intubated_apache', label: 'Intubado', type: 'select', options: ['0', '1'], required: true },
    { name: 'ventilated_apache', label: 'Ventilación Mecánica', type: 'select', options: ['0', '1'], required: true },
    { name: 'arf_apache', label: 'Falla Renal Aguda', type: 'select', options: ['0', '1'] },

    // Signos vitales Apache
    { name: 'heart_rate_apache', label: 'Frecuencia Cardíaca', type: 'number', min: 20, max: 300 },
    { name: 'map_apache', label: 'Presión Arterial Media', type: 'number', min: 20, max: 200 },
    { name: 'resprate_apache', label: 'Frecuencia Respiratoria', type: 'number', min: 5, max: 80 },
    { name: 'temp_apache', label: 'Temperatura (°C)', type: 'number', min: 32, max: 45 },

    // Comorbilidades CRÍTICAS (alta mortalidad)
    { name: 'aids', label: 'SIDA/VIH', type: 'select', options: ['0', '1'], critical: true },
    { name: 'cirrhosis', label: 'Cirrosis', type: 'select', options: ['0', '1'], critical: true },
    { name: 'hepatic_failure', label: 'Falla Hepática', type: 'select', options: ['0', '1'], critical: true },
    { name: 'leukemia', label: 'Leucemia', type: 'select', options: ['0', '1'], critical: true },
    { name: 'lymphoma', label: 'Linfoma', type: 'select', options: ['0', '1'], critical: true },
    { name: 'solid_tumor_with_metastasis', label: 'Tumor Sólido con Metástasis', type: 'select', options: ['0', '1'], critical: true },
    { name: 'diabetes_mellitus', label: 'Diabetes Mellitus', type: 'select', options: ['0', '1'] },
    { name: 'immunosuppression', label: 'Inmunosupresión', type: 'select', options: ['0', '1'] },

    // Día 1 - Presión arterial
    { name: 'd1_diasbp_max', label: 'Día 1 - Presión Diastólica Máx', type: 'number', min: 20, max: 200 },
    { name: 'd1_diasbp_min', label: 'Día 1 - Presión Diastólica Mín', type: 'number', min: 20, max: 200 },
    { name: 'd1_sysbp_max', label: 'Día 1 - Presión Sistólica Máx', type: 'number', min: 50, max: 300 },
    { name: 'd1_sysbp_min', label: 'Día 1 - Presión Sistólica Mín', type: 'number', min: 50, max: 300 },

    // Día 1 - Otros signos vitales
    { name: 'd1_heartrate_max', label: 'Día 1 - FC Máxima', type: 'number', min: 20, max: 300 },
    { name: 'd1_heartrate_min', label: 'Día 1 - FC Mínima', type: 'number', min: 20, max: 300 },
    { name: 'd1_resprate_max', label: 'Día 1 - FR Máxima', type: 'number', min: 5, max: 80 },
    { name: 'd1_resprate_min', label: 'Día 1 - FR Mínima', type: 'number', min: 5, max: 80 },
    { name: 'd1_spo2_max', label: 'Día 1 - SpO2 Máxima', type: 'number', min: 50, max: 100 },
    { name: 'd1_spo2_min', label: 'Día 1 - SpO2 Mínima', type: 'number', min: 50, max: 100 },
    { name: 'd1_temp_max', label: 'Día 1 - Temperatura Máx', type: 'number', min: 32, max: 45 },
    { name: 'd1_temp_min', label: 'Día 1 - Temperatura Mín', type: 'number', min: 32, max: 45 },

    // Laboratorios
    { name: 'd1_glucose_max', label: 'Día 1 - Glucosa Máx', type: 'number', min: 20, max: 1000 },
    { name: 'd1_glucose_min', label: 'Día 1 - Glucosa Mín', type: 'number', min: 20, max: 1000 },
    { name: 'd1_potassium_max', label: 'Día 1 - Potasio Máx', type: 'number', min: 1, max: 10 },
    { name: 'd1_potassium_min', label: 'Día 1 - Potasio Mín', type: 'number', min: 1, max: 10 },

    // Diagnósticos Apache
    { name: 'apache_2_diagnosis', label: 'Diagnóstico Apache II', type: 'number', placeholder: 'Ej: 113' },
    { name: 'apache_3j_diagnosis', label: 'Diagnóstico Apache III', type: 'number', placeholder: 'Ej: 502.01' },
    { name: 'apache_3j_bodysystem', label: 'Sistema Corporal Apache III', type: 'select',
      options: ['Cardiovascular', 'Respiratory', 'Neurological', 'Gastrointestinal', 'Metabolic', 'Hematological', 'Genitourinary', 'Trauma', 'Sepsis', 'Other'] },
    { name: 'apache_2_bodysystem', label: 'Sistema Corporal Apache II', type: 'select',
      options: ['Cardiovascular', 'Respiratory', 'Neurological', 'Gastrointestinal', 'Metabolic', 'Hematological', 'Genitourinary', 'Trauma', 'Sepsis', 'Other'] }

    // NOTA: apache_4a_hospital_death_prob y apache_4a_icu_death_prob ELIMINADOS
    // Estos son OUTPUTS del modelo Apache, no deben ser inputs para nuestro modelo
  ];

  useEffect(() => {
    loadModelInfo();
  }, []);

  const loadModelInfo = async () => {
    try {
      const response = await fetch('/api/model-info');
      const data = await response.json();
      setModelInfo(data);
    } catch (error) {
      console.error('Error cargando info del modelo:', error);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;

    if (type === 'number') {
      if (value === '') {
        setFormData(prev => ({ ...prev, [name]: null }));
        return;
      }

      // Normalizar el valor: reemplazar comas por puntos
      let normalizedValue = value.toString().replace(',', '.');

      // Verificar si es un número válido
      const numValue = parseFloat(normalizedValue);
      if (!isNaN(numValue)) {
        setFormData(prev => ({ ...prev, [name]: numValue }));
      } else {
        // Si no es un número válido, mantener el valor original como string
        // El backend se encargará de la validación
        setFormData(prev => ({ ...prev, [name]: value }));
      }
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const loadPredefinedCase = (caseKey) => {
    const selectedCase = predefinedCases[caseKey];
    setFormData(prev => ({
      ...defaultValues,  // Empezar con valores por defecto
      ...selectedCase.datos  // Sobrescribir con datos específicos del caso
    }));
    setPrediction(null);  // Limpiar predicción anterior
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (data.status === 'success') {
        setPrediction(data);
      } else {
        setError(data.message || 'Error en la predicción');
      }
    } catch (error) {
      setError('Error de conexión con el servidor');
    } finally {
      setIsLoading(false);
    }
  };

  const trainModel = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/train', {
        method: 'POST',
      });

      const data = await response.json();

      if (data.status === 'success') {
        alert('¡Modelo entrenado exitosamente!');
        loadModelInfo();
      } else {
        setError(data.message || 'Error entrenando modelo');
      }
    } catch (error) {
      setError('Error de conexión durante entrenamiento');
    } finally {
      setIsLoading(false);
    }
  };

  const renderField = (field) => {
    const value = formData[field.name] || '';

    if (field.type === 'select') {
      return (
        <select
          name={field.name}
          value={value}
          onChange={handleInputChange}
          className={`form-control ${field.critical ? 'critical-field' : ''}`}
          required={field.required}
        >
          <option value="">Seleccionar...</option>
          {field.options.map(option => (
            <option key={option} value={option}>{option}</option>
          ))}
        </select>
      );
    }

    return (
      <input
        type={field.type}
        name={field.name}
        value={value}
        onChange={handleInputChange}
        className={`form-control ${field.critical ? 'critical-field' : ''}`}
        min={field.min}
        max={field.max}
        step={field.type === 'number' ? 'any' : undefined}
        required={field.required}
        placeholder={field.placeholder || `Ej: ${field.min || ''}`}
        title="Puede usar punto (.) o coma (,) como separador decimal"
      />
    );
  };

  const renderPredictionResult = () => {
    if (!prediction) return null;

    return (
      <div className="prediction-results">
        <h3>🏥 RESULTADOS DE LA PREDICCIÓN</h3>

        {/* SALIDA 1: Clasificación Binaria */}
        <div className={`result-card ${prediction.resultado_binario.prediction === 1 ? 'death-prediction' : 'survival-prediction'}`}>
          <h4>📊 SALIDA 1: Clasificación Binaria</h4>
          <div className="binary-result">
            <strong>{prediction.resultado_binario.result_text}</strong>
          </div>
          <small>Código: {prediction.resultado_binario.prediction} (0=Sobrevive, 1=Muere)</small>
        </div>

        {/* SALIDA 2: Probabilidades Detalladas */}
        <div className="result-card">
          <h4>📈 SALIDA 2: Probabilidades Detalladas</h4>
          <div className="probability-bars">
            <div className="prob-item">
              <span>💀 Probabilidad de Muerte:</span>
              <div className="progress">
                <div
                  className="progress-bar bg-danger"
                  style={{width: `${prediction.probabilidades.prob_muerte}%`}}
                >
                  {prediction.probabilidades.prob_muerte}%
                </div>
              </div>
            </div>
            <div className="prob-item">
              <span>💚 Probabilidad de Supervivencia:</span>
              <div className="progress">
                <div
                  className="progress-bar bg-success"
                  style={{width: `${prediction.probabilidades.prob_supervivencia}%`}}
                >
                  {prediction.probabilidades.prob_supervivencia}%
                </div>
              </div>
            </div>
            <div className="confidence">
              <strong>Confianza del Modelo: {prediction.probabilidades.confianza}%</strong>
            </div>
          </div>
        </div>

        {/* SALIDA 3: Niveles de Riesgo */}
        <div className={`result-card risk-level-${prediction.evaluacion_riesgo.nivel_riesgo.toLowerCase().replace(' ', '-')}`}>
          <h4>⚠️ SALIDA 3: Evaluación de Riesgo</h4>
          <div className="risk-level">
            <h5>{prediction.evaluacion_riesgo.nivel_riesgo}</h5>
            <p>Probabilidad de muerte: <strong>{prediction.evaluacion_riesgo.probabilidad_muerte}%</strong></p>
          </div>

          <div className="recommendations">
            <h6>📋 Recomendaciones Médicas:</h6>
            <ul>
              {prediction.evaluacion_riesgo.recomendaciones.map((rec, index) => (
                <li key={index}>{rec}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Información del Modelo */}
        <div className="result-card model-info-card">
          <h4>🤖 Información del Modelo</h4>
          <p><strong>Algoritmo usado:</strong> {prediction.modelo_info.algoritmo_usado}</p>
          <div className="important-variables">
            <h6>Variables más importantes:</h6>
            <ol>
              {prediction.modelo_info.variables_mas_importantes.slice(0, 5).map(([variable, importance], index) => (
                <li key={index}>
                  <strong>{variable}</strong>: {(importance * 100).toFixed(2)}%
                </li>
              ))}
            </ol>
          </div>
          <small>Predicción realizada: {new Date(prediction.timestamp).toLocaleString()}</small>
        </div>
      </div>
    );
  };

  // Agrupar campos por categoría
  const fieldGroups = {
    'Datos Demográficos': medicalFields.filter(f =>
      ['age', 'gender', 'ethnicity', 'height', 'weight', 'bmi'].includes(f.name)
    ),
    'Comorbilidades CRÍTICAS': medicalFields.filter(f => f.critical),
    'Estado Neurológico (Glasgow)': medicalFields.filter(f =>
      f.name.includes('gcs_')
    ),
    'Soporte Vital': medicalFields.filter(f =>
      ['intubated_apache', 'ventilated_apache', 'arf_apache'].includes(f.name)
    ),
    'Signos Vitales Apache': medicalFields.filter(f =>
      ['heart_rate_apache', 'map_apache', 'resprate_apache', 'temp_apache'].includes(f.name)
    ),
    'Hospitalización': medicalFields.filter(f =>
      ['elective_surgery', 'apache_post_operative', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'pre_icu_los_days'].includes(f.name)
    ),
    'Signos Vitales Día 1': medicalFields.filter(f =>
      f.name.startsWith('d1_') && !f.name.includes('glucose') && !f.name.includes('potassium')
    ),
    'Laboratorios': medicalFields.filter(f =>
      f.name.includes('glucose') || f.name.includes('potassium')
    ),
    'Diagnósticos Apache': medicalFields.filter(f =>
      ['apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_3j_bodysystem', 'apache_2_bodysystem'].includes(f.name)
    )
  };

  return (
    <div className="App">
      <nav className="navbar">
        <div className="nav-brand">
          <h1>🏥 Medical ML Predictor</h1>
          <p>Sistema de Predicción de Supervivencia en UCI</p>
        </div>
        <div className="nav-tabs">
          <button
            className={activeTab === 'predict' ? 'active' : ''}
            onClick={() => setActiveTab('predict')}
          >
            Predicción
          </button>
          <button
            className={activeTab === 'model' ? 'active' : ''}
            onClick={() => setActiveTab('model')}
          >
            Modelo
          </button>
        </div>
      </nav>

      <div className="container">
        {activeTab === 'predict' && (
          <div className="prediction-tab">
            <div className="row">
              <div className="col-md-6">
                <div className="form-container">
                  <h2>📋 Datos del Paciente</h2>

                  {/* Botones de casos predefinidos */}
                  <div className="predefined-cases">
                    <h4>🚀 Casos de Prueba Rápida</h4>
                    <div className="case-buttons">
                      {Object.entries(predefinedCases).map(([key, case_]) => (
                        <button
                          key={key}
                          type="button"
                          className="btn btn-outline-secondary btn-sm case-btn"
                          onClick={() => loadPredefinedCase(key)}
                          title={case_.descripcion}
                        >
                          {case_.nombre}
                        </button>
                      ))}
                    </div>
                    <small className="text-muted">
                      💡 Usa estos botones para cargar casos predefinidos y probar rápidamente el modelo
                    </small>
                  </div>

                  {error && <div className="alert alert-danger">{error}</div>}

                  <form onSubmit={handleSubmit}>
                    {Object.entries(fieldGroups).map(([groupName, fields]) => (
                      <div key={groupName} className="field-group">
                        <h4 className="group-title">{groupName}</h4>
                        <div className="row">
                          {fields.map(field => (
                            <div key={field.name} className="col-md-6 mb-3">
                              <label className="form-label">
                                {field.label}
                                {field.required && <span className="required">*</span>}
                                {field.critical && <span className="critical">⚠️</span>}
                              </label>
                              {renderField(field)}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}

                    <div className="form-actions">
                      <button
                        type="submit"
                        className="btn btn-primary btn-lg"
                        disabled={isLoading}
                      >
                        {isLoading ? '🔄 Analizando...' : '🔍 Realizar Predicción'}
                      </button>
                    </div>
                  </form>
                </div>
              </div>

              <div className="col-md-6">
                {renderPredictionResult()}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'model' && (
          <div className="model-tab">
            <h2>🤖 Información del Modelo</h2>

            <div className="model-actions">
              <button
                onClick={trainModel}
                className="btn btn-success btn-lg"
                disabled={isLoading}
              >
                {isLoading ? '🔄 Entrenando...' : '🚀 Entrenar Modelo'}
              </button>
            </div>

            {modelInfo && (
              <div className="model-info">
                <h3>Estado del Modelo</h3>
                <div className="info-card">
                  <p><strong>Modelo Cargado:</strong> {modelInfo.model_loaded ? '✅ Sí' : '❌ No'}</p>
                  {modelInfo.model_loaded && (
                    <>
                      <p><strong>Algoritmo:</strong> {modelInfo.algorithm}</p>
                      <p><strong>Características:</strong> {modelInfo.features_count}</p>
                      {modelInfo.top_features && (
                        <div className="top-features">
                          <h4>Variables más importantes:</h4>
                          <ol>
                            {modelInfo.top_features.map((feature, index) => (
                              <li key={index}>
                                <strong>{feature.feature}</strong>: {(feature.importance * 100).toFixed(2)}%
                              </li>
                            ))}
                          </ol>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}

            <div className="instructions">
              <h3>📖 Instrucciones de Uso</h3>
              <ol>
                <li><strong>Entrenar Modelo:</strong> Hacer clic en "Entrenar Modelo" (requiere dataset en data/dataset.csv)</li>
                <li><strong>Completar Formulario:</strong> Ingresar los datos del paciente en la pestaña "Predicción"</li>
                <li><strong>Obtener Resultados:</strong> El sistema genera 3 tipos de predicciones:
                  <ul>
                    <li><strong>Binaria:</strong> Sobrevive/Muere</li>
                    <li><strong>Probabilidades:</strong> % específicos</li>
                    <li><strong>Nivel de Riesgo:</strong> BAJO/MODERADO/ALTO/CRÍTICO + recomendaciones</li>
                  </ul>
                </li>
              </ol>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;