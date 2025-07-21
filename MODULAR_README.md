# NYC Taxi Tips Classifier - Modular Version

Este proyecto contiene un clasificador de propinas para taxis de NYC transformado de un notebook monolítico a una arquitectura modular y reutilizable.

## Estructura del Proyecto

```
src/
├── taxi_tips_classifier/
│   ├── __init__.py          # Punto de entrada del paquete
│   ├── data.py              # Carga y limpieza de datos
│   ├── features.py          # Ingeniería de características (build_features)
│   ├── train.py             # Entrenamiento del modelo
│   ├── predict.py           # Predicción y evaluación
│   ├── model.py             # Clase principal y pipeline completo
│   └── plots.py             # Visualizaciones
examples/
├── complete_pipeline_example.py  # Ejemplo completo del pipeline
scripts/
├── train_simple.py         # Script simple de entrenamiento
└── evaluate_model.py       # Script de evaluación
models/                     # Modelos entrenados guardados
```

## Módulos Principales

### 1. `data.py` - Gestión de Datos
- **`load_taxi_data()`**: Carga datos desde archivos Parquet o URLs
- **`clean_data()`**: Limpieza básica (eliminar fare_amount <= 0)
- **`create_target_variable()`**: Crea variable objetivo (propina > 20%)
- **`preprocess_data()`**: Pipeline completo de preprocesamiento

### 2. `features.py` - Ingeniería de Características  
- **`create_time_features()`**: Características temporales (día, hora, etc.)
- **`create_trip_features()`**: Características del viaje (duración, velocidad)
- **`create_all_features()`**: Crea todas las características
- **Constantes**: `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, `ALL_FEATURES`

### 3. `train.py` - Entrenamiento
- **`create_model()`**: Crea modelo RandomForest
- **`train_model()`**: Entrena el modelo
- **`evaluate_model()`**: Evalúa en datos de prueba
- **`save_model()` / `load_model()`**: Persistencia del modelo
- **`full_training_pipeline()`**: Pipeline completo de entrenamiento

### 4. `predict.py` - Predicción y Evaluación
- **`predict_single_trip()`**: Predicción para un viaje individual
- **`predict_batch()`**: Predicción para lotes de datos
- **`evaluate_predictions()`**: Evaluación con métricas (F1, accuracy, etc.)
- **`compare_datasets()`**: Comparar rendimiento entre datasets

### 5. `model.py` - Clase Principal
- **`TaxiTipClassifier`**: Clase principal que encapsula todo el pipeline
- **`train_taxi_tip_model()`**: Función conveniente para entrenar
- **`evaluate_model_on_data()`**: Función conveniente para evaluar

### 6. `plots.py` - Visualizaciones
- **`plot_confusion_matrix()`**: Matrix de confusión
- **`plot_feature_importance()`**: Importancia de características
- **`plot_prediction_distribution()`**: Distribución de probabilidades
- **`plot_model_comparison()`**: Comparación entre datasets

## Uso Rápido

### Instalación
```bash
# Instalar el paquete en modo desarrollo
uv pip install -e .

# O con pip
pip install -e .
```

### Ejemplo Básico
```python
from taxi_tips_classifier.model import TaxiTipClassifier

# Crear y entrenar el clasificador
classifier = TaxiTipClassifier()
classifier.train("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet", 
                 sample_size=50000)

# Evaluar en datos de febrero
results = classifier.evaluate("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet")
print(f"F1 Score: {results['f1_score']:.4f}")

# Guardar modelo
classifier.save_model("models/my_model.joblib")
```

### Uso de Funciones Convenientes
```python
from taxi_tips_classifier.model import train_taxi_tip_model, evaluate_model_on_data

# Entrenar con una línea
classifier = train_taxi_tip_model(
    "path/to/train_data.parquet",
    sample_size=100000,
    save_path="models/model.joblib"
)

# Evaluar con una línea
results = evaluate_model_on_data("models/model.joblib", "path/to/test_data.parquet")
```

### Uso Modular
```python
from taxi_tips_classifier.data import load_taxi_data, preprocess_data
from taxi_tips_classifier.features import create_all_features
from taxi_tips_classifier.train import create_model, train_model
from taxi_tips_classifier.predict import predict_and_evaluate

# Cargar y procesar datos
raw_data = load_taxi_data("data.parquet")
data_with_features = create_all_features(raw_data)
processed_data = preprocess_data(data_with_features)

# Entrenar modelo
model = create_model(n_estimators=200, max_depth=15)
trained_model, info = train_model(model, processed_data[features], processed_data["high_tip"])

# Evaluar
results = predict_and_evaluate(trained_model, test_data, features)
```

## Scripts de Ejemplo

### Entrenamiento Simple
```bash
python scripts/train_simple.py
```

### Evaluación de Modelo
```bash
python scripts/evaluate_model.py
```

### Pipeline Completo
```bash
python examples/complete_pipeline_example.py
```

## Características del Modelo

### Variables Predictoras (Features)
**Numéricas:**
- `pickup_weekday`: Día de la semana (0-6)
- `pickup_hour`: Hora del día (0-23) 
- `pickup_minute`: Minuto de la hora (0-59)
- `work_hours`: Indicador de horario laboral (0/1)
- `passenger_count`: Número de pasajeros
- `trip_distance`: Distancia del viaje en millas
- `trip_time`: Duración del viaje en segundos
- `trip_speed`: Velocidad del viaje (millas/segundo)

**Categóricas:**
- `PULocationID`: ID de zona de recogida
- `DOLocationID`: ID de zona de destino  
- `RatecodeID`: Tipo de tarifa

### Variable Objetivo
- `high_tip`: 1 si propina > 20% del fare_amount, 0 en caso contrario

## Flujo de Datos Original (del Notebook)

1. **Carga de datos**: Enero 2020 para entrenamiento
2. **Preprocesamiento**: 
   - Filtrar fare_amount > 0
   - Crear variable objetivo (tip_fraction > 0.2)
   - Crear características temporales y de viaje
3. **Entrenamiento**: RandomForest (100 trees, max_depth=10)
4. **Evaluación**: F1-score en febrero 2020
5. **Serialización**: Guardar modelo con joblib
6. **Prueba temporal**: Evaluar en mayo 2020 (degradación esperada)

## Mejoras Implementadas

1. **Modularidad**: Separación clara de responsabilidades
2. **Reutilización**: Funciones independientes y parametrizables
3. **Flexibilidad**: Parámetros configurables para modelos y datos
4. **Robustez**: Manejo de errores y validación de datos
5. **Documentación**: Docstrings detallados para todas las funciones
6. **Extensibilidad**: Fácil agregar nuevas características o modelos
7. **Testing**: Estructura preparada para pruebas unitarias

## Próximos Pasos

- [ ] Agregar pruebas unitarias
- [ ] Implementar validación cruzada
- [ ] Agregar más tipos de modelos
- [ ] Mejorar visualizaciones
- [ ] Optimización de hiperparámetros
- [ ] Pipeline de CI/CD
- [ ] Monitoreo de deriva del modelo
