# 🚕 NYC Taxi Tips Classifier

Un sistema completo de machine learning para predecir propinas en viajes de taxi de Nueva York, con capacidades de evaluación temporal y análisis de robustez del modelo.

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación y Configuración](#instalación-y-configuración)
- [Uso Rápido](#uso-rápido)
- [Scripts Disponibles](#scripts-disponibles)
- [Evaluación Temporal](#evaluación-temporal)
- [Resultados y Métricas](#resultados-y-métricas)
- [Contribución](#contribución)

## 🎯 Descripción del Proyecto

Este proyecto implementa un **clasificador Random Forest** para predecir si un viaje en taxi recibirá una propina alta (>20% del costo del viaje). El sistema incluye:

- **Módulos modulares y reutilizables** para procesamiento de datos, entrenamiento y predicción
- **Evaluación temporal automatizada** para analizar la robustez del modelo a lo largo del tiempo
- **Pipeline completo** desde datos crudos hasta modelo en producción
- **Análisis del impacto de COVID-19** en el comportamiento de propinas

### 🔍 Características Técnicas

- **Target**: Clasificación binaria de propinas altas (threshold: 20%)
- **Algoritmo**: Random Forest Classifier optimizado
- **Features**: Características temporales, geográficas y de viaje
- **Datos**: NYC Taxi & Limousine Commission (2020)
- **Métricas**: F1-score, Accuracy, Precision, Recall

## 📁 Estructura del Proyecto

```
Producto-datos-lab1/
├── 📊 src/taxi_tips_classifier/          # Paquete principal
│   ├── data.py                          # Carga y preprocesamiento
│   ├── features.py                      # Ingeniería de características
│   ├── train.py                         # Entrenamiento de modelos
│   ├── predict.py                       # Predicción y evaluación
│   ├── model.py                         # Clase principal del modelo
│   ├── plots.py                         # Visualizaciones
│   ├── temporal_evaluation.py           # Evaluación temporal
│   └── __init__.py                      # Configuración del paquete
├── 🚀 scripts/                          # Scripts ejecutables
│   ├── train_simple.py                  # Entrenamiento básico
│   ├── evaluate_model.py                # Evaluación de modelos
│   └── temporal_evaluation_script.py    # Análisis temporal automatizado
├── 📓 notebook/                         # Análisis interactivos
│   └── temporal_evaluation.ipynb        # Notebook de evaluación temporal
├── 📈 temporal_analysis_results/        # Resultados de análisis temporal
│   ├── resultados_temporales.csv        # Tabla de métricas por mes
│   ├── temporal_performance.png         # Gráficos de rendimiento
│   └── reporte_completo.joblib          # Reporte completo serializado
├── 📖 examples/                         # Ejemplos de uso
├── 🤖 random_forest.joblib              # Modelo entrenado
├── ⚙️ pyproject.toml                    # Configuración del proyecto
└── 📄 README.md                         # Este archivo
```

## 🛠️ Instalación y Configuración

### Prerrequisitos

- **Python 3.8+**
- **uv** (recomendado) o pip para gestión de dependencias

### Instalación Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/Godoca2/Producto-datos-lab1.git
cd Producto-datos-lab1

# 2. Instalar dependencias con uv (recomendado)
uv sync

# O con pip
pip install -e .

# 3. Verificar instalación
uv run python -c "from taxi_tips_classifier import TaxiTipClassifier; print('✅ Instalación exitosa')"
```

### Dependencias Principales

```toml
pandas = ">=1.3.0"
scikit-learn = ">=1.0.0"
joblib = ">=1.0.0"
pyarrow = ">=5.0.0"
matplotlib = ">=3.5.0"
seaborn = ">=0.11.0"
numpy = ">=1.20.0"
```

## 🚀 Uso Rápido

### 1. Entrenamiento Básico

```bash
# Entrenar modelo con datos de enero 2020
uv run python scripts/train_simple.py

# Salida esperada:
# ✅ Modelo entrenado y guardado en: random_forest.joblib
# 📊 F1-Score: 0.4004 | Accuracy: 0.5042
```

### 2. Evaluación del Modelo

```bash
# Evaluar modelo en datos de febrero 2020
uv run python scripts/evaluate_model.py

# Con datos personalizados
uv run python scripts/evaluate_model.py --data-url "your_data_url.parquet"
```

### 3. Análisis Temporal (Evaluación Mensual)

```bash
# Evaluación rápida (3 meses)
uv run python scripts/temporal_evaluation_script.py --quick

# Evaluación completa (6 meses)
uv run python scripts/temporal_evaluation_script.py

# Evaluación personalizada
uv run python scripts/temporal_evaluation_script.py \
  --months 2020-01 2020-02 2020-03 2020-04 \
  --sample-size 15000
```

### 4. Uso Programático

```python
from taxi_tips_classifier import TaxiTipClassifier

# Crear y entrenar modelo
classifier = TaxiTipClassifier()
classifier.train("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet")

# Evaluar en nuevos datos
results = classifier.evaluate("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet")
print(f"F1-Score: {results['f1_score']:.4f}")

# Predicción en lote
predictions = classifier.predict("path/to/new_data.parquet")
```

## 📜 Scripts Disponibles

### 🎯 `train_simple.py` - Entrenamiento Básico

```bash
uv run python scripts/train_simple.py [opciones]
```

**Funcionalidad**: Entrena un modelo Random Forest con datos de enero 2020.

### 📊 `evaluate_model.py` - Evaluación de Modelos

```bash
uv run python scripts/evaluate_model.py [--model-path] [--data-url] [--sample-size]
```

**Funcionalidad**: Evalúa un modelo entrenado en nuevos datos.

### ⏰ `temporal_evaluation_script.py` - Análisis Temporal

```bash
uv run python scripts/temporal_evaluation_script.py [opciones]
```

**Opciones principales**:
- `--quick`: Evaluación rápida (3 meses, 20k muestras)
- `--months`: Meses específicos a evaluar
- `--sample-size`: Número de muestras por mes
- `--no-plots`: Solo generar tabla sin gráficos

**Salidas generadas**:
- `temporal_analysis_results/resultados_temporales.csv`
- `temporal_analysis_results/temporal_performance.png`
- `temporal_analysis_results/reporte_completo.joblib`

## ⏰ Evaluación Temporal

### 🎯 Objetivo

Demostrar y analizar el comportamiento del modelo a lo largo del tiempo, identificando:
- Degradación del rendimiento temporal
- Factores que afectan la robustez
- Recomendaciones para mejora continua

### 📊 Resultados Clave (Ejemplo)

| Mes     | F1-Score | Accuracy | Precision | Recall | Propinas Altas | Cambio vs Baseline |
|---------|----------|----------|-----------|--------|----------------|--------------------|
| 2020-01 | 0.4004   | 0.5042   | 0.6313    | 0.2932 | 56.5%          | Baseline           |
| 2020-02 | 0.3642   | 0.4849   | 0.6230    | 0.2573 | 57.3%          | -9.0%              |
| 2020-03 | 0.3926   | 0.5022   | 0.6186    | 0.2875 | 55.9%          | -1.9%              |
| 2020-04 | 0.5197   | 0.6043   | 0.5225    | 0.5169 | 41.4%          | +29.8%             |
| 2020-05 | 0.5129   | 0.6355   | 0.5313    | 0.4958 | 38.7%          | +28.1%             |

### 🔍 Insights Principales

1. **Correlación fuerte**: -0.98 entre F1-score y tasa de propinas altas
2. **Paradoja COVID-19**: El modelo mejoró durante la pandemia
3. **Factor principal**: Cambios en distribución de propinas afectan el rendimiento
4. **Recomendación**: Reentrenamiento automático mensual necesario

### 📈 Visualizaciones Generadas

- **Evolución temporal de métricas**: Líneas de tendencia por mes
- **Distribución de clases**: Cambios en porcentaje de propinas altas
- **Análisis de degradación**: Cambios relativos respecto al baseline
- **Intervalos de confianza**: F1-score con márgenes de error

## 📊 Resultados y Métricas

### 🎯 Rendimiento del Modelo Base

- **F1-Score**: 0.40 (baseline enero 2020)
- **Accuracy**: 0.50
- **Precision**: 0.63
- **Recall**: 0.29

### 📈 Análisis Temporal

- **Degradación máxima**: 9% (febrero 2020)
- **Mejora máxima**: 29.8% (abril 2020)
- **Variabilidad**: 15.7% coeficiente de variación
- **Tendencia**: Mejora durante COVID-19, degradación post-pandemia

### 🎖️ Factores de Éxito Identificados

1. **Balanceamiento de clases**: Mejores resultados con distribución 40/60
2. **Estabilidad temporal**: Necesario reentrenamiento cada 2-3 meses
3. **Características clave**: Hora del día, día de semana, duración del viaje

## 🤝 Contribución

### 🛠️ Desarrollo

```bash
# 1. Fork y clonar
git clone https://github.com/YOUR-USERNAME/Producto-datos-lab1.git
cd Producto-datos-lab1

# 2. Crear entorno de desarrollo
uv sync --dev

# 3. Ejecutar tests
uv run pytest

# 4. Verificar código
uv run ruff check src/
uv run black src/
```

### 📝 Guidelines

- **Código modular**: Una responsabilidad por módulo
- **Documentación**: Docstrings en todas las funciones públicas
- **Tests**: Cobertura mínima del 80%
- **Type hints**: Usar anotaciones de tipo en funciones públicas

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👨‍💻 Autor

**César Godoy Delaigue**
- GitHub: [@Godoca2](https://github.com/Godoca2)
- Email: cesar.delaigue@gmail.com

---

## 🔗 Enlaces Útiles

- **📊 Notebook Original**: `00_nyc_taxi_model.ipynb`
- **📖 Documentación Modular**: `MODULAR_README.md`
- **📈 Análisis Temporal**: `EVALUACION_TEMPORAL_ESTRATEGIA.md`
- **🎯 NYC Taxi Data**: [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

**¿Tienes preguntas?** Abre un [issue](https://github.com/Godoca2/Producto-datos-lab1/issues) o revisa la documentación en el repositorio.
