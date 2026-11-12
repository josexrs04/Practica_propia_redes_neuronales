# Practica propia redes neuronales
Apuntes y ejercicios para practicar y comprender cómo funcionan las redes neuronales desde cero y de manera autodidacta.

## Contenido

| Carpeta | Descripción |
|---|---|
| `Apuntes redes neuronales/` | Apuntes teóricos sobre PyTorch y librerías de redes neuronales |
| `red_neuronal_MNIST/` | Clasificación de dígitos manuscritos |
| `red_neuronal_titanic/` | Predicción de supervivencia con datos tabulares |
| `red_neuronal_cats_vd_dogs/` | CNN desde cero, con data augmentation y transfer learning (ResNet18) |
| `red_neuronal_IMDB/` | Análisis de sentimiento con LSTM, Transformer y DistilBERT |
| `red_neuronal_series_temporales/` | Predicción de series temporales con el dataset Jena Climate |

## Requisitos

- Python 3.14 (el usado en desarrollo)
- Proyectos: `pip install -r requirements.txt`
- Apuntes (opcional, requiere jax, keras, etc.): `pip install -r requirements-apuntes.txt`
- Para usar GPU, instala PyTorch según https://pytorch.org/get-started/locally/
  (las versiones de `requirements.txt` corresponden a la build con CUDA 13.2, `+cu132`).


## Archivos excluidos del repositorio (`.gitignore`)

Algunos archivos no están en GitHub porque son muy pesados, se pueden volver a descargar o se regeneran al ejecutar los notebooks. GitHub rechaza archivos de más de 100 MB.

### Datasets y datos

| Ruta | Tamaño aprox. | Motivo | Cómo obtenerlo |
|---|---|---|---|
| `red_neuronal_cats_vd_dogs/dogs-vs-cats/` | varios cientos de MB | Dataset de imágenes, demasiado pesado | Competición *Dogs vs. Cats* de Kaggle |
| `red_neuronal_IMDB/imdb/` | ~440 MB | Contiene GloVe (331 MB), el CSV de reseñas (63 MB) y datos preparados (`.pkl`, ~48 MB) | GloVe 6B (100d) de Stanford y *IMDB Dataset of 50K Movie Reviews* de Kaggle. Los `.pkl` se regeneran con `red_imdb.ipynb` |
| `red_neuronal_MNIST/data/` | ~65 MB | Se descarga automáticamente | Se crea sola al ejecutar el notebook (torchvision) |
| `red_neuronal_series_temporales/jena_climate_2009_2016.csv` y `.csv.zip` | 41 MB y 13 MB | Dataset público | Dataset *Jena Climate* (se descarga desde los ejemplos de Keras/TensorFlow) |

### Modelos entrenados

| Ruta | Tamaño | Motivo | Cómo obtenerlo |
|---|---|---|---|
| `red_neuronal_IMDB/distilbert_6_6.pt` | 253 MB | Supera el límite de 100 MB de GitHub | Se regenera ejecutando `red_imdb.ipynb` |

El resto de modelos (`.pt` de gatos/perros e IMDB) sí están incluidos porque son ligeros.

### Archivos temporales y del sistema

| Patrón | Motivo |
|---|---|
| `__pycache__/`, `*.pyc` | Caché que genera Python automáticamente |
| `.ipynb_checkpoints/` | Copias automáticas de Jupyter |
| `venv/`, `.venv/`, `env/` | Entornos virtuales: son enormes y dependen de cada equipo |
| `.DS_Store`, `Thumbs.db` | Archivos que generan macOS y Windows |
| `.vscode/` | Configuración personal del editor |
| `.env` | Puede contener contraseñas o claves; nunca debe subirse |

### Cómo reproducir el proyecto

1. Clona el repositorio.
2. Descarga los datasets de la primera tabla y colócalos en las rutas indicadas.
3. Ejecuta los notebooks de cada carpeta; los modelos excluidos se regenerarán.
