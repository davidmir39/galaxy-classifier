# Clasificación Morfológica de Galaxias con Deep Learning

Trabajo Final de la asignatura **Aprendizaje Profundo** (Máster en IA).

Sistema de clasificación de imágenes de galaxias en 10 tipos morfológicos mediante redes
neuronales convolucionales (CNN) y *Transfer Learning*. Se compara una CNN entrenada desde
cero con dos backbones preentrenados en ImageNet (**ResNet50** y **EfficientNetB0**),
aplicando Transfer Learning en fases de descongelado creciente.

## Resultado principal

El mejor modelo (**ResNet50 con fine-tuning extendido**) alcanza en el conjunto de test:

| Métrica | Valor |
|---------|:-----:|
| F1 macro | **0,787** |
| Accuracy | **0,801** |

## Dataset

**Galaxy10 DECaLS**: 17.736 imágenes en color de 256×256 px (bandas fotométricas g, r, z),
etiquetadas en 10 clases morfológicas a partir del consenso de voluntarios de *Galaxy Zoo*.
Tras eliminar 121 imágenes duplicadas con etiquetas contradictorias (detectadas en el EDA),
se trabaja con **17.615 imágenes**.

Descarga (~2,5 GB): el fichero `Galaxy10_DECals.h5` debe colocarse en `data/raw/`.
Fuente oficial: https://github.com/henrysky/Galaxy10

| Clase | Nombre | Nº imágenes |
|:-----:|--------|:-----------:|
| 0 | Disturbed (perturbadas) | 1.081 |
| 1 | Merging (en fusión) | 1.853 |
| 2 | Round Smooth | 2.645 |
| 3 | In-between Round Smooth | 2.027 |
| 4 | Cigar Shaped Smooth | 334 |
| 5 | Barred Spiral | 2.043 |
| 6 | Unbarred Tight Spiral | 1.829 |
| 7 | Unbarred Loose Spiral | 2.628 |
| 8 | Edge-on without Bulge | 1.423 |
| 9 | Edge-on with Bulge | 1.873 |

> El dataset está **desbalanceado** (la clase 4 tiene ~8x menos imágenes que la 2). Se aborda
> con particionado estratificado, ponderación de clases y la métrica F1 macro.

## Estructura del repositorio
galaxy-classifier/
├── data/raw/        # Galaxy10_DECals.h5 (NO versionado)
├── models/          # pesos entrenados .pt (NO versionado)
├── notebooks/       # 01_eda.ipynb y 02_pipeline_y_modelado.ipynb
├── src/             # código reutilizable (importado desde los notebooks)
├── reports/figures/ # figuras generadas
└── requirements.txt

## Instalación (Windows + GPU NVIDIA)

```bash
python -m venv .venv
.venv\Scripts\activate

# PyTorch con soporte CUDA (coger el comando exacto en pytorch.org/get-started/locally)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Resto de dependencias
pip install -r requirements.txt
```

## Ejecución

1. Descargar el dataset en `data/raw/`.
2. Ejecutar `notebooks/01_eda.ipynb` (análisis exploratorio).
3. Ejecutar `notebooks/02_pipeline_y_modelado.ipynb` (entrenamiento y evaluación).

Los cuadernos se entregan con sus salidas ya ejecutadas, por lo que pueden leerse sin
reentrenar (reentrenar requiere GPU).

## Documentación

Para una explicación detallada de los conceptos (astronomía y Deep Learning) y del código,
ver `DOCUMENTACION_PROYECTO.pdf`.

## Equipo

- **David Miguel Miranda Rodríguez**
- **Alejandro Córcoles Roldán**