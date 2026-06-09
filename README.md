# Clasificación morfológica de galaxias

Trabajo de la asignatura **Aprendizaje Profundo** (Máster en IA).

Clasificamos imágenes del dataset Galaxy10 DECaLS en 10 tipos morfológicos. Probamos:

1. Una CNN entrenada desde cero (referencia).
2. ResNet50 con transfer learning (congelar → fine-tuning).
3. EfficientNetB0 con el mismo esquema.

**Mejor modelo:** ResNet50 con fine-tuning de `layer3` y `layer4` → F1 macro **0,784** y accuracy **0,801** en test.

## Dataset

- Fuente: [Galaxy10](https://github.com/henrysky/Galaxy10), alojado en [Zenodo](https://zenodo.org/records/10845026) (de donde se descarga automáticamente)
- Fichero: `Galaxy10_DECals.h5` (~2,5 GB) dentro de `data/raw/`
- 17.736 imágenes originales; tras quitar duplicados con etiquetas contradictorias nos quedan **17.615**
- Las clases están desbalanceadas (la 4, *Cigar Shaped*, es la minoritaria)

## Estructura

```
notebooks/     # 01_eda.ipynb y 02_pipeline_y_modelado.ipynb
src/           # funciones que importamos desde el notebook de entrenamiento
reports/figures/
data/raw/      # dataset (no versionado)
models/        # pesos .pt (no versionado)
```

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Ejecución

1. Abrir `notebooks/01_eda.ipynb` para ver el análisis exploratorio.
2. Abrir `notebooks/02_pipeline_y_modelado.ipynb` para entrenar y evaluar (GPU recomendada).

La primera vez, el notebook **descarga el dataset solo** (`asegurar_dataset`): baja el `.h5` (~2,5 GB) desde Zenodo a `data/raw/` y comprueba su SHA256. Si el fichero ya existe, no vuelve a descargarlo.

Los notebooks incluyen las salidas ya ejecutadas.

**Autores:** David Miguel Miranda Rodríguez, Alejandro Córcoles Roldán
