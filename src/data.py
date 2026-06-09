"""
Módulo de DATOS.

Contiene toda la lógica del pipeline de datos del proyecto:
  - carga del dataset Galaxy10 DECaLS desde el fichero HDF5 y limpieza de duplicados,
  - partición estratificada train/val/test,
  - tuberías de transformación (con y sin augmentation),
  - Dataset de PyTorch y construcción de los DataLoaders.
"""

import hashlib

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Nombres legibles de las 10 clases, indexados por su etiqueta numérica (0-9).
NOMBRES_CLASES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]

# Constantes de normalización de ImageNet (fijas; las usan los backbones preentrenados).
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


def cargar_y_limpiar(ruta_h5):
    """Carga imágenes y etiquetas del .h5 y elimina las imágenes duplicadas.

    Los duplicados exactos del dataset tienen etiquetas contradictorias (ruido de
    etiquetado), así que se eliminan TODAS sus copias para no introducir ruido ni
    provocar fuga de datos entre particiones.

    Returns:
        (imagenes, clases): array uint8 (N, 256, 256, 3) y array int (N,).
    """
    with h5py.File(ruta_h5, "r") as f:
        imagenes = f["images"][:]
        clases = f["ans"][:].astype(int)

    hashes = np.array([hashlib.md5(imagenes[i].tobytes()).hexdigest()
                       for i in range(len(imagenes))])
    valores, conteos = np.unique(hashes, return_counts=True)
    hashes_duplicados = set(valores[conteos > 1])
    mask_unicas = np.array([h not in hashes_duplicados for h in hashes])

    return imagenes[mask_unicas], clases[mask_unicas]


def dividir_estratificado(clases, semilla=42):
    """Devuelve (idx_train, idx_val, idx_test) con partición estratificada 70/15/15.

    Estratificar conserva la proporción de cada clase en los tres subconjuntos, algo
    importante por el desbalance (clases minoritarias representadas de forma fiable).
    """
    indices = np.arange(len(clases))
    idx_temp, idx_test = train_test_split(
        indices, test_size=0.15, stratify=clases, random_state=semilla)
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=0.1765, stratify=clases[idx_temp], random_state=semilla)
    return idx_train, idx_val, idx_test


def construir_transforms(tam=224):
    """Crea las dos tuberías de transformación: (transform_train, transform_eval).

    - train: redimensiona, normaliza y aplica augmentation (rotaciones y espejos),
      transformaciones que preservan la clase de la galaxia.
    - eval: solo redimensiona y normaliza (determinista, para val y test).
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((tam, tam), antialias=True),
        transforms.RandomRotation(degrees=180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(MEDIA_IMAGENET, STD_IMAGENET),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((tam, tam), antialias=True),
        transforms.Normalize(MEDIA_IMAGENET, STD_IMAGENET),
    ])
    return transform_train, transform_eval


class GalaxyDataset(Dataset):
    """Sirve (imagen, etiqueta) para una partición concreta del dataset.

    No copia las imágenes: guarda una referencia al array completo y la lista de
    índices de su partición, aplicando la transformación en cada acceso.
    """

    def __init__(self, imagenes, clases, indices, transform):
        self.imagenes = imagenes
        self.clases = clases
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        idx = self.indices[pos]                 # posición local -> índice global
        imagen = self.transform(self.imagenes[idx])
        etiqueta = int(self.clases[idx])
        return imagen, etiqueta


def crear_dataloaders(imagenes, clases, idx_train, idx_val, idx_test,
                      transform_train, transform_eval, batch=32):
    """Construye los tres DataLoaders (train con shuffle; val y test sin él)."""
    ds_train = GalaxyDataset(imagenes, clases, idx_train, transform_train)
    ds_val = GalaxyDataset(imagenes, clases, idx_val, transform_eval)
    ds_test = GalaxyDataset(imagenes, clases, idx_test, transform_eval)

    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True,
                          num_workers=0, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False,
                        num_workers=0, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch, shuffle=False,
                         num_workers=0, pin_memory=True)
    return dl_train, dl_val, dl_test
