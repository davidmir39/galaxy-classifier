# Carga del HDF5, limpieza, partición train/val/test y DataLoaders de PyTorch.

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

NOMBRES_CLASES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]

# Media y std de ImageNet: las usan ResNet y EfficientNet preentrenadas.
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

# El dataset no se versiona (pesa ~2,5 GB), así que lo bajamos de Zenodo la primera
# vez. Guardamos el SHA256 oficial para confirmar que la descarga llegó completa.
URL_DATASET = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5?download=1"
SHA256_DATASET = "19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571"


def _sha256(ruta, bloque=1024 * 1024):
    """SHA256 de un fichero leyéndolo a trozos (el .h5 no cabe entero en memoria)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(bloque), b""):
            h.update(chunk)
    return h.hexdigest()


def asegurar_dataset(ruta_h5, verificar=True):
    """Descarga el .h5 de Zenodo la primera vez; si ya está en disco, no hace nada."""
    ruta = Path(ruta_h5)
    if ruta.exists():
        return str(ruta)

    ruta.parent.mkdir(parents=True, exist_ok=True)

    # Bajamos a un .part y renombramos solo al final. Así, si se corta la descarga,
    # no queda un .h5 incompleto que en la próxima ejecución parezca ya descargado.
    tmp = ruta.with_suffix(ruta.suffix + ".part")
    print(f"Dataset no encontrado. Descargando ~2,5 GB desde Zenodo a {ruta} ...")
    with urllib.request.urlopen(URL_DATASET) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                        desc="Galaxy10_DECals.h5") as barra:
            # De 1 MB en 1 MB para no cargar los 2,5 GB de golpe en memoria.
            for chunk in iter(lambda: resp.read(1024 * 1024), b""):
                f.write(chunk)
                barra.update(len(chunk))

    # Si el hash no cuadra el fichero está corrupto: lo borramos en vez de dejar
    # un dataset a medias que rompería el entrenamiento más adelante sin avisar.
    if verificar:
        print("Verificando integridad (SHA256) ...")
        digest = _sha256(tmp)
        if digest != SHA256_DATASET:
            tmp.unlink()
            raise RuntimeError(
                f"El SHA256 no coincide (esperado {SHA256_DATASET}, obtenido {digest}). "
                "Se ha descartado la descarga; vuelve a intentarlo.")
        print("Integridad verificada.")

    tmp.rename(ruta)
    return str(ruta)


def cargar_y_limpiar(ruta_h5):
    """Lee el .h5 y elimina imágenes duplicadas (mismo contenido, distinta etiqueta)."""
    with h5py.File(ruta_h5, "r") as f:
        imagenes = f["images"][:]
        clases = f["ans"][:].astype(int)

    # Hash por imagen para detectar copias exactas (lo vimos en el EDA).
    hashes = np.array([hashlib.md5(imagenes[i].tobytes()).hexdigest()
                       for i in range(len(imagenes))])
    valores, conteos = np.unique(hashes, return_counts=True)
    hashes_duplicados = set(valores[conteos > 1])

    # Quitamos todas las copias: no sabemos cuál etiqueta es la buena y además
    # podrían acabar en train y test a la vez (fuga de datos).
    mask_unicas = np.array([h not in hashes_duplicados for h in hashes])
    return imagenes[mask_unicas], clases[mask_unicas]


def dividir_estratificado(clases, semilla=42):
    """Split 70 % train / 15 % val / 15 % test, manteniendo proporción de clases."""
    indices = np.arange(len(clases))

    idx_temp, idx_test = train_test_split(
        indices, test_size=0.15, stratify=clases, random_state=semilla)

    # Del 85 % restante, nos quedamos con 15/85 ≈ 0,1765 para validación.
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=0.1765, stratify=clases[idx_temp], random_state=semilla)
    return idx_train, idx_val, idx_test


def construir_transforms(tam=224):
    """Devuelve (transform_train, transform_eval)."""
    # En train: augmentation geométrica (la orientación no define la clase de galaxia).
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((tam, tam), antialias=True),
        transforms.RandomRotation(degrees=180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(MEDIA_IMAGENET, STD_IMAGENET),
    ])

    # En val/test no aleatorizamos nada.
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((tam, tam), antialias=True),
        transforms.Normalize(MEDIA_IMAGENET, STD_IMAGENET),
    ])
    return transform_train, transform_eval


class GalaxyDataset(Dataset):
    """Dataset que indexa sobre el array completo sin duplicarlo en memoria."""

    def __init__(self, imagenes, clases, indices, transform):
        self.imagenes = imagenes
        self.clases = clases
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        idx = self.indices[pos]
        imagen = self.transform(self.imagenes[idx])
        etiqueta = int(self.clases[idx])
        return imagen, etiqueta


def crear_dataloaders(imagenes, clases, idx_train, idx_val, idx_test,
                      transform_train, transform_eval, batch=32):
    ds_train = GalaxyDataset(imagenes, clases, idx_train, transform_train)
    ds_val = GalaxyDataset(imagenes, clases, idx_val, transform_eval)
    ds_test = GalaxyDataset(imagenes, clases, idx_test, transform_eval)

    # num_workers=0: en Windows nos daba problemas con multiprocessing.
    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True,
                          num_workers=0, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False,
                        num_workers=0, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch, shuffle=False,
                         num_workers=0, pin_memory=True)
    return dl_train, dl_val, dl_test    