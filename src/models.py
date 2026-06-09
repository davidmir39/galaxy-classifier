"""
Módulo de MODELOS.

  - CNNBaseline: red sencilla entrenada desde cero (línea base de referencia).
  - construir_resnet50 / construir_efficientnet_b0: backbones preentrenados en
    ImageNet con el backbone congelado y una cabeza nueva (fase de feature extraction).
  - descongelar_*: ayudantes para la fase de fine-tuning.
"""

import torch.nn as nn
from torchvision import models


class CNNBaseline(nn.Module):
    """CNN sencilla entrenada desde cero, como referencia frente al Transfer Learning."""

    def __init__(self, num_clases=10):
        super().__init__()
        # Extractor: 3 bloques (conv -> ReLU -> maxpool). 224 -> 112 -> 56 -> 28.
        self.caracteristicas = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Clasificador denso.
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_clases),
        )

    def forward(self, x):
        x = self.caracteristicas(x)
        x = self.clasificador(x)
        return x


def construir_resnet50(num_clases=10, dispositivo="cpu"):
    """ResNet50 preentrenada con backbone congelado y cabeza nueva (feature extraction)."""
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_clases)   # cabeza nueva, entrenable
    return m.to(dispositivo)


def descongelar_resnet(modelo):
    """Descongela el último bloque convolucional (layer4) para el fine-tuning."""
    for p in modelo.layer4.parameters():
        p.requires_grad = True


def construir_efficientnet_b0(num_clases=10, dispositivo="cpu"):
    """EfficientNetB0 preentrenada con backbone congelado y cabeza nueva."""
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    in_features = m.classifier[1].in_features        # 1280 en B0
    m.classifier[1] = nn.Linear(in_features, num_clases)
    return m.to(dispositivo)


def descongelar_efficientnet(modelo, n_bloques=2):
    """Descongela los últimos n bloques de features para el fine-tuning."""
    for bloque in modelo.features[-n_bloques:]:
        for p in bloque.parameters():
            p.requires_grad = True
