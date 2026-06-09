# Definición de la CNN baseline y de los modelos con transfer learning.

import torch.nn as nn
from torchvision import models


class CNNBaseline(nn.Module):
    """Red convolucional sencilla para comparar con transfer learning."""

    def __init__(self, num_clases=10):
        super().__init__()
        # Tres bloques conv + pool: 224 -> 112 -> 56 -> 28 píxeles.
        self.caracteristicas = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
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
    """Fase 1: backbone ImageNet congelado y cabeza nueva de 10 clases."""
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_clases)  # solo esto se entrena al principio
    return m.to(dispositivo)


def descongelar_resnet(modelo):
    """Fase 2 de fine-tuning: entrenar también el último bloque residual."""
    for p in modelo.layer4.parameters():
        p.requires_grad = True


def construir_efficientnet_b0(num_clases=10, dispositivo="cpu"):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_clases)
    return m.to(dispositivo)


def descongelar_efficientnet(modelo, n_bloques=2):
    for bloque in modelo.features[-n_bloques:]:
        for p in bloque.parameters():
            p.requires_grad = True
