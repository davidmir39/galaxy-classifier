"""
Módulo de VISUALIZACIÓN.

  - graficar_historial: curvas de pérdida y de F1 de validación por época.
  - matriz_confusion: matriz de confusión normalizada.

Ambas guardan la figura en disco si se indica 'ruta' (además de mostrarla).
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def graficar_historial(historiales, ruta=None):
    """Dibuja loss y F1 de validación por época.

    Args:
        historiales: dict {nombre_modelo: historial}, donde historial es la lista
            de dicts que devuelve entrenar().
        ruta: si se indica, guarda la figura en ese path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for nombre, h in historiales.items():
        epocas = [r["epoca"] for r in h]
        ax1.plot(epocas, [r["loss"] for r in h], marker="o", label=nombre)
        ax2.plot(epocas, [r["f1_val"] for r in h], marker="o", label=nombre)
    ax1.set_title("Pérdida de entrenamiento")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Loss"); ax1.legend()
    ax2.set_title("F1 macro (validación)")
    ax2.set_xlabel("Época"); ax2.set_ylabel("F1 macro"); ax2.legend()
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.show()


def matriz_confusion(y_true, y_pred, nombres_clases, ruta=None,
                     titulo="Matriz de confusión normalizada"):
    """Dibuja la matriz de confusión normalizada por fila (recall por clase)."""
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=nombres_clases, normalize="true",
        xticks_rotation=45, cmap="Blues", ax=ax, values_format=".2f")
    ax.set_title(titulo)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.show()
