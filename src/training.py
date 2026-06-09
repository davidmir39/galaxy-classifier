"""
Módulo de ENTRENAMIENTO.

  - calcular_pesos: pesos de clase para mitigar el desbalance.
  - evaluar: F1 macro y accuracy sobre un cargador (sin actualizar pesos).
  - entrenar: bucle de entrenamiento con early stopping, scheduler de learning rate
    y guardado del mejor modelo según F1 de validación.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


def calcular_pesos(clases, idx_train, dispositivo, num_clases=10):
    """Pesos inversamente proporcionales a la frecuencia de cada clase en train."""
    pesos = compute_class_weight("balanced", classes=np.arange(num_clases),
                                 y=clases[idx_train])
    return torch.tensor(pesos, dtype=torch.float32, device=dispositivo)


@torch.no_grad()
def evaluar(modelo, cargador, dispositivo):
    """Devuelve (f1_macro, accuracy, y_true, y_pred) sobre el cargador dado."""
    modelo.eval()
    y_true, y_pred = [], []
    for imgs, labs in cargador:
        logits = modelo(imgs.to(dispositivo))
        y_pred.append(logits.argmax(dim=1).cpu())
        y_true.append(labs)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return f1, acc, y_true, y_pred


def entrenar(modelo, dl_train, dl_val, criterio, dispositivo,
             epocas, lr, etiqueta="modelo", paciencia=5, usar_scheduler=True,
             ruta_modelos="../models"):
    """Entrena el modelo y guarda el mejor (por F1 de validación) en ruta_modelos.

    - Solo optimiza los parámetros entrenables (en Transfer Learning, la cabeza/capas
      descongeladas).
    - Scheduler ReduceLROnPlateau: baja el LR si el F1 de validación se estanca.
    - Early stopping: corta si no mejora en 'paciencia' épocas.
    """
    optim = torch.optim.Adam(
        [p for p in modelo.parameters() if p.requires_grad], lr=lr)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=2) if usar_scheduler else None)

    mejor_f1, epocas_sin_mejora, historial = 0.0, 0, []
    for ep in range(1, epocas + 1):
        modelo.train()
        loss_acum = 0.0
        for imgs, labs in dl_train:
            imgs, labs = imgs.to(dispositivo), labs.to(dispositivo)
            optim.zero_grad()                  # borrar gradientes previos
            logits = modelo(imgs)              # forward
            perdida = criterio(logits, labs)   # loss
            perdida.backward()                 # backward
            optim.step()                       # actualizar pesos
            loss_acum += perdida.item()

        f1v, accv, _, _ = evaluar(modelo, dl_val, dispositivo)
        if scheduler is not None:
            scheduler.step(f1v)
        historial.append({"epoca": ep, "loss": loss_acum / len(dl_train),
                          "f1_val": f1v, "acc_val": accv})
        print(f"Época {ep:2d} | loss {loss_acum/len(dl_train):.3f} | "
              f"F1 val {f1v:.3f} | acc val {accv:.3f}")

        if f1v > mejor_f1:
            mejor_f1 = f1v
            epocas_sin_mejora = 0
            torch.save(modelo.state_dict(), f"{ruta_modelos}/{etiqueta}.pt")
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= paciencia:
                print(f"Early stopping en época {ep} "
                      f"(sin mejora en {paciencia} épocas). Mejor F1: {mejor_f1:.3f}")
                break

    return historial
