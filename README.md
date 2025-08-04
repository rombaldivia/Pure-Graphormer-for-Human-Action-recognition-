# Graphormer-Spatial 🦾 — HAR sobre NTU-RGB-D-120
A pure Graphormer implementation for Human Action Recognition (HAR) from 3-D skeletons.Trains with PyTorch ≥ 2.1 and runs on CPU, GPU or Jetson (ONNX / TensorRT).

<div align="center">
  <img src="docs/figures/pipeline.svg" width="620" alt="Pipeline">
</div>

---

## ✨ Características
- **Token Dropout** y **unfreezing progresivo** listos para CLI.
- Reanuda checkpoints → `--resume best_graphormer.pth`.
- Hot-reload de LR vía `hyper_lr.txt` sin reiniciar el entrenamiento.
- Confusion matrix y curva de pérdidas en `./metrics/`.
- Split 70 / 20 / 10 estratificado con el script `tools/split_ntu.py`.
- Compatible con multi-GPU (`DataParallel`) y mixed-precision (AMP).

## Requisitos
```bash
