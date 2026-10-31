"""Convertit un U-Net Keras (.keras/.h5/.hdf5, p. ex. celui du TP4) en ONNX + fichier descriptif .json.

Usage :  pip install tensorflow tf2onnx
         python tools/export_keras_to_onnx.py unet_bs16_ep50.hdf5 models/unet_busi.onnx [--norm tp4_l2]
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--norm", default="tp4_l2", choices=["unit", "tp4_l2", "zscore", "imagenet"],
                help="prétraitement utilisé à l'entraînement (TP4 : tp4_l2 = tf.keras.utils.normalize)")
ap.add_argument("--name", default=None)
a = ap.parse_args()

import tensorflow as tf, tf2onnx  # noqa: E401
model = tf.keras.models.load_model(a.src, compile=False)
_, h, w, c = model.input_shape
spec = (tf.TensorSpec((1, h, w, c), tf.float32, name="input"),)
tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=a.dst)
meta = {"name": a.name or Path(a.dst).stem, "input_size": h, "channels": c, "norm": a.norm, "layout": "nhwc",
        "threshold": 0.5, "description": f"Exporté depuis {Path(a.src).name}"}
Path(a.dst).with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print("OK ->", a.dst)
