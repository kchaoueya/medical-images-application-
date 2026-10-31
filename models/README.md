# Dossier des modèles Deep Learning

Déposez ici vos modèles ; ils apparaissent automatiquement dans « Apprentissage profond ».

| Format | Dépendance | Remarque |
|---|---|---|
| `.onnx` | `onnxruntime` (déjà dans requirements) | recommandé — léger, portable |
| `.keras` `.h5` `.hdf5` | `tensorflow` | chargement direct, ex. le U-Net du TP4 |
| MedSAM | `torch transformers` | poids téléchargés depuis Hugging Face au premier usage |

## Fichier descriptif facultatif `<nom>.json`

```json
{ "name": "U-Net BUSI (TP4)", "input_size": 256, "channels": 1,
  "norm": "tp4_l2", "layout": "nhwc", "threshold": 0.5, "output_channel": 0,
  "description": "U-Net entraîné sur Breast Ultrasound Images" }
```

`norm` : `unit` (x/255, défaut), `tp4_l2` (`tf.keras.utils.normalize(axis=1)` utilisé au TP4), `zscore`, `imagenet`.
Pour les fichiers Keras dont le nom commence par `unet`, `tp4_l2` est supposé si aucun `.json` n'est fourni.

## Utiliser le U-Net du TP4
1. Sur Kaggle, récupérez `unet_..._ep50.hdf5` (onglet *Output*) et copiez-le ici.
2. Soit `pip install tensorflow` (chargement direct), soit convertissez-le :
   `python tools/export_keras_to_onnx.py models/unet_..._ep50.hdf5 models/unet_busi.onnx --name "U-Net BUSI (TP4)"`.

Attention : un modèle n'est valide que pour la modalité sur laquelle il a été entraîné (BUSI = échographie mammaire).
