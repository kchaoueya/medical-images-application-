# MedSeg — analyse et segmentation d'images médicales (web)

Application web (Flask + HTML/JS) : import d'une image médicale → prétraitement → segmentation
conventionnelle et par apprentissage profond → comparaison visuelle et quantitative.

## Lancer
```bash
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # puis ouvrir http://127.0.0.1:5000
```
Tests : `pip install pytest onnx && python -m pytest tests -q`.

## Fonctionnalités
**1. Prétraitement** (pipeline ordonnable, paramétrable, aperçu en direct, PSNR/SSIM, histogrammes) :
bruit gaussien / poivre et sel (simulation, TP2), filtres médian, gaussien, moyenneur, bilatéral, non-local means,
CLAHE, égalisation, étirement de contraste, gamma, unsharp, morphologie, inversion.

**2. Segmentation conventionnelle** : Otsu, seuil manuel, seuillage adaptatif, multi-Otsu, K-means, Mean-shift (TP3),
Chan-Vese (TP3), contour actif géodésique, ligne de partage des eaux (marqueurs auto ou germe), croissance de région.
Post-traitement commun : exclusion du fond noir, ouverture, plus grande composante, remplissage, surface minimale (démarche du TP3).

**3. Deep Learning pré-entraîné** : modèles ONNX / Keras déposés dans `models/` (U-Net du TP4), MedSAM en option. Voir `models/README.md`.

**4. Entraîner un U-Net (TP4)** — onglet dédié, nécessite `pip install tensorflow` (Python 3.11/3.12 recommandé sous Windows) :
jeu de données image + masque (zip ou dossier local, format BUSI `nom.png` / `nom_mask.png`, masques multiples fusionnés),
redimensionnement 256×256, normalisation du TP4, binarisation des masques, découpage apprentissage/validation (90/10, `random_state=0`),
U-Net identique au notebook (16→256 filtres, 1 940 817 paramètres), Adam, perte BCE / Dice / BCE+Dice, arrêt anticipé
(`EarlyStopping`, `restore_best_weights`), miroir horizontal optionnel, suivi en direct des courbes (loss, accuracy, Dice, IoU),
alerte sur le déséquilibre de classes et sur l'accuracy trompeuse, prédictions sur la validation, comparaison de plusieurs
entraînements (10 vs 50 époques vs arrêt anticipé). Le modèle est sauvegardé dans `models/` (+ historique) et devient utilisable
immédiatement dans « Apprentissage profond ».

**5. Visualisation / comparaison** : cartes de résultats (superposition ↔ masque), contours superposés de plusieurs méthodes,
curseur avant/après, tableau (temps, aire, périmètre, circularité), matrice d'accord Dice/IoU entre méthodes,
Dice / IoU / Hausdorff contre un masque de référence (si fourni), diagrammes, export ZIP (masques, superpositions, CSV).

## Architecture
```
app.py                 API Flask (/api/upload, preprocess, segment, compare, export)
core/preprocess.py     opérateurs de prétraitement
core/segmentation.py   méthodes conventionnelles + post-traitement
core/dl.py             ONNX / Keras / MedSAM
core/unet.py, core/training.py   U-Net du TP4 et entraînement (tâche de fond, suivi par l'API /api/train/*)
core/metrics.py        Dice, IoU, Hausdorff, descripteurs de forme
static/                interface (index.html, app.js, style.css) — pilotée par le schéma exposé par /api/schema
samples/ tools/ tests/ fantômes synthétiques, export Keras→ONNX, tests
```
Ajouter une méthode : écrire la fonction dans `core/segmentation.py`, l'ajouter à `METHODS` et à `SCHEMA` — l'interface la prend en compte seule.

## Limites à connaître
- Images converties en niveaux de gris, réduites à 1024 px max. Pas de DICOM (convertir en PNG/TIFF).
- L'entraînement tourne sur CPU/GPU de la machine qui exécute l'application (compter plusieurs minutes pour 647 images à 256×256) ; un seul à la fois.
- Les images sont gardées en mémoire (20 max, perdues au redémarrage). Usage local, pas d'authentification : ne pas exposer sur Internet.
- Outil pédagogique, pas un dispositif médical.
- Sans masque de référence, Dice/IoU mesurent l'accord entre méthodes, pas la justesse (comme au TP3).
