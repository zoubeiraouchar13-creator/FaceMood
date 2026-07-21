# FaceMood — Real-Time Facial Emotion Detection and Classification

Cascaded computer vision pipeline combining **YOLOv8** (person detection) and a **YOLOv8n-cls** emotion classifier fine-tuned on **FER-2013**, optimized for real-time CPU inference via **OpenVINO**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-purple)
![OpenVINO](https://img.shields.io/badge/Inference-OpenVINO%20FP16-teal)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Description

**FaceMood** analyzes a video feed and displays, in real time, the dominant emotion of each person detected on screen (anger, disgust, fear, joy, neutral, sadness, surprise).

The system relies on a **two-stage cascaded architecture**:

1. A generic **YOLOv8n** model (pre-trained on COCO) detects the people present in each image.
2. A **YOLOv8n-cls** model, trained via transfer learning on **FER-2013**, classifies the facial expression of each detected region.

The classifier is then exported to **OpenVINO (FP16)** format to ensure smooth execution on a machine **without a dedicated GPU**.

```
Video ──► Person detection (YOLOv8n) ──► ROI cropping ──► Emotion classification (OpenVINO) ──► Real-time display```

## 🎯 Features

- 📥 Automatic download and preparation of the **FER-2013** dataset (via Kaggle)
- 🧠 Emotion classifier training using transfer learning (YOLOv8n-cls)
- ⚡ Model export to **OpenVINO** in **FP16** precision for CPU inference
- 🎥 Real-time video analysis pipeline using OpenCV
- 🇫🇷 Automatic translation of emotion labels into French
- 📦 Display of bounding boxes, emotion labels, and confidence scores

## 📁 Project Structure

```
FaceMood/
├── train_yolo.py     # FER-2013 download + YOLOv8n-cls training
├── export_yolo.py     # Export du modèle entraîné vers OpenVINO (FP16)
├── main_video.py       # Pipeline d'inférence vidéo temps réel
├── requirements.txt   # Dépendances Python
├── video_1.mp4        # Vidéo d'entrée (à fournir par l'utilisateur)
├── fer2013/           # Dataset (généré automatiquement à l'entraînement)
└── runs/classify/train/weights/
    ├── best.pt                 # Poids PyTorch entraînés
    └── best_openvino_model/    # Modèle exporté (OpenVINO, FP16)
```

## ⚙️ Installation

```bash
git clone https://github.com/<votre-nom-utilisateur>/FaceMood.git
cd FaceMood
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ Un compte **Kaggle** avec un token API configuré (`~/.kaggle/kaggle.json` ou variables d'environnement `KAGGLE_USERNAME` / `KAGGLE_KEY`) est nécessaire pour le téléchargement automatique du dataset FER-2013.

## 🚀 Utilisation

Les trois étapes s'exécutent dans l'ordre suivant :

### 1. Entraîner le classificateur d'émotions

```bash
python train_yolo.py
```

Télécharge FER-2013 si nécessaire, puis entraîne YOLOv8n-cls (15 épochs, CPU) et sauvegarde les poids dans `runs/classify/train/weights/best.pt`.

### 2. Exporter le modèle vers OpenVINO

```bash
python export_yolo.py
```

Convertit `best.pt` en modèle optimisé `best_openvino_model/` (précision FP16, adapté au CPU).

### 3. Lancer l'analyse vidéo temps réel

```bash
python main_video.py
```

Placez votre vidéo sous le nom `video_1.mp4` à la racine du projet (ou modifiez `video_path` dans le script). Appuyez sur **`q`** pour quitter la fenêtre d'affichage.

## 🧠 Modèle & Dataset

| Élément | Détail |
|---|---|
| Détecteur de personnes | `yolov8n.pt` (pré-entraîné COCO, classe `person`) |
| Classificateur d'émotions | `yolov8n-cls` fine-tuné sur FER-2013 |
| Dataset | [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) — 7 classes, images 48×48 en niveaux de gris |
| Format d'inférence | OpenVINO IR, précision FP16 |
| Device d'entraînement | CPU |

## ⚠️ Limites connues

- Le recadrage utilisé pour la classification provient d'un détecteur de **personnes**, pas d'un détecteur de **visages** dédié.
- FER-2013 est déséquilibré (classe *dégoût* sous-représentée), ce qui peut affecter sa reconnaissance.
- Aucun lissage temporel entre images : l'émotion affichée peut varier d'une frame à l'autre.
- Traite actuellement un fichier vidéo enregistré (l'usage webcam en direct nécessite de remplacer le chemin vidéo par l'indice de la caméra, ex. `cv2.VideoCapture(0)`).

## 🔭 Perspectives

- Intégrer un détecteur de visages dédié
- Ajouter un tracking + lissage temporel des prédictions
- Support de la webcam en direct
- Augmentation de données pour les classes sous-représentées

## 🛠️ Technologies utilisées

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)
- [KaggleHub](https://github.com/Kaggle/kagglehub)

## 👤 Auteur

**Zoubeir Aouchar**

## 📄 Licence

Ce projet est distribué sous licence MIT — voir le fichier [LICENSE](LICENSE) pour plus de détails.
