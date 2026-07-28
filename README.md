# FaceMood — Real-Time Facial Emotion Detection and Classification

A cascaded computer vision pipeline combining **YOLOv8** (person detection) and a **YOLOv8s-cls** emotion classifier fine-tuned on **FER-2013**, optimized for real-time CPU inference via **OpenVINO**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-purple)
![OpenVINO](https://img.shields.io/badge/Inference-OpenVINO%20FP16-teal)
![License](https://img.shields.io/badge/License-CC%20BY%20NC-blue)

---

## 📋 Description

**FaceMood** analyzes a video feed and displays, in real time, the dominant emotion of each person detected on screen (anger, disgust, fear, joy, neutral, sadness, surprise).

The system is based on a **two-stage cascaded architecture**:
1. A generic **YOLOv8s** model (pre-trained on COCO) detects the people present in each image.
2. A **YOLOv8s-cls** model, trained via transfer learning on **FER-2013**, classifies the facial expression of each detected region.

The classifier is then exported in **OpenVINO (FP16)** format to ensure smooth execution on a machine **without a dedicated GPU**.
```bash
Video ──► Person detection (YOLOv8s) ──► ROI cropping ──► Emotion classification (OpenVINO) ──► Real-time display
```

## Features

- Automatic download and preparation of the **FER-2013** dataset (via Kaggle)
- Emotion classifier training using transfer learning (YOLOv8s-cls)
- Model export to **OpenVINO** in **FP16** precision for CPU inference
- Real-time video analysis pipeline using OpenCV
- 🇫🇷 Automatic translation of emotion labels into French
- Display of bounding boxes, emotion labels, and confidence scores

## Project Structure

```bash
FaceMood/
├── data_verification.py     # Check the composition of the FER2013 dataset
├── balance_dataset.py     # Balanced the dataset based on the median.
├── train_yolo.py     # FER-2013 download + YOLOv8s-cls training
├── export_yolo.py     # Exporting the trained model to OpenVINO (FP16)
├── main_video.py       # Real-time video inference pipeline
├── requirements.txt   # Python dependencies
├── video_1.mp4        # Intro video (to be provided by the user)
├── fer2013/           # Dataset (automatically generated during training)
└── runs/classify/train/weights/
    ├── best.pt                 # Pre-trained PyTorch weights
    └── best_openvino_model/    # Exported model (OpenVINO, FP16)
```

## ⚙️ Installation

```bash
git clone https://github.com/<votre-nom-utilisateur>/FaceMood.git
cd FaceMood
python -m venv .venv
source venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The three steps are executed in the following order:

### 1. Train the emotion classifier

```bash
python train_yolo.py
```

Download FER-2013 if necessary, then train YOLOv8s-cls (15 epochs, CPU) and save the weights to `runs/classify/train/weights/best.pt`.

### 2. Export the model to OpenVINO

```bash
python export_yolo.py
```

Converts `best.pt` into an optimized model `best_openvino_model/` (FP16 precision, optimized for CPU).

### 3. Start real-time video analysis

```bash
python main_video.py
```

Place your video, named `video_1.mp4`, at the project root (or modify `video_path` in the script). Press **`q`** to close the display window.

## Modèle & Dataset

| Élément | Détail |
|---|---|
| Person Detector | `yolov8s.pt` (pre-entrained COCO, classe `person`) |
| Emotion Classificator | `yolov8s-cls` fine-tuned using FER-2013 |
| Dataset | [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) — 7 classes, 48×48 grayscale images |
| Inference Format | OpenVINO IR, precision FP16 |
| Training Device | CPU |


## 🛠️ Used Technologies

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)
- [KaggleHub](https://github.com/Kaggle/kagglehub)
