from pathlib import Path
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent

# 1. Charger le détecteur de visage/personne (standard) et le classificateur d'émotions (votre modèle exporté)
detector = YOLO("yolov8n.pt")
EMOTION_MODEL_PATH = PROJECT_ROOT / "runs" / "classify" / "train" / "weights" / "best_openvino_model"
emotion_classifier = YOLO(str(EMOTION_MODEL_PATH))

# 2. Ouvrir la vidéo MP4 (remplacez par le nom de votre vidéo)
video_path = PROJECT_ROOT / "video_1.mp4" 
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"❌ Impossible d'ouvrir la vidéo à : {video_path}")
    exit()

# Dictionnaire de traduction (optionnel)
EMOTION_TRANSLATE = {
    "angry": "Colere", "disgust": "Degout", "fear": "Peur",
    "happy": "Joie", "neutral": "Neutre", "sad": "Tristesse", "surprise": "Surprise"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Détection avec YOLO
    results = detector(frame, verbose=False)[0]

    for box in results.boxes:
        # Ne garder que la classe 'person' (cls == 0)
        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Inférence de l'émotion via OpenVINO
        emo_res = emotion_classifier(roi, verbose=False)[0]
        top1_idx = emo_res.probs.top1
        raw_label = emo_res.names[top1_idx]
        label_fr = EMOTION_TRANSLATE.get(raw_label, raw_label)
        confidence = emo_res.probs.top1conf.item() * 100

        # Rendu visuel
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label_fr} ({confidence:.1f}%)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Analyse des Expressions - YOLO OpenVINO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()