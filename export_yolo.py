from pathlib import Path
from ultralytics import YOLO

# Définition dynamique des chemins
PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "runs" / "classify" / "train" / "weights" / "best.pt"

if not WEIGHTS_PATH.exists():
    print(f"❌ Fichier weights introuvable à : {WEIGHTS_PATH}")
else:
    print(f"📦 Chargement du modèle depuis : {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))

    print("⚡ Exportation vers le format OpenVINO (mode FP16 pour CPU)...")
    model.export(format="openvino", half=True)
    print("🎉 Exportation réussie ! Vous avez maintenant votre dossier OpenVINO ready pour le temps réel.")