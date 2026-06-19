"""
Pipeline de détection d'objets et segmentation sur le dataset Digimon.

Étapes :
1. Chargement des images depuis le dataset HuggingFace (MatrixStudio/digimon-profile)
2. Détection d'objets avec YOLOv8 (ultralytics)
3. Crop de chaque objet détecté
4. Segmentation de chaque crop avec CLIPSeg (CIDAS/clipseg-rd64-refined)
5. Projection des contours de segmentation sur l'image originale
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import numpy as np
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from datasets import load_dataset
import cv2

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

DATASET_NAME     = "MatrixStudio/digimon-profile"
CLIPSEG_MODEL    = "CIDAS/clipseg-rd64-refined"
YOLO_MODEL       = "yolov8n.pt"          # téléchargé automatiquement par ultralytics
OUTPUT_DIR       = "output"
N_IMAGES         = 10                    # nombre d'images à traiter depuis le dataset

# Prompt CLIPSeg : décrit ce qu'on cherche à segmenter dans le crop
CLIPSEG_PROMPT   = "a character"

# Seuil de confiance YOLO (0–1)
YOLO_CONFIDENCE  = 0.3

# Couleur et épaisseur des contours (BGR pour OpenCV)
CONTOUR_COLOR    = (180, 0, 180)
CONTOUR_THICKNESS = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# CHARGEMENT DES MODÈLES
# ──────────────────────────────────────────────

def load_models():
    """Charge YOLOv8 et CLIPSeg."""
    print("[1/2] Chargement de YOLOv8...")
    yolo = YOLO(YOLO_MODEL)

    print("[2/2] Chargement de CLIPSeg...")
    processor = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL)
    clipseg   = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL)
    clipseg.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipseg.to(device)
    print(f"  → CLIPSeg sur : {device}")

    return yolo, processor, clipseg, device


# ──────────────────────────────────────────────
# CHARGEMENT DU DATASET
# ──────────────────────────────────────────────

def load_images(n: int) -> list[tuple[str, Image.Image]]:
    """
    Charge les n premières images du dataset HuggingFace.
    Retourne une liste de (nom, image PIL RGB).
    """
    print(f"\nChargement du dataset '{DATASET_NAME}' ({n} images)...")
    ds = load_dataset(DATASET_NAME, split="train", verification_mode="no_checks")

    images = []
    for i, sample in enumerate(ds):
        if i >= n:
            break
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        name = f"digimon_{i:03d}"
        images.append((name, img))
        print(f"  Image {i+1}/{n} chargée : {img.size}")

    return images


# ──────────────────────────────────────────────
# DÉTECTION D'OBJETS — YOLOv8
# ──────────────────────────────────────────────

def detect_objects(yolo, image: Image.Image) -> list[dict]:
    """
    Détecte les objets dans l'image avec YOLOv8.
    Retourne une liste de dicts : {label, confidence, bbox: (x1,y1,x2,y2)}.
    """
    results = yolo(image, conf=YOLO_CONFIDENCE, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        label = results.names[int(box.cls[0])]
        detections.append({
            "label":      label,
            "confidence": conf,
            "bbox":       (x1, y1, x2, y2),
        })

    return detections


# ──────────────────────────────────────────────
# SEGMENTATION — CLIPSeg
# ──────────────────────────────────────────────

def segment_crop(
    processor,
    clipseg,
    device: str,
    crop: Image.Image,
    prompt: str = CLIPSEG_PROMPT,
) -> np.ndarray:
    """
    Applique CLIPSeg sur un crop PIL et retourne un masque binaire NumPy
    de la même taille que le crop (valeurs 0 ou 255).
    """
    inputs = processor(
        text=[prompt],
        images=[crop],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = clipseg(**inputs)

    # Logits → sigmoid → redimensionnement à la taille du crop
    logits = outputs.logits[0]                          # (H, W)
    mask_prob = torch.sigmoid(logits).cpu().numpy()     # valeurs [0, 1]

    # Redimensionnement au format original du crop
    mask_resized = cv2.resize(
        mask_prob,
        (crop.width, crop.height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Binarisation par seuil médian
    threshold = float(np.median(mask_resized)) + 0.1
    binary_mask = (mask_resized > threshold).astype(np.uint8) * 255

    return binary_mask


# ──────────────────────────────────────────────
# CONTOURS ET PROJECTION
# ──────────────────────────────────────────────

def draw_contours_on_image(
    original_image: Image.Image,
    detections: list[dict],
    masks: list[np.ndarray],
) -> Image.Image:
    """
    Pour chaque (detection, mask), calcule les contours du masque
    et les projette sur l'image originale aux coordonnées de la bbox.
    Retourne l'image annotée.
    """
    # Conversion PIL → OpenCV (BGR)
    img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    for det, mask in zip(detections, masks):
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf  = det["confidence"]

        # Extraction des contours dans le repère du crop
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Décalage des contours vers la position de la bbox dans l'image originale
        shifted = [c + np.array([[[x1, y1]]]) for c in contours]

        # Dessin des contours sur l'image originale
        cv2.drawContours(img_cv, shifted, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

        # Dessin de la bbox et du label
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 100, 0), 1)
        cv2.putText(
            img_cv,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 100, 0),
            1,
            cv2.LINE_AA,
        )

    # Reconversion BGR → PIL RGB
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def process_image(
    name: str,
    image: Image.Image,
    yolo,
    processor,
    clipseg,
    device: str,
) -> Image.Image:
    """
    Traite une image complète : détection → crop → segmentation → contours.
    Sauvegarde le résultat dans OUTPUT_DIR et retourne l'image annotée.
    """
    print(f"\n{'─'*50}")
    print(f"  Traitement : {name}  ({image.size})")

    # 1. Détection
    detections = detect_objects(yolo, image)
    print(f"  Objets détectés : {len(detections)}")
    for d in detections:
        print(f"    • {d['label']} ({d['confidence']:.2f})  bbox={d['bbox']}")

    if not detections:
        print("  Aucun objet détecté, image sauvegardée sans annotation.")
        image.save(os.path.join(OUTPUT_DIR, f"{name}_no_detection.jpg"))
        return image

    # 2. Crop + segmentation pour chaque objet
    masks = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # Clamp aux dimensions de l'image
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image.width, x2); y2 = min(image.height, y2)

        crop = image.crop((x1, y1, x2, y2))

        if crop.width < 4 or crop.height < 4:
            # Crop trop petit → masque vide
            masks.append(np.zeros((crop.height, crop.width), dtype=np.uint8))
            continue

        mask = segment_crop(processor, clipseg, device, crop)
        masks.append(mask)
        print(f"    → Segmentation OK pour '{det['label']}'")

    # 3. Projection des contours sur l'image originale
    annotated = draw_contours_on_image(image, detections, masks)

    # 4. Sauvegarde
    out_path = os.path.join(OUTPUT_DIR, f"{name}_result.jpg")
    annotated.save(out_path)
    print(f"  ✓ Sauvegardé : {out_path}")

    return annotated


def main():
    # Chargement des modèles
    yolo, processor, clipseg, device = load_models()

    # Chargement des images
    images = load_images(N_IMAGES)

    # Traitement de chaque image
    for name, image in images:
        process_image(name, image, yolo, processor, clipseg, device)

    print(f"\n{'='*50}")
    print(f"Pipeline terminé. Résultats dans : ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()