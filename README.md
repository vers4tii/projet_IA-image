# Pipeline Détection & Segmentation — Digimon

Pipeline complet qui, à partir d'images du dataset **MatrixStudio/digimon-profile** :
1. Détecte les objets avec **YOLOv8**
2. Croppe chaque objet détecté
3. Segmente chaque crop avec **CLIPSeg**
4. Projette les contours de segmentation sur l'image originale

---

## Prérequis — Version Python compatible

> Version de python à utiliser : **Python 3.13** (supporte PyTorch 2.6+) ou **Python 3.12**.

Vérifie les versions installées sur ta machine :
```powershell
py -0
```

---

## Installation

**1. Crée un environnement virtuel :**

```powershell
# Avec Python 3.13
py -3.13 -m venv venv313
venv313\Scripts\activate

# Ou avec Python 3.12 (si 3.13 absent)
py -3.12 -m venv venv312
venv312\Scripts\activate

python --version   # vérifie la version active
```

> Le préfixe `(venv313)` dans le terminal confirme que le venv est actif.  
> À chaque nouvelle session, réactive-le avec `venv313\Scripts\activate`.

**2. Installe PyTorch puis les dépendances :**

```powershell
# Avec GPU NVIDIA (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Ou CPU uniquement (sans GPU NVIDIA)
pip install torch torchvision

# Puis les autres dépendances
pip install -r requirements.txt
```

> Si Python 3.12 et 3.13 sont absents, télécharge Python 3.12 ici :  
> 👉 https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe

---

## Utilisation

```bash
python pipeline.py
```

Les images annotées sont sauvegardées dans le dossier `output/`.

---

## Configuration

Toutes les options sont regroupées en haut de `pipeline.py` :

| Variable | Valeur par défaut | Description |
|---|---|---|
| `N_IMAGES` | `5` | Nombre d'images à traiter |
| `YOLO_CONFIDENCE` | `0.3` | Seuil de confiance YOLO (0–1) |
| `CLIPSEG_PROMPT` | `"a character"` | Prompt texte pour CLIPSeg |
| `CONTOUR_COLOR` | `(0, 255, 0)` | Couleur des contours (BGR) |
| `CONTOUR_THICKNESS` | `2` | Épaisseur des contours (px) |
| `OUTPUT_DIR` | `"output"` | Dossier de sortie |

---

## Modèles utilisés

| Rôle | Modèle | Source |
|---|---|---|
| Détection d'objets | YOLOv8n | [Ultralytics/YOLOv8](https://huggingface.co/Ultralytics/YOLOv8) |
| Segmentation | CLIPSeg RD64 | [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) |
| Dataset | Digimon Profile | [MatrixStudio/digimon-profile](https://huggingface.co/datasets/MatrixStudio/digimon-profile) |

---

## Structure du projet

```
PROJET_IA-IMAGE/
├── pipeline.py        # Script principal
├── requirements.txt   # Dépendances Python
├── README.md
└── output/            # Images annotées (créé automatiquement)
```

---

## Notes

- **YOLOv8** est pré-entraîné sur COCO (80 classes génériques). Il détecte des personnes, animaux, objets courants. Sur des images de type illustration/manga comme les Digimon, les détections peuvent être partielles — ajuster `YOLO_CONFIDENCE` à la baisse (ex: `0.2`) pour plus de détections.
- **CLIPSeg** est un modèle *zero-shot* : il segmente ce que tu décris en texte via `CLIPSEG_PROMPT`. Adapter le prompt selon le contenu (ex: `"a digital monster"`, `"an animal"`, `"a character"`).
- Le pipeline fonctionne sur CPU mais sera plus rapide avec un GPU CUDA.