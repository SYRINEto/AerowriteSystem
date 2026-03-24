#  Air Canvas — Dessiner dans l'air avec Python & Computer Vision

> Dessinez dans l'air avec vos gestes, sans toucher l'écran.  
> Application temps réel basée sur **OpenCV**, **MediaPipe** et **NumPy**.

---

## Aperçu

Air Canvas capture le flux de ta webcam, détecte ta main en temps réel grâce à **MediaPipe Hands**, et te permet de dessiner librement dans l'air en levant l'index.  
La **détection faciale** (Face Mesh) déclenche automatiquement une capture d'écran quand tu ouvres la bouche.

---

## Fonctionnalités

- **Dessin gestuel** — index levé seul = mode dessin
- **Navigation** — index + majeur levés = sélectionner couleur sans tracer
- **7 couleurs** — Rouge, Orange, Jaune, Vert, Bleu, Violet, Blanc
- **4 tailles de pinceau** — réglables avec `+` / `-`
- **Gomme intelligente** — sélectionnable dans la toolbar
- **Effets de particules** — émises à chaque trait
- **Screenshot automatique** — déclenché par ouverture de la bouche (Face Mesh)
- **Screenshot manuel** — touche `S`
- **Effacement du canvas** — touche `C`
- **FPS affiché** en temps réel

---

## Technologies utilisées

| Outil | Rôle |
|---|---|
| Python 3.8+ | Langage principal |
| OpenCV | Capture webcam, rendu graphique |
| MediaPipe Hands | Détection et tracking de la main |
| MediaPipe Face Mesh | Détection de l'ouverture de la bouche |
| NumPy | Gestion des pixels et optimisation |

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/SYRINEto/Air-Canvas.git
cd air-canvas
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l'application

```bash
python air_canvas.py
```

> Une webcam est requise. Résolution conseillée : 1280×720.

---

## Gestes et contrôles

### Gestes de la main

| Geste | Action |
|---|---|
| ☝️ Index seul levé | Mode **DESSIN** — trace un trait |
| ✌️ Index + Majeur levés | Mode **NAVIGATION** — sélectionner sans tracer |
| 👆 Pointer sur la toolbar | Sélectionner une couleur / gomme |

### Clavier

| Touche | Action |
|---|---|
| `C` | Effacer le canvas |
| `S` | Screenshot manuel |
| `+` / `=` | Augmenter la taille du pinceau |
| `-` | Réduire la taille du pinceau |
| `Q` ou `Esc` | Quitter |

### Détection faciale

| Geste | Action |
|---|---|
|  Bouche ouverte | Screenshot automatique (cooldown 3s) |

---

## Structure du projet

```
air-canvas/
├── air_canvas.py       # Application principale
├── requirements.txt    # Dépendances Python
├── README.md           # Documentation
└── screenshots/        # Captures auto-générées (créé au lancement)
```

---

## Comment ça marche

```
Webcam
  │
  ▼
OpenCV (flip + BGR→RGB)
  │
  ├──▶ MediaPipe Hands
  │        └── Landmarks 21 points → position index → dessin sur canvas NumPy
  │
  └──▶ MediaPipe Face Mesh
           └── Distance lèvres → bouche ouverte → screenshot
  │
  ▼
Fusion canvas + frame (masque binaire)
  │
  ▼
Particules + Toolbar + HUD → affichage OpenCV
```

---

## Auteur

**Toumi Syrine**  
LinkedIn : https://www.linkedin.com/in/syrine-toumi-36a11425b/?skipRedirect=true  
 GitHub :  https://github.com/SYRINEto

