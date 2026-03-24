"""
Air Canvas - Dessiner dans l'air avec OpenCV et MediaPipe
Compatible MediaPipe 0.10+ (nouvelle API mediapipe.tasks)
"""

import cv2
import numpy as np
import time
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision



MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}

def download_model(filename, url):
    if os.path.exists(filename):
        print(f"[OK] {filename} deja present.")
        return True
    print(f"[DOWNLOAD] Telechargement de {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"[OK] {filename} telecharge.")
        return True
    except Exception as e:
        print(f"[ERREUR] Impossible de telecharger {filename} : {e}")
        return False

print("=" * 55)
print("   Air Canvas - Verification des modeles...")
print("=" * 55)

hand_ok = download_model("hand_landmarker.task", MODELS["hand_landmarker.task"])
face_ok = download_model("face_landmarker.task", MODELS["face_landmarker.task"])

if not hand_ok:
    print("[FATAL] Le modele Hand Landmarker est requis. Verifie ta connexion internet.")
    exit(1)

# CONFIGURATION

ERASER_RADIUS  = 40
PARTICLE_COUNT = 6
SCREENSHOT_DIR = "screenshots"

COLORS = {
    "Rouge":  (0,   0,   220),
    "Orange": (0,  140,  255),
    "Jaune":  (0,  220,  220),
    "Vert":   (0,  200,   80),
    "Bleu":   (220, 80,    0),
    "Violet": (200,  0,  200),
    "Blanc":  (255, 255,  255),
    "Gomme":  None,
}

BRUSH_SIZES = [3, 6, 12, 20]

# INIT MEDIAPIPE TASKS

# Hand Landmarker (LIVE_STREAM)
hand_result_global = None

def hand_callback(result, output_image, timestamp_ms):
    global hand_result_global
    hand_result_global = result

hand_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
    result_callback=hand_callback
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
print("[OK] Hand Landmarker initialise.")

# Face Landmarker (optionnel)
USE_FACE = False
face_landmarker = None
face_result_global = None

if face_ok:
    try:
        def face_callback(result, output_image, timestamp_ms):
            global face_result_global
            face_result_global = result

        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=face_callback
        )
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)
        USE_FACE = True
        print("[OK] Face Landmarker initialise.")
    except Exception as e:
        print(f"[WARN] Face Landmarker desactive : {e}")

# CLASSE PARTICULE

class Particle:
    def __init__(self, x, y, color):
        self.x     = float(x)
        self.y     = float(y)
        self.vx    = np.random.uniform(-3, 3)
        self.vy    = np.random.uniform(-4, -1)
        self.life  = 1.0
        self.r     = int(np.random.randint(2, 5))
        self.color = color

    def update(self):
        self.x    += self.vx
        self.y    += self.vy
        self.vy   += 0.2
        self.life -= 0.07
        return self.life > 0

    def draw(self, frame):
        if self.life <= 0:
            return
        c = tuple(int(v * self.life) for v in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), max(self.r, 1), c, -1, cv2.LINE_AA)

# CLASSE PRINCIPALE


class AirCanvas:

    def __init__(self, width=1280, height=720):
        self.w = width
        self.h = height

        self.canvas    = np.zeros((height, width, 3), dtype=np.uint8)
        self.particles = []

        self.prev_x = None
        self.prev_y = None

        self.color_name = "Rouge"
        self.color      = COLORS["Rouge"]
        self.brush_idx  = 1
        self.brush_size = BRUSH_SIZES[self.brush_idx]

        self.toolbar_h  = 90
        self.color_keys = list(COLORS.keys())
        self.col_w      = width // len(self.color_keys)

        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        self.last_screenshot_time = 0
        self.screenshot_cooldown  = 3.0
        self.screenshot_flash     = 0

        self.fps_time  = time.time()
        self.fps       = 0
        self.frame_cnt = 0

    # ── Detection gestes ──────────────────────

    def get_finger_state(self, landmarks):
        """Index levé seul = dessin | Index+Majeur = navigation"""
        index_up  = landmarks[8].y  < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        return index_up, middle_up

    def get_index_tip(self, landmarks):
        lm = landmarks[8]
        return int(lm.x * self.w), int(lm.y * self.h)

    def check_mouth_open(self, face_landmarks):
        """Bouche ouverte si écart lèvres > seuil"""
        if not face_landmarks:
            return False
        # landmarks 13 (lèvre sup intérieure) et 14 (lèvre inf intérieure)
        ul = face_landmarks[13].y
        ll = face_landmarks[14].y
        return (ll - ul) > 0.04

    # ── Dessin ────────────────────────────────

    def draw_stroke(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return
        if self.color_name == "Gomme":
            cv2.circle(self.canvas, (x, y), ERASER_RADIUS, (0, 0, 0), -1, cv2.LINE_AA)
            for _ in range(3):
                px = x + int(np.random.randint(-ERASER_RADIUS, ERASER_RADIUS))
                py = y + int(np.random.randint(-ERASER_RADIUS, ERASER_RADIUS))
                self.particles.append(Particle(px, py, (150, 150, 150)))
        else:
            cv2.line(self.canvas,
                     (self.prev_x, self.prev_y), (x, y),
                     self.color, self.brush_size, cv2.LINE_AA)
            for _ in range(PARTICLE_COUNT):
                px = x + int(np.random.randint(-self.brush_size, self.brush_size + 1))
                py = y + int(np.random.randint(-self.brush_size, self.brush_size + 1))
                self.particles.append(Particle(px, py, self.color))
        self.prev_x, self.prev_y = x, y

    def update_particles(self, frame):
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles:
            p.draw(frame)

    def clear_canvas(self):
        self.canvas[:] = 0
        self.particles.clear()
        self.prev_x = self.prev_y = None

    # ── Screenshot ────────────────────────────

    def take_screenshot(self, frame):
        now = time.time()
        if now - self.last_screenshot_time < self.screenshot_cooldown:
            return
        self.last_screenshot_time = now
        self.screenshot_flash = 8
        fname = os.path.join(SCREENSHOT_DIR, f"air_canvas_{int(now)}.png")
        cv2.imwrite(fname, frame)
        print(f"[CAPTURE] {fname}")

    # ── Toolbar ───────────────────────────────

    def draw_toolbar(self, frame):
        overlay = frame.copy()
      #  cv2.rectangle(overlay, (0, 0), (self.w, self.toolbar_h), (20, 20, 20), -1)
      #  poujr que la barre d'outils soit plus claire et visible 
        cv2.rectangle(overlay, (0, 0), (self.w, self.toolbar_h), (240, 240, 240), -1)

        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        for i, name in enumerate(self.color_keys):
            x1 = i * self.col_w
            x2 = x1 + self.col_w
            cy = self.toolbar_h // 2
            color_bgr = (80, 80, 80) if name == "Gomme" else COLORS[name]
            is_active = (name == self.color_name)

            cv2.rectangle(frame, (x1+4, 8), (x2-4, self.toolbar_h-8),
                          color_bgr, -1 if is_active else 2, cv2.LINE_AA)
            if is_active:
                cv2.rectangle(frame, (x1+2, 6), (x2-2, self.toolbar_h-6),
                              (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, name[:6], (x1+8, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Taille: {self.brush_size}px",
                    (self.w - 160, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "[+][-] taille  [C] effacer  [S] capture  [Q] quitter",
                    (self.w - 460, self.toolbar_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    def check_toolbar_click(self, x, y):
        if y > self.toolbar_h:
            return
        idx = x // self.col_w
        if 0 <= idx < len(self.color_keys):
            self.color_name = self.color_keys[idx]
            self.color      = COLORS[self.color_name]
            self.prev_x     = self.prev_y = None

    # ── HUD ───────────────────────────────────

    def draw_hud(self, frame, drawing_mode):
        self.frame_cnt += 1
        now = time.time()
        if now - self.fps_time >= 1.0:
            self.fps       = self.frame_cnt
            self.frame_cnt = 0
            self.fps_time  = now

        mode_label = "DESSIN" if drawing_mode else "NAVIGATION"
        mode_color = (0, 220, 80) if drawing_mode else (220, 180, 0)

        cv2.putText(frame, f"FPS: {self.fps}",
                    (10, self.toolbar_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Mode: {mode_label}",
                    (10, self.toolbar_h + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Index seul = dessin | Index+Majeur = navigation",
                    (10, self.h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
        if USE_FACE:
            cv2.putText(frame, "Bouche ouverte = screenshot auto",
                        (10, self.h - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        if self.screenshot_flash > 0:
            wh = np.ones_like(frame) * 255
            a  = self.screenshot_flash / 8
            cv2.addWeighted(wh, a * 0.4, frame, 1 - a * 0.4, 0, frame)
            cv2.putText(frame, "CAPTURE !",
                        (self.w // 2 - 80, self.h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 80), 3, cv2.LINE_AA)
            self.screenshot_flash -= 1

    def draw_cursor(self, frame, x, y):
        if self.color_name == "Gomme":
            cv2.circle(frame, (x, y), ERASER_RADIUS, (180, 180, 180), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), self.brush_size + 4, self.color, 2, cv2.LINE_AA)
            # pour que le curseur soit plus visible 
            cv2.circle(frame, (x, y), 3, (0, 0, 0), -1, cv2.LINE_AA)
           # cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
           
            


    # ── Boucle principale ─────────────────────

    def run(self):
        global hand_result_global, face_result_global

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        if not cap.isOpened():
            print("[ERREUR] Webcam introuvable.")
            return

        print("=" * 55)
        print("   Air Canvas demarre !  (Q pour quitter)")
        print("=" * 55)
        print("  ☝  Index seul leve    -> DESSIN")
        print("  ✌  Index + Majeur     -> NAVIGATION")
        if USE_FACE:
            print("  😮 Bouche ouverte     -> Screenshot auto")
        print("  C=effacer | S=capture | +/-=taille | Q=quitter")
        print("=" * 55)

        timestamp = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERREUR] Impossible de lire la webcam.")
                break

            frame = cv2.flip(frame, 1)
            h_frame, w_frame = frame.shape[:2]

            # Adapter si la webcam ne supporte pas 1280x720
            if w_frame != self.w or h_frame != self.h:
                self.w = w_frame
                self.h = h_frame
                self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                self.col_w  = self.w // len(self.color_keys)

            # Convertir en RGB pour MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp += 1
            hand_landmarker.detect_async(mp_image, timestamp)
            if USE_FACE and face_landmarker:
                face_landmarker.detect_async(mp_image, timestamp)

            # Lire les resultats du callback
            drawing_mode = False
            ix, iy = None, None

            if hand_result_global and hand_result_global.hand_landmarks:
                landmarks = hand_result_global.hand_landmarks[0]
                ix, iy = self.get_index_tip(landmarks)
                index_up, middle_up = self.get_finger_state(landmarks)

                if index_up and not middle_up:
                    drawing_mode = True
                    if iy > self.toolbar_h:
                        self.draw_stroke(ix, iy)
                    else:
                        self.check_toolbar_click(ix, iy)
                        self.prev_x = self.prev_y = None
                elif index_up and middle_up:
                    self.prev_x = self.prev_y = None
                    if iy <= self.toolbar_h:
                        self.check_toolbar_click(ix, iy)
                else:
                    self.prev_x = self.prev_y = None

            if USE_FACE and face_result_global and face_result_global.face_landmarks:
                face_lm = face_result_global.face_landmarks[0]
                if self.check_mouth_open(face_lm):
                    self.take_screenshot(frame)

            # Fusion canvas + frame
           # canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
           # _, mask     = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
           # mask_inv    = cv2.bitwise_not(mask)
           # bg          = cv2.bitwise_and(frame, frame, mask=mask_inv)
           # draw_layer  = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
           # frame       = cv2.add(bg, draw_layer)

           # self.update_particles(frame)
           # self.draw_toolbar(frame)

           # if ix is not None:
            #    self.draw_cursor(frame, ix, iy)

           # self.draw_hud(frame, drawing_mode)

           # cv2.imshow("Air Canvas", frame)

           # Tableau blanc — fond blanc au lieu de la vidéo webcam
            whiteboard = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255

            # Coller le dessin sur le fond blanc
            canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask     = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
            draw_layer  = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            whiteboard[mask > 0] = draw_layer[mask > 0]

            self.update_particles(whiteboard)
            self.draw_toolbar(whiteboard)

            if ix is not None:
                self.draw_cursor(whiteboard, ix, iy)

            self.draw_hud(whiteboard, drawing_mode)


            cv2.imshow("Tableau Blanc", whiteboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                self.clear_canvas()
            elif key == ord('s'):
                self.take_screenshot(frame)
                self.screenshot_flash = 8
            elif key in (ord('+'), ord('=')):
                self.brush_idx  = min(self.brush_idx + 1, len(BRUSH_SIZES) - 1)
                self.brush_size = BRUSH_SIZES[self.brush_idx]
            elif key == ord('-'):
                self.brush_idx  = max(self.brush_idx - 1, 0)
                self.brush_size = BRUSH_SIZES[self.brush_idx]

        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        if face_landmarker:
            face_landmarker.close()
        print("Air Canvas ferme.")
if __name__ == "__main__":
    app = AirCanvas(width=1280, height=720)
    app.run()