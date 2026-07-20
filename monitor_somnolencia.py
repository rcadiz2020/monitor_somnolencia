import sys
import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# ---------------------------------------------------------------------------
# Resolución y FPS
# ---------------------------------------------------------------------------
CAPTURE_W, CAPTURE_H = 640, 480
TARGET_FPS            = 20
FRAME_INTERVAL_MS     = 1000 // TARGET_FPS

# Haar sobre imagen reducida: 640×480 × 0.5 → 320×240
PROC_SCALE = 0.5

# Haar cada N frames
DETECT_EVERY_N = 3

# ---------------------------------------------------------------------------
# Parámetros EAR
# Con CALIBRACION_MODE=True el programa imprime los valores reales en consola.
# ---------------------------------------------------------------------------
EAR_THRESH       = 0.20
CALIBRACION_MODE = True   # cambiar a False una vez calibrado

EAR_CONSEC_TRIGGER = 4
EAR_CONSEC_RESET   = 3
CABECEO_FRAMES     = 20

# ---------------------------------------------------------------------------
# Corrección de pose de cabeza
# POSE_CORRECTION=True: rota los landmarks según la inclinación real de la
# cabeza antes de calcular EAR → funciona bien en ángulos ±30°.
# MAX_HEAD_TILT_DEG: si la inclinación supera este valor, el EAR se
# descarta (cabeza muy girada, medición no fiable).
# ---------------------------------------------------------------------------
POSE_CORRECTION    = True
MAX_HEAD_TILT_DEG  = 30.0   # ignorar EAR si la cabeza está muy inclinada

# ---------------------------------------------------------------------------
# Umbral EAR adaptativo
# ADAPTIVE_THRESH=True: el umbral se ajusta automáticamente al EAR "de
# referencia" del usuario (ojos abiertos) durante los primeros segundos.
# ADAPT_FRAMES: cuántos frames de "ojos abiertos" usar para la referencia.
# ADAPT_RATIO: fracción del EAR de referencia que marca "cerrado"
#   p.ej. 0.75 → umbral = 0.75 × EAR_ref  (típicamente 0.21–0.24)
# ---------------------------------------------------------------------------
ADAPTIVE_THRESH  = True
ADAPT_FRAMES     = 60    # ~3 s a 20 FPS
ADAPT_RATIO      = 0.75

LEFT_EYE_IDX  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]


def yuv420_to_bgr(yuv_frame):
    """
    Picamera2 entrega YUV420 con shape (H*3//2, W) = (720, 640).
    cv2.cvtColor espera shape (H*3//2, W) con COLOR_YUV2BGR_I420.
    """
    return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)


# ===========================================================================
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitor Somnolencia — 640×480 @ 20 FPS")
        self.setMinimumSize(700, 480)
        self.setStyleSheet("background:#0e0e0e; color:white;")

        self.stat_lbl = QLabel("Iniciando...")
        self.stat_lbl.setAlignment(Qt.AlignCenter)
        self.stat_lbl.setStyleSheet(
            "font-size:17px; font-weight:bold; padding:6px; color:yellow;")

        self.diag_lbl = QLabel("")
        self.diag_lbl.setAlignment(Qt.AlignCenter)
        self.diag_lbl.setStyleSheet("font-size:12px; color:#aaaaaa; padding:2px;")

        self.v_lbl = QLabel()
        self.v_lbl.setAlignment(Qt.AlignCenter)

        lay = QVBoxLayout()
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        lay.addWidget(self.stat_lbl)
        lay.addWidget(self.diag_lbl)
        lay.addWidget(self.v_lbl, stretch=1)
        c = QWidget()
        c.setLayout(lay)
        self.setCentralWidget(c)

        # --- Haar ---
        self.face_det    = cv2.CascadeClassifier("face.xml")
        self.profile_det = cv2.CascadeClassifier("profile.xml")
        if self.face_det.empty():
            print("[ERROR] face.xml no cargado")
        if self.profile_det.empty():
            print("[ERROR] profile.xml no cargado")

        # --- LBF ---
        self.facemark           = None
        self.facemark_available = False
        self._cargar_facemark("lbfmodel.yaml")

        # --- Cámara en YUV420 ---
        self.picam2 = None
        self._iniciar_camara()

        # --- Estado ---
        self.consec_cerrados   = 0
        self.consec_abiertos   = 0
        self.frames_sin_rostro = 0
        self.en_alerta         = False
        self.causa_alerta      = ""
        self.ultimo_ear        = None
        self._ear_log          = []

        self.detect_count = 0
        self.frame_count  = 0
        self._t0          = time.time()
        self._fps_real    = 0.0

        # Umbral adaptativo
        self._adapt_samples   = []   # EAR de ojos abiertos para calibración
        self._ear_thresh_live = EAR_THRESH  # umbral activo (se actualiza)
        self._adapt_done      = False

        self._cached_face_small = None
        self._cached_modo       = None

        # Tamaño de display fijo para evitar resize caro en cada frame
        self._disp_w = CAPTURE_W
        self._disp_h = CAPTURE_H

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.procesar_frame)
        self.timer.start(FRAME_INTERVAL_MS)

    # -----------------------------------------------------------------------
    def _iniciar_camara(self):
        try:
            self.picam2 = Picamera2()
            us  = 1_000_000 // TARGET_FPS
            cfg = self.picam2.create_video_configuration(
                main={"size": (CAPTURE_W, CAPTURE_H), "format": "YUV420"},
                controls={"FrameDurationLimits": (us, us)},
                buffer_count=2
            )
            self.picam2.configure(cfg)
            self.picam2.start()
            print(f"[INFO] Camara YUV420: {CAPTURE_W}x{CAPTURE_H} @ {TARGET_FPS} FPS")
        except Exception as e:
            print(f"[ERROR] Picamera2: {e}")
            self.picam2 = None

    # -----------------------------------------------------------------------
    def _cargar_facemark(self, ruta):
        if not hasattr(cv2, "face"):
            print("[WARNING] cv2.face no disponible")
            return
        ruta = os.path.abspath(ruta)
        if not os.path.isfile(ruta):
            print(f"[WARNING] lbfmodel.yaml no encontrado: {ruta}")
            return
        try:
            fm = cv2.face.createFacemarkLBF()
            fm.loadModel(ruta)
            self.facemark           = fm
            self.facemark_available = True
            print("[INFO] LBF cargado OK")
        except Exception as e:
            print(f"[ERROR] LBF: {e}")

    # -----------------------------------------------------------------------
    def _validar_rect(self, x, y, w, h, shape):
        fh, fw = shape[:2]
        return w > 0 and h > 0 and x >= 0 and y >= 0 \
               and (x + w) <= fw and (y + h) <= fh

    # -----------------------------------------------------------------------
    def _detectar_rostro(self, gray_small):
        kw_f = dict(scaleFactor=1.1, minNeighbors=4,
                    minSize=(30, 30), maxSize=(300, 300))
        kw_p = dict(scaleFactor=1.1, minNeighbors=3,
                    minSize=(30, 30), maxSize=(300, 300))

        faces = self.face_det.detectMultiScale(gray_small, **kw_f)
        if len(faces):
            return max(faces, key=lambda r: r[2] * r[3]), "Frontal"

        profs = self.profile_det.detectMultiScale(gray_small, **kw_p)
        if len(profs):
            return profs[0], "Perfil"

        flip   = cv2.flip(gray_small, 1)
        profs2 = self.profile_det.detectMultiScale(flip, **kw_p)
        if len(profs2):
            x, y, w, h = profs2[0]
            return [gray_small.shape[1] - (x + w), y, w, h], "Perfil"

        return None, None

    # -----------------------------------------------------------------------
    def _ear(self, pts, idxs):
        p = pts[idxs]
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return (A + B) / (2.0 * C + 1e-6)

    # -----------------------------------------------------------------------
    def _head_tilt_deg(self, pts):
        """
        Calcula el ángulo de inclinación lateral de la cabeza en grados,
        usando los centros de los dos ojos como referencia horizontal.
        Positivo = cabeza inclinada a la derecha (desde el punto de vista
        del sistema), negativo = izquierda.
        """
        left_center  = pts[np.array(LEFT_EYE_IDX)].mean(axis=0)
        right_center = pts[np.array(RIGHT_EYE_IDX)].mean(axis=0)
        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    def _rotar_puntos(self, pts, angulo_deg, centro):
        """
        Rota un array Nx2 de puntos `angulo_deg` grados alrededor de `centro`.
        """
        rad   = np.radians(-angulo_deg)           # corrección: quitar la inclinación
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        R     = np.array([[cos_a, -sin_a],
                           [sin_a,  cos_a]], dtype=np.float32)
        return (pts - centro) @ R.T + centro

    # -----------------------------------------------------------------------
    def _evaluar_landmarks(self, gray_full, face_small, frame_bgr):
        """
        face_small: rect en coordenadas de gray_small (PROC_SCALE).
        gray_full y frame_bgr: escala completa (CAPTURE_W x CAPTURE_H).
        """
        if not self.facemark_available:
            return None, None

        inv = 1.0 / PROC_SCALE
        x = int(face_small[0] * inv)
        y = int(face_small[1] * inv)
        w = int(face_small[2] * inv)
        h = int(face_small[3] * inv)

        pad = int(h * 0.20)
        y1  = max(0, y - pad)
        h1  = min(gray_full.shape[0] - y1, h + pad * 2)

        if not self._validar_rect(x, y1, w, h1, gray_full.shape):
            if not self._validar_rect(x, y, w, h, gray_full.shape):
                return None, None
            y1, h1 = y, h

        faces_np = np.array([[x, y1, w, h1]], dtype=np.int32)

        try:
            ok, landmarks = self.facemark.fit(gray_full, faces_np)
        except Exception as e:
            print(f"[DBG] LBF fit error: {e}")
            return None, None

        if not ok or not landmarks or len(landmarks) == 0:
            return None, None

        try:
            pts = landmarks[0].reshape(-1, 2).astype(np.float32)
        except Exception:
            return None, None

        if pts.shape[0] < 48:
            return None, None

        # --- Corrección de pose de cabeza ---
        tilt = self._head_tilt_deg(pts)

        if POSE_CORRECTION and abs(tilt) > MAX_HEAD_TILT_DEG:
            # Inclinación excesiva: medición no fiable, ignorar
            if CALIBRACION_MODE:
                print(f"[POSE] Inclinación {tilt:+.1f}° > {MAX_HEAD_TILT_DEG}° — EAR descartado")
            return None, None

        pts_eval = pts
        if POSE_CORRECTION and abs(tilt) > 1.0:
            # Centro de rotación: punto medio entre los dos ojos
            centro = (pts[np.array(LEFT_EYE_IDX)].mean(axis=0) +
                      pts[np.array(RIGHT_EYE_IDX)].mean(axis=0)) / 2.0
            pts_eval = self._rotar_puntos(pts, tilt, centro)

        try:
            ear_l = self._ear(pts_eval, np.array(LEFT_EYE_IDX,  dtype=np.int32))
            ear_r = self._ear(pts_eval, np.array(RIGHT_EYE_IDX, dtype=np.int32))
        except Exception:
            return None, None

        ear = (ear_l + ear_r) / 2.0

        if not (0.02 <= ear <= 0.60):
            return None, None

        # --- Umbral adaptativo ---
        umbral = self._ear_thresh_live
        if ADAPTIVE_THRESH and not self._adapt_done:
            # Acumular muestras de ojos abiertos (EAR alto = ojos abiertos)
            if ear > 0.23:   # valor mínimo para considerar "claramente abierto"
                self._adapt_samples.append(ear)
            if len(self._adapt_samples) >= ADAPT_FRAMES:
                ref = float(np.percentile(self._adapt_samples, 25))  # percentil bajo del "abierto"
                self._ear_thresh_live = round(ref * ADAPT_RATIO, 3)
                self._adapt_done = True
                print(f"[ADAPT] Calibración completada: EAR_ref={ref:.3f}  "
                      f"umbral_adaptado={self._ear_thresh_live:.3f}")

        # Dibujar landmarks oculares
        color = (0, 80, 255) if ear < umbral else (0, 255, 100)
        for i in (*LEFT_EYE_IDX, *RIGHT_EYE_IDX):
            cv2.circle(frame_bgr, (int(pts[i][0]), int(pts[i][1])), 3, color, -1)

        ex = int(pts[RIGHT_EYE_IDX[3]][0]) + 6
        ey = int(pts[RIGHT_EYE_IDX[3]][1])
        cv2.putText(frame_bgr, f"EAR {ear:.2f}", (ex, ey),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if CALIBRACION_MODE:
            adapt_str = f"adapt={umbral:.3f}" if ADAPTIVE_THRESH else ""
            print(f"[POSE] tilt={tilt:+.1f}°  EAR={ear:.3f}  {adapt_str}")

        return bool(ear < umbral), ear

    # -----------------------------------------------------------------------
    def _actualizar_alerta_ear(self, ojo_cerrado):
        if ojo_cerrado:
            self.consec_cerrados += 1
            self.consec_abiertos  = 0
        else:
            self.consec_abiertos += 1
            self.consec_cerrados  = 0

        if not self.en_alerta and self.consec_cerrados >= EAR_CONSEC_TRIGGER:
            self.en_alerta    = True
            self.causa_alerta = "EAR"
            print(f"[ALERTA] SOMNOLENCIA (EAR={self.ultimo_ear:.3f})")

        if self.en_alerta and self.causa_alerta == "EAR":
            if self.consec_abiertos >= EAR_CONSEC_RESET:
                self.en_alerta    = False
                self.causa_alerta = ""
                self.consec_cerrados = 0

    # -----------------------------------------------------------------------
    def procesar_frame(self):
        if self.picam2 is None:
            self.stat_lbl.setText("ERROR: camara no disponible")
            return

        try:
            yuv = self.picam2.capture_array()
        except Exception as e:
            print(f"[WARNING] capture_array: {e}")
            return
        if yuv is None:
            return

        # YUV420 (720,640) → BGR (480,640,3)
        frame_bgr = yuv420_to_bgr(yuv)

        # Contadores FPS
        self.detect_count += 1
        self.frame_count  += 1
        now     = time.time()
        elapsed = now - self._t0
        if elapsed >= 2.0:
            self._fps_real   = self.frame_count / elapsed
            self.frame_count = 0
            self._t0         = now

        # ===================================================================
        # PASO 1 — Haar cada DETECT_EVERY_N frames
        # ===================================================================
        if self.detect_count % DETECT_EVERY_N == 0 or self._cached_face_small is None:
            small = cv2.resize(frame_bgr, (0, 0),
                               fx=PROC_SCALE, fy=PROC_SCALE,
                               interpolation=cv2.INTER_LINEAR)
            # Extraer canal Y directamente del YUV en vez de convertir BGR→GRAY
            # yuv shape (720,640): las primeras 480 filas son el plano Y
            gray_small = yuv[:CAPTURE_H, :].copy()
            gray_small = cv2.resize(gray_small,
                                    (int(CAPTURE_W * PROC_SCALE),
                                     int(CAPTURE_H * PROC_SCALE)),
                                    interpolation=cv2.INTER_LINEAR)
            cv2.equalizeHist(gray_small, gray_small)

            face_small, modo = self._detectar_rostro(gray_small)
            self._cached_face_small = face_small
            self._cached_modo       = modo

        face_small = self._cached_face_small
        modo       = self._cached_modo

        # ===================================================================
        # PASO 2 — Cabeceo
        # ===================================================================
        if face_small is None or modo != "Frontal":
            self.frames_sin_rostro += 1
            if self.en_alerta and self.causa_alerta == "EAR" \
                    and self.frames_sin_rostro > CABECEO_FRAMES * 2:
                self.en_alerta    = False
                self.causa_alerta = ""
            if self.frames_sin_rostro >= CABECEO_FRAMES:
                self.en_alerta    = True
                self.causa_alerta = "CABECEO"
        else:
            self.frames_sin_rostro = 0
            if self.en_alerta and self.causa_alerta == "CABECEO":
                self.en_alerta    = False
                self.causa_alerta = ""

        # ===================================================================
        # PASO 3 — LBF + EAR (solo con frontal)
        # ===================================================================
        if face_small is not None and modo == "Frontal":
            # Usar plano Y del YUV directamente como gray_full (sin cvtColor)
            gray_full = yuv[:CAPTURE_H, :].copy()

            ojo_cerrado, ear = self._evaluar_landmarks(gray_full, face_small, frame_bgr)

            if ear is not None:
                self.ultimo_ear = ear
                self._ear_log.append(ear)
                if len(self._ear_log) > 30:
                    self._ear_log.pop(0)

                if CALIBRACION_MODE:
                    estado = "OJO_CERRADO" if ojo_cerrado else "ojo_abierto"
                    adapt_info = (f"  umbral_adapt={self._ear_thresh_live:.3f}"
                                  f"({'OK' if self._adapt_done else f'{len(self._adapt_samples)}/{ADAPT_FRAMES}'})"
                                  if ADAPTIVE_THRESH else "")
                    print(f"[EAR] {ear:.3f}  {estado}  "
                          f"cerr:{self.consec_cerrados}  "
                          f"min30={min(self._ear_log):.3f}  "
                          f"max30={max(self._ear_log):.3f}"
                          f"{adapt_info}")

            if ojo_cerrado is not None:
                self._actualizar_alerta_ear(ojo_cerrado)

        # ===================================================================
        # PASO 4 — Dibujar rostro
        # ===================================================================
        if face_small is not None:
            inv = 1.0 / PROC_SCALE
            rx  = int(face_small[0] * inv)
            ry  = int(face_small[1] * inv)
            rw  = int(face_small[2] * inv)
            rh  = int(face_small[3] * inv)
            rc  = (0, 0, 255) if self.en_alerta else \
                  ((0, 220, 0) if modo == "Frontal" else (30, 140, 255))
            cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), rc, 2)
            cv2.putText(frame_bgr, modo, (rx, ry - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rc, 2)

        # Barra EAR
        if self.ultimo_ear is not None:
            bar_x, bar_y, bar_w, bar_h = 10, 10, 180, 16
            fill      = int(np.clip(self.ultimo_ear / 0.40, 0, 1) * bar_w)
            color_bar = (0, 80, 255) if self.ultimo_ear < EAR_THRESH else (0, 200, 80)
            cv2.rectangle(frame_bgr, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            cv2.rectangle(frame_bgr, (bar_x, bar_y),
                          (bar_x + fill, bar_y + bar_h), color_bar, -1)
            cv2.putText(frame_bgr, f"EAR {self.ultimo_ear:.3f}",
                        (bar_x + bar_w + 6, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bar, 1)

        # ===================================================================
        # PASO 5 — Etiquetas estado
        # ===================================================================
        if self.en_alerta:
            msg    = "!  CABECEO DETECTADO  !" if self.causa_alerta == "CABECEO" \
                     else "!  SOMNOLENCIA DETECTADA  !"
            estilo = ("color:red; font-size:19px; font-weight:bold; padding:6px;"
                      "background:#300000; border-radius:4px;")
        elif face_small is None:
            msg    = "Buscando rostro..."
            estilo = "color:yellow; font-size:16px; padding:6px;"
        elif modo == "Perfil":
            msg    = "Perfil — gira hacia la camara"
            estilo = "color:orange; font-size:15px; padding:6px;"
        else:
            msg    = "Despierto"
            estilo = "color:#00cc44; font-size:17px; font-weight:bold; padding:6px;"

        self.stat_lbl.setText(msg)
        self.stat_lbl.setStyleSheet(estilo)

        parts = []
        if self.ultimo_ear is not None:
            umbral_actual = self._ear_thresh_live if ADAPTIVE_THRESH else EAR_THRESH
            adapt_tag = "" if not ADAPTIVE_THRESH else \
                        ("✓adapt" if self._adapt_done else
                         f"cal {len(self._adapt_samples)}/{ADAPT_FRAMES}")
            parts.append(
                f"EAR {self.ultimo_ear:.3f} "
                f"({'BAJO' if self.ultimo_ear < umbral_actual else 'OK'} "
                f"umbral={umbral_actual:.3f}{' ' + adapt_tag if adapt_tag else ''})"
            )
        parts.append(
            f"cerr:{self.consec_cerrados}/{EAR_CONSEC_TRIGGER} "
            f"abie:{self.consec_abiertos}/{EAR_CONSEC_RESET}"
        )
        if self.frames_sin_rostro > 0:
            parts.append(f"sin_rostro:{self.frames_sin_rostro}/{CABECEO_FRAMES}")
        if self._fps_real > 0:
            parts.append(f"FPS:{self._fps_real:.1f}")
        if not self.facemark_available:
            parts.append("[SIN LBF]")
        self.diag_lbl.setText("   ".join(parts))

        # ===================================================================
        # PASO 6 — Display: resize solo si el widget cambió de tamaño
        # ===================================================================
        dw = self.v_lbl.width()  or CAPTURE_W
        dh = self.v_lbl.height() or CAPTURE_H

        if dw != self._disp_w or dh != self._disp_h:
            self._disp_w = dw
            self._disp_h = dh

        if self._disp_w == CAPTURE_W and self._disp_h == CAPTURE_H:
            disp = frame_bgr
        else:
            disp = cv2.resize(frame_bgr, (self._disp_w, self._disp_h),
                              interpolation=cv2.INTER_LINEAR)

        # BGR → RGB para Qt
        rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            np.ascontiguousarray(rgb).data,
            rgb.shape[1], rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888
        ).copy()
        self.v_lbl.setPixmap(QPixmap.fromImage(qimg))

    # -----------------------------------------------------------------------
    def closeEvent(self, e):
        self.timer.stop()
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        e.accept()


# ===========================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())