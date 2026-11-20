import sys
import os
import time
import ctypes
from ctypes import wintypes
import tempfile
import cv2
import mss
import numpy as np
from PIL import Image
from nudenet import NudeDetector
from imgutils.validate import nsfw_pred_score
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QRect, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QImage

# Fix Windows DPI scaling BEFORE QApplication starts
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

# Real-life NSFW thresholds (NudeNet)
CLASS_THRESHOLDS = {
    "FEMALE_GENITALIA_EXPOSED": 0.40,
    "MALE_GENITALIA_EXPOSED": 0.40,
    "FEMALE_BREAST_EXPOSED": 0.42,
    "BUTTOCKS_EXPOSED": 0.50,
    "ANUS_EXPOSED": 0.45,
}

# Anime/2D NSFW threshold (dghs-imgutils)
ANIME_HENTAI_THRESHOLD = 0.55  # Raise to 0.65 if normal anime triggers false blurs

GRID_ROWS = 3
GRID_COLS = 3
TILE_OVERLAP = 0.12
MIN_BOX_SIZE = 18
PADDING_PERCENT = 0.30
MIN_PADDING_PX = 25
HOLD_SECONDS = 0.35
CHANGE_THRESHOLD = 4.0

# Windows hotkey constants
WM_HOTKEY = 0x0312
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
VK_P = 0x50
VK_Q = 0x51
HOTKEY_PAUSE_ID = 1
HOTKEY_QUIT_ID = 2


def fast_blur(region):
    h, w = region.shape[:2]
    small_w = max(1, w // 10)
    small_h = max(1, h // 10)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    small = cv2.GaussianBlur(small, (7, 7), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class ScannerThread(QThread):
    patches_ready = pyqtSignal(list)

    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.running = True
        self.paused = False

    def generate_tiles(self, w_img, h_img):
        tiles = [(0, 0, w_img, h_img)]  # Index 0: full screen
        step_x = w_img / GRID_COLS
        step_y = h_img / GRID_ROWS
        ov_x = int(step_x * TILE_OVERLAP)
        ov_y = int(step_y * TILE_OVERLAP)

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                tx1 = max(0, int(c * step_x) - ov_x)
                ty1 = max(0, int(r * step_y) - ov_y)
                tx2 = min(w_img, int((c + 1) * step_x) + ov_x)
                ty2 = min(h_img, int((r + 1) * step_y) + ov_y)
                tiles.append((tx1, ty1, tx2, ty2))
        return tiles

    def detect_tile(self, detector, tile, use_numpy, temp_path):
        if use_numpy:
            return detector.detect(tile)
        cv2.imwrite(temp_path, tile, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return detector.detect(temp_path)

    def run(self):
        detector = NudeDetector()
        sct = mss.mss()
        temp_path = os.path.join(tempfile.gettempdir(), "nudenet_tile.jpg")

        use_numpy = False
        try:
            detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
            use_numpy = True
        except Exception:
            pass

        prev_thumbs = {}
        cached_dets = {}
        last_valid_patches = []
        last_detection_time = 0.0

        while self.running:
            if self.paused:
                self.patches_ready.emit([])
                time.sleep(0.1)
                continue

            raw = sct.grab(self.monitor)
            frame_bgr = np.array(raw)[:, :, :3]
            h_img, w_img, _ = frame_bgr.shape

            tiles = self.generate_tiles(w_img, h_img)

            for idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                if not self.running or self.paused:
                    break

                tile = frame_bgr[ty1:ty2, tx1:tx2]
                thumb = cv2.resize(tile, (32, 32), interpolation=cv2.INTER_AREA)
                prev = prev_thumbs.get(idx)
                if prev is not None:
                    if np.mean(cv2.absdiff(thumb, prev)) < CHANGE_THRESHOLD:
                        continue

                prev_thumbs[idx] = thumb
                detections = self.detect_tile(detector, tile, use_numpy, temp_path)

                tile_hits = []
                for det in detections:
                    cls = det["class"]
                    score = float(det["score"])
                    threshold = CLASS_THRESHOLDS.get(cls)
                    if threshold is not None and score >= threshold:
                        x, y, w, h = det["box"]
                        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                            continue
                        tile_hits.append(
                            ([int(x + tx1), int(y + ty1), int(w), int(h)], score, False)
                        )

                # If NudeNet found nothing in this grid tile, check for anime NSFW
                if idx > 0 and not tile_hits:
                    pil_tile = Image.fromarray(tile[:, :, ::-1])
                    scores = nsfw_pred_score(pil_tile)
                    hentai_score = float(scores.get("hentai", 0.0))
                    if hentai_score >= ANIME_HENTAI_THRESHOLD:
                        tile_hits.append(
                            ([tx1, ty1, tx2 - tx1, ty2 - ty1], hentai_score, True)
                        )

                cached_dets[idx] = tile_hits

            raw_boxes, raw_scores, is_tile_flags = [], [], []
            for hits in cached_dets.values():
                for box, score, is_tile in hits:
                    raw_boxes.append(box)
                    raw_scores.append(score)
                    is_tile_flags.append(is_tile)

            new_patches = []
            if raw_boxes:
                indices = cv2.dnn.NMSBoxes(raw_boxes, raw_scores, 0.35, 0.4)
                if len(indices) > 0:
                    for i in np.array(indices).flatten():
                        x, y, w, h = raw_boxes[i]

                        if is_tile_flags[i]:
                            x1, y1, x2, y2 = x, y, x + w, y + h
                        else:
                            pad_x = max(MIN_PADDING_PX, int(w * PADDING_PERCENT))
                            pad_y = max(MIN_PADDING_PX, int(h * PADDING_PERCENT))
                            x1 = max(0, x - pad_x)
                            y1 = max(0, y - pad_y)
                            x2 = min(w_img, x + w + pad_x)
                            y2 = min(h_img, y + h + pad_y)

                        region = frame_bgr[y1:y2, x1:x2]
                        if region.size == 0:
                            continue

                        blurred = fast_blur(region)
                        blurred_rgb = np.ascontiguousarray(blurred[:, :, ::-1])

                        rh, rw, ch = blurred_rgb.shape
                        q_img = QImage(
                            blurred_rgb.data, rw, rh, ch * rw, QImage.Format_RGB888
                        ).copy()
                        new_patches.append((QRect(x1, y1, rw, rh), q_img))

            now = time.time()
            if new_patches:
                last_valid_patches = new_patches
                last_detection_time = now
                self.patches_ready.emit(new_patches)
            elif now - last_detection_time < HOLD_SECONDS:
                self.patches_ready.emit(last_valid_patches)
            else:
                self.patches_ready.emit([])

    def stop(self):
        self.running = False
        self.wait()


class BlurOverlay(QWidget):
    def __init__(self):
        super().__init__()
        with mss.mss() as sct:
            self.monitor = sct.monitors[1]

        self.blurred_patches = []

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(
            self.monitor["left"],
            self.monitor["top"],
            self.monitor["width"],
            self.monitor["height"],
        )

        if sys.platform == "win32":
            hwnd = int(self.winId())
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)

            ctypes.windll.user32.RegisterHotKey(
                hwnd, HOTKEY_PAUSE_ID, MOD_CONTROL | MOD_SHIFT, VK_P
            )
            ctypes.windll.user32.RegisterHotKey(
                hwnd, HOTKEY_QUIT_ID, MOD_CONTROL | MOD_SHIFT, VK_Q
            )

        self.show()

        self.scanner = ScannerThread(self.monitor)
        self.scanner.patches_ready.connect(self.update_patches)
        self.scanner.start()

    def nativeEvent(self, eventType, message):
        if sys.platform == "win32":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY:
                if msg.wParam == HOTKEY_PAUSE_ID:
                    self.scanner.paused = not self.scanner.paused
                    if self.scanner.paused:
                        self.blurred_patches = []
                        self.update()
                elif msg.wParam == HOTKEY_QUIT_ID:
                    self.close()
        return super().nativeEvent(eventType, message)

    def update_patches(self, patches):
        self.blurred_patches = patches
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        for rect, q_img in self.blurred_patches:
            painter.drawImage(rect, q_img)

    def closeEvent(self, event):
        if sys.platform == "win32":
            hwnd = int(self.winId())
            ctypes.windll.user32.UnregisterHotKey(hwnd, HOTKEY_PAUSE_ID)
            ctypes.windll.user32.UnregisterHotKey(hwnd, HOTKEY_QUIT_ID)
        self.scanner.stop()
        QApplication.quit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = BlurOverlay()
    sys.exit(app.exec_())
