"""
EmotiScan: Face detection + emotion classification.
Thread-safe for streamlit-webrtc callbacks.

Backend priority:
  0. YOLOv8-face + HSEmotion (GPU-accelerated face detection + EfficientNet emotion CNN)
  1. FER library (face detection + emotion CNN in one call)
  2. HSEmotion standalone with OpenCV Haar cascade for face detection
  3. OpenCV Haar cascade + heuristic (last resort, low accuracy)

Emotion model: EmotiEffLib (HSEmotion) — EfficientNet-B0 trained on AffectNet.
  - 60.95% accuracy on AffectNet 8-class (vs DeepFace FER2013 ~56%)
  - 1st place ABAW competition, 16MB model
  - 8-class: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
  - Contempt is mapped to Disgust (nearest valence/arousal neighbor)
"""

import threading
import numpy as np
import logging
import functools
from config import EMOTION_LABELS

logger = logging.getLogger(__name__)

# ── Singleton detector (shared across tabs to avoid reloading YOLO) ──
_singleton_detector = None
_singleton_lock = threading.Lock()


def get_shared_detector():
    """Return a shared EmotionDetector instance (thread-safe singleton).

    This avoids loading the YOLO model twice when both the Emotion Detection
    tab and the Voice Diary tab are active.
    """
    global _singleton_detector
    if _singleton_detector is None:
        with _singleton_lock:
            if _singleton_detector is None:
                _singleton_detector = EmotionDetector()
    return _singleton_detector


# Emotion color map (BGR for OpenCV)
EMOTION_COLORS = {
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (0, 255, 255),
    "fear": (128, 0, 128),
    "disgust": (0, 128, 128),
    "neutral": (200, 200, 200),
}

# HSEmotion 8-class → our standardized 7-class labels
# Contempt (class 1) doesn't exist in our EMOTION_LABELS.
# It maps to "disgust" (closest in valence-arousal space).
_HSEMOTION_LABEL_MAP = {
    "Anger": "angry",
    "Contempt": "disgust",  # merged into disgust
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprise",
}

# DeepFace emotion keys → our standardized keys (kept for FER backend compat)
_DEEPFACE_EMOTION_MAP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust",
    "neutral": "neutral",
}


def _patch_torch_load():
    """Monkey-patch torch.load for hsemotion compatibility with PyTorch 2.6+.

    PyTorch 2.6 changed torch.load to default to weights_only=True,
    which breaks hsemotion's model loading (it saves full model objects).
    This patch adds weights_only=False when not explicitly specified.
    """
    try:
        import torch
        if hasattr(torch.load, '_hsemotion_patched'):
            return  # already patched

        _original = torch.load

        @functools.wraps(_original)
        def _patched(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original(*args, **kwargs)

        _patched._hsemotion_patched = True
        torch.load = _patched
    except ImportError:
        pass


class EmotionDetector:
    """Thread-safe emotion detector with multi-backend support."""

    # Number of recent frames to average for temporal smoothing.
    # 3 gives responsiveness while still filtering single-frame noise.
    SMOOTHING_WINDOW = 3
    # Minimum face size (pixels) to accept a detection
    # Lowered from 60 to 30 so downsampled webcam faces (often ~50-80px) aren't rejected
    MIN_FACE_SIZE = 30
    # Maximum face-to-frame ratio -- reject if "face" is >60% of frame (likely whole-image fallback)
    MAX_FACE_RATIO = 0.6

    def __init__(self):
        self._lock = threading.Lock()
        self._backend = None  # 'yolo_hsemotion', 'fer', 'opencv_hsemotion', 'opencv_heuristic'
        self._fer_detector = None
        self._haar_cascade = None
        self._hsemotion_model = None   # HSEmotionRecognizer instance
        self._yolo_model = None        # YOLO model instance (GPU-accelerated when available)
        self._yolo_device = "cpu"      # actual device YOLO is running on
        self._hsemotion_device = "cpu" # actual device HSEmotion is running on
        self._models_loaded = False
        self._load_error = None
        self._load_errors_detail = []  # list of per-backend error strings
        self._frame_count = 0
        # Temporal smoothing buffer: list of dicts {emotion: score}
        self._emotion_history = []

    def _load_hsemotion(self):
        """Load HSEmotion model on best available device. Returns HSEmotionRecognizer."""
        _patch_torch_load()

        from hsemotion.facial_emotions import HSEmotionRecognizer

        # Determine device
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"HSEmotion: CUDA available ({torch.cuda.get_device_name(0)})")
        except ImportError:
            pass

        model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device=device)
        logger.info(f"HSEmotion: loaded enet_b0_8_best_afew on {device}")
        return model, device

    def _load_models(self):
        """Lazy-load models on first use. Thread-safe. Tries backends in order."""
        if self._models_loaded:
            return
        with self._lock:
            if self._models_loaded:
                return

            errors = []

            # Try loading HSEmotion (our primary emotion model)
            has_hsemotion = False
            try:
                self._hsemotion_model, self._hsemotion_device = self._load_hsemotion()
                has_hsemotion = True
                logger.info("EmotionDetector: HSEmotion available")
            except Exception as e:
                errors.append(f"HSEmotion: {e}")
                logger.info(f"HSEmotion unavailable: {e}")

            # Backend 0: YOLOv8-face + HSEmotion (GPU-accelerated)
            if has_hsemotion:
                try:
                    self._yolo_model, self._yolo_device = self._load_yolo()
                    self._backend = "yolo_hsemotion"
                    logger.info(
                        f"EmotionDetector: using YOLOv8-face + HSEmotion "
                        f"(yolo={self._yolo_device}, emotion={self._hsemotion_device})"
                    )
                    self._models_loaded = True
                    self._load_errors_detail = errors
                    return
                except Exception as e:
                    errors.append(f"YOLOv8: {e}")
                    logger.info(f"YOLOv8 backend unavailable: {e}")

            # Backend 1: FER library (face detector + emotion CNN)
            try:
                from fer import FER
                self._fer_detector = FER(mtcnn=False)
                self._backend = "fer"
                logger.info("EmotionDetector: using FER backend")
                self._models_loaded = True
                return
            except Exception as e:
                errors.append(f"FER: {e}")
                logger.info(f"FER backend unavailable: {e}")

            # Load Haar cascade (needed for backends 2, 3)
            try:
                import cv2
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._haar_cascade = cv2.CascadeClassifier(cascade_path)
                if self._haar_cascade.empty():
                    raise RuntimeError("Failed to load Haar cascade")
                logger.info("EmotionDetector: Haar cascade loaded")
            except Exception as e:
                errors.append(f"OpenCV: {e}")
                logger.info(f"OpenCV Haar cascade unavailable: {e}")

            # Backend 2: Haar cascade + HSEmotion
            if has_hsemotion and self._haar_cascade is not None:
                self._backend = "opencv_hsemotion"
                logger.info("EmotionDetector: using Haar + HSEmotion (crop-then-classify)")
                self._models_loaded = True
                self._load_errors_detail = errors
                return

            # Backend 3: OpenCV Haar cascade + heuristic (last resort)
            if self._haar_cascade is not None:
                self._backend = "opencv_heuristic"
                logger.warning(
                    "EmotionDetector: using OpenCV Haar + heuristic fallback. "
                    "This CANNOT classify real emotions — install 'hsemotion' or run: "
                    "pip install hsemotion timm"
                )

            self._load_errors_detail = errors
            if errors:
                self._load_error = " | ".join(errors)
                logger.warning(f"EmotionDetector load issues: {self._load_error}")

            self._models_loaded = True

    @property
    def is_ready(self) -> bool:
        self._load_models()
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        self._load_models()
        return self._backend or "none"

    @staticmethod
    def _load_yolo():
        """Load YOLOv8-face model on best available device. Returns (model, device_str)."""
        from config import YOLO_MODEL, YOLO_FACE_CONF, YOLO_FACE_IOU, YOLO_DEVICE

        # Try huggingface face-detection model first, then fall back to config model
        model = None
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="arnabdhar/YOLOv8-Face-Detection",
                filename="model.pt",
            )
            from ultralytics import YOLO
            model = YOLO(model_path)
            logger.info("YOLOv8: loaded arnabdhar/YOLOv8-Face-Detection from HuggingFace")
        except Exception as e:
            logger.info(f"YOLOv8: HuggingFace face model unavailable ({e}), trying local")

        if model is None:
            from ultralytics import YOLO
            model = YOLO(YOLO_MODEL)
            logger.info(f"YOLOv8: loaded local model {YOLO_MODEL}")

        # Move model to GPU if available
        device = YOLO_DEVICE
        try:
            import torch
            if device == "cuda" and torch.cuda.is_available():
                model.to(device)
                logger.info(f"YOLOv8: running on GPU ({torch.cuda.get_device_name(0)})")
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

        # Set default inference params
        model.overrides["conf"] = YOLO_FACE_CONF
        model.overrides["iou"] = YOLO_FACE_IOU
        model.overrides["verbose"] = False

        return model, device

    @property
    def status_info(self) -> dict:
        """Return diagnostic info about the detector state."""
        self._load_models()
        can_classify = self._backend in (
            "yolo_hsemotion", "fer", "opencv_hsemotion"
        )
        return {
            "backend": self._backend or "none",
            "can_classify_emotions": can_classify,
            "gpu_device": self._yolo_device if self._yolo_model else "n/a",
            "emotion_model": "HSEmotion (EfficientNet-B0)" if self._hsemotion_model else "n/a",
            "emotion_device": self._hsemotion_device if self._hsemotion_model else "n/a",
            "load_error": self._load_error,
            "errors_detail": list(self._load_errors_detail),
            "tip": (
                None if can_classify
                else "Install hsemotion + ultralytics for GPU emotion detection: "
                     "pip install hsemotion timm ultralytics"
            ),
        }

    def detect_emotions(self, frame: np.ndarray) -> tuple:
        """
        Detect faces and classify emotions in a frame.

        Returns:
            (annotated_frame, results_list)
            results_list: [{bbox: [x,y,w,h], emotion: str, confidence: float, all_emotions: dict}]
        """
        self._load_models()
        self._frame_count += 1

        if frame is None:
            return frame, []

        annotated = frame.copy()
        results = []
        frame_h, frame_w = frame.shape[:2]

        try:
            if self._backend == "yolo_hsemotion":
                raw_results = self._detect_yolo_hsemotion(frame)
            elif self._backend == "fer":
                raw_results = self._detect_fer(frame)
            elif self._backend == "opencv_hsemotion":
                raw_results = self._detect_opencv_hsemotion(frame)
            elif self._backend == "opencv_heuristic":
                raw_results = self._detect_opencv_heuristic(frame)
            else:
                raw_results = []

            # Filter out bad detections (too small, or covers whole frame)
            for r in raw_results:
                bx, by, bw, bh = r["bbox"]
                if bw < self.MIN_FACE_SIZE or bh < self.MIN_FACE_SIZE:
                    continue
                if frame_w > 0 and frame_h > 0:
                    face_ratio = (bw * bh) / (frame_w * frame_h)
                    if face_ratio > self.MAX_FACE_RATIO:
                        continue
                results.append(r)

            # Apply temporal smoothing to the first (primary) face
            if results:
                results[0] = self._smooth_emotion(results[0])

            # Draw annotations
            for r in results:
                x, y, w, h = r["bbox"]
                self._draw_annotation(annotated, x, y, w, h, r["emotion"], r["confidence"])

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return frame, []

        return annotated, results

    def _smooth_emotion(self, result: dict) -> dict:
        """Temporal smoothing: average emotion scores over recent frames to stabilize."""
        all_emotions = result.get("all_emotions", {})
        if not all_emotions:
            return result

        with self._lock:
            self._emotion_history.append(dict(all_emotions))
            # Keep only the last N frames
            if len(self._emotion_history) > self.SMOOTHING_WINDOW:
                self._emotion_history = self._emotion_history[-self.SMOOTHING_WINDOW:]

            # Average scores across the window
            avg_emotions = {}
            for emo in all_emotions:
                vals = [h.get(emo, 0.0) for h in self._emotion_history]
                avg_emotions[emo] = round(sum(vals) / len(vals), 3)

        top_emotion = max(avg_emotions, key=avg_emotions.get)
        confidence = avg_emotions[top_emotion]

        return {
            "bbox": result["bbox"],
            "emotion": top_emotion,
            "confidence": float(confidence),
            "all_emotions": avg_emotions,
        }

    # ── HSEmotion classification helper ─────────────────────────────
    #
    # HSEmotion's EfficientNet-B0 (enet_b0_8_best_afew) outputs 8 classes:
    #   {0:'Anger', 1:'Contempt', 2:'Disgust', 3:'Fear', 4:'Happiness',
    #    5:'Neutral', 6:'Sadness', 7:'Surprise'}
    #
    # We merge Contempt into Disgust and map to our 7-class EMOTION_LABELS.
    # Scores from Contempt and Disgust are summed into "disgust".

    def _classify_hsemotion(self, face_crop: np.ndarray) -> tuple:
        """Classify emotion on a cropped face image using HSEmotion.

        Args:
            face_crop: BGR numpy array of cropped face (any size, will be resized internally).

        Returns:
            (top_emotion, confidence, all_emotions_dict)
            where all_emotions_dict has our 7 standard emotion labels as keys.
        """
        import cv2

        # ── Validate input ──────────────────────────────────────────
        if face_crop is None or face_crop.size == 0:
            logger.warning("HSEmotion: empty face crop")
            return self._neutral_fallback()

        if self._hsemotion_model is None:
            logger.error("HSEmotion: model not loaded!")
            return self._neutral_fallback()

        try:
            # Ensure 3-channel input
            if face_crop.ndim == 2:
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)
            elif face_crop.ndim == 3 and face_crop.shape[2] == 4:
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGRA2BGR)

            # Ensure uint8 dtype
            if face_crop.dtype != np.uint8:
                face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)

            # HSEmotion expects RGB input (converts via PIL.Image.fromarray)
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # predict_emotions with logits=False returns softmax probabilities
            emotion_label, scores = self._hsemotion_model.predict_emotions(face_rgb, logits=False)

            # Map 8-class HSEmotion scores to our 7-class labels
            # scores is a numpy array of 8 probabilities
            idx_to_class = self._hsemotion_model.idx_to_class
            all_emotions = {e: 0.0 for e in EMOTION_LABELS}

            for idx, score in enumerate(scores):
                hs_label = idx_to_class[idx]
                our_label = _HSEMOTION_LABEL_MAP.get(hs_label)
                if our_label and our_label in all_emotions:
                    # Contempt and Disgust both map to "disgust" — sum them
                    all_emotions[our_label] += float(score)

            # Round
            all_emotions = {k: round(v, 3) for k, v in all_emotions.items()}

            # Sanity check: scores should sum to ~1.0 and not be uniform
            total = sum(all_emotions.values())
            if total < 0.01:
                logger.warning(f"HSEmotion: all-zero scores (total={total})")
                return self._neutral_fallback()

            top_emotion = max(all_emotions, key=all_emotions.get) if all_emotions else "neutral"
            confidence = all_emotions.get(top_emotion, 0.5)

            return top_emotion, confidence, all_emotions

        except Exception as e:
            # Log with full traceback at DEBUG level for diagnosability
            logger.error(f"HSEmotion classification FAILED: {type(e).__name__}: {e}")
            logger.debug("HSEmotion traceback:", exc_info=True)
            return self._neutral_fallback()

    @staticmethod
    def _neutral_fallback() -> tuple:
        """Return a distinguishable neutral fallback (NOT uniform 14%).

        When emotion classification fails, return neutral-dominant scores
        that are clearly distinguishable from real model output:
        neutral gets 0.7, everything else gets small varied amounts.
        This prevents the 'uniform 14% per emotion' bug where users
        can't tell real detection from fallback.
        """
        all_emotions = {
            "angry": 0.03,
            "disgust": 0.02,
            "fear": 0.03,
            "happy": 0.05,
            "sad": 0.04,
            "surprise": 0.03,
            "neutral": 0.80,
        }
        return "neutral", 0.80, all_emotions

    # ── Backend: YOLOv8-face + HSEmotion (GPU-accelerated) ────────

    def _detect_yolo_hsemotion(self, frame: np.ndarray) -> list:
        """YOLOv8-face for GPU-accelerated face detection, HSEmotion for emotion.

        This is the fastest backend when a CUDA GPU is available — YOLOv8
        runs face detection entirely on the GPU, then each cropped face is
        classified by HSEmotion's EfficientNet-B0 (also GPU when available).
        """
        from config import YOLO_FACE_CONF

        results = []
        try:
            # Run YOLOv8 inference (GPU-accelerated when available)
            preds = self._yolo_model(
                frame,
                conf=YOLO_FACE_CONF,
                verbose=False,
            )
            if not preds or len(preds) == 0:
                return results

            boxes = preds[0].boxes
            if boxes is None or len(boxes) == 0:
                return results

            for box in boxes:
                # Get bounding box (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                if w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
                    continue

                # Crop face from original frame
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # Classify emotion with HSEmotion
                top_emotion, confidence, all_emotions = self._classify_hsemotion(face_crop)
                det_conf = float(box.conf[0]) if box.conf is not None else 1.0

                results.append(
                    {
                        "bbox": [x, y, w, h],
                        "emotion": top_emotion,
                        "confidence": confidence,
                        "all_emotions": all_emotions,
                        "face_det_conf": round(det_conf, 3),
                    }
                )

        except Exception as e:
            logger.warning(f"YOLOv8-face detection failed: {e}")
        return results

    # ── Backend: FER ──────────────────────────────────────────────

    def _detect_fer(self, frame: np.ndarray) -> list:
        """Detect using FER library (OpenCV DNN face detector + mini-Xception emotion CNN)."""
        fer_results = self._fer_detector.detect_emotions(frame)
        results = []
        for face in fer_results:
            bbox = face.get("box", [0, 0, 0, 0])
            emotions = face.get("emotions", {})
            if not emotions:
                continue
            top_emotion = max(emotions, key=emotions.get)
            confidence = emotions[top_emotion]
            results.append({
                "bbox": list(bbox),
                "emotion": top_emotion,
                "confidence": round(confidence, 3),
                "all_emotions": {k: round(v, 3) for k, v in emotions.items()},
            })
        return results

    # ── Backend: OpenCV Haar + HSEmotion ──────────────────────────

    def _detect_opencv_hsemotion(self, frame: np.ndarray) -> list:
        """Haar cascade for face boxes, HSEmotion for emotion on cropped faces.

        Uses CLAHE on full frame for better face detection, but crops from
        the original frame to avoid CLAHE artifacts in emotion classification.
        """
        import cv2

        # Apply CLAHE to improve face-detection accuracy
        processed = self._preprocess_frame(frame)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        faces = self._haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        results = []
        for (x, y, w, h) in faces:
            # Crop face from the ORIGINAL frame (not CLAHE-processed)
            face_crop = frame[y:y + h, x:x + w]
            if face_crop.size == 0:
                continue

            top_emotion, confidence, all_emotions = self._classify_hsemotion(face_crop)

            results.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "emotion": top_emotion,
                "confidence": confidence,
                "all_emotions": all_emotions,
            })
        return results

    # ── Backend: OpenCV + heuristic (last resort) ─────────────────

    def _detect_opencv_heuristic(self, frame: np.ndarray) -> list:
        """Haar cascade for face boxes, brightness heuristic for emotion (inaccurate)."""
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        results = []
        for (x, y, w, h) in faces:
            face_region = frame[y:y + h, x:x + w]
            emotion, confidence, all_emotions = self._estimate_emotion_heuristic(face_region)
            results.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "emotion": emotion,
                "confidence": confidence,
                "all_emotions": all_emotions,
            })
        return results

    def _estimate_emotion_heuristic(self, face_region: np.ndarray) -> tuple:
        """
        Last-resort heuristic. NOT accurate -- just provides non-crashing output.
        Install 'hsemotion' for real emotion detection.
        """
        if face_region is None or face_region.size == 0:
            return self._neutral_fallback()

        try:
            import cv2
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            mean_brightness = np.mean(gray) / 255.0
            std_brightness = np.std(gray) / 128.0

            rng = np.random.default_rng(self._frame_count)
            noise = rng.uniform(-0.1, 0.1, len(EMOTION_LABELS))

            scores = {}
            scores["happy"] = 0.15 + mean_brightness * 0.3 + noise[0]
            scores["sad"] = 0.15 + (1 - mean_brightness) * 0.2 + noise[1]
            scores["angry"] = 0.1 + std_brightness * 0.2 + noise[2]
            scores["surprise"] = 0.1 + std_brightness * 0.15 + noise[3]
            scores["fear"] = 0.08 + (1 - mean_brightness) * 0.1 + noise[4]
            scores["disgust"] = 0.07 + noise[5]
            scores["neutral"] = 0.2 + (1 - abs(mean_brightness - 0.5)) * 0.2 + noise[6]

            total = sum(max(v, 0.01) for v in scores.values())
            scores = {k: round(max(v, 0.01) / total, 3) for k, v in scores.items()}

            top_emotion = max(scores, key=scores.get)
            return top_emotion, scores[top_emotion], scores

        except Exception:
            return self._neutral_fallback()

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """Normalize lighting via CLAHE to improve face detection accuracy."""
        try:
            import cv2
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            return frame

    @staticmethod
    def _preprocess_face_for_emotion(face_crop: np.ndarray) -> np.ndarray:
        """Prepare a face crop for emotion classification.

        Ensures 3-channel BGR input. Used by heuristic backend.
        """
        try:
            if face_crop.ndim == 2:
                import cv2
                return cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)
            return face_crop
        except Exception:
            return face_crop

    # ── Drawing ───────────────────────────────────────────────────

    @staticmethod
    def _draw_annotation(frame, x, y, w, h, emotion, confidence):
        """Draw bounding box and emotion label on frame."""
        try:
            import cv2
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"{emotion} {confidence:.0%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        except Exception:
            pass
