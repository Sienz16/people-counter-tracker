"""
Hybrid Re-Identification Component
Primary: Face recognition (ONNX models - no InsightFace dependency)
Fallback: Body features (color + texture + shape)
"""

import cv2
import numpy as np
import os
import requests
import zipfile
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import onnxruntime as ort

from .reid import ReIDDatabase
from config import SIMILARITY_THRESHOLD


@dataclass
class PersonFeatures:
    """Combined features for a person."""
    face_embedding: Optional[np.ndarray] = None
    body_features: Optional[dict] = None
    has_face: bool = False


def download_buffalo_models(weights_dir: str) -> Tuple[str, str]:
    """
    Download buffalo_sc model pack from InsightFace.

    Returns:
        Tuple of (det_model_path, rec_model_path)
    """
    os.makedirs(weights_dir, exist_ok=True)

    det_model = os.path.join(weights_dir, "det_500m.onnx")
    rec_model = os.path.join(weights_dir, "w600k_mbf.onnx")

    # Check if already downloaded
    if os.path.exists(det_model) and os.path.exists(rec_model):
        return det_model, rec_model

    # Download buffalo_sc.zip
    zip_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
    zip_path = os.path.join(weights_dir, "buffalo_sc.zip")

    if not os.path.exists(zip_path):
        print("Downloading face recognition models (buffalo_sc)...")
        print("This may take a few minutes (~180MB)...")

        response = requests.get(zip_url, stream=True)
        response.raise_for_status()

        total = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = (downloaded / total) * 100
                    print(f"\rDownloading: {pct:.1f}% ({downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB)", end="", flush=True)
        print("\nDownload complete!")

    # Extract needed files
    print("Extracting models...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List contents to find the model files
        for name in zf.namelist():
            if name.endswith("det_500m.onnx"):
                # Extract to weights dir
                with zf.open(name) as src, open(det_model, 'wb') as dst:
                    dst.write(src.read())
                print(f"  Extracted: det_500m.onnx")
            elif name.endswith("w600k_mbf.onnx"):
                with zf.open(name) as src, open(rec_model, 'wb') as dst:
                    dst.write(src.read())
                print(f"  Extracted: w600k_mbf.onnx")

    # Clean up zip file to save space
    if os.path.exists(det_model) and os.path.exists(rec_model):
        os.remove(zip_path)
        print("Extraction complete!")

    return det_model, rec_model


class FaceDetector:
    """SCRFD face detector using ONNX Runtime."""

    def __init__(self, model_path: str):
        self.input_size = (640, 640)
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load ONNX model."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        if input_shape[2] is not None and input_shape[3] is not None:
            self.input_size = (input_shape[2], input_shape[3])

        print(f"Face detector loaded ({self.session.get_providers()[0]})")

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect faces in image.

        Returns:
            List of face bboxes [x1, y1, x2, y2, score]
        """
        h, w = image.shape[:2]

        # Preprocess
        img, scale = self._preprocess(image)

        # Inference
        outputs = self.session.run(None, {self.input_name: img})

        # Postprocess
        faces = self._postprocess(outputs, scale, (w, h))
        return faces

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess image for SCRFD."""
        h, w = image.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((self.input_size[0], self.input_size[1], 3), 127, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Normalize and transpose
        img = (padded.astype(np.float32) - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]

        return img, scale

    def _postprocess(self, outputs, scale: float, orig_size: Tuple[int, int]) -> List[np.ndarray]:
        """Postprocess SCRFD outputs."""
        faces = []

        # SCRFD det_500m outputs: 9 tensors (3 scales x 3 outputs each: scores, bboxes, kps)
        # Structure: [scores_8, scores_16, scores_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
        num_outputs = len(outputs)

        if num_outputs == 9:
            # Standard SCRFD output
            fmc = 3
            for idx in range(fmc):
                scores = outputs[idx]
                bboxes = outputs[idx + fmc]

                if len(scores.shape) == 3:
                    scores = scores[0]
                if len(bboxes.shape) == 3:
                    bboxes = bboxes[0]

                # Get indices above threshold
                if scores.shape[-1] == 1:
                    score_vals = scores[:, 0]
                else:
                    score_vals = scores.flatten()

                pos_inds = np.where(score_vals > self.conf_threshold)[0]

                for i in pos_inds:
                    score = score_vals[i]

                    # Bboxes are in format [x1, y1, x2, y2] or [cx, cy, w, h]
                    if bboxes.shape[-1] == 4:
                        bbox = bboxes[i]
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    else:
                        continue

                    # Scale back to original
                    x1 = x1 / scale
                    y1 = y1 / scale
                    x2 = x2 / scale
                    y2 = y2 / scale

                    # Clip to image bounds
                    x1 = max(0, min(x1, orig_size[0]))
                    y1 = max(0, min(y1, orig_size[1]))
                    x2 = max(0, min(x2, orig_size[0]))
                    y2 = max(0, min(y2, orig_size[1]))

                    if x2 > x1 and y2 > y1:
                        faces.append(np.array([x1, y1, x2, y2, score]))

        if len(faces) == 0:
            return []

        # NMS
        faces = np.array(faces)
        keep = self._nms(faces[:, :4], faces[:, 4], self.nms_threshold)
        return [faces[i] for i in keep]

    def _nms(self, boxes, scores, thresh):
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= thresh)[0]
            order = order[inds + 1]

        return keep


class FaceRecognizer:
    """ArcFace face recognition using ONNX Runtime."""

    def __init__(self, model_path: str):
        self.input_size = (112, 112)
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load ONNX model."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"Face recognizer loaded ({self.session.get_providers()[0]})")

    def get_embedding(self, image: np.ndarray, face_bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from image.

        Args:
            image: Full image (BGR)
            face_bbox: Face bounding box [x1, y1, x2, y2, score]

        Returns:
            512-dim face embedding or None
        """
        # Crop face with margin
        x1, y1, x2, y2 = map(int, face_bbox[:4])
        h, w = image.shape[:2]

        # Add margin
        face_w = x2 - x1
        face_h = y2 - y1
        margin_x = int(face_w * 0.3)
        margin_y = int(face_h * 0.3)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return None

        # Resize to model input size
        face_resized = cv2.resize(face_crop, self.input_size)

        # Preprocess: BGR -> RGB, normalize, transpose
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_norm = (face_rgb.astype(np.float32) - 127.5) / 127.5
        face_input = face_norm.transpose(2, 0, 1)[np.newaxis, ...]

        # Get embedding
        embedding = self.session.run(None, {self.input_name: face_input})[0][0]

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class HybridReID:
    """
    Hybrid Re-ID system using face recognition with body fallback.

    - If face is detected: Use face embedding (high accuracy)
    - If no face: Fall back to body features (color/texture)
    """

    # Face similarity threshold (cosine similarity, higher = stricter)
    FACE_THRESHOLD = 0.4  # Typical range: 0.3-0.5

    def __init__(self, body_threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialize hybrid Re-ID system.

        Args:
            body_threshold: Threshold for body feature matching (fallback)
        """
        self.weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")

        # Initialize face models
        self.face_detector = None
        self.face_recognizer = None
        self._init_face_models()

        # Body Re-ID as fallback
        self.body_reid = ReIDDatabase(threshold=body_threshold)

        # Database of known people
        self.face_db: List[np.ndarray] = []  # Face embeddings
        self.body_db: List[dict] = []  # Body features (for people without faces)

        # Stats
        self.face_matches = 0
        self.body_matches = 0

    def _init_face_models(self):
        """Initialize face detection and recognition models."""
        try:
            det_model, rec_model = download_buffalo_models(self.weights_dir)
            self.face_detector = FaceDetector(det_model)
            self.face_recognizer = FaceRecognizer(rec_model)
            print("Face recognition system ready!")
        except Exception as e:
            print(f"WARNING: Failed to load face models: {e}")
            print("Falling back to body Re-ID only.")
            self.face_detector = None
            self.face_recognizer = None

    def extract_features(self, frame: np.ndarray, bbox: np.ndarray) -> PersonFeatures:
        """
        Extract features from a person detection.

        Args:
            frame: Full frame image (BGR)
            bbox: Person bounding box [x1, y1, x2, y2]

        Returns:
            PersonFeatures with face embedding and/or body features
        """
        features = PersonFeatures()

        # Try to extract face embedding
        if self.face_detector is not None and self.face_recognizer is not None:
            face_embedding = self._extract_face(frame, bbox)
            if face_embedding is not None:
                features.face_embedding = face_embedding
                features.has_face = True

        # Always extract body features (for fallback)
        features.body_features = self.body_reid.extract_features(frame, bbox)

        return features

    def _extract_face(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from person crop.

        Args:
            frame: Full frame
            bbox: Person bounding box

        Returns:
            Face embedding (512-dim) or None if no face found
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Expand bbox slightly to ensure face is included
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Crop person region
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return None

        # Detect faces in the person crop
        try:
            faces = self.face_detector.detect(person_crop)

            if len(faces) > 0:
                # Use the largest face (most likely the person)
                largest_face = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))

                # Get embedding
                embedding = self.face_recognizer.get_embedding(person_crop, largest_face)
                return embedding

        except Exception:
            # Face detection can fail for various reasons - fall back to body
            pass

        return None

    def _compute_face_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between face embeddings."""
        return float(np.dot(emb1, emb2))

    def is_duplicate(self, features: PersonFeatures) -> Tuple[bool, float, str]:
        """
        Check if person already exists in database.

        Args:
            features: PersonFeatures object

        Returns:
            Tuple of (is_duplicate, similarity_score, method_used)
        """
        # Strategy 1: If we have a face, use face matching (most accurate)
        if features.has_face and features.face_embedding is not None:
            is_dup, sim = self._check_face_duplicate(features.face_embedding)
            if is_dup:
                self.face_matches += 1
                return True, sim, "face"
            # If no face match, still might be a new person - continue to add
            return False, sim, "face"

        # Strategy 2: No face - use body features (fallback)
        if features.body_features is not None:
            is_dup, sim = self.body_reid.is_duplicate(features.body_features)
            if is_dup:
                self.body_matches += 1
                return True, sim, "body"
            return False, sim, "body"

        # No features at all
        return False, -1.0, "none"

    def _check_face_duplicate(self, face_embedding: np.ndarray) -> Tuple[bool, float]:
        """Check if face exists in database."""
        if len(self.face_db) == 0:
            return False, -1.0

        max_similarity = -1.0
        for stored_embedding in self.face_db:
            similarity = self._compute_face_similarity(face_embedding, stored_embedding)
            if similarity > max_similarity:
                max_similarity = similarity

        is_dup = max_similarity > self.FACE_THRESHOLD

        if is_dup:
            print(f"  [FACE DUPLICATE] Similarity: {max_similarity:.3f} > {self.FACE_THRESHOLD}")
        else:
            print(f"  [FACE NEW] Best similarity: {max_similarity:.3f} < {self.FACE_THRESHOLD}")

        return is_dup, max_similarity

    def add_person(self, features: PersonFeatures):
        """
        Add person to database.

        Args:
            features: PersonFeatures to store
        """
        # Prefer face if available
        if features.has_face and features.face_embedding is not None:
            self.face_db.append(features.face_embedding.copy())
            print(f"  Added face to database (total: {len(self.face_db)})")
        # Otherwise store body features
        elif features.body_features is not None:
            self.body_db.append(features.body_features)
            self.body_reid.add_person(features.body_features)
            print(f"  Added body features to database (total: {len(self.body_db)})")

    def clear(self):
        """Clear all databases."""
        self.face_db.clear()
        self.body_db.clear()
        self.body_reid.clear()
        self.face_matches = 0
        self.body_matches = 0

    @property
    def total_count(self) -> int:
        """Total unique people in database."""
        return len(self.face_db) + len(self.body_db)

    def get_stats(self) -> Dict:
        """Get Re-ID statistics."""
        return {
            "face_db_size": len(self.face_db),
            "body_db_size": len(self.body_db),
            "face_matches": self.face_matches,
            "body_matches": self.body_matches,
        }
