"""
YOLOX Detector Component
Apache 2.0 Licensed - Safe for commercial use
"""

import cv2
import numpy as np
import os
import requests
import onnxruntime as ort
import supervision as sv


class YOLOXDetector:
    """YOLOX person detector using ONNX Runtime."""

    MODELS = {
        "nano": "yolox_nano.onnx",
        "tiny": "yolox_tiny.onnx",
        "s": "yolox_s.onnx",
        "m": "yolox_m.onnx",
        "l": "yolox_l.onnx",
        "x": "yolox_x.onnx",
    }

    BASE_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"

    def __init__(
        self,
        model_size: str = "s",
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45
    ):
        """
        Initialize YOLOX detector.

        Args:
            model_size: Model size - nano, tiny, s, m, l, x
            conf_thresh: Confidence threshold
            nms_thresh: NMS threshold
        """
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = (640, 640)
        self.model_size = model_size

        # Generate grids for decoding
        self.grids, self.strides = self._generate_grids()

        model_path = self._ensure_model(model_size)
        self._load_model(model_path)

    def _generate_grids(self):
        """Generate grid coordinates for YOLOX decoding."""
        grids = []
        strides = []

        # YOLOX uses 3 feature maps with strides 8, 16, 32
        for stride in [8, 16, 32]:
            h = self.input_size[0] // stride
            w = self.input_size[1] // stride

            # Create grid
            yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid = np.stack([xv, yv], axis=2).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((h * w, 1), stride))

        grids = np.concatenate(grids, axis=0).astype(np.float32)
        strides = np.concatenate(strides, axis=0).astype(np.float32)

        return grids, strides

    def _ensure_model(self, model_size: str) -> str:
        """Download model if not present."""
        weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
        os.makedirs(weights_dir, exist_ok=True)

        model_file = self.MODELS.get(model_size)
        if not model_file:
            raise ValueError(f"Unknown model: {model_size}. Options: {list(self.MODELS.keys())}")

        model_path = os.path.join(weights_dir, model_file)

        if not os.path.exists(model_path):
            self._download_model(model_file, model_path)

        return model_path

    def _download_model(self, model_file: str, model_path: str):
        """Download model from GitHub."""
        print(f"Downloading YOLOX-{self.model_size} model...")
        url = f"{self.BASE_URL}/{model_file}"

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\rDownloading: {pct:.1f}%", end="", flush=True)

        print("\nDownload complete!")

    def _load_model(self, model_path: str):
        """Load ONNX model."""
        print(f"Loading YOLOX-{self.model_size} model...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"Model loaded! Using: {self.session.get_providers()[0]}")

    def _preprocess(self, frame: np.ndarray) -> tuple:
        """Preprocess image for YOLOX."""
        h, w = frame.shape[:2]
        ratio = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * ratio), int(w * ratio)

        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        img = padded.astype(np.float32).transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        return img, ratio

    def _decode_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """Decode YOLOX raw outputs to boxes."""
        # outputs shape: [8400, 85]
        # Decode box coordinates using grid and stride
        outputs[:, :2] = (outputs[:, :2] + self.grids) * self.strides
        outputs[:, 2:4] = np.exp(outputs[:, 2:4]) * self.strides
        return outputs

    def _postprocess(self, outputs, ratio: float) -> sv.Detections:
        """Convert YOLOX output to supervision Detections."""
        output = outputs[0]  # First output tensor

        # Remove batch dimension if present
        if len(output.shape) == 3:
            predictions = output[0]
        else:
            predictions = output

        if len(predictions) == 0:
            return sv.Detections.empty()

        # Decode the raw outputs
        predictions = self._decode_outputs(predictions.copy())

        # Extract components
        boxes = predictions[:, :4]
        obj_conf = predictions[:, 4]
        class_scores = predictions[:, 5:]

        # Final score = objectness * class_score
        scores = obj_conf[:, np.newaxis] * class_scores
        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        # Filter by confidence and person class (class 0)
        mask = (max_scores > self.conf_thresh) & (class_ids == 0)
        boxes = boxes[mask]
        max_scores = max_scores[mask]

        if len(boxes) == 0:
            return sv.Detections.empty()

        # Convert from center format to corner format and scale to original image
        x_center, y_center = boxes[:, 0], boxes[:, 1]
        width, height = boxes[:, 2], boxes[:, 3]

        x1 = (x_center - width / 2) / ratio
        y1 = (y_center - height / 2) / ratio
        x2 = (x_center + width / 2) / ratio
        y2 = (y_center + height / 2) / ratio

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Clip to image bounds (optional but safer)
        boxes = np.clip(boxes, 0, None)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            max_scores.tolist(),
            self.conf_thresh,
            self.nms_thresh
        )

        if len(indices) == 0:
            return sv.Detections.empty()

        indices = indices.flatten()
        return sv.Detections(
            xyxy=boxes[indices],
            confidence=max_scores[indices],
            class_id=np.zeros(len(indices), dtype=int)
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect persons in frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            sv.Detections with person detections
        """
        img, ratio = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: img})
        return self._postprocess(outputs, ratio)
