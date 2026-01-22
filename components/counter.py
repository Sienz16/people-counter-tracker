"""
People Counter Component
Main counting logic: tracking, line crossing detection, counting
"""

import supervision as sv
from typing import Dict, Set, Optional
from dataclasses import dataclass, field

from .detector import YOLOXDetector
from .reid import ReIDDatabase
from .ui import UIRenderer, CounterStats

import config


@dataclass
class TrackedPerson:
    """Data for a tracked person."""
    positions: list = field(default_factory=list)
    features: Optional[object] = None


class PeopleCounter:
    """
    Main people counter class.
    Handles detection, tracking, counting, and Re-ID.
    """

    def __init__(self):
        """Initialize the people counter."""
        # Detection
        self.detector = YOLOXDetector(
            model_size=config.MODEL_SIZE,
            conf_thresh=config.CONF_THRESHOLD,
            nms_thresh=config.NMS_THRESHOLD
        )

        # Tracking
        self.tracker = sv.ByteTrack()

        # Re-ID
        self.reid_db = ReIDDatabase(threshold=config.SIMILARITY_THRESHOLD)

        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.6)

        # Counting state
        self.in_count = 0
        self.out_count = 0
        self.duplicate_blocked = 0

        # Tracking state
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.crossed_ids: Set[int] = set()

        # Configuration (mutable at runtime)
        self.line_x = config.FRAME_WIDTH // 2
        self.max_capacity = config.MAX_CAPACITY
        self.direction_normal = config.DIRECTION_NORMAL

        # Frame dimensions
        self.frame_width = config.FRAME_WIDTH
        self.frame_height = config.FRAME_HEIGHT

        # UI
        self.ui = UIRenderer(self.frame_width, self.frame_height)

    def update_frame_size(self, width: int, height: int):
        """Update frame dimensions and recenter line."""
        self.frame_width = width
        self.frame_height = height
        self.line_x = width // 2
        self.ui.update_dimensions(width, height)

    def process_frame(self, frame):
        """
        Process a single frame.

        Args:
            frame: BGR image (will be flipped/mirrored)

        Returns:
            Annotated frame with UI overlay
        """
        import cv2

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Detect people
        detections = self.detector.detect(frame)

        # Track detections
        detections = self.tracker.update_with_detections(detections)

        # Process each tracked person
        self._process_detections(frame, detections)

        # Create labels
        labels = []
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            labels = [f"#{tid}" for tid in detections.tracker_id]

        # First pass: annotate frame with boxes and labels
        frame = self.box_annotator.annotate(frame, detections)
        if labels:
            frame = self.label_annotator.annotate(frame, detections, labels=labels)

        # Render UI overlay (applies transparency)
        stats = self.get_stats()
        frame = self.ui.render(frame, stats, self.line_x)

        # Redraw boxes AND labels on top (so they're not transparent)
        frame = self.box_annotator.annotate(frame, detections)
        if labels:
            frame = self.label_annotator.annotate(frame, detections, labels=labels)

        return frame

    def _process_detections(self, frame, detections: sv.Detections):
        """Process detections for line crossing."""
        if detections.tracker_id is None:
            return

        for i, track_id in enumerate(detections.tracker_id):
            bbox = detections.xyxy[i]
            center_x = (bbox[0] + bbox[2]) / 2

            # Get or create tracked person
            if track_id not in self.tracked_persons:
                self.tracked_persons[track_id] = TrackedPerson()

            person = self.tracked_persons[track_id]

            # Update position history
            person.positions.append(center_x)
            if len(person.positions) > 10:
                person.positions.pop(0)

            # Update features
            features = self.reid_db.extract_features(frame, bbox)
            if features is not None:
                person.features = features

            # Check for line crossing
            self._check_crossing(track_id, person)

    def _check_crossing(self, track_id: int, person: TrackedPerson):
        """Check if person crossed the line."""
        positions = person.positions

        if len(positions) < 2 or track_id in self.crossed_ids:
            return

        prev_x = positions[-2]
        curr_x = positions[-1]

        # Determine crossing direction
        if self.direction_normal:
            crossing_in = prev_x > self.line_x and curr_x <= self.line_x
            crossing_out = prev_x < self.line_x and curr_x >= self.line_x
        else:
            crossing_in = prev_x < self.line_x and curr_x >= self.line_x
            crossing_out = prev_x > self.line_x and curr_x <= self.line_x

        if crossing_in:
            self._handle_crossing(track_id, person, is_in=True)
        elif crossing_out:
            self._handle_crossing(track_id, person, is_in=False)

    def _handle_crossing(self, track_id: int, person: TrackedPerson, is_in: bool):
        """Handle a line crossing event."""
        self.crossed_ids.add(track_id)
        direction = "IN" if is_in else "OUT"

        if person.features is not None:
            is_duplicate, similarity = self.reid_db.is_duplicate(person.features)

            if is_duplicate:
                self.duplicate_blocked += 1
                print(f"BLOCKED: Duplicate detected (similarity: {similarity:.2f})")
            else:
                if is_in:
                    self.in_count += 1
                else:
                    self.out_count += 1
                self.reid_db.add_person(person.features)
                print(f"{direction}: New person (Total: {self.in_count if is_in else self.out_count})")
        else:
            if is_in:
                self.in_count += 1
            else:
                self.out_count += 1
            print(f"{direction}: Person crossed (no features)")

    def get_stats(self) -> CounterStats:
        """Get current counter statistics."""
        return CounterStats(
            in_count=self.in_count,
            out_count=self.out_count,
            duplicate_blocked=self.duplicate_blocked,
            current_occupancy=max(0, self.in_count - self.out_count),
            max_capacity=self.max_capacity,
            direction_normal=self.direction_normal
        )

    def reset(self):
        """Reset all counts and tracking state."""
        self.in_count = 0
        self.out_count = 0
        self.duplicate_blocked = 0
        self.crossed_ids.clear()
        self.tracked_persons.clear()
        self.reid_db.clear()
        print(">>> ALL COUNTS RESET! <<<")

    def move_line(self, delta: int):
        """Move counting line by delta pixels."""
        self.line_x = max(100, min(self.frame_width - 100, self.line_x + delta))
        print(f"Line moved to: {self.line_x}")

    def adjust_capacity(self, delta: int):
        """Adjust max capacity by delta."""
        self.max_capacity = max(10, self.max_capacity + delta)
        print(f"Capacity: {self.max_capacity}")

    def swap_direction(self):
        """Swap IN/OUT direction."""
        self.direction_normal = not self.direction_normal
        direction_text = "Right->Left = IN" if self.direction_normal else "Left->Right = IN"
        print(f">>> Direction swapped: {direction_text} <<<")

    @property
    def current_occupancy(self) -> int:
        """Current occupancy count."""
        return max(0, self.in_count - self.out_count)

    @property
    def is_at_capacity(self) -> bool:
        """Check if at max capacity."""
        return self.current_occupancy >= self.max_capacity
