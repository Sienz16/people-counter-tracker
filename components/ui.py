"""
UI Drawing Component
Handles all visual elements: panels, stats, counting line, alerts
"""

import cv2
import time
import numpy as np
from dataclasses import dataclass

from config import Colors


@dataclass
class CounterStats:
    """Current counter statistics for display."""
    in_count: int
    out_count: int
    duplicate_blocked: int
    current_occupancy: int
    max_capacity: int
    direction_normal: bool


class UIRenderer:
    """Renders the UI overlay on the video frame."""

    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize UI renderer.

        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

    def update_dimensions(self, width: int, height: int):
        """Update frame dimensions."""
        self.frame_width = width
        self.frame_height = height

    def render(
        self,
        frame: np.ndarray,
        stats: CounterStats,
        line_x: int
    ) -> np.ndarray:
        """
        Render complete UI overlay.

        Args:
            frame: Video frame to draw on
            stats: Current counter statistics
            line_x: X position of counting line

        Returns:
            Frame with UI overlay
        """
        overlay = frame.copy()

        occupancy_percent = (stats.current_occupancy / stats.max_capacity * 100
                            if stats.max_capacity > 0 else 0)

        # Determine status
        status_color, status_text = self._get_status(occupancy_percent)

        # Draw UI elements
        self._draw_main_panel(overlay, stats, occupancy_percent, status_color, status_text)
        self._draw_counting_line(overlay, line_x, stats)
        self._draw_direction_arrows(overlay, line_x, stats.direction_normal)
        self._draw_help_bar(overlay)

        if stats.current_occupancy >= stats.max_capacity:
            self._draw_capacity_warning(overlay)

        # Apply transparency
        alpha = 0.85
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return result

    def _get_status(self, occupancy_percent: float) -> tuple:
        """Get status color and text based on occupancy."""
        if occupancy_percent < 70:
            return Colors.GREEN, "NORMAL"
        elif occupancy_percent < 90:
            return Colors.ORANGE, "BUSY"
        elif occupancy_percent < 100:
            return Colors.RED, "ALMOST FULL"
        else:
            return Colors.RED, "FULL"

    def _draw_main_panel(
        self,
        overlay: np.ndarray,
        stats: CounterStats,
        occupancy_percent: float,
        status_color: tuple,
        status_text: str
    ):
        """Draw the main statistics panel."""
        panel_x, panel_y = 15, 15
        panel_w, panel_h = 380, 360

        # Panel background
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), Colors.BG, -1)
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + 45), status_color, -1)

        # Header
        cv2.putText(overlay, "PEOPLE COUNTER", (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.WHITE, 2)

        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(overlay, status_text,
                   (panel_x + panel_w - status_size[0] - 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.WHITE, 1)

        # "INSIDE NOW" label
        cv2.putText(overlay, "INSIDE NOW", (panel_x + 15, panel_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.GRAY, 1)

        # Current occupancy - big number
        occ_y = panel_y + 140
        cv2.putText(overlay, str(stats.current_occupancy), (panel_x + 15, occ_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, status_color, 4)

        num_size = cv2.getTextSize(str(stats.current_occupancy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
        cv2.putText(overlay, f"/ {stats.max_capacity}",
                   (panel_x + 20 + num_size[0], occ_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, Colors.GRAY, 2)

        # Progress bar
        bar_y = occ_y + 25
        bar_w = panel_w - 30
        bar_h = 25
        cv2.rectangle(overlay, (panel_x + 15, bar_y),
                     (panel_x + 15 + bar_w, bar_y + bar_h), Colors.BG_LIGHT, -1)

        fill_w = int((min(occupancy_percent, 100) / 100) * bar_w)
        if fill_w > 0:
            cv2.rectangle(overlay, (panel_x + 15, bar_y),
                         (panel_x + 15 + fill_w, bar_y + bar_h), status_color, -1)

        cv2.rectangle(overlay, (panel_x + 15, bar_y),
                     (panel_x + 15 + bar_w, bar_y + bar_h), Colors.GRAY, 1)
        cv2.putText(overlay, f"{occupancy_percent:.0f}%",
                   (panel_x + bar_w - 30, bar_y + 19),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.WHITE, 1)

        # Divider
        div_y = bar_y + 45
        cv2.line(overlay, (panel_x + 15, div_y),
                (panel_x + panel_w - 15, div_y), Colors.BG_LIGHT, 2)

        # IN/OUT boxes
        stats_y = div_y + 40

        # IN box
        cv2.rectangle(overlay, (panel_x + 15, stats_y),
                     (panel_x + 180, stats_y + 60), Colors.BG_LIGHT, -1)
        cv2.rectangle(overlay, (panel_x + 15, stats_y),
                     (panel_x + 20, stats_y + 60), Colors.GREEN, -1)
        cv2.putText(overlay, "IN", (panel_x + 30, stats_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.GRAY, 1)
        cv2.putText(overlay, str(stats.in_count), (panel_x + 30, stats_y + 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, Colors.WHITE, 2)

        # OUT box
        cv2.rectangle(overlay, (panel_x + 200, stats_y),
                     (panel_x + 365, stats_y + 60), Colors.BG_LIGHT, -1)
        cv2.rectangle(overlay, (panel_x + 200, stats_y),
                     (panel_x + 205, stats_y + 60), Colors.ORANGE, -1)
        cv2.putText(overlay, "OUT", (panel_x + 215, stats_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.GRAY, 1)
        cv2.putText(overlay, str(stats.out_count), (panel_x + 215, stats_y + 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, Colors.WHITE, 2)

        # Blocked count
        blocked_y = stats_y + 75
        cv2.putText(overlay, f"Duplicates Blocked: {stats.duplicate_blocked}",
                   (panel_x + 15, blocked_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.GRAY, 1)

    def _draw_counting_line(
        self,
        overlay: np.ndarray,
        line_x: int,
        stats: CounterStats
    ):
        """Draw the counting line."""
        line_color = (Colors.GREEN if stats.current_occupancy < stats.max_capacity
                     else Colors.RED)

        # Dashed line effect
        dash_length = 20
        for y in range(0, self.frame_height, dash_length * 2):
            cv2.line(overlay, (line_x, y),
                    (line_x, min(y + dash_length, self.frame_height)),
                    line_color, 3)

    def _draw_direction_arrows(
        self,
        overlay: np.ndarray,
        line_x: int,
        direction_normal: bool
    ):
        """Draw direction indicator arrows."""
        arrow_y = self.frame_height // 2

        if direction_normal:
            # IN arrow (right to left)
            cv2.rectangle(overlay, (line_x - 100, arrow_y - 50),
                         (line_x - 10, arrow_y - 10), Colors.GREEN, -1)
            cv2.putText(overlay, "IN", (line_x - 75, arrow_y - 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, Colors.WHITE, 2)
            cv2.arrowedLine(overlay, (line_x + 50, arrow_y - 30),
                           (line_x - 5, arrow_y - 30), Colors.GREEN, 3, tipLength=0.4)

            # OUT arrow (left to right)
            cv2.rectangle(overlay, (line_x + 10, arrow_y + 10),
                         (line_x + 100, arrow_y + 50), Colors.ORANGE, -1)
            cv2.putText(overlay, "OUT", (line_x + 25, arrow_y + 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.WHITE, 2)
            cv2.arrowedLine(overlay, (line_x - 50, arrow_y + 30),
                           (line_x + 5, arrow_y + 30), Colors.ORANGE, 3, tipLength=0.4)
        else:
            # IN arrow (left to right)
            cv2.rectangle(overlay, (line_x + 10, arrow_y - 50),
                         (line_x + 100, arrow_y - 10), Colors.GREEN, -1)
            cv2.putText(overlay, "IN", (line_x + 35, arrow_y - 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, Colors.WHITE, 2)
            cv2.arrowedLine(overlay, (line_x - 50, arrow_y - 30),
                           (line_x + 5, arrow_y - 30), Colors.GREEN, 3, tipLength=0.4)

            # OUT arrow (right to left)
            cv2.rectangle(overlay, (line_x - 100, arrow_y + 10),
                         (line_x - 10, arrow_y + 50), Colors.ORANGE, -1)
            cv2.putText(overlay, "OUT", (line_x - 85, arrow_y + 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.WHITE, 2)
            cv2.arrowedLine(overlay, (line_x + 50, arrow_y + 30),
                           (line_x - 5, arrow_y + 30), Colors.ORANGE, 3, tipLength=0.4)

    def _draw_help_bar(self, overlay: np.ndarray):
        """Draw the bottom help bar."""
        help_h = 40
        help_y = self.frame_height - help_h
        cv2.rectangle(overlay, (0, help_y),
                     (self.frame_width, self.frame_height), Colors.BG, -1)

        help_text = ("[Q] Quit  |  [F] Fullscreen  |  [R] Reset  |  "
                    "[S] Swap Direction  |  [Arrows] Move Line / Adjust Capacity")
        cv2.putText(overlay, help_text, (20, help_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.GRAY, 1)

    def _draw_capacity_warning(self, overlay: np.ndarray):
        """Draw flashing capacity warning."""
        if int(time.time() * 2) % 2 == 0:
            warning_h = 80
            warning_y = 80
            cv2.rectangle(overlay, (0, warning_y),
                         (self.frame_width, warning_y + warning_h), Colors.RED, -1)

            warning_text = "CAPACITY FULL - NO MORE ENTRY"
            text_size = cv2.getTextSize(warning_text,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (self.frame_width - text_size[0]) // 2
            cv2.putText(overlay, warning_text, (text_x, warning_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, Colors.WHITE, 3)
