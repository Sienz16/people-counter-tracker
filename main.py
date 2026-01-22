"""
People Counter System (Commercial Version)
Uses YOLOX (Apache 2.0) - Safe for commercial use

Features:
- Person detection using YOLOX (ONNX Runtime)
- Tracking with ByteTrack
- IN/OUT counting with line crossing
- Re-ID to prevent duplicate counting
- Capacity alerts with visual/audio feedback

Controls:
- Q: Quit
- F: Toggle fullscreen
- R: Reset counts
- S: Swap IN/OUT direction
- Left/Right arrows: Move counting line
- Up/Down arrows: Adjust capacity
"""

import cv2
import sys

try:
    import winsound
except ImportError:
    winsound = None

import config
from components import PeopleCounter


def main():
    """Main application entry point."""
    print("=" * 60)
    print("       PEOPLE COUNTER SYSTEM (Commercial - YOLOX)")
    print("=" * 60)

    # Initialize counter
    counter = PeopleCounter()

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Get actual dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counter.update_frame_size(frame_width, frame_height)

    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"Max Capacity: {counter.max_capacity}")
    print("-" * 60)
    print("CONTROLS:")
    print("  [Q] Quit              [F] Toggle fullscreen")
    print("  [R] Reset counts      [S] Swap IN/OUT direction")
    print("  [LEFT/RIGHT] Move line position")
    print("  [UP/DOWN] Adjust capacity (+/-10)")
    print("=" * 60)

    # Setup window
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                         cv2.WINDOW_FULLSCREEN)

    # Capacity alert state
    capacity_alert_shown = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame
            frame = counter.process_frame(frame)

            # Capacity alert
            if counter.is_at_capacity and not capacity_alert_shown:
                print(">>> CAPACITY FULL! <<<")
                if winsound:
                    try:
                        winsound.Beep(1000, 500)
                    except:
                        pass
                capacity_alert_shown = True
            elif not counter.is_at_capacity:
                capacity_alert_shown = False

            # Display
            cv2.imshow(config.WINDOW_NAME, frame)

            # Handle input
            key = cv2.waitKeyEx(1)
            if not _handle_input(key, counter):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Print final stats
        stats = counter.get_stats()
        print("\n" + "=" * 60)
        print("FINAL COUNTS")
        print("=" * 60)
        print(f"Total IN:  {stats.in_count}")
        print(f"Total OUT: {stats.out_count}")
        print(f"Final Occupancy: {stats.current_occupancy}")
        print(f"Duplicates Blocked: {stats.duplicate_blocked}")
        print("=" * 60)

        cap.release()
        cv2.destroyAllWindows()


def _handle_input(key: int, counter: PeopleCounter) -> bool:
    """
    Handle keyboard input.

    Args:
        key: Key code from waitKeyEx
        counter: PeopleCounter instance

    Returns:
        False if should quit, True otherwise
    """
    if key == ord('q'):
        return False

    elif key == ord('f'):
        current = cv2.getWindowProperty(config.WINDOW_NAME,
                                        cv2.WND_PROP_FULLSCREEN)
        if current == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(config.WINDOW_NAME,
                                 cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(config.WINDOW_NAME,
                                 cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    elif key == ord('r'):
        counter.reset()

    elif key == ord('s'):
        counter.swap_direction()

    elif key == 2424832:  # Left arrow
        counter.move_line(-config.LINE_MOVE_STEP)

    elif key == 2555904:  # Right arrow
        counter.move_line(config.LINE_MOVE_STEP)

    elif key == 2490368:  # Up arrow
        counter.adjust_capacity(10)

    elif key == 2621440:  # Down arrow
        counter.adjust_capacity(-10)

    return True


if __name__ == "__main__":
    main()
