"""
Configuration settings for People Counter
"""

# ============== CAPACITY SETTINGS ==============
MAX_CAPACITY = 50  # Maximum people allowed inside
LINE_MOVE_STEP = 20  # Pixels to move line per keypress

# ============== DETECTION SETTINGS ==============
MODEL_SIZE = "s"  # YOLOX model: nano, tiny, s, m, l, x
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

# ============== RE-ID SETTINGS ==============
# Threshold for combined color+texture matching (higher = less restrictive)
# Two people must match on BOTH color AND texture to be considered duplicate
SIMILARITY_THRESHOLD = 0.65

# ============== DIRECTION SETTINGS ==============
# True: Right->Left = IN, Left->Right = OUT
# False: Left->Right = IN, Right->Left = OUT
DIRECTION_NORMAL = True

# ============== DISPLAY SETTINGS ==============
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "People Counter"

# ============== UI COLORS (BGR) ==============
class Colors:
    BG = (30, 30, 30)
    BG_LIGHT = (50, 50, 50)
    ACCENT = (255, 140, 0)
    GREEN = (0, 200, 100)
    ORANGE = (0, 165, 255)
    RED = (60, 60, 220)
    WHITE = (255, 255, 255)
    GRAY = (150, 150, 150)
