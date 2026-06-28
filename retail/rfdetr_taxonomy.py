"""
RF-DETR Taxonomy — single source of truth for class IDs.

RF-DETR uses non-contiguous 1-indexed COCO IDs internally.
The detector subtracts 1, producing 0-indexed IDs that are STILL
non-contiguous (gaps at 11, 25, 28-29, 65, 67-68, 70, 82).

This module defines every class ID as it exits the detector
(0-indexed). All retail subsystems MUST import from here.
Never hardcode class IDs elsewhere.

Derivation:
    rfdetr raw COCO ID    0-indexed detector output
    ─────────────────     ─────────────────────────
    person        = 1     →  0
    backpack      = 27    → 26
    umbrella      = 28    → 27
    handbag       = 31    → 30
    tie           = 32    → 31
    suitcase      = 33    → 32
    sports ball   = 37    → 36
    bottle        = 44    → 43
    wine glass    = 46    → 45
    cup           = 47    → 46
    laptop        = 73    → 72
    keyboard      = 76    → 75
    cell phone    = 77    → 76
    book          = 84    → 83
"""

# ── Detector output class IDs (0-indexed) ──────────────────────
PERSON      = 0
BACKPACK    = 26
UMBRELLA    = 27
HANDBAG     = 30
TIE         = 31
SUITCASE    = 32
SPORTS_BALL = 36
BOTTLE      = 43
WINE_GLASS  = 45
CUP         = 46
LAPTOP      = 72
KEYBOARD    = 75
CELL_PHONE  = 76
BOOK        = 83

# Category-agnostic sentinel for shelf change detections
UNKNOWN_OBJECT = -1

# ── Grouped sets ───────────────────────────────────────────────
BAG_CLASSES = {BACKPACK, HANDBAG, SUITCASE}

RETAIL_OBJECT_CLASSES = {BOTTLE, CUP, CELL_PHONE, BOOK, LAPTOP, KEYBOARD, SPORTS_BALL, WINE_GLASS}

ALL_TRACKED_CLASSES = RETAIL_OBJECT_CLASSES | BAG_CLASSES | {UNKNOWN_OBJECT}

RETAIL_TARGET_CLASSES = [PERSON] + sorted(ALL_TRACKED_CLASSES - {UNKNOWN_OBJECT})

# ── Human-readable names ──────────────────────────────────────
CLASS_NAMES = {
    PERSON:         "Person",
    BACKPACK:       "Backpack",
    UMBRELLA:       "Umbrella",
    HANDBAG:        "Handbag",
    SUITCASE:       "Suitcase",
    SPORTS_BALL:    "Sports Ball",
    BOTTLE:         "Bottle",
    WINE_GLASS:     "Wine Glass",
    CUP:            "Cup",
    LAPTOP:         "Laptop",
    KEYBOARD:       "Keyboard",
    CELL_PHONE:     "Cell Phone",
    BOOK:           "Book",
    UNKNOWN_OBJECT: "Unknown",
}

def class_label(class_id):
    return CLASS_NAMES.get(class_id, f"cls{class_id}")
