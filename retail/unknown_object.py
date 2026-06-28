"""
Unknown Object — sentinel class_id for category-agnostic shelf change detections.

Objects detected by the ShelfChangeDetector have no COCO class.
They use UNKNOWN_OBJECT = -1 as their class_id throughout the pipeline.
"""

UNKNOWN_OBJECT = -1
