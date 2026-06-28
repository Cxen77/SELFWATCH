import config
from retail import RetailTheftEngine

# ── Simulate a person approaching a shelf and picking up an item ──
config.RETAIL_ZONES = [('ZONE_A', 100, 100, 400, 300)]
config.SHELF_CHANGE_ENABLED = False

engine = RetailTheftEngine()

display = {
    1: {'box': [200, 50, 350, 450], 'vel': [0, 0], 'class_id': 0},
}
print('=== Frame 1-5: Person at shelf ===')
for f in range(1, 6):
    res = engine.process_frame(display, {1: {'state': 0}}, f, shelf_change_dets=[])

print('\n=== Frame 6-10: Shelf detects item pickup ===')
# Shelf change detects something!
proposals = [{'box': [250, 180, 310, 250], 'confidence': 0.6, 'zone': 'ZONE_A', 'class_id': -1}]
for f in range(6, 11):
    res = engine.process_frame(display, {1: {'state': 0}}, f, shelf_change_dets=proposals)

print('\n=== Frame 11-30: Person moves away, item CARRIED ===')
for f in range(11, 31):
    # Smooth movement
    display[1]['box'][0] += 10
    display[1]['box'][2] += 10
    proposals = [{'box': [250 + (f-5)*10, 180, 310 + (f-5)*10, 250], 'confidence': 0.5, 'zone': 'ZONE_A', 'class_id': -1}]
    res = engine.process_frame(display, {1: {'state': 0}}, f, shelf_change_dets=proposals)

print('\n=== Frame 31-50: Item visual dropout (concealment) ===')
proposals = []  # Item no longer detected
for f in range(31, 51):
    display[1]['box'][0] += 5
    display[1]['box'][2] += 5
    res = engine.process_frame(display, {1: {'state': 0}}, f, shelf_change_dets=proposals)

print('\n=== Frame 51-60: Person exits store ===')
for f in range(51, 61):
    display[1]['box'][1] += 15  # Move down towards exit
    display[1]['box'][3] += 15
    res = engine.process_frame(display, {1: {'state': 0}}, f, shelf_change_dets=proposals)

print('\n')
engine.close()
print('\n=== TEST COMPLETE ===')
