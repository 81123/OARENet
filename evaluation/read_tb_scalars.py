from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

LOG_ROOT = Path(r"D:\project\pythonProject\Road_Identification\OARENet\logs")

for d in ["OARENET_20260309_171645", "OARENET_20260309_171457", "OARENET_20260709_224820"]:
    path = LOG_ROOT / d
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    print("==", d)
    tags = acc.Tags().get("scalars", [])
    if not tags:
        print("  (no scalar data)")
    for tag in tags:
        evs = acc.Scalars(tag)
        vals = [(e.step, round(e.value, 4)) for e in evs]
        if len(vals) <= 15:
            print(f"  {tag} -> {vals}")
        else:
            print(f"  {tag} -> {len(vals)} pts, first={vals[:2]}, last={vals[-3:]}")
