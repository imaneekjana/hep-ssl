from pathlib import Path
import gc

from colliderml.core import load_tables, collect_tables


data_dir = Path("colliderml-data").resolve()
data_dir.mkdir(parents=True, exist_ok=True)

channels = ["ttbar", "dihiggs", "ggf"]

for channel in channels:
    print(f"Downloading channel: {channel}", flush=True)

    config = {
        "dataset_id": "CERN/ColliderML-Release-1",
        "channels": channel,
        "pileup": "pu0",
        "objects": ["calo_hits"],
        "split": "train",
        "lazy": False,
        "max_events": 1500,
        "data_dir": str(data_dir),
    }

    tables = load_tables(config)
    frames = collect_tables(tables)

    event_count = len(frames["calo_hits"])
    print(
        f"Finished channel: {channel}, events: {event_count}",
        flush=True,
    )

    del tables
    del frames
    gc.collect()

print(f"ColliderML data directory: {data_dir}", flush=True)
print("All requested channels downloaded successfully.", flush=True)
