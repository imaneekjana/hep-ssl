#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path

from colliderml.core import collect_tables, load_tables
from colliderml.core.hf_download import DownloadSpec, download_config


CHANNELS = ("ttbar", "dihiggs", "ggf")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extend an existing ColliderML cache and verify at least 2500 "
            "pu0 calo-hit events for each pairwise-training channel."
        )
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--events-per-channel", default=2500, type=int)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    event_counts = {}

    for channel in CHANNELS:
        print(f"Downloading/checking channel: {channel}", flush=True)

        config_name = f"{channel}_pu0_calo_hits"

        download_result = download_config(
            DownloadSpec(
                dataset_id="CERN/ColliderML-Release-1",
                config=config_name,
                split="train",
                max_events=args.events_per_channel,
            ),
            out_dir=data_dir,
        )

        print(
            f"Downloaded/verified shards: "
            f"{config_name} -> {download_result.local_dir}",
            flush=True,
        )

        config = {
            "dataset_id": "CERN/ColliderML-Release-1",
            "channels": channel,
            "pileup": "pu0",
            "objects": ["calo_hits"],
            "split": "train",
            "lazy": False,
            "max_events": args.events_per_channel,
            "data_dir": str(data_dir),
        }

        tables = load_tables(config)
        frames = collect_tables(tables)
        count = len(frames["calo_hits"])

        if count < args.events_per_channel:
            raise RuntimeError(
                f"Channel {channel}: requested {args.events_per_channel} "
                f"events, but only loaded {count}."
            )

        event_counts[channel] = count
        print(f"Verified channel {channel}: {count} events", flush=True)

        del frames
        del tables
        gc.collect()

    manifest = {
        "dataset_id": "CERN/ColliderML-Release-1",
        "pileup": "pu0",
        "object": "calo_hits",
        "split": "train",
        "requested_events_per_channel": args.events_per_channel,
        "verified_event_counts": event_counts,
    }

    manifest_path = data_dir / "pairwise_2500_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote manifest: {manifest_path}", flush=True)
    print("All pairwise channels verified successfully.", flush=True)


if __name__ == "__main__":
    main()