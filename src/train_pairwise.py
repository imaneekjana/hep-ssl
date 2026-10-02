"""Train or exactly resume the common pairwise pipeline."""
import argparse
from datetime import datetime
from pathlib import Path

from src.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON configuration (defaults to project baseline)")
    parser.add_argument("--prepared", help="Prepared artifact directory; relocation is verified by fingerprint")
    parser.add_argument("--run-dir", help="Fresh output directory; required if resuming a relocated run")
    parser.add_argument("--resume", help="A phase-1 last.pt or best.pt checkpoint")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop after this many total epochs without changing scheduler horizon")
    args = parser.parse_args(argv)
    from src.training.trainer import train
    config = load_config(args.config) if args.config else None
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.resume:
        run_dir = Path(args.resume).resolve().parent.parent
    else:
        stamp = datetime.now()
        run_dir = Path("experiments") / stamp.strftime("%m_%d training") / "pretraining" / stamp.strftime("run_%H%M%S")
    if not args.resume and not (args.prepared or (config and config["prepared_dir"])):
        parser.error("--prepared or config.prepared_dir is required; run prepare_pairwise first")
    train(config, args.prepared, run_dir, resume=args.resume, stop_after_epoch=args.stop_after_epoch)


if __name__ == "__main__":
    main()
