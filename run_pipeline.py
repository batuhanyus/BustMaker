"""Bust Forge CLI entrypoint.

    python run_pipeline.py --input ./input/subject_01 [--output ...] \
        [--mode auto|generative|capture|depth_relief] [--preset fast|balanced|high]

Runs the pipeline end-to-end and writes ``report.json`` into the job output
directory. Exit code 0 = pipeline completed (success or graceful failure with
report); 2 = usage/validation error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import Config
from core.logging import get_logger, setup_logging
from core.paths import JobPaths
from core.pipeline import MODES, classify_input, run_pipeline

log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Local-only video/photo -> print-ready 3D bust STL pipeline.",
    )
    p.add_argument("--input", required=True, help="Video file, image file, or folder of images.")
    p.add_argument("--output", default=None, help="Job output directory (default: output/<input-name>/<timestamp>).")
    p.add_argument("--mode", default="auto", choices=sorted(MODES), help="Pipeline strategy (default: auto).")
    p.add_argument("--preset", default="balanced", choices=["fast", "balanced", "high"], help="Quality preset.")
    p.add_argument("--config", default=None, help="Path to config.yaml (default: project config.yaml).")
    p.add_argument("--debug", action="store_true", help="Keep debug artifacts and verbose logs.")
    p.add_argument("--resume", action="store_true",
                   help="Skip stages already successful in the output dir's previous report.json.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = Config.load(args.config)
    setup_logging(cfg.logging_level, log_dir=cfg.resolve_path("paths.logs", "./logs"))

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input path does not exist: %s", input_path)
        return 2
    itype = classify_input(input_path)
    if itype == "unknown":
        log.error(
            "Unsupported input type for %s (want video, image, or image folder).",
            input_path,
        )
        return 2

    job = JobPaths.create(cfg, input_path, output_dir=Path(args.output) if args.output else None)
    log.info("Job %s | input=%s (%s) | mode=%s | preset=%s", job.job_id, input_path, itype, args.mode, args.preset)
    log.info("Output dir: %s", job.job_dir)

    report = run_pipeline(
        cfg,
        job,
        mode=args.mode,
        preset=args.preset,
        cli={"debug": args.debug, "resume": args.resume},
    )
    ok = report.summary.get("success", False)
    log.info("Pipeline finished: success=%s | report=%s", ok, job.report_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
