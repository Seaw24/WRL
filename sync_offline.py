"""Offline sync: pair camera/video frames with Pupil Neon gaze labels.

The rotated/flipped eye videos are postprocessed copies of the original 96x96
videos, so their frame numbers still match camera_frames.csv. This script
therefore syncs using the original camera frame timestamps and Neon gaze
timestamps, then writes labels that point at the corrected eye video.

Examples:
    ./venv/bin/python sync_offline.py
    ./venv/bin/python sync_offline.py --postprocessed-root postprocessed_eye_videos
    ./venv/bin/python sync_offline.py --video postprocessed_eye_videos/joey/joey_final_96x96_dataset_rotated_cw45_flip_h.mp4 --source-dir joey
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


NEON_GAZE_WIDTH = 1600.0
NEON_GAZE_HEIGHT = 1200.0


@dataclass(frozen=True)
class SyncJob:
    name: str
    video_path: Path | None
    source_dir: Path
    camera_csv: Path
    gaze_csv: Path
    output_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync camera/video frames to Neon gaze with a strict tolerance."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root. Default: current directory.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Original session folder containing camera/gaze CSVs. For manual single-video use.",
    )
    parser.add_argument(
        "--camera-csv",
        type=Path,
        help="Manual camera_frames CSV path.",
    )
    parser.add_argument(
        "--gaze-csv",
        type=Path,
        help="Manual Neon gaze CSV path.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Manual corrected eye video path. Frame numbers are assumed to match camera_frames.csv.",
    )
    parser.add_argument(
        "--postprocessed-root",
        type=Path,
        default=Path("postprocessed_eye_videos"),
        help="Root containing corrected videos. Default: postprocessed_eye_videos.",
    )
    parser.add_argument(
        "--video-pattern",
        default="*_rotated_cw45_flip_h.mp4",
        help="Corrected videos to sync when scanning postprocessed-root.",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="Only sync matching session folder/name. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Manual output CSV path for a single job.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_synced_labels_10ms.csv",
        help="Suffix appended to corrected video stems for scanned outputs.",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=10.0,
        help="Drop frames whose nearest Neon sample is above this gap. Default: 10.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="adjusted_ts",
        help="Neon timestamp column to sync against cam_ts. Default: adjusted_ts.",
    )
    parser.add_argument(
        "--out-of-frame",
        choices=("drop", "clip", "keep"),
        default="drop",
        help="How to handle gaze outside the Neon scene frame. Default: drop.",
    )
    parser.add_argument(
        "--neon-width",
        type=float,
        default=NEON_GAZE_WIDTH,
        help="Neon scene-camera gaze coordinate width. Default: 1600.",
    )
    parser.add_argument(
        "--neon-height",
        type=float,
        default=NEON_GAZE_HEIGHT,
        help="Neon scene-camera gaze coordinate height. Default: 1200.",
    )
    parser.add_argument(
        "--legacy-current-dir",
        action="store_true",
        help="Use camera_frames.csv/neon_gaze_raw.csv in the current directory and write final_synced_labels.csv.",
    )
    return parser.parse_args()


def first_match(root: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def resolve_source_csvs(source_dir: Path) -> tuple[Path, Path]:
    camera_csv = first_match(source_dir, ("*camera_frames.csv",))
    gaze_csv = first_match(source_dir, ("*neon_gaze_raw.csv",))
    if camera_csv is None:
        raise FileNotFoundError(f"Missing camera frames CSV in {source_dir}")
    if gaze_csv is None:
        raise FileNotFoundError(f"Missing Neon gaze CSV in {source_dir}")
    return camera_csv, gaze_csv


def name_for_video(video_path: Path) -> str:
    return video_path.parent.name or video_path.stem


def output_for_video(video_path: Path, output_suffix: str) -> Path:
    return video_path.with_name(video_path.stem + output_suffix)


def discover_jobs(args: argparse.Namespace) -> list[SyncJob]:
    root = args.root.expanduser().resolve()

    if args.legacy_current_dir:
        source_dir = root
        camera_csv = args.camera_csv or (source_dir / "camera_frames.csv")
        gaze_csv = args.gaze_csv or (source_dir / "neon_gaze_raw.csv")
        output_csv = args.output or (source_dir / "final_synced_labels.csv")
        return [
            SyncJob(
                name=source_dir.name,
                video_path=None,
                source_dir=source_dir,
                camera_csv=camera_csv,
                gaze_csv=gaze_csv,
                output_csv=output_csv,
            )
        ]

    if args.video:
        video_path = args.video.expanduser().resolve()
        source_dir = (args.source_dir or (root / video_path.parent.name)).expanduser().resolve()
        camera_csv = args.camera_csv.expanduser().resolve() if args.camera_csv else None
        gaze_csv = args.gaze_csv.expanduser().resolve() if args.gaze_csv else None
        if camera_csv is None or gaze_csv is None:
            found_camera_csv, found_gaze_csv = resolve_source_csvs(source_dir)
            camera_csv = camera_csv or found_camera_csv
            gaze_csv = gaze_csv or found_gaze_csv
        output_csv = args.output.expanduser().resolve() if args.output else output_for_video(video_path, args.output_suffix)
        return [
            SyncJob(
                name=name_for_video(video_path),
                video_path=video_path,
                source_dir=source_dir,
                camera_csv=camera_csv,
                gaze_csv=gaze_csv,
                output_csv=output_csv,
            )
        ]

    postprocessed_root = args.postprocessed_root.expanduser()
    if not postprocessed_root.is_absolute():
        postprocessed_root = root / postprocessed_root
    wanted = {session.lower() for session in args.session}

    jobs: list[SyncJob] = []
    for video_path in sorted(postprocessed_root.glob(f"*/{args.video_pattern}")):
        name = name_for_video(video_path)
        if wanted and name.lower() not in wanted and video_path.stem.lower() not in wanted:
            continue
        source_dir = root / name
        camera_csv, gaze_csv = resolve_source_csvs(source_dir)
        jobs.append(
            SyncJob(
                name=name,
                video_path=video_path,
                source_dir=source_dir,
                camera_csv=camera_csv,
                gaze_csv=gaze_csv,
                output_csv=output_for_video(video_path, args.output_suffix),
            )
        )

    if not jobs:
        raise FileNotFoundError(
            f"No videos matched {args.video_pattern!r} under {postprocessed_root}"
        )
    return jobs


def require_columns(df: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def load_streams(job: SyncJob, timestamp_column: str):
    camera_df = pd.read_csv(job.camera_csv)
    neon_df = pd.read_csv(job.gaze_csv)

    if camera_df.empty:
        raise ValueError("Camera CSV is empty. Nothing to sync.")
    if neon_df.empty:
        raise ValueError("Neon CSV is empty. Was the Neon connected during recording?")

    require_columns(camera_df, ("frame", "cam_ts"), job.camera_csv)
    require_columns(neon_df, ("ts", timestamp_column, "gaze_x", "gaze_y"), job.gaze_csv)

    camera_df = camera_df.dropna(subset=["frame", "cam_ts"]).copy()
    camera_df["frame"] = camera_df["frame"].astype(int)
    camera_df = camera_df.sort_values("cam_ts").reset_index(drop=True)

    neon_df = neon_df.dropna(subset=[timestamp_column, "gaze_x", "gaze_y"]).copy()
    neon_df = neon_df.sort_values(timestamp_column).reset_index(drop=True)
    return camera_df, neon_df


def pair_and_interpolate(
    camera_df,
    neon_df,
    timestamp_column: str,
    neon_width: float,
    neon_height: float,
):
    """For each camera frame, find the two surrounding gaze samples and
    linearly interpolate gaze_x / gaze_y. Also compute distance to nearest
    gaze sample for reporting."""

    cam_ts = camera_df["cam_ts"].to_numpy(dtype=np.float64)
    gaze_ts = neon_df[timestamp_column].to_numpy(dtype=np.float64)
    gaze_x = neon_df["gaze_x"].to_numpy(dtype=np.float64)
    gaze_y = neon_df["gaze_y"].to_numpy(dtype=np.float64)
    worn = neon_df["worn"].to_numpy() if "worn" in neon_df.columns else None

    # Linear interpolation at camera timestamps.
    # np.interp clamps to endpoint values for cam_ts outside [gaze_ts[0], gaze_ts[-1]],
    # which is the safest behavior (no extrapolation).
    interp_x = np.interp(cam_ts, gaze_ts, gaze_x)
    interp_y = np.interp(cam_ts, gaze_ts, gaze_y)

    # Find the nearest gaze sample for each camera frame (for sync_diff reporting).
    # searchsorted gives the insertion point; nearest is either that index or the one before.
    idx_right = np.searchsorted(gaze_ts, cam_ts, side="left")
    idx_right = np.clip(idx_right, 0, len(gaze_ts) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(gaze_ts) - 1)

    dist_left = np.abs(cam_ts - gaze_ts[idx_left])
    dist_right = np.abs(cam_ts - gaze_ts[idx_right])
    choose_left = dist_left <= dist_right
    nearest_idx = np.where(choose_left, idx_left, idx_right)
    nearest_dist = np.minimum(dist_left, dist_right)

    # Mark camera frames that fall outside the Neon time range.
    # For these, interpolation has clamped to the endpoint - not ideal but not catastrophic.
    out_of_range = (cam_ts < gaze_ts[0]) | (cam_ts > gaze_ts[-1])

    # Normalize to [-1, 1] to match the training target space
    norm_x = (interp_x / neon_width) * 2.0 - 1.0
    norm_y = (interp_y / neon_height) * 2.0 - 1.0

    out = camera_df.copy()
    out["neon_ts"] = gaze_ts[nearest_idx]
    out["neon_raw_ts"] = neon_df["ts"].to_numpy(dtype=np.float64)[nearest_idx]
    out["gaze_x"] = interp_x
    out["gaze_y"] = interp_y
    out["norm_x"] = norm_x
    out["norm_y"] = norm_y
    out["sync_diff_seconds"] = nearest_dist
    out["sync_diff_ms"] = nearest_dist * 1000.0
    out["out_of_range"] = out_of_range
    if worn is not None:
        out["worn"] = worn[nearest_idx]
    return out


def validate_video_frame_count(video_path: Path | None, camera_df: pd.DataFrame) -> None:
    if video_path is None:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open corrected video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count and frame_count != len(camera_df):
        raise ValueError(
            f"{video_path} has {frame_count} frames, but {len(camera_df)} camera timestamp rows. "
            "The rotated/flipped video must preserve frame count."
        )


def finalize_training_labels(
    job: SyncJob,
    paired: pd.DataFrame,
    tolerance_ms: float,
    out_of_frame_handling: str,
    neon_width: float,
    neon_height: float,
) -> pd.DataFrame:
    kept = paired[paired["sync_diff_ms"] <= tolerance_ms].copy()

    if job.video_path is not None:
        kept.insert(0, "video_path", str(job.video_path))
        kept.insert(1, "video_frame", kept["frame"].astype(int))

    out_of_frame_mask = (
        (kept["gaze_x"] < 0) | (kept["gaze_x"] > neon_width)
        | (kept["gaze_y"] < 0) | (kept["gaze_y"] > neon_height)
    )

    if out_of_frame_handling == "drop":
        kept = kept[~out_of_frame_mask].copy()
    elif out_of_frame_handling == "clip":
        kept["gaze_x"] = kept["gaze_x"].clip(0.0, neon_width)
        kept["gaze_y"] = kept["gaze_y"].clip(0.0, neon_height)
        kept["norm_x"] = kept["norm_x"].clip(-1.0, 1.0)
        kept["norm_y"] = kept["norm_y"].clip(-1.0, 1.0)
    elif out_of_frame_handling != "keep":
        raise ValueError("Unknown out-of-frame handling: " + repr(out_of_frame_handling))

    kept = kept.drop(columns=["out_of_range"], errors="ignore")
    return kept


def print_report(job: SyncJob, paired_df, kept_df, tolerance_ms: float, out_of_frame_handling: str):
    total = len(paired_df)
    kept = len(kept_df)
    dropped = total - kept
    diffs_ms = paired_df["sync_diff_seconds"] * 1000.0

    perfect = int((diffs_ms <= 15.0).sum())
    good = int((diffs_ms <= 30.0).sum())
    within_tol = int((diffs_ms <= tolerance_ms).sum())
    out_of_range = int(paired_df["out_of_range"].sum())

    print("=" * 48)
    print("OFFLINE SYNC REPORT: " + job.name)
    print("=" * 48)
    if job.video_path is not None:
        print("Corrected video:         " + str(job.video_path))
    print("Camera CSV:              " + str(job.camera_csv))
    print("Neon CSV:                " + str(job.gaze_csv))
    print("Output CSV:              " + str(job.output_csv))
    print("Total camera frames:     " + str(total))
    print("Kept for training:       " + str(kept)
          + "  (" + ("%.1f" % (100.0 * kept / total)) + "%)")
    print("Dropped:                 " + str(dropped))
    print("-" * 48)
    print("Perfect   (<= 15 ms):    " + str(perfect)
          + "  (" + ("%.1f" % (100.0 * perfect / total)) + "%)")
    print("Good      (<= 30 ms):    " + str(good)
          + "  (" + ("%.1f" % (100.0 * good / total)) + "%)")
    print("Within " + ("%.1f" % tolerance_ms) + " ms:         " + str(within_tol)
          + "  (" + ("%.1f" % (100.0 * within_tol / total)) + "%)")
    print("-" * 48)
    print("Mean sync gap:           " + ("%.2f" % diffs_ms.mean()) + " ms")
    print("Median sync gap:         " + ("%.2f" % diffs_ms.median()) + " ms")
    print("Max sync gap:            " + ("%.2f" % diffs_ms.max()) + " ms")
    print("Out-of-frame handling:   " + out_of_frame_handling)
    if out_of_range:
        print("Out-of-range frames:     " + str(out_of_range)
              + "  (clamped to endpoint gaze)")
    print("=" * 48)

    if total and dropped:
        print("\nWorst 3 dropped frames:")
        worst = paired_df.nlargest(3, "sync_diff_seconds")
        for _, row in worst.iterrows():
            print("  frame " + str(int(row["frame"]))
                  + "  gap=" + ("%.1f" % (row["sync_diff_seconds"] * 1000.0)) + " ms")


def sync_job(job: SyncJob, args: argparse.Namespace) -> pd.DataFrame:
    camera_df, neon_df = load_streams(job, args.timestamp_column)
    validate_video_frame_count(job.video_path, camera_df)

    print("Loaded " + str(len(camera_df)) + " camera frames, "
          + str(len(neon_df)) + " Neon gaze samples for " + job.name + ".")

    paired = pair_and_interpolate(
        camera_df,
        neon_df,
        args.timestamp_column,
        args.neon_width,
        args.neon_height,
    )
    kept = finalize_training_labels(
        job,
        paired,
        args.tolerance_ms,
        args.out_of_frame,
        args.neon_width,
        args.neon_height,
    )

    job.output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(job.output_csv, index=False)
    print("Wrote " + str(len(kept)) + " training rows to '" + str(job.output_csv) + "'.")

    print_report(job, paired, kept, args.tolerance_ms, args.out_of_frame)
    return kept


def main():
    args = parse_args()
    try:
        jobs = discover_jobs(args)
        for job in jobs:
            sync_job(job, args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit("ERROR: " + str(exc)) from exc


if __name__ == "__main__":
    main()
