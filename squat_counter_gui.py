#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import serial

from eval_posture_realtime import (
    format_scores,
    records_to_posture_data,
    save_frame_stream,
)
from get_range_profile import load_configuration, parse_range_config, send_configuration
from near_field_gesture_viewer import (
    remove_leading_sensor_stop,
    stop_and_drain,
    warm_reset_demo,
)
from posture_lab_common import (
    INPUT_TYPE,
    SESSIONS_DIR,
    cloud_arrays,
    extract_posture_feature_vector,
    filter_posture_points,
    now_text,
    prediction_confidence,
    prediction_scores,
    read_point_cloud_frame,
    safe_label,
    timestamp,
    write_json,
)


DEFAULT_CFG = Path("xwrL64xx-evm/point_cloud.cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count squats from a trained real-time posture model."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds. Use 0 to run until Ctrl+C.",
    )
    parser.add_argument("--window-seconds", type=float, help="Default: model setting.")
    parser.add_argument("--step-seconds", type=float, default=0.5)
    parser.add_argument("--min-window-frames", type=int, help="Default: model setting or 4.")
    parser.add_argument("--standing-label", default="standing")
    parser.add_argument("--squat-label", default="squat")
    parser.add_argument(
        "--stable-predictions",
        type=int,
        default=2,
        help="Consecutive matching predictions required before state changes.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.45,
        help="Ignore predictions below this confidence when probabilities exist.",
    )
    parser.add_argument(
        "--min-count-interval",
        type=float,
        default=1.0,
        help="Minimum seconds between counted squats.",
    )
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--out-root", default=str(SESSIONS_DIR))
    parser.add_argument("--session-name")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--no-config", action="store_true")
    parser.add_argument("--no-warm-reset", action="store_true")
    return parser.parse_args()


@dataclass
class SquatCounter:
    standing_label: str
    squat_label: str
    min_count_interval_s: float
    state: str = "wait_for_standing"
    count: int = 0
    last_count_time_s: float = -1e9

    def reset(self) -> None:
        self.state = "wait_for_standing"
        self.count = 0
        self.last_count_time_s = -1e9

    def update(self, stable_label: str | None, time_s: float) -> str:
        """Advance standing -> squat -> standing state machine.

        Returns an event string describing what happened:
        - "" (empty): no state change
        - "stand": transitioned to standing_ready
        - "squat": transitioned to in_squat
        - "count": counted a squat (standing -> squat -> standing complete)
        """
        if stable_label is None:
            return ""

        if stable_label not in (self.standing_label, self.squat_label):
            return ""

        event = ""

        if self.state == "wait_for_standing":
            if stable_label == self.standing_label:
                self.state = "standing_ready"
                event = "stand"

        elif self.state == "standing_ready":
            if stable_label == self.squat_label:
                self.state = "in_squat"
                event = "squat"

        elif self.state == "in_squat":
            if stable_label == self.standing_label:
                elapsed = time_s - self.last_count_time_s
                if elapsed >= self.min_count_interval_s:
                    self.count += 1
                    self.last_count_time_s = time_s
                    event = "count"
                self.state = "standing_ready"
                if not event:
                    event = "stand"

        return event


class ConsecutivePredictionFilter:
    def __init__(self, required_count: int, confidence_threshold: float):
        self.required_count = max(1, int(required_count))
        self.confidence_threshold = float(confidence_threshold)
        self.history: deque[str | None] = deque(maxlen=self.required_count)

    def reset(self) -> None:
        self.history.clear()

    def update(self, prediction, confidence: float | None) -> str | None:
        """Debounce noisy predictions by requiring consecutive agreement.

        Returns the stable label when all entries in the history match,
        or None if the prediction is unstable or confidence is too low.
        """
        # 1. Reject low-confidence predictions
        if confidence is not None and confidence < self.confidence_threshold:
            self.history.append(None)
            return None

        # 2. Keep the most recent required_count labels
        self.history.append(prediction)

        # 3. Return a label only when all entries agree
        if len(self.history) < self.required_count:
            return None

        first = self.history[0]
        if first is None:
            return None

        for entry in self.history:
            if entry != first:
                return None

        return str(first)


class SquatCounterPlot:
    def __init__(self, feature_params: dict, standing_label: str, squat_label: str):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.plt = plt
        self.reset_requested = False
        self.quit_requested = False
        self.feature_params = feature_params
        plt.ion()
        self.figure = plt.figure(figsize=(10.5, 6.5))
        grid = self.figure.add_gridspec(
            2,
            2,
            width_ratios=(1.25, 0.9),
            height_ratios=(1.0, 0.14),
        )
        self.cloud_axis = self.figure.add_subplot(grid[0, 0])
        self.status_axis = self.figure.add_subplot(grid[0, 1])
        self.reset_axis = self.figure.add_subplot(grid[1, 0])
        self.quit_axis = self.figure.add_subplot(grid[1, 1])

        x_limit = float(feature_params.get("x_limit_m", 2.0))
        min_range = float(feature_params.get("min_range_m", 0.2))
        max_range = float(feature_params.get("max_range_m", 5.0))
        self.cloud_axis.set_xlim(-x_limit, x_limit)
        self.cloud_axis.set_ylim(min_range, max_range)
        self.cloud_axis.set_aspect("equal", adjustable="box")
        self.cloud_axis.set_xlabel("x left/right (m)")
        self.cloud_axis.set_ylabel("y range (m)")
        self.cloud_axis.grid(True)
        self.scatter = self.cloud_axis.scatter([], [], c=[], cmap="viridis", s=45)
        self.scatter.set_clim(min_range, max_range)
        colorbar = self.figure.colorbar(self.scatter, ax=self.cloud_axis, pad=0.02)
        colorbar.set_label("y range (m)")

        self.status_axis.axis("off")
        self.count_text = self.status_axis.text(
            0.5,
            0.78,
            "0",
            ha="center",
            va="center",
            fontsize=96,
            fontweight="bold",
        )
        self.status_text = self.status_axis.text(
            0.02,
            0.42,
            "",
            ha="left",
            va="top",
            fontsize=13,
            linespacing=1.5,
        )
        self.hint_text = self.status_axis.text(
            0.02,
            0.02,
            f"Cycle: {standing_label} -> {squat_label} -> {standing_label}\n"
            "Keys: r reset, q quit",
            ha="left",
            va="bottom",
            fontsize=11,
            color="0.35",
        )

        self.reset_button = Button(self.reset_axis, "Reset")
        self.quit_button = Button(self.quit_axis, "Quit")
        self.reset_button.on_clicked(self._request_reset)
        self.quit_button.on_clicked(self._request_quit)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.tight_layout()
        self.figure.show()

    def _request_reset(self, _event=None) -> None:
        self.reset_requested = True

    def _request_quit(self, _event=None) -> None:
        self.quit_requested = True
        self.plt.close(self.figure)

    def _on_key(self, event) -> None:
        if event.key == "r":
            self._request_reset()
        elif event.key == "q":
            self._request_quit()

    def is_open(self) -> bool:
        return self.plt.fignum_exists(self.figure.number) and not self.quit_requested

    def update(
        self,
        xyz: np.ndarray,
        counter: SquatCounter,
        prediction=None,
        confidence: float | None = None,
        stable_label: str | None = None,
        event: str = "",
    ) -> None:
        filtered = filter_posture_points(np.asarray(xyz, dtype=float), self.feature_params)
        offsets = filtered[:, :2] if len(filtered) else np.empty((0, 2), dtype=float)
        colors = filtered[:, 1] if len(filtered) else np.empty(0, dtype=float)
        self.scatter.set_offsets(offsets)
        self.scatter.set_array(colors)
        self.cloud_axis.set_title(f"ROI points: {len(filtered)}")

        if confidence is None:
            prediction_text = str(prediction) if prediction is not None else "waiting"
        elif prediction is None:
            prediction_text = "waiting"
        else:
            prediction_text = f"{prediction} ({confidence:.2f})"
        self.count_text.set_text(str(counter.count))
        self.status_text.set_text(
            f"prediction: {prediction_text}\n"
            f"stable: {stable_label or '-'}\n"
            f"state: {counter.state}\n"
            f"event: {event or '-'}"
        )
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)


def validate_args(args: argparse.Namespace) -> None:
    if args.step_seconds <= 0:
        raise SystemExit("--step-seconds must be positive.")
    if args.stable_predictions < 1:
        raise SystemExit("--stable-predictions must be at least 1.")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be in [0, 1].")
    if args.min_count_interval < 0:
        raise SystemExit("--min-count-interval must be nonnegative.")
    if args.standing_label == args.squat_label:
        raise SystemExit("--standing-label and --squat-label must be different.")


def main() -> int:
    args = parse_args()
    validate_args(args)

    try:
        import joblib
    except ImportError as exc:
        raise SystemExit(f"Missing joblib/sklearn environment: {exc}") from exc

    model_path = Path(args.model).expanduser().resolve()
    payload = joblib.load(model_path)
    input_type = payload.get("input_type", INPUT_TYPE)
    if input_type != INPUT_TYPE:
        raise SystemExit(f"This counter cannot run model input_type: {input_type}")

    model = payload["model"]
    feature_params = payload.get("feature_params") or {}
    feature_names = payload.get("feature_names") or []
    classifier_label = payload.get("classifier_label", payload.get("classifier", "unknown"))
    labels_order = list(payload.get("labels_order") or getattr(model, "classes_", []))
    window_seconds = float(args.window_seconds or payload.get("window_seconds", 2.0))
    min_window_frames = int(args.min_window_frames or payload.get("min_segment_frames", 4) or 4)
    if window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive.")
    if min_window_frames < 2:
        raise SystemExit("--min-window-frames must be at least 2.")

    required_labels = {args.standing_label, args.squat_label}
    if labels_order and not required_labels.issubset({str(label) for label in labels_order}):
        print(
            "Warning: model labels do not appear to include both "
            f"{sorted(required_labels)}. Model labels: {labels_order}"
        )

    commands = load_configuration(args.cfg)
    range_config = parse_range_config(commands)
    if range_config is None:
        raise SystemExit("Could not compute range-bin spacing from cfg.")

    session_name = args.session_name or f"squat_counter_{safe_label(model_path.stem)}_{timestamp()}"
    session_dir = Path(args.out_root).expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = session_dir / "squat_counter_predictions.csv"
    frames_path = session_dir / "squat_counter_frames.npz"
    summary_path = session_dir / "squat_counter_summary.json"

    write_json(
        session_dir / "session_metadata.json",
        {
            "kind": "squat_counter_gui",
            "input_type": INPUT_TYPE,
            "model": str(model_path),
            "classifier_label": classifier_label,
            "feature_params": feature_params,
            "cfg_path": str(args.cfg),
            "duration_s": args.duration,
            "window_seconds": window_seconds,
            "step_seconds": args.step_seconds,
            "min_window_frames": min_window_frames,
            "standing_label": args.standing_label,
            "squat_label": args.squat_label,
            "stable_predictions": args.stable_predictions,
            "confidence_threshold": args.confidence_threshold,
            "min_count_interval_s": args.min_count_interval,
            "range_bin_count": int(range_config.num_range_bins),
            "range_bin_spacing_m": float(range_config.bin_spacing_m),
            "started_at": now_text(),
        },
    )

    print(f"Session folder: {session_dir}")
    print(f"Loaded classifier: {classifier_label}")
    print(f"Window: {window_seconds:.2f} s; step: {args.step_seconds:.2f} s")
    print(
        "Squat logic: "
        f"{args.standing_label} -> {args.squat_label} -> {args.standing_label} = 1"
    )

    try:
        serial_port = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as error:
        raise SystemExit(f"Could not open serial port {args.port}: {error}") from None

    plot = None
    if not args.no_plot:
        try:
            plot = SquatCounterPlot(feature_params, args.standing_label, args.squat_label)
        except ImportError as exc:
            print(f"Plot disabled; missing plotting dependency: {exc}")

    counter = SquatCounter(
        standing_label=args.standing_label,
        squat_label=args.squat_label,
        min_count_interval_s=args.min_count_interval,
    )
    stable_filter = ConsecutivePredictionFilter(
        args.stable_predictions,
        args.confidence_threshold,
    )
    all_records: list[dict] = []
    recent_records: deque[dict] = deque()
    prediction_count = 0
    count_events = 0
    interrupted = False
    failed = False

    with predictions_path.open("w", newline="") as prediction_file:
        writer = csv.DictWriter(
            prediction_file,
            fieldnames=[
                "time_s",
                "frame_number",
                "window_frames",
                "prediction",
                "confidence",
                "stable_label",
                "state",
                "event",
                "squat_count",
                "scores_json",
                "point_count",
            ],
        )
        writer.writeheader()

        with serial_port as port:
            if args.no_config:
                port.reset_input_buffer()
            else:
                print(f"Using cfg: {args.cfg}")
                stop_and_drain(port)
                if not args.no_warm_reset:
                    warm_reset_demo(port)
                try:
                    send_configuration(
                        port,
                        remove_leading_sensor_stop(commands),
                        use_cfg_baud_rate=False,
                    )
                except (RuntimeError, ValueError) as error:
                    raise SystemExit(f"Could not configure radar: {error}") from None

            try:
                print("Start standing, squat, then return to standing. Use Ctrl+C to stop.")
                start_time = time.monotonic()
                last_prediction_time = start_time
                consecutive_warnings = 0
                feature_mismatch_reported = False
                display_prediction = None
                display_confidence = None
                display_stable_label = None
                display_event = ""

                while args.duration <= 0 or time.monotonic() - start_time < args.duration:
                    if plot is not None and not plot.is_open():
                        break
                    if plot is not None and plot.reset_requested:
                        counter.reset()
                        stable_filter.reset()
                        plot.reset_requested = False
                        display_stable_label = None
                        display_event = "reset"
                        print("Squat counter reset.")

                    try:
                        frame_number, cloud = read_point_cloud_frame(
                            port,
                            args.frame_timeout,
                        )
                    except (TimeoutError, ValueError, RuntimeError) as error:
                        consecutive_warnings += 1
                        print(f"\nFrame parse warning: {error}", flush=True)
                        if consecutive_warnings >= 3:
                            raise RuntimeError("No valid frames received.") from error
                        continue

                    consecutive_warnings = 0
                    now = time.monotonic()
                    elapsed_s = now - start_time
                    xyz, velocity = cloud_arrays(cloud)
                    record = {
                        "time_s": elapsed_s,
                        "frame_number": frame_number,
                        "points_xyz": xyz,
                        "points_velocity": velocity,
                    }
                    all_records.append(record)
                    recent_records.append(record)
                    while recent_records and elapsed_s - recent_records[0]["time_s"] > window_seconds:
                        recent_records.popleft()

                    if (
                        now - last_prediction_time >= args.step_seconds
                        and len(recent_records) >= min_window_frames
                    ):
                        window_data = records_to_posture_data(list(recent_records))
                        if window_data is not None:
                            features, names = extract_posture_feature_vector(
                                window_data,
                                feature_params,
                            )
                        else:
                            features, names = None, []

                        if features is not None and feature_names and len(features) != len(feature_names):
                            if not feature_mismatch_reported:
                                print(
                                    "\nFeature length mismatch: "
                                    f"model expects {len(feature_names)}, got {len(features)}."
                                )
                                feature_mismatch_reported = True
                        elif features is not None:
                            prediction = model.predict([features])[0]
                            confidence = prediction_confidence(model, features, prediction)
                            scores = prediction_scores(model, features)
                            stable_label = stable_filter.update(prediction, confidence)
                            event = counter.update(stable_label, elapsed_s)
                            if event == "count":
                                count_events += 1

                            display_prediction = prediction
                            display_confidence = confidence
                            display_stable_label = stable_label
                            display_event = event
                            prediction_count += 1
                            last_prediction_time = now

                            writer.writerow(
                                {
                                    "time_s": f"{elapsed_s:.3f}",
                                    "frame_number": frame_number,
                                    "window_frames": len(recent_records),
                                    "prediction": prediction,
                                    "confidence": "" if confidence is None else f"{confidence:.4f}",
                                    "stable_label": "" if stable_label is None else stable_label,
                                    "state": counter.state,
                                    "event": event,
                                    "squat_count": counter.count,
                                    "scores_json": json.dumps(scores),
                                    "point_count": len(xyz),
                                }
                            )
                            prediction_file.flush()

                            print(
                                f"count={counter.count} | state={counter.state} | "
                                f"prediction={prediction}"
                                + ("" if confidence is None else f" ({confidence:.2f})")
                                + f" | stable={stable_label or '-'} | event={event or '-'} | "
                                f"scores: {format_scores(scores)}",
                                flush=True,
                            )

                    if plot is not None:
                        plot.update(
                            xyz,
                            counter,
                            display_prediction,
                            display_confidence,
                            display_stable_label,
                            display_event,
                        )

            except KeyboardInterrupt:
                interrupted = True
                print("\nInterrupted. Stopping radar...")
            except RuntimeError as error:
                failed = True
                print(f"\nSquat counter failed: {error}")
            finally:
                if not args.no_config:
                    print("> sensorStop 0")
                    stop_and_drain(port)

    if not args.no_save_frames:
        save_frame_stream(frames_path, all_records, model_path, args.cfg)

    write_json(
        summary_path,
        {
            "input_type": INPUT_TYPE,
            "model": str(model_path),
            "predictions_csv": str(predictions_path),
            "frames_npz": "" if args.no_save_frames else str(frames_path),
            "frame_count": len(all_records),
            "prediction_count": prediction_count,
            "squat_count": counter.count,
            "count_events": count_events,
            "final_state": counter.state,
            "interrupted": interrupted,
            "failed": failed,
            "finished_at": now_text(),
        },
    )
    print(f"Final squat count: {counter.count}")
    print(f"Predictions saved to: {predictions_path}")
    if not args.no_save_frames:
        print(f"Frames saved to: {frames_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
