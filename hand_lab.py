#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial

from get_range_profile import (
    RANGE_PROFILE_MAJOR,
    RANGE_PROFILE_MINOR,
    load_configuration,
    parse_range_config,
    read_frame,
    send_configuration,
    write_cli_command,
)


def range_profile_from_tlvs(tlvs: list[tuple[int, bytes]]) -> np.ndarray | None:
    for tlv_type, payload in tlvs:
        if tlv_type in {RANGE_PROFILE_MAJOR, RANGE_PROFILE_MINOR}:
            return np.frombuffer(payload, dtype="<u4").astype(float)
    return None


def estimate_distance(
    profile: np.ndarray,
    background: np.ndarray,
    bin_spacing_m: float,
    min_range_m: float,
    max_range_m: float,
    peak_ratio: float,
) -> tuple[float | None, np.ndarray]:
    difference = subtract_background(profile, background)

    start_bin = max(1, int(math.ceil(min_range_m / bin_spacing_m)))
    stop_bin = min(len(difference), int(math.floor(max_range_m / bin_spacing_m)) + 1)
    window = difference[start_bin:stop_bin]

    if len(window) == 0:
        return None, difference

    # Find the strongest peak inside the window.
    peak_bin_offset = int(np.argmax(window))
    peak_value = float(window[peak_bin_offset])

    # Reject weak peaks: compare peak against a typical window level (median).
    # The peak_ratio parameter defines the minimum multiple of the median level.
    window_level = float(np.median(window))
    if window_level <= 0:
        window_level = float(np.mean(window))
    if window_level <= 0 or peak_value < peak_ratio * window_level:
        return None, difference

    # Convert the peak bin index to meters.
    peak_bin = peak_bin_offset + start_bin
    peak_m = peak_bin * bin_spacing_m
    return peak_m, difference



def db_scale(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, 0.0) + 1.0)


def read_lab_frame(
    port: serial.Serial,
    frame_timeout: float,
    expected_range_profile_bytes: int,
) -> tuple[int, np.ndarray]:
    while True:
        frame_number, tlvs = read_frame(
            port,
            frame_timeout,
            expected_range_profile_bytes,
        )
        profile = range_profile_from_tlvs(tlvs)
        if profile is not None:
            return frame_number, profile


def collect_background_profiles(
    port: serial.Serial,
    frame_count: int,
    frame_timeout: float,
    expected_range_profile_bytes: int,
) -> list[np.ndarray]:
    """Collect empty-scene range profiles before the hand enters the scene."""
    profiles = []
    while len(profiles) < frame_count:
        _, profile = read_lab_frame(port, frame_timeout, expected_range_profile_bytes)
        if profile is not None:
            profiles.append(profile)
    return profiles

def make_background(profiles: list[np.ndarray]) -> np.ndarray:
    """Compute one stable background profile from the empty-scene profiles."""
    stacked = np.stack(profiles)
    return np.median(stacked, axis=0)

def subtract_background(profile: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Return the positive range-profile change after background subtraction."""
    return np.max(profile - background, 0)


def smooth_distance(
    history: deque[float],
    distance: float | None,
    method: str,
) -> float | None:
    """Smooth the latest distance estimate with a short mean or median filter.

    Mean smoothing responds more quickly but a single bad peak can pull
    the estimate away from the true distance. Median smoothing is more
    robust to outliers because it ignores extreme values.
    """
    if distance is None:
        return None

    history.append(float(distance))
    if method == "mean":
        return float(np.mean(history))
    elif method == "median":
        return float(np.median(history))

    return float(np.median(history))


def stop_and_drain(port: serial.Serial) -> None:
    """Stop a previous run if it is still streaming binary frames."""
    port.reset_input_buffer()

    deadline = time.monotonic() + 4.0
    last_rx = time.monotonic()
    next_stop = 0.0
    text_tail = ""

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_stop:
            port.write(b"sensorStop 0\r\n")
            port.flush()
            next_stop = now + 0.25

        waiting = port.in_waiting
        if waiting:
            data = port.read(waiting)
            text_tail = (text_tail + data.decode("ascii", errors="ignore"))[-512:]
            last_rx = time.monotonic()
        elif "done" in text_tail.lower() and time.monotonic() - last_rx > 0.25:
            break
        elif "mmwdemo:/>" in text_tail.lower() and time.monotonic() - last_rx > 0.5:
            break
        else:
            time.sleep(0.02)

    quiet_deadline = time.monotonic() + 0.6
    while time.monotonic() < quiet_deadline:
        waiting = port.in_waiting
        if waiting:
            port.read(waiting)
            quiet_deadline = time.monotonic() + 0.6
        else:
            time.sleep(0.02)

    port.reset_input_buffer()
    time.sleep(0.2)


def remove_leading_sensor_stop(commands: list[str]) -> list[str]:
    if commands and commands[0].split()[0] == "sensorStop":
        return commands[1:]
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal hand distance lab.")
    parser.add_argument("--port", required=True)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("xwrL64xx-evm/hand_distance.cfg"),
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--background-frames", type=int, default=20)
    parser.add_argument("--min-range", type=float, default=0.15)
    parser.add_argument("--max-range", type=float, default=2.0)
    parser.add_argument("--peak-ratio", type=float, default=3.0)
    parser.add_argument("--smooth-frames", type=int, default=5)
    parser.add_argument("--smooth-method", choices=["mean", "median"], default="median")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument(
        "--db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dB display scale. Pass --no-db for linear strength.",
    )
    args = parser.parse_args()

    print(f"Using cfg: {args.cfg}")
    commands = load_configuration(args.cfg)
    range_config = parse_range_config(commands)
    if range_config is None:
        raise RuntimeError("Could not compute range-bin spacing from cfg.")
    if args.background_frames < 1:
        raise RuntimeError("--background-frames must be at least 1.")
    if args.smooth_frames < 1:
        raise RuntimeError("--smooth-frames must be at least 1.")

    expected_bytes = range_config.num_range_bins * 4
    ranges = np.arange(range_config.num_range_bins) * range_config.bin_spacing_m
    range_mask = (ranges >= args.min_range) & (ranges <= args.max_range)
    if not np.any(range_mask):
        raise RuntimeError("The requested range window has no FFT bins.")
    plot_ranges = ranges[range_mask]

    with serial.Serial(args.port, args.baud, timeout=0.2) as port:
        stop_and_drain(port)
        try:
            send_configuration(
                port,
                remove_leading_sensor_stop(commands),
                use_cfg_baud_rate=False,
            )
        except (RuntimeError, ValueError) as error:
            raise SystemExit(f"Could not configure radar: {error}") from None

        try:
            print(
                f"Keep the scene empty for {args.background_frames} frames "
                "to capture the background."
            )
            backgrounds = collect_background_profiles(
                port,
                args.background_frames,
                args.frame_timeout,
                expected_bytes,
            )
            background = make_background(backgrounds)
            print("Background captured. Put your hand in front of the radar.")

            plt.ion()
            figure, axis = plt.subplots()
            (line,) = axis.plot(plot_ranges, np.zeros_like(plot_ranges))
            axis.set_xlabel("Range (m)")
            if args.db:
                axis.set_ylabel("Background-subtracted strength (dB)")
            else:
                axis.set_ylabel("Background-subtracted strength")
            axis.set_xlim(args.min_range, args.max_range)
            axis.set_ylim(0.0, 10.0)
            axis.grid(True)

            frame_count = 0
            distance_history: deque[float] = deque(maxlen=args.smooth_frames)
            while plt.fignum_exists(figure.number):
                frame_number, profile = read_lab_frame(
                    port,
                    args.frame_timeout,
                    expected_bytes,
                )
                distance, difference = estimate_distance(
                    profile,
                    background,
                    range_config.bin_spacing_m,
                    args.min_range,
                    args.max_range,
                    args.peak_ratio,
                )
                smoothed_distance = smooth_distance(
                    distance_history,
                    distance,
                    args.smooth_method,
                )
                if args.db:
                    display_strength = db_scale(difference[range_mask])
                else:
                    display_strength = difference[range_mask]
                line.set_ydata(display_strength)
                if args.db:
                    axis.set_ylim(
                        0.0,
                        max(10.0, float(np.percentile(display_strength, 98)) * 1.2),
                    )
                else:
                    axis.set_ylim(
                        0.0,
                        max(5e5, float(np.percentile(display_strength, 98)) * 1.2),
                    )

                if smoothed_distance is None:
                    message = f"frame {frame_number}: no hand target"
                else:
                    message = (
                        f"frame {frame_number}: distance {smoothed_distance:5.2f} m "
                        f"({args.smooth_method} smoothed)"
                    )

                axis.set_title(message)
                print("\r" + message, end="", flush=True)
                figure.canvas.draw_idle()
                plt.pause(0.001)

                frame_count += 1
                if args.frames and frame_count >= args.frames:
                    break

            print()

        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            print("> sensorStop 0")
            stop_and_drain(port)


if __name__ == "__main__":
    main()
