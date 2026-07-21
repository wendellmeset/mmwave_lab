#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
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


def read_range_profile_frame(
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


def stop_and_drain(port: serial.Serial) -> None:
    """Recover the CLI even if binary frames are still streaming."""
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


def db_scale(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, 0.0) + 1.0)


def motion_residual(
    profile: np.ndarray,
    background: np.ndarray,
    residual_mode: str,
) -> np.ndarray:
    residual = profile - background
    if residual_mode == "positive":
        return np.maximum(residual, 0.0)
    return np.abs(residual)


def update_background(
    background: np.ndarray,
    profile: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Update the background with an exponential moving average."""
    # TODO: implement the exponential background update:
    # new_background = (1 - alpha) * background + alpha * profile
    # Try several alpha values and explain the tradeoff.
    new_background = (1 - alpha) * background + alpha * profile
    return new_background


def estimate_motion_target(
    motion: np.ndarray,
    bin_spacing_m: float,
    min_range_m: float,
    max_range_m: float,
    peak_ratio: float,
) -> tuple[float | None, float]:
    start_bin = max(1, int(math.ceil(min_range_m / bin_spacing_m)))
    stop_bin = min(len(motion), int(math.floor(max_range_m / bin_spacing_m)) + 1)
    window = motion[start_bin:stop_bin]

    if len(window) == 0:
        return None, 0.0

    peak_index = start_bin + int(np.argmax(window))
    peak_strength = float(motion[peak_index])
    typical_strength = float(np.median(window)) + 1.0

    if peak_strength < peak_ratio * typical_strength:
        return None, peak_strength

    left = max(start_bin, peak_index - 1)
    right = min(stop_bin, peak_index + 2)
    indices = np.arange(left, right, dtype=float)
    weights = motion[left:right].astype(float)
    weight_sum = float(np.sum(weights))
    if weight_sum > 0:
        refined_bin = float(np.sum(indices * weights) / weight_sum)
    else:
        refined_bin = float(peak_index)

    return refined_bin * bin_spacing_m, peak_strength


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Moving-hand detector using exponential background subtraction."
    )
    parser.add_argument("--port", required=True)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("xwrL64xx-evm/hand_distance.cfg"),
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--init-frames",
        type=int,
        default=20,
        help="Initial empty-scene frames used to seed the background.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="EMA update rate. Smaller values keep motion visible longer.",
    )
    parser.add_argument("--min-range", type=float, default=0.15)
    parser.add_argument("--max-range", type=float, default=2.0)
    parser.add_argument("--peak-ratio", type=float, default=3.0)
    parser.add_argument(
        "--residual",
        choices=["absolute", "positive"],
        default="absolute",
        help="absolute detects increases and decreases; positive detects new peaks.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=80,
        help="Number of recent frames shown in the waterfall plot.",
    )
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument(
        "--db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dB display scale. Pass --no-db for linear strength.",
    )
    args = parser.parse_args()

    if args.init_frames < 1:
        raise SystemExit("--init-frames must be at least 1.")
    if args.history_frames < 2:
        raise SystemExit("--history-frames must be at least 2.")
    if not 0.0 < args.alpha <= 1.0:
        raise SystemExit("--alpha must be in the interval (0, 1].")
    if args.min_range >= args.max_range:
        raise SystemExit("--min-range must be smaller than --max-range.")

    print(f"Using cfg: {args.cfg}")
    commands = load_configuration(args.cfg)
    range_config = parse_range_config(commands)
    if range_config is None:
        raise SystemExit("Could not compute range-bin spacing from cfg.")

    expected_bytes = range_config.num_range_bins * 4
    ranges = np.arange(range_config.num_range_bins) * range_config.bin_spacing_m
    range_mask = (ranges >= args.min_range) & (ranges <= args.max_range)
    if not np.any(range_mask):
        raise SystemExit("The requested range window has no FFT bins.")
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
                f"Keep the scene still for {args.init_frames} frames "
                "to initialize the background."
            )
            seed_profiles = []
            while len(seed_profiles) < args.init_frames:
                _frame_number, profile = read_range_profile_frame(
                    port,
                    args.frame_timeout,
                    expected_bytes,
                )
                seed_profiles.append(profile)

            background = np.median(np.vstack(seed_profiles), axis=0)
            print("Background initialized. Move your hand in front of the radar.")

            plt.ion()
            figure, (axis, waterfall_axis) = plt.subplots(
                2,
                1,
                figsize=(9, 7),
                sharex=True,
                gridspec_kw={"height_ratios": (1.0, 1.2)},
            )
            (line,) = axis.plot(plot_ranges, np.zeros_like(plot_ranges))
            if args.db:
                strength_label = "Motion strength (dB)"
                default_y_max = 10.0
            else:
                strength_label = "Motion strength"
                default_y_max = 5e5
            axis.set_ylabel(strength_label)
            axis.set_xlim(args.min_range, args.max_range)
            axis.set_ylim(0.0, default_y_max)
            axis.grid(True)

            history = np.zeros((args.history_frames, len(plot_ranges)), dtype=float)
            waterfall = waterfall_axis.imshow(
                history,
                aspect="auto",
                origin="lower",
                extent=(plot_ranges[0], plot_ranges[-1], -args.history_frames, 0),
                cmap="magma",
                vmin=0.0,
                vmax=default_y_max,
            )
            waterfall_axis.set_xlabel("Range (m)")
            waterfall_axis.set_ylabel("Recent frames")
            colorbar = figure.colorbar(waterfall, ax=waterfall_axis, pad=0.02)
            colorbar.set_label(strength_label)

            frame_count = 0
            while plt.fignum_exists(figure.number):
                frame_number, profile = read_range_profile_frame(
                    port,
                    args.frame_timeout,
                    expected_bytes,
                )

                motion = motion_residual(profile, background, args.residual)
                distance, peak_strength = estimate_motion_target(
                    motion,
                    range_config.bin_spacing_m,
                    args.min_range,
                    args.max_range,
                    args.peak_ratio,
                )

                if args.db:
                    display_strength = db_scale(motion[range_mask])
                else:
                    display_strength = motion[range_mask]
                line.set_ydata(display_strength)
                y_max = max(
                    default_y_max,
                    float(np.percentile(display_strength, 98)) * 1.2,
                )
                axis.set_ylim(0.0, y_max)

                history = np.roll(history, -1, axis=0)
                history[-1, :] = display_strength
                waterfall.set_data(history)
                waterfall.set_clim(
                    0.0,
                    max(default_y_max, float(np.percentile(history, 99)) * 1.2),
                )

                if distance is None:
                    message = f"frame {frame_number}: no moving target"
                else:
                    if args.db:
                        strength_text = (
                            f"{db_scale(np.array([peak_strength]))[0]:4.1f} dB"
                        )
                    else:
                        strength_text = f"{peak_strength:.0f}"
                    message = (
                        f"frame {frame_number}: moving target {distance:5.2f} m, "
                        f"strength {strength_text}"
                    )

                axis.set_title(message)
                print("\r" + message, end="", flush=True)
                figure.canvas.draw_idle()
                plt.pause(0.001)

                background = update_background(background, profile, args.alpha)

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
