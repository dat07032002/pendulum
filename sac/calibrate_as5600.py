"""
AS5600 calibration helper for the Furuta pendulum.

1. Flash sac/firmware/as5600_calibrate/as5600_calibrate.ino to the ESP32.
2. Run:  python sac/calibrate_as5600.py --port COM5
3. Let the pendulum hang freely when prompted.
4. The script prints the UPRIGHT_RAW value to put in nidec_policy.ino.
"""
import argparse
import time
import serial


def read_raw(ser: serial.Serial, n_samples: int = 30, timeout: float = 10.0) -> float:
    samples = []
    deadline = time.time() + timeout
    ser.reset_input_buffer()
    while len(samples) < n_samples and time.time() < deadline:
        line = ser.readline().decode(errors="ignore").strip()
        if not line.startswith("raw="):
            continue
        try:
            raw_str = line.split()[0].split("=")[1]
            samples.append(int(raw_str))
            print(f"  sample {len(samples):2d}/{n_samples}  raw={samples[-1]}")
        except (IndexError, ValueError):
            pass
    if not samples:
        raise RuntimeError("No samples received — check port and baud rate.")
    return sum(samples) / len(samples)


def main():
    parser = argparse.ArgumentParser(description="AS5600 calibration for Furuta pendulum.")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--baud", type=int, default=921600)
    args = parser.parse_args()

    print(f"Connecting to {args.port} at {args.baud} baud...")
    with serial.Serial(args.port, args.baud, timeout=1.0) as ser:
        time.sleep(1.5)  # let ESP32 boot

        print("\n=== STEP 1: HANGING position ===")
        print("Let the pendulum hang freely (do NOT touch it).")
        input("Press Enter when it is still...")
        print("Reading raw values...")
        raw_hang = read_raw(ser)
        print(f"  -> mean raw (hanging) = {raw_hang:.1f}")

        # UPRIGHT_RAW = (raw_hanging - 2048 + 4096) % 4096
        upright_raw = int(round((raw_hang - 2048 + 4096))) % 4096

        print("\n=== STEP 2: UPRIGHT verification (optional) ===")
        ans = input("Hold the pendulum UPRIGHT and press Enter to verify, or skip (S): ").strip().lower()
        if ans != "s":
            print("Reading raw values (hold steady)...")
            raw_up = read_raw(ser, n_samples=20)
            print(f"  -> mean raw (upright) = {raw_up:.1f}")
            diff = abs(raw_up - upright_raw)
            if diff > 2048:
                diff = 4096 - diff
            print(f"  -> difference from computed UPRIGHT_RAW: {diff:.1f} counts "
                  f"= {diff * 360 / 4096:.2f} deg")
            if diff < 50:
                print("  ✓ Consistent (< ~4 deg error) — computed value is reliable.")
            else:
                print("  ⚠ Large discrepancy — use the direct upright reading instead.")
                upright_raw = int(round(raw_up)) % 4096
                print(f"  -> Using direct upright raw = {upright_raw}")

    print("\n" + "=" * 50)
    print(f"  UPRIGHT_RAW = {upright_raw}")
    print("=" * 50)
    print(f"\nIn nidec_policy.ino, update line:")
    print(f"  const int UPRIGHT_RAW = {upright_raw};")
    print("\nThen re-flash nidec_policy.ino to the ESP32.")


if __name__ == "__main__":
    main()
