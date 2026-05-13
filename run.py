"""FEM-NET submission runner (Windows-friendly).

Commands:
  python run.py serve
  python run.py simulate --condition <cond> --rounds <n>
  python run.py train --condition <cond> --epochs <n>
  python run.py preprocess --condition <cond>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from core.constants import SUPPORTED_CONDITIONS


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _run_serve() -> int:
    # Start via `python -m uvicorn ...` for reliable Windows module resolution.
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    return subprocess.call(cmd, cwd=_repo_root())


def _run_simulate(condition: str, rounds: int) -> int:
    from simulation.run_simulation import run_federated_simulation

    run_federated_simulation(condition=condition, num_rounds=rounds)
    return 0


def _run_train(condition: str, epochs: int) -> int:
    from training.train_model import train

    train(condition=condition, epochs=epochs)
    return 0


def _run_preprocess(condition: str) -> int:
    script = os.path.join(_repo_root(), "data", f"preprocess_{condition}.py")
    if not os.path.exists(script):
        print(f"[ERROR] Preprocess script not found: data/preprocess_{condition}.py")
        print("        Available conditions:")
        for c in SUPPORTED_CONDITIONS:
            print(f"        - {c}")
        return 2

    try:
        subprocess.check_call([sys.executable, script], cwd=_repo_root())
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Preprocess failed (exit code {exc.returncode}).")
        return exc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run.py", description="FEM-NET submission runner")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="Start FastAPI server (serves dashboard)")

    p_sim = sub.add_parser("simulate", help="Run in-process federated simulation")
    p_sim.add_argument("--condition", required=True, choices=SUPPORTED_CONDITIONS)
    p_sim.add_argument("--rounds", required=True, type=int)

    p_train = sub.add_parser("train", help="Train a local model for one condition")
    p_train.add_argument("--condition", required=True, choices=SUPPORTED_CONDITIONS)
    p_train.add_argument("--epochs", required=True, type=int)

    p_pre = sub.add_parser("preprocess", help="Run preprocessing script for one condition")
    p_pre.add_argument("--condition", required=True)

    args = parser.parse_args(argv)

    if args.command == "serve":
        return _run_serve()
    if args.command == "simulate":
        return _run_simulate(args.condition, args.rounds)
    if args.command == "train":
        return _run_train(args.condition, args.epochs)
    if args.command == "preprocess":
        return _run_preprocess(args.condition)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
