#!/usr/bin/env python3
"""Run all production PLM-only S2F experiments in dependency order."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


S2F_ROOT = Path(__file__).resolve().parents[1]
S2F_PYTHON = Path("/home/marcelo_baez/anaconda3/envs/S2F/bin/python")
S2F_INSTALLATION = Path("/run/media/marcelo_baez/HD_Disc1/.S2F")
RUNS = (
    ("83333_plm_knn_k160", "conf/plm_only/83333_plm_knn_k160.conf"),
    ("83333_plm_kde", "conf/plm_only/83333_plm_kde.conf"),
    ("1111708_plm_knn_k80", "conf/plm_only/1111708_plm_knn_k80.conf"),
    ("1111708_plm_kde", "conf/plm_only/1111708_plm_kde.conf"),
)


def run_and_log(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=S2F_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    for alias, config in RUNS:
        run_and_log(
            [
                str(S2F_PYTHON), "-u", "S2F.py", "predict",
                "--run-config", config,
            ],
            S2F_INSTALLATION / "output" / f"{alias}.log",
        )
    subprocess.run(
        [str(S2F_PYTHON), "scripts/evaluate_plm_only_s2f.py"],
        cwd=S2F_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
