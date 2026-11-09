"""Pull the crash frames' forward reading down without moving the canyon.

The route 0 canyon is an anchor: its readout must stay put. Nothing is flown.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.aerial.rl.clearance_stick import _opening
from experiments.aerial.rl.clearance_view import ClearanceView
from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.scripts.train_clearance_view import _center_stats, _depth, _eval, _raw_rows, _report
from experiments.aerial.scripts.train_posterior_gate import _batch, _clear


def _close_wall(tr) -> bool:
    mean_m, _p10 = _center_stats(_depth(tr.obs))
    _fwd, left, right = _clear(tr.obs)
    return float(mean_m) < 12.0 and not _opening(mean_m, left, right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True)
    parser.add_argument("--street", required=True)
    parser.add_argument("--crash", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    street = load_episode(Path(args.street))
    canyon = [tr for tr in street if float(np.asarray(tr.obs.position)[1]) <= 50.0]
    crash = [tr for tr in load_episode(Path(args.crash)) if _close_wall(tr)]
    print(f"anchor={len(street)} crash_close={len(crash)} canyon={len(canyon)}")
    if len(crash) < 5 or len(canyon) < 10:
        raise SystemExit("refuse: not enough frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = ClearanceView().to(device)
    teacher.load_state_dict(torch.load(args.init, map_location=device)["model"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    student = ClearanceView().to(device)
    student.load_state_dict(teacher.state_dict())
    opt = torch.optim.Adam(student.parameters(), lr=float(args.lr))

    anchor_rgb, _ = _batch(street, None)
    crash_rgb, _ = _batch(crash, None)
    crash_target = torch.as_tensor(_raw_rows(crash)[:, 0], dtype=torch.float32)
    student.train()
    for step in range(int(args.steps)):
        opt.zero_grad()
        a_pick = torch.randint(0, anchor_rgb.shape[0], (16,))
        c_pick = torch.randint(0, crash_rgb.shape[0], (16,))
        with torch.no_grad():
            teacher_out = teacher(anchor_rgb[a_pick].to(device))
        student_anchor = student(anchor_rgb[a_pick].to(device))
        student_crash = student(crash_rgb[c_pick].to(device))
        anchor_loss = F.mse_loss(student_anchor, teacher_out)
        crash_loss = F.mse_loss(student_crash[:, 0], torch.log1p(crash_target[c_pick].to(device)))
        loss = anchor_loss + crash_loss
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 50 == 0:
            print(
                f"step={step} loss={float(loss.detach()):.4f} "
                f"anchor={float(anchor_loss.detach()):.4f} crash={float(crash_loss.detach()):.4f}"
            )

    student.eval()
    pred, raw = _eval(student, canyon)
    acc, corr, sign_n = _report("canyon", pred, raw)
    crash_pred, crash_raw = _eval(student, crash)
    crash_m = np.expm1(crash_pred[:, 0])
    print(
        f"crash pred_med={float(np.median(crash_m)):.1f} "
        f"true_med={float(np.median(crash_raw[:, 0])):.1f} "
        f"frac<12={float((crash_m < 12).mean()):.2f}"
    )
    ok = acc >= 0.8 and corr >= 0.5 and sign_n >= 10 and float(np.median(crash_m)) < 12.0
    out = Path(args.out)
    if ok:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": student.state_dict()}, str(out))
        print(f"PASS saved {out}")
    else:
        print("FAIL near anchor")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
