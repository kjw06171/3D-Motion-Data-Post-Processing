import os
import numpy as np
import pandas as pd
from glob import glob
import os, sys
import warnings
from scipy.signal import savgol_filter, butter, filtfilt   # [ADDED] smoothing
from scipy.ndimage import uniform_filter1d                # [ADDED] moving average
warnings.filterwarnings("ignore", message="All-NaN slice")   # [ADDED] NaN = missing frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RAW_KEYPOINTS_DIR = os.path.join(ROOT, "data", "raw_keypoints")
OUT_DIR = os.path.join(ROOT, "data", "processed")

os.makedirs(OUT_DIR, exist_ok=True)

VISUALIZE = True    # [ADDED] set False to skip the viser preview
COMPARE = True      # [ADDED] save before/after figures to data/figures
AUTO_SMOOTH = True  # [ADDED] pick the best smoothing filter per video (cross-validation)
HAND_STABILIZE = True  # [ADDED] straighten hands along the forearm (stable palms in Unity)
PALM_FLIP = {"left": False, "right": False}   # [ADDED] True → turn that palm 180° (if hands look reversed in Unity)
FPS = 30    # fallback frame rate (used only if the .npz has no saved fps)


# =====================================================================
# Original code (unchanged)
# =====================================================================
def load_raw_keypoints(npz_path):
    """Load BlazePose raw data (T,33,3)."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame","landmark","x","y","z","visibility"])

    pts_seq = []

    for f, group in df.groupby("frame"):
        group = group.sort_values("landmark")
        pts = group[["x","y","z"]].values  # (33,3)
        pts_seq.append(pts)

    return np.array(pts_seq)   # (T,33,3)


def create_height_corrected_target(raw_seq):
    """
    raw_seq: (T,33,3)
    Return target height corrected (T,33,3)
    """

    seq = raw_seq.copy()

    # BlazePose foot joints
    LEFT_FOOT = 31
    RIGHT_FOOT = 32

    # 각 프레임에서 두 발끝 중 낮은 y값 → 그 프레임의 ground level
    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])

    # reshape to (T,1) → broadcast to (T,33)
    frame_ground = foot_y[:, None]

    # 전체 시퀀스를 ground=0으로 맞춤
    seq[:, :, 1] -= frame_ground

    # 땅 아래(-)로 떨어지지 않게 clamp
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)

    return seq


# =====================================================================
# [ADDED] Refinement pipeline
#   1) restore dropped frames   2) fix left/right swaps
#   3) remove outliers          4) temporal smoothing (stronger on z)
#   5) constant + symmetric bone lengths
#   6) hand stabilization (hand continues the forearm)
#   7) ground correction without clamp
# =====================================================================

# BlazePose kinematic tree: child -> parent (hips 23/24 are the root)
PARENT = {
    11: 23, 13: 11, 15: 13, 17: 15, 19: 15, 21: 15,     # left arm
    12: 24, 14: 12, 16: 14, 18: 16, 20: 16, 22: 16,     # right arm
    25: 23, 27: 25, 29: 27, 31: 27,                      # left leg
    26: 24, 28: 26, 30: 28, 32: 28,                      # right leg
}
FACE = list(range(0, 11))                 # moved rigidly with the shoulders
FOOT_JOINTS = [27, 28, 29, 30, 31, 32]    # ankles, heels, toes

# left/right pairs checked for label swaps (shoulders/hips excluded)
SWAP_GROUPS = {
    "arms": [(13, 14), (15, 16), (17, 18), (19, 20), (21, 22)],
    "legs": [(25, 26), (27, 28), (29, 30), (31, 32)],
}


def load_raw_with_meta(npz_path):
    """
    Like load_raw_keypoints, but keeps the frame index and visibility.
    Frames where MediaPipe found no person are inserted as NaN,
    so the time axis has no hidden jumps.

    return
      pts      : (T_full,33,3)  NaN on missing frames
      vis      : (T_full,33)    0 on missing frames
      detected : (T_full,)      True if the frame was detected
      frame_ids: (T_full,)      original video frame numbers
    """
    data = np.load(npz_path, allow_pickle=True)["data"].astype(float)
    df = pd.DataFrame(data, columns=["frame","landmark","x","y","z","visibility"])

    f0, f1 = int(df.frame.min()), int(df.frame.max())
    T_full = f1 - f0 + 1

    pts = np.full((T_full, 33, 3), np.nan)
    vis = np.zeros((T_full, 33))

    t = df.frame.values.astype(int) - f0
    j = df.landmark.values.astype(int)
    pts[t, j] = df[["x","y","z"]].values
    vis[t, j] = df["visibility"].values

    detected = ~np.isnan(pts[:, 0, 0])
    frame_ids = np.arange(f0, f1 + 1)
    return pts, vis, detected, frame_ids


def get_fps(npz_path):
    """Frame rate saved by p1_create_raw_keypoints (falls back to FPS for old files)."""
    data = np.load(npz_path, allow_pickle=True)
    return float(data["fps"]) if "fps" in data.files else FPS


def _interpolate_nan(seq, max_gap=None, fallback=None):
    """
    Linear interpolation over time for every joint/axis.
    If max_gap is set, NaN runs longer than max_gap are restored
    from `fallback` instead of being interpolated.
    """
    out = seq.copy()
    T = seq.shape[0]
    t = np.arange(T)
    flat = out.reshape(T, -1)
    fb = None if fallback is None else fallback.reshape(T, -1)

    for c in range(flat.shape[1]):
        col = flat[:, c]
        bad = np.isnan(col)
        if not bad.any() or bad.all():
            continue
        col[bad] = np.interp(t[bad], t[~bad], col[~bad])

        if max_gap is not None and fb is not None:
            # find NaN runs that are too long and put the original values back
            edges = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
            starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
            for s, e in zip(starts, ends):
                if e - s > max_gap:
                    col[s:e] = fb[s:e, c]
    return out


def fill_missing_frames(pts):
    """Step 1: interpolate frames where no person was detected."""
    return _interpolate_nan(pts)


def fix_left_right_swaps(pts, ratio=0.5):
    """
    Step 2: MediaPipe sometimes swaps left and right limbs for a few frames
    (the arm "teleports" to the other side). For each frame, compare
      keep : distance of each limb to the same limb in the previous frame
      swap : distance if left and right labels are exchanged
    If swapping is clearly better (swap < ratio * keep), swap the labels.
    """
    out = pts.copy()
    n_fixed = 0
    for pairs in SWAP_GROUPS.values():
        L = [a for a, b in pairs]
        R = [b for a, b in pairs]
        for t in range(1, len(out)):
            prev = out[t - 1]
            keep = (np.linalg.norm(out[t, L] - prev[L], axis=1).sum()
                    + np.linalg.norm(out[t, R] - prev[R], axis=1).sum())
            swap = (np.linalg.norm(out[t, R] - prev[L], axis=1).sum()
                    + np.linalg.norm(out[t, L] - prev[R], axis=1).sum())
            if swap < ratio * keep:
                out[t, L + R] = out[t, R + L]
                n_fixed += 1
    return out, n_fixed


def remove_outliers(pts, vis, vis_thresh=0.3, k=6.0, max_gap=10):
    """
    Step 3: mark suspicious joint positions and re-interpolate them.
      - spike : speed into AND out of the frame are both abnormally high
                (median + k * MAD per joint)  → single-frame jump
      - low visibility (< vis_thresh)
    Only gaps up to max_gap frames are replaced.
    """
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=2)          # (T-1,33)
    med = np.median(speed, axis=0)
    mad = np.median(np.abs(speed - med), axis=0) * 1.4826 + 1e-6
    fast = speed > (med + k * mad)                                # (T-1,33)

    spike = np.zeros(pts.shape[:2], dtype=bool)
    spike[1:-1] = fast[:-1] & fast[1:]

    low_vis = (vis < vis_thresh) & (vis > 0)   # vis==0 → missing frame (already filled)
    bad = spike | low_vis

    cleaned = pts.copy()
    cleaned[bad] = np.nan
    cleaned = _interpolate_nan(cleaned, max_gap=max_gap, fallback=pts)

    info = {"spikes": int(spike.sum()), "low_visibility": int(low_vis.sum())}
    return cleaned, info


def _savgol(x, window, polyorder):
    T = x.shape[0]
    if T < 5:
        return x.copy()
    window = min(window, T if T % 2 == 1 else T - 1)
    polyorder = min(polyorder, window - 1)
    return savgol_filter(x, window_length=window, polyorder=polyorder, axis=0)


def smooth_motion(pts, window=9, polyorder=2, z_window=15):
    """
    Step 4: Savitzky-Golay filter along time.
    Removes jitter while keeping peaks (e.g. top of an arm raise)
    better than a moving average.
    Depth (z) from a single camera is the noisiest axis,
    so it gets a longer window.
    """
    out = pts.copy()
    out[..., :2] = _savgol(pts[..., :2], window, polyorder)     # x, y
    out[..., 2] = _savgol(pts[..., 2], z_window, polyorder)     # z
    return out


def enforce_bone_lengths(pts):
    """
    Step 5: real bones don't stretch, and left/right bones are equal.
    Each bone gets the median length of the whole clip (averaged with
    the mirrored bone); the direction from the data is kept.
    Processed from the hips outward.
    """
    out = pts.copy()
    lengths = {c: np.median(np.linalg.norm(pts[:, c] - pts[:, p], axis=1))
               for c, p in PARENT.items()}
    # left (odd) / right (even) symmetry: 13<->14, 15<->16, ...
    lengths = {c: (lengths[c] + lengths[c + 1 if c % 2 else c - 1]) / 2
               for c in lengths}

    for c, p in PARENT.items():          # dict order = parents before children
        d = pts[:, c] - pts[:, p]
        n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
        out[:, c] = out[:, p] + d / n * lengths[c]

    # face follows the shift of the shoulder midpoint
    shift = (out[:, [11, 12]] - pts[:, [11, 12]]).mean(axis=1, keepdims=True)
    out[:, FACE] += shift
    return out


# (wrist, elbow, pinky, index, thumb) for each side
HANDS = {"left": (15, 13, 17, 19, 21), "right": (16, 14, 18, 20, 22)}


def stabilize_hands(pts, palm_window=31):
    """
    Step 6: MediaPipe gives only 3 small hand points (pinky, index, thumb).
    They are often hidden and jump around, so the palm direction computed
    in Unity flips → wrists bend in impossible ways.
    Here each hand is rebuilt as a straight continuation of the forearm:
      - forward : forearm direction (elbow → wrist)
      - sideways: thumb–pinky direction, smoothed over ~1 s and made
                  perpendicular to the forearm (keeps the rough palm
                  orientation, removes frame-to-frame flips)
      - lengths : median hand size of the clip (constant)
    """
    out = pts.copy()
    for name, (wrist, elbow, pinky, index, thumb) in HANDS.items():
        W = pts[:, wrist]
        fwd = W - pts[:, elbow]
        fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-8

        side = _savgol(pts[:, thumb] - pts[:, pinky], palm_window, 2)
        side -= (side * fwd).sum(axis=1, keepdims=True) * fwd       # ⟂ forearm
        n = np.linalg.norm(side, axis=1, keepdims=True)
        # fallback when degenerate: shoulder line made ⟂ forearm
        sh = pts[:, 11] - pts[:, 12]
        sh -= (sh * fwd).sum(axis=1, keepdims=True) * fwd
        side = np.where(n > 1e-4, side / (n + 1e-8),
                        sh / (np.linalg.norm(sh, axis=1, keepdims=True) + 1e-8))
        if PALM_FLIP.get(name):          # swap thumb/pinky side → palm turned 180°
            side = -side

        L_index = np.median(np.linalg.norm(pts[:, index] - W, axis=1))
        L_pinky = np.median(np.linalg.norm(pts[:, pinky] - W, axis=1))
        L_thumb = np.median(np.linalg.norm(pts[:, thumb] - W, axis=1))
        half = np.median(np.linalg.norm(pts[:, thumb] - pts[:, pinky], axis=1)) / 2

        out[:, index] = W + fwd * L_index
        out[:, pinky] = W + fwd * np.sqrt(max(L_pinky**2 - half**2, 1e-6)) - side * half
        out[:, thumb] = W + fwd * np.sqrt(max(L_thumb**2 - half**2, 1e-6)) + side * half
    return out


def ground_correction_refined(pts):
    """
    Step 7: same idea as create_height_corrected_target, but
      - uses ankles + heels + toes (not toes only), so the lowest
        foot point is exactly on y=0 → no joint goes below ground
      - therefore no clamp → no flattened heels
    """
    seq = pts.copy()
    ground = seq[:, FOOT_JOINTS, 1].min(axis=1)
    seq[:, :, 1] -= ground[:, None]
    return seq


def refine_motion(pts, vis, smoother=None, fps=FPS):
    """
    Run steps 1-7. Returns refined (T_full,33,3) and a log dict.
    smoother: (filter name, strength) from select_smoothing.
              None → default Savitzky-Golay (stronger on z).
    """
    info = {"missing_frames": int(np.isnan(pts[:, 0, 0]).sum())}
    x = fill_missing_frames(pts)
    x, info["lr_swaps"] = fix_left_right_swaps(x)
    x, out_info = remove_outliers(x, vis)
    info.update(out_info)
    if smoother is None:
        x = smooth_motion(x)
    else:
        func, _ = FILTERS[smoother[0]]
        x = func(x, smoother[1], fps)
        info["smoother"] = f"{smoother[0]} ({smoother[1]:g})"
    x = enforce_bone_lengths(x)
    if HAND_STABILIZE:
        x = stabilize_hands(x)
    x = ground_correction_refined(x)
    return x, info


# =====================================================================
# [ADDED] Before / after comparison  (figures for the report)
#   data/figures/<name>_problems.png  - slide 2: problems in raw data
#   data/figures/<name>_compare.png   - slide 4: raw vs provided vs refined
#   data/figures/<name>_skeleton.png  - one frame drawn three ways
#   data/figures/<name>_metrics.csv   - numbers for the table
#   data/figures/<name>_filters.png   - smoothing filter comparison (+ .csv)
# =====================================================================
FIG_DIR = os.path.join(ROOT, "data", "figures")

WRIST = 16                       # right wrist: moves a lot in most exercises
BONE = (14, 16)                  # right forearm
BONE_NAME = "right forearm"

COLORS = {"raw": "#9a9893", "provided": "#2a78d6", "refined": "#eb6834"}
LABELS = {"raw": "Raw (MediaPipe)", "provided": "Provided (height corr.)",
          "refined": "Refined (ours)"}

# skeleton lines for drawing
DRAW_BONES = list(PARENT.items()) + [(23, 24), (11, 12)]

PLOT_STYLE = {
    "axes.unicode_minus": False,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
    "axes.edgecolor": "#8a8984", "axes.titleweight": "bold",
    "axes.titlesize": 11, "legend.frameon": False, "lines.linewidth": 2,
}


# ---------------------------------------------------------------------
# metrics (all NaN-aware: missing frames are skipped)
# ---------------------------------------------------------------------
def jitter_mm(seq):
    """Mean acceleration magnitude in mm/frame^2 (lower = smoother)."""
    acc = seq[2:] - 2 * seq[1:-1] + seq[:-2]
    return float(np.nanmean(np.linalg.norm(acc, axis=2)) * 1000)


def bone_cv_percent(seq):
    """Mean coefficient of variation of bone lengths in % (0 = rigid)."""
    cvs = []
    for c, p in PARENT.items():
        L = np.linalg.norm(seq[:, c] - seq[:, p], axis=1)
        cvs.append(np.nanstd(L) / np.nanmean(L))
    return float(np.mean(cvs) * 100)


def ground_error_mm(seq):
    """Mean distance of the lowest foot point from y=0 in mm."""
    return float(np.nanmean(np.abs(np.nanmin(seq[:, FOOT_JOINTS, 1], axis=1))) * 1000)


def deviation_mm(seq, raw):
    """
    How far the body shape moved from the raw data, in mm.
    Both are compared relative to the hip center, so the ground shift
    doesn't count. Small = we removed noise without changing the motion.
    """
    rel = lambda s: s - s[:, [23, 24]].mean(axis=1, keepdims=True)
    return float(np.nanmean(np.linalg.norm(rel(seq) - rel(raw), axis=2)) * 1000)


def bone_length(seq, bone=BONE):
    c, p = bone
    return np.linalg.norm(seq[:, c] - seq[:, p], axis=1)


def to_full_timeline(seq_detected, detected):
    """Put a (T_detected,...) array back on the full timeline with NaN gaps."""
    full = np.full((len(detected),) + seq_detected.shape[1:], np.nan)
    full[detected] = seq_detected
    return full


# ---------------------------------------------------------------------
def analyze(name, pts, vis, detected, frame_ids, refined, info, fps=FPS, selection=None):
    """Compare raw / provided / refined and save figures + metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    globals()["plt"] = plt
    plt.rcParams.update(PLOT_STYLE)
    # use a Korean font if one is installed (video titles are often Korean)
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if font in installed:
            plt.rcParams["font.family"] = font
            break
    os.makedirs(FIG_DIR, exist_ok=True)

    raw_det = pts[detected]                                  # what the class code sees
    provided_det = create_height_corrected_target(raw_det)

    # how many joint values the provided method clamps to 0 (flattened)
    foot_y = np.minimum(raw_det[:, 31, 1], raw_det[:, 32, 1])
    below = raw_det[:, :, 1] - foot_y[:, None] < 0
    clamp_frames = float(below.any(axis=1).mean() * 100)

    seqs = {
        "raw": pts,                                          # NaN at missing frames
        "provided": to_full_timeline(provided_det, detected),
        "refined": refined,
    }
    t = (frame_ids - frame_ids[0]) / fps

    # ---------------- metrics table ----------------
    rows = []
    for k, s in seqs.items():
        rows.append({
            "method": k,
            "jitter (mm/frame^2)": jitter_mm(s),
            "bone length CV (%)": bone_cv_percent(s),
            "ground error (mm)": ground_error_mm(s),
            "deviation from raw (mm)": deviation_mm(s, pts),
            "missing frames": int(np.isnan(s[:, 0, 0]).sum()),
            "frames with clamped joints (%)": clamp_frames if k == "provided" else 0.0,
        })
    metrics = pd.DataFrame(rows).set_index("method")
    metrics.to_csv(os.path.join(FIG_DIR, f"{name}_metrics.csv"))

    print(f"\n=== {name} ===")
    print(f"frames: {len(t)}  detected: {detected.sum()}  log: {info}")
    print(metrics.round(2).to_string())

    plot_problems(name, t, pts, vis, detected, info)
    plot_compare(name, t, seqs, metrics)
    plot_skeleton(name, seqs, raw_det, detected)
    if selection is not None:
        best, table, x = selection
        plot_filters(name, t, pts, x, table, best, fps)
    return metrics


# ---------------------------------------------------------------------
# figure 1: problems in raw data (slide 2)
# ---------------------------------------------------------------------
def shade_missing(ax, t, detected):
    miss = ~detected
    edges = np.diff(np.concatenate([[0], miss.astype(int), [0]]))
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        half = 0.5 * (t[1] - t[0])
        ax.axvspan(t[s] - half, t[e - 1] + half, color="#e34948", alpha=0.15, lw=0)


def plot_problems(name, t, pts, vis, detected, info):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    c = COLORS["raw"]

    ax = axes[0, 0]
    ax.plot(t, pts[:, WRIST, 2], color="#52514e", lw=1.4)
    shade_missing(ax, t, detected)
    ax.set_title("① Jitter & missing frames (right wrist depth z)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("z (m)")
    ax.text(0.01, 0.97, f"red = no detection ({info['missing_frames']} frames)",
            transform=ax.transAxes, va="top", fontsize=9, color="#b3302f")

    ax = axes[0, 1]
    acc = np.linalg.norm(pts[2:] - 2 * pts[1:-1] + pts[:-2], axis=2) * 1000
    ax.plot(t[1:-1], np.nanmax(acc, axis=1), color="#52514e", lw=1.2)
    ax.set_title("② Sudden jumps (max joint acceleration per frame)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("mm / frame²")

    ax = axes[1, 0]
    L = bone_length(pts) * 100
    ax.plot(t, L, color="#52514e", lw=1.4)
    ax.axhline(np.nanmedian(L), color="#8a8984", ls="--", lw=1)
    ax.set_title(f"③ Bone length changes ({BONE_NAME})")
    ax.set_xlabel("time (s)"); ax.set_ylabel("length (cm)")
    ax.text(0.01, 0.97, f"range {np.nanmin(L):.1f} – {np.nanmax(L):.1f} cm (real bone: constant)",
            transform=ax.transAxes, va="top", fontsize=9)

    ax = axes[1, 1]
    ax.plot(t, np.nanmin(pts[:, FOOT_JOINTS, 1], axis=1), color="#52514e", lw=1.4)
    ax.axhline(0, color="#0b0b0b", lw=1)
    ax.set_title("④ Feet not on the ground (lowest foot y)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("y (m)")
    ax.text(0.01, 0.5, "hip-centered coords → ground is at y~-0.9, not 0",
            transform=ax.transAxes, va="center", fontsize=9)

    fig.suptitle(f"Problems in raw 3D motion — {name}", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"{name}_problems.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------
# figure 2: before / after (slide 4)
# ---------------------------------------------------------------------
def plot_compare(name, t, seqs, metrics):
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])

    # wrist depth over time — raw vs refined (provided does not change x/z)
    ax = fig.add_subplot(gs[0, :3])
    for k in ["raw", "refined"]:
        ax.plot(t, seqs[k][:, WRIST, 2], color=COLORS[k], label=LABELS[k],
                lw=1.3 if k == "raw" else 2)
    ax.set_title("Right wrist depth z — jitter and gaps removed")
    ax.set_xlabel("time (s)"); ax.set_ylabel("z (m)")
    ax.legend(loc="upper right")

    # bone length
    ax = fig.add_subplot(gs[0, 3])
    for k in ["raw", "refined"]:
        ax.plot(t, bone_length(seqs[k]) * 100, color=COLORS[k], label=LABELS[k],
                lw=1.3 if k == "raw" else 2)
    ax.set_title(f"{BONE_NAME} length")
    ax.set_xlabel("time (s)"); ax.set_ylabel("cm")
    ax.legend(loc="upper right")

    # metric bars: one panel per metric (different units → no shared axis)
    items = [("jitter (mm/frame^2)", "Jitter ↓", "mm/frame²"),
             ("bone length CV (%)", "Bone length variation ↓", "%"),
             ("ground error (mm)", "Ground error ↓", "mm"),
             ("deviation from raw (mm)", "Change from raw (keep small)", "mm")]
    for i, (col, title, unit) in enumerate(items):
        ax = fig.add_subplot(gs[1, i])
        vals = metrics[col]
        bars = ax.bar(range(3), vals.values, color=[COLORS[k] for k in vals.index],
                      width=0.6, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(3), ["Raw", "Provided", "Refined"])
        ax.set_title(title); ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)

    fig.suptitle(f"Before vs after post-processing — {name}", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"{name}_compare.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------
# [ADDED] Smoothing filter comparison + automatic selection
#   4 filters x 6 strengths are run on the same input (after steps 1-3).
#   There is no ground truth for a real video, so each setting is scored
#   by cross-validation:
#     hide every other frame (replace it with the average of its
#     neighbours), filter, and check how well the filter predicts the
#     hidden raw frame. Noise in a hidden frame never enters the filter,
#     so the setting with the lowest error is the one that removes noise
#     without destroying real motion.
#   (Checked on synthetic data with known truth: rank correlation 0.98.)
# ---------------------------------------------------------------------
def filter_moving_average(x, size, fps):
    return uniform_filter1d(x, size=int(size), axis=0, mode="nearest")


def filter_butterworth(x, cutoff_hz, fps):
    b, a = butter(4, min(cutoff_hz, 0.45 * fps) / (fps / 2))
    return filtfilt(b, a, x, axis=0)                 # forward + backward → no lag


def filter_one_euro(x, min_cutoff, fps, beta=1.0, d_cutoff=1.0):
    """One-Euro filter: adaptive low-pass, strong when slow, light when fast.
    Real-time (causal) filter → it only looks at past frames."""
    def alpha(cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * fps)
    y = np.empty_like(x)
    y[0] = x[0]
    dx_hat = np.zeros_like(x[0])
    a_d = alpha(d_cutoff)
    for t in range(1, len(x)):
        dx = (x[t] - y[t - 1]) * fps
        dx_hat = a_d * dx + (1 - a_d) * dx_hat
        a = alpha(min_cutoff + beta * np.abs(dx_hat))
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def filter_savgol(x, window, fps):
    return _savgol(x, int(window), 2)


FILTERS = {
    #  name               function                strengths (weak → strong)
    "Moving average":    (filter_moving_average, [3, 5, 7, 9, 13, 17]),       # window
    "Butterworth":       (filter_butterworth,    [10, 8, 6, 4, 3, 2]),        # cutoff Hz
    "One-Euro":          (filter_one_euro,       [4, 2, 1, 0.5, 0.3, 0.2]),   # min cutoff Hz
    "Savitzky-Golay":    (filter_savgol,         [5, 7, 9, 13, 17, 21]),      # window
}
FILTER_COLORS = {"Moving average": "#1baf7a", "Butterworth": "#6250d6",
                 "One-Euro": "#eda100", "Savitzky-Golay": "#eb6834"}


def cv_error_mm(x, func, strength, fps):
    """Hold-out-every-other-frame prediction error (RMS, mm)."""
    errs = []
    for off in (0, 1):
        held = np.arange(1 + off, len(x) - 1, 2)
        x_in = x.copy()
        x_in[held] = (x[held - 1] + x[held + 1]) / 2
        y = func(x_in, strength, fps)
        errs.append(np.sum((y[held] - x[held]) ** 2, axis=2).ravel())
    return float(np.sqrt(np.mean(np.concatenate(errs))) * 1000)


def select_smoothing(pts, vis, fps):
    """
    Score every filter/strength and return
      best : (name, strength) with the lowest CV error
      table: all scores (also saved for the report)
      x    : the common input (after steps 1-3)
    """
    x = fill_missing_frames(pts)
    x, _ = fix_left_right_swaps(x)
    x, _ = remove_outliers(x, vis)

    rows = []
    for fname, (func, strengths) in FILTERS.items():
        for st in strengths:
            y = func(x, st, fps)
            rows.append({"filter": fname, "strength": st,
                         "jitter (mm/frame^2)": jitter_mm(y),
                         "CV error (mm)": cv_error_mm(x, func, st, fps)})
    table = pd.DataFrame(rows)
    b = table.loc[table["CV error (mm)"].idxmin()]
    best = (b["filter"], float(b["strength"]))

    print("\n--- smoothing filter selection (lower CV error = better) ---")
    per_filter = table.loc[table.groupby("filter")["CV error (mm)"].idxmin()]
    print(per_filter.set_index("filter").round(2).to_string())
    print(f"→ selected: {best[0]} ({best[1]:g})")
    return best, table, x


def plot_filters(name, t, pts, x, table, best, fps):
    os.makedirs(FIG_DIR, exist_ok=True)
    table.to_csv(os.path.join(FIG_DIR, f"{name}_filters.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.4]})

    # left: CV error vs smoothness, one curve per filter
    ax = axes[0]
    for fname in FILTERS:
        d = table[table["filter"] == fname]
        ax.plot(d["jitter (mm/frame^2)"], d["CV error (mm)"], "-o", ms=6,
                color=FILTER_COLORS[fname], label=fname, lw=2)
    bb = table[(table["filter"] == best[0]) & (table["strength"] == best[1])].iloc[0]
    ax.scatter(bb["jitter (mm/frame^2)"], bb["CV error (mm)"], s=260, marker="*",
               color=FILTER_COLORS[best[0]], edgecolor="#0b0b0b", zorder=5,
               label=f"selected: {best[0]} ({best[1]:g})")
    ax.set_xscale("log")
    from matplotlib.ticker import LogLocator, FuncFormatter
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("jitter (mm/frame², log)  ← smoother")
    ax.set_ylabel("cross-validation error (mm)  ↓ better")
    ax.set_title("Which filter keeps real motion best?")
    ax.legend(fontsize=8)

    # right: busiest 3 s of the wrist, each filter at its own best setting
    ax = axes[1]
    acc = np.linalg.norm(np.diff(x[:, WRIST], 2, axis=0), axis=1)
    win = min(int(3 * fps), len(acc))
    c = int(np.argmax(np.convolve(acc, np.ones(win), "valid")))
    sl = slice(c, min(c + win, len(t)))
    ax.plot(t[sl], pts[sl, WRIST, 2], color="#9a9893", lw=1.3, label="Raw")
    best_per = table.loc[table.groupby("filter")["CV error (mm)"].idxmin()].set_index("filter")
    for fname, (func, _) in FILTERS.items():
        st = best_per.loc[fname, "strength"]
        ax.plot(t[sl], func(x, st, fps)[sl, WRIST, 2], color=FILTER_COLORS[fname],
                lw=2, label=f"{fname} ({st:g})")
    ax.set_title("Right wrist z, each filter at its best setting (busiest 3 s)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("z (m)")
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle(f"Smoothing filter comparison — {name}", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"{name}_filters.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------
# figure 3: one frame, three versions (front view x-y)
# ---------------------------------------------------------------------
def plot_skeleton(name, seqs, raw_det, detected):
    # pick the frame where the provided method clamps the most joints
    foot_y = np.minimum(raw_det[:, 31, 1], raw_det[:, 32, 1])
    n_below = (raw_det[:, :, 1] - foot_y[:, None] < 0).sum(axis=1)
    f_det = int(np.argmax(n_below)) if n_below.max() > 0 else len(raw_det) // 2
    f = int(np.where(detected)[0][f_det])

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    for ax, k in zip(axes, ["raw", "provided", "refined"]):
        p = seqs[k][f]
        for a, b in DRAW_BONES:
            ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color=COLORS[k], lw=2.5,
                    solid_capstyle="round")
        ax.scatter(p[:, 0], p[:, 1], s=14, color="#0b0b0b", zorder=3)
        ax.axhline(0, color="#0b0b0b", lw=1)
        ax.set_aspect("equal"); ax.set_title(LABELS[k])
        ax.set_xlabel("x (m)")
        if k != "raw":
            ax.set_ylim(-0.15, None)
    axes[0].set_ylabel("y (m)")
    fig.suptitle(f"Frame {f} — ground line y=0", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"{name}_skeleton.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("saved", out)

# =====================================================================
# Original code (unchanged)
# =====================================================================
def export_unity_csv(motion, output_path):
    """
    motion: (T, 33, 3)
    Export Unity-compatible CSV.
    """

    T, n_landmarks, _ = motion.shape

    rows = []

    for t in range(T):
        for landmark in range(n_landmarks):

            x, y, z = motion[t, landmark]

            rows.append([
                t,
                landmark,
                float(x),
                float(y),
                float(z),
                1.0
            ])

    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "landmark",
            "x",
            "y",
            "z",
            "visibility"
        ]
    )

    # 문자열 경로 처리
    output_path = os.path.abspath(output_path)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"Unity CSV saved: {output_path}")
    print(f"Frames: {T}")
    print(f"Landmarks: {n_landmarks}")
    print(f"Total rows: {len(df)}")


def main():
    paths = glob(f"{RAW_KEYPOINTS_DIR}/*.npz")
    print(f"Found {len(paths)} raw keypoint files")

    for path in paths:
        base = os.path.basename(path).replace(".npz", "")

        # -------------------------------
        # 1) Load raw BlazePose sequence
        # -------------------------------
        raw_seq = load_raw_keypoints(path)      # (T,33,3)
        raw_out_path = os.path.join(OUT_DIR, f"{base}_raw.npy")
        np.save(raw_out_path, raw_seq)
        print(f"Saved raw → {raw_out_path}")

        # ---------------------------------------
        # 2) Create height-corrected target (GT)
        # ---------------------------------------
        target_seq = create_height_corrected_target(raw_seq)
        target_out_path = os.path.join(OUT_DIR, f"{base}_target.npy")
        np.save(target_out_path, target_seq)
        print(f"Saved target → {target_out_path}")

        OUTPUT_PATH = (
                os.path.join(OUT_DIR,
                f"{base}_target.csv")
            )
        export_unity_csv(target_seq,OUTPUT_PATH)

        # ---------------------------------------
        # 3) [ADDED] Refined motion (steps 1-7)
        # ---------------------------------------
        pts, vis, detected, frame_ids = load_raw_with_meta(path)
        fps = get_fps(path)                                  # [ADDED]
        print(f"fps → {fps:.1f}")
        selection = select_smoothing(pts, vis, fps) if AUTO_SMOOTH else None   # [ADDED]
        refined_seq, info = refine_motion(pts, vis,
                                          smoother=selection[0] if selection else None,
                                          fps=fps)
        refined_out_path = os.path.join(OUT_DIR, f"{base}_refined.npy")
        np.save(refined_out_path, refined_seq)
        print(f"Saved refined → {refined_out_path}  {info}")

        export_unity_csv(refined_seq, os.path.join(OUT_DIR, f"{base}_refined.csv"))

        # ---------------------------------------
        # 4) [ADDED] Before / after comparison
        # ---------------------------------------
        if COMPARE:
            analyze(base, pts, vis, detected, frame_ids, refined_seq, info, fps, selection)

    print("\nDone processing all keypoint files.")

    # IF you want to visualize the output
    if VISUALIZE and paths:
        from utils.viser_test import PoseViser
        vis = PoseViser(fps=fps)    # [CHANGED] 60 → fps saved in the .npz
        refined = np.load(os.path.join(OUT_DIR, f"{base}_refined.npy"))
        print(refined.shape)

        # [ADDED] start camera: front view at waist height (data is y-up)
        CAM_POS = (0.0, 1.0, -4.0)     # camera position (x, y, z) — z sign flips front/back
        CAM_LOOK = (0.0, 0.9, 0.0)     # point the camera looks at (body center)
        try:
            vis.server.scene.set_up_direction("+y")      # orbit around the y axis
        except Exception:
            pass

        @vis.server.on_client_connect
        def _set_camera(client):
            try:
                client.camera.up_direction = (0.0, 1.0, 0.0)
                client.camera.position = CAM_POS
                client.camera.look_at = CAM_LOOK
            except Exception as e:
                print("camera setup skipped:", e)

        vis.play_sequence(refined)


if __name__ == "__main__":
    main()