"""Hand-drawn schematic figures (no data): assay, task, network, timeline,
metric definitions, and the experimental sample-structure diagram."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figlib import PALETTE, add_box, arrow, save


def figure_paired_clone_assay(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.set_axis_off()
    add_box(ax, 0.05, 0.62, 0.20, 0.16, "naive\nSNN state", facecolor="#F3F4F6", weight="bold")
    add_box(ax, 0.33, 0.62, 0.22, 0.16, "train to\ncriterion", facecolor="#DBEAFE", weight="bold")
    add_box(ax, 0.64, 0.62, 0.26, 0.16, "clone full\ntrained state", facecolor="#DCFCE7", weight="bold")
    arrow(ax, (0.25, 0.70), (0.33, 0.70))
    arrow(ax, (0.55, 0.70), (0.64, 0.70))
    add_box(ax, 0.09, 0.26, 0.30, 0.18, "stimulation branch\ncandidate reset protocol", facecolor="#FEE2E2", weight="bold")
    add_box(ax, 0.53, 0.26, 0.30, 0.18, "matched branch\nplasticity-on rest", facecolor="#F3F4F6", weight="bold")
    arrow(ax, (0.77, 0.62), (0.24, 0.44), color=PALETTE["vermillion"])
    arrow(ax, (0.77, 0.62), (0.68, 0.44), color=PALETTE["gray"])
    add_box(ax, 0.30, 0.03, 0.34, 0.14, "paired comparison\nbehavior, weights, activity, relearning", facecolor="#FEF3C7", weight="bold")
    arrow(ax, (0.24, 0.26), (0.43, 0.17), color=PALETTE["vermillion"])
    arrow(ax, (0.68, 0.26), (0.50, 0.17), color=PALETTE["gray"])
    ax.text(
        0.50,
        0.91,
        "Paired-clone reset assay",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )
    ax.text(
        0.50,
        0.54,
        "clone includes synaptic weights, membrane variables, thresholds, homeostatic state, eligibility traces, and readout state",
        ha="center",
        va="center",
        fontsize=8.5,
        color=PALETTE["gray"],
    )
    save(fig, out_dir, "M1_paired_clone_reset_assay")


def figure_task_schematics(out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(-5, 110)
        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks([])
        ax.set_xlabel("trial time (ms)")
        ax.grid(axis="x", alpha=0.18)
    ax = axes[0]
    ax.set_title("Conditioned electrode association")
    ax.text(-3, 3.0, "training", ha="right", va="center", fontsize=8)
    ax.text(-3, 1.2, "probe", ha="right", va="center", fontsize=8)
    ax.broken_barh([(0, 2.5)], (2.78, 0.24), facecolors=PALETTE["blue"])
    ax.text(1.2, 3.22, "input A", fontsize=8, color=PALETTE["blue"])
    ax.broken_barh([(12, 2.5)], (2.28, 0.24), facecolors=PALETTE["green"])
    ax.text(13.2, 2.58, "target B", fontsize=8, color=PALETTE["green"])
    ax.broken_barh([(0, 2.5)], (1.28, 0.24), facecolors=PALETTE["blue"])
    ax.axvspan(4, 45, ymin=0.02, ymax=0.34, color=PALETTE["orange"], alpha=0.16)
    ax.text(8, 0.65, "score B response in 4-45 ms window", fontsize=8, color=PALETTE["black"])
    ax.text(70, 1.35, "success: A alone evokes B\nfailure: no B response or sham response", fontsize=8)
    ax.axhline(0.0, color=PALETTE["light_gray"], lw=0.8)
    ax = axes[1]
    ax.set_title("Pattern discrimination")
    ax.text(-3, 3.0, "train A", ha="right", va="center", fontsize=8)
    ax.text(-3, 2.2, "train B", ha="right", va="center", fontsize=8)
    ax.text(-3, 1.0, "probe", ha="right", va="center", fontsize=8)
    ax.broken_barh([(0, 2.5)], (2.88, 0.22), facecolors=PALETTE["blue"])
    ax.broken_barh([(12, 2.5)], (2.56, 0.22), facecolors=PALETTE["green"])
    ax.text(3, 3.18, "input A -> target A", fontsize=8)
    ax.broken_barh([(0, 2.5)], (2.08, 0.22), facecolors=PALETTE["sky"])
    ax.broken_barh([(12, 2.5)], (1.76, 0.22), facecolors=PALETTE["orange"])
    ax.text(3, 2.38, "input B -> target B", fontsize=8)
    ax.broken_barh([(0, 2.5)], (0.96, 0.22), facecolors=PALETTE["blue"])
    ax.broken_barh([(0, 2.5)], (0.56, 0.22), facecolors=PALETTE["sky"])
    ax.axvspan(4, 50, ymin=0.04, ymax=0.43, color=PALETTE["orange"], alpha=0.14)
    ax.text(8, 0.16, "score correct targets over crossed targets", fontsize=8)
    ax.text(67, 0.80, "criterion = positive target responses\nminus crossed-target responses", fontsize=8)
    save(fig, out_dir, "M2_M3_task_schematics")


def figure_network_architecture(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.set_axis_off()
    ax.text(0.5, 0.94, "Network and perturbed substrate", ha="center", va="center", fontsize=12, weight="bold")
    add_box(ax, 0.04, 0.56, 0.18, 0.18, "64 MEA\nelectrodes", facecolor="#DBEAFE", weight="bold")
    add_box(ax, 0.04, 0.24, 0.18, 0.18, "task cues\nand targets", facecolor="#E0F2FE")
    add_box(ax, 0.38, 0.46, 0.26, 0.28, "10k recurrent\nspiking culture\nE/I hidden state", facecolor="#DCFCE7", weight="bold")
    add_box(ax, 0.75, 0.56, 0.18, 0.18, "probe readout\nscore", facecolor="#FEF3C7", weight="bold")
    add_box(ax, 0.38, 0.16, 0.26, 0.16, "plasticity state\nSTDP traces + homeostasis", facecolor="#F3E8FF")
    arrow(ax, (0.22, 0.65), (0.38, 0.62), color=PALETTE["blue"])
    arrow(ax, (0.22, 0.33), (0.38, 0.51), color=PALETTE["sky"])
    arrow(ax, (0.64, 0.61), (0.75, 0.65), color=PALETTE["green"])
    arrow(ax, (0.51, 0.46), (0.51, 0.32), color=PALETTE["purple"])
    arrow(ax, (0.51, 0.32), (0.51, 0.46), color=PALETTE["purple"])
    ax.annotate(
        "candidate reset stimulation\nacts through electrode pulses",
        xy=(0.22, 0.65),
        xytext=(0.24, 0.82),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["vermillion"], lw=1.1),
        fontsize=8,
        color=PALETTE["vermillion"],
        ha="left",
    )
    ax.annotate(
        "reported displacement metric includes\npaired recurrent weight vector change",
        xy=(0.51, 0.60),
        xytext=(0.35, 0.04),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["gray"], lw=1.0),
        fontsize=8,
        color=PALETTE["gray"],
        ha="center",
    )
    save(fig, out_dir, "M4_network_architecture_schematic")


def figure_experiment_timeline(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 2.8))
    ax.set_axis_off()
    xs = np.linspace(0.07, 0.93, 7)
    labels = [
        "naive",
        "train\nplasticity on",
        "criterion\nreached",
        "1 s rest\nplasticity on",
        "clone",
        "reset or\nmatched rest",
        "post-test\nrelearn assay",
    ]
    colors = ["#F3F4F6", "#DBEAFE", "#DCFCE7", "#F3E8FF", "#DCFCE7", "#FEE2E2", "#FEF3C7"]
    for index, (x, label, color) in enumerate(zip(xs, labels, colors, strict=True)):
        add_box(ax, x - 0.055, 0.44, 0.11, 0.24, label, facecolor=color, fontsize=8, weight="bold")
        if index < len(xs) - 1:
            arrow(ax, (x + 0.055, 0.56), (xs[index + 1] - 0.055, 0.56))
    ax.text(0.5, 0.86, "Experimental epochs", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(
        0.5,
        0.22,
        "Post-reset score is measured before relearning count; relearning trials are zero when the reset branch is already above criterion.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=PALETTE["gray"],
    )
    save(fig, out_dir, "M6_training_reset_relearn_timeline")


def figure_metric_definitions(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.set_axis_off()
    ax.text(0.5, 0.92, "Endpoint definitions", ha="center", va="center", fontsize=12, weight="bold")
    equations = [
        ("Score dent", "D_score = S_no-reset - S_reset"),
        ("Criterion margin", "M_reset = S_reset - C_task"),
        ("Criterion-level forgetting", "F = 1[S_no-reset >= C_task and S_reset < C_task]"),
        ("Relearning burden", "B_relearn = trials needed to regain criterion"),
        ("Paired weight displacement", "D_W = ||W_reset - W_no-reset||"),
        ("Training-axis erasure fraction", "E = -((W_reset - W_no-reset) dot (W_trained - W_naive)) / ||W_trained - W_naive||^2"),
    ]
    y = 0.76
    for label, equation in equations:
        add_box(ax, 0.06, y - 0.045, 0.26, 0.09, label, facecolor="#F3F4F6", fontsize=8.5, weight="bold")
        ax.text(0.36, y, equation, ha="left", va="center", fontsize=9, family="monospace")
        y -= 0.115
    save(fig, out_dir, "M12_metric_definitions")


def figure_sample_structure(raw: pd.DataFrame, out_dir: Path) -> None:
    n_tasks = raw["task_label"].nunique()
    n_seeds = raw["seed"].nunique()
    n_protocols = raw["protocol_id"].nunique()
    n_evaluations = len(raw)
    eval_reps = 8
    if "trained_positive_response_probability" in raw.columns:
        eval_reps = 8
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.set_axis_off()
    labels = [
        (f"{n_tasks} learned tasks", "#DBEAFE"),
        (f"{n_seeds} trained network seeds", "#DCFCE7"),
        (f"{n_protocols} stimulation protocols", "#FEF3C7"),
        (f"{n_evaluations} paired reset evaluations", "#FEE2E2"),
        (f"{eval_reps} probe repetitions per evaluation", "#F3E8FF"),
    ]
    xs = np.linspace(0.10, 0.90, len(labels))
    for index, (x, (label, color)) in enumerate(zip(xs, labels, strict=True)):
        add_box(ax, x - 0.08, 0.48, 0.16, 0.18, label, facecolor=color, fontsize=8.5, weight="bold")
        if index < len(xs) - 1:
            arrow(ax, (x + 0.08, 0.57), (xs[index + 1] - 0.08, 0.57))
    ax.text(0.5, 0.84, "Nested experimental sample structure", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(
        0.5,
        0.24,
        "The independent trained-network unit is seed within task; protocol evaluations are paired branches from those trained states.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=PALETTE["gray"],
    )
    save(fig, out_dir, "M13_experimental_sample_structure")

