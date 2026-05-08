"""
Evaluation logger — writes every run to JSON + Markdown and prints a
colour-coded dashboard to the terminal so anyone can read the scores at a
glance without knowing ML.
"""

import json
import os
import subprocess
from datetime import datetime


LOG_JSON = "logs/evaluation_log.json"
LOG_MD   = "logs/evaluation_log.md"

# ─── ANSI colours ──────────────────────────────────────────────────────────────
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_BLUE   = "\033[94m"
_MAGENTA= "\033[95m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

def _c(text, colour): return f"{colour}{text}{_RESET}"
def _b(text):         return f"{_BOLD}{text}{_RESET}"


# ─── Grading thresholds ────────────────────────────────────────────────────────
_GRADES = {
    # (metric_key, good_thresh, fair_thresh, low_is_better)
    "precision_at_10": (0.25, 0.10, False),
    "recall_at_10":    (0.05, 0.02, False),
    "ndcg_at_10":      (0.25, 0.10, False),
    "rmse":            (0.80, 0.95, True),   # lower RMSE is better
}

_GRADE_LABELS = {
    "precision_at_10": ("GOOD",      "FAIR",      "WEAK"),
    "recall_at_10":    ("GOOD",      "FAIR",      "WEAK"),
    "ndcg_at_10":      ("GOOD",      "FAIR",      "WEAK"),
    "rmse":            ("EXCELLENT", "GOOD",      "POOR"),
}

_DESCRIPTIONS = {
    "precision_at_10": (
        "Out of every 10 recommendations, how many the user actually likes.",
        {
            "good":  "Excellent — roughly 1 in 4+ recommendations is a perfect match",
            "fair":  "Decent  — 1 in 10 recommendations is relevant",
            "weak":  "Needs work — fewer than 1 in 10 recommendations hits the mark",
        }
    ),
    "recall_at_10": (
        "What fraction of all the user's liked movies are we surfacing in the top 10.",
        {
            "good":  "Good coverage — finding at least 5% of all liked movies",
            "fair":  "Some coverage — finding 2-5% of liked movies",
            "weak":  "Low coverage — missing most of the user's taste",
        }
    ),
    "ndcg_at_10": (
        "Are the correct movies appearing near the TOP of the list, not buried at #9?",
        {
            "good":  "Great ranking — the best matches appear at the top",
            "fair":  "Moderate ranking — hits are somewhat well-placed",
            "weak":  "Poor ranking — correct movies aren't rising to the top",
        }
    ),
    "rmse": (
        "How accurately do we predict the star rating a user would give (scale 1–5)?",
        {
            "good":  "Excellent — predictions off by less than ±0.80 stars",
            "fair":  "Good — predictions off by ±0.80–0.95 stars",
            "weak":  "Below target — predictions off by more than ±0.95 stars",
        }
    ),
}


def _grade(key, value):
    """Returns (colour, label_str, description_str) for a metric value."""
    good_t, fair_t, low_better = _GRADES[key]
    g_lbl, f_lbl, w_lbl = _GRADE_LABELS[key]
    _, desc_map = _DESCRIPTIONS[key]

    if low_better:
        if value < good_t:  colour, label, dk = _GREEN,  g_lbl, "good"
        elif value < fair_t: colour, label, dk = _YELLOW, f_lbl, "fair"
        else:                colour, label, dk = _RED,    w_lbl, "weak"
    else:
        if value >= good_t:  colour, label, dk = _GREEN,  g_lbl, "good"
        elif value >= fair_t: colour, label, dk = _YELLOW, f_lbl, "fair"
        else:                colour, label, dk = _RED,    w_lbl, "weak"

    return colour, label, desc_map[dk]


# ─── Git helpers ───────────────────────────────────────────────────────────────
def _git(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def _get_git_info():
    return _git(["git","rev-parse","--short","HEAD"]), _git(["git","log","-1","--pretty=%s"])


# ─── Core log writer ───────────────────────────────────────────────────────────
def log_run(precision, recall, ndcg, rmse, n_users, changes_note=""):
    """
    Append one evaluation run to the JSON and Markdown logs, then print a
    full colour-coded dashboard to the terminal.

    Args:
        precision   : Precision@10
        recall      : Recall@10
        ndcg        : NDCG@10
        rmse        : Root Mean Squared Error of rating predictions
        n_users     : number of users evaluated
        changes_note: optional description; defaults to latest git commit message
    """
    os.makedirs("logs", exist_ok=True)

    history = []
    if os.path.exists(LOG_JSON):
        with open(LOG_JSON) as f:
            history = json.load(f)

    def pct(new, old):
        if old is None or old == 0: return None
        return round((new - old) / abs(old) * 100, 2)

    prev   = history[-1] if history else None
    prev_p = prev["scores"]["precision_at_10"] if prev else None
    prev_r = prev["scores"]["recall_at_10"]    if prev else None
    prev_n = prev["scores"]["ndcg_at_10"]      if prev else None
    prev_e = prev["scores"].get("rmse")        if prev else None

    commit_hash, commit_msg = _get_git_info()
    changes = changes_note.strip() or commit_msg

    entry = {
        "run":             len(history) + 1,
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit":          commit_hash,
        "changes":         changes,
        "users_evaluated": n_users,
        "scores": {
            "precision_at_10": round(precision, 4),
            "recall_at_10":    round(recall,    4),
            "ndcg_at_10":      round(ndcg,      4),
            "rmse":            round(rmse,       4) if rmse == rmse else None,  # NaN check
        },
        "improvement_vs_previous": {
            "precision_pct": pct(precision, prev_p),
            "recall_pct":    pct(recall,    prev_r),
            "ndcg_pct":      pct(ndcg,      prev_n),
            "rmse_pct":      pct(rmse,      prev_e),
        },
    }
    history.append(entry)

    with open(LOG_JSON, "w") as f:
        json.dump(history, f, indent=2)

    _write_markdown(history)
    _print_dashboard(entry)


# ─── Terminal dashboard ────────────────────────────────────────────────────────
def _print_dashboard(entry):
    s   = entry["scores"]
    imp = entry["improvement_vs_previous"]
    W   = 70

    def border(ch="═"): print(_c(ch * W, _CYAN))
    def row(left="", right=""):
        pad = W - len(left) - len(right) - 4
        print(f"  {left}{' '*max(pad,1)}{right}")

    def delta_str(val):
        if val is None: return _c("baseline", _DIM)
        arrow = "▲" if val >= 0 else "▼"
        colour = _GREEN if val >= 0 else _RED
        return _c(f"{arrow} {abs(val):.1f}%", colour)

    border()
    print(_b(_c(f"  🎬  MOVIE RECOMMENDATION SYSTEM — RUN #{entry['run']}", _CYAN)))
    print(_c(f"  {entry['timestamp']}  |  commit: {entry['commit']}", _DIM))
    print(_c(f"  {entry['changes'][:65]}", _DIM))
    border("─")

    # ── Recommendation quality block ───────────────────────────────────────────
    print(_b("  RECOMMENDATION QUALITY"))
    print(_c("  How often do we suggest the right movie?\n", _DIM))

    metrics_q = [
        ("Precision@10", "precision_at_10", s["precision_at_10"], imp["precision_pct"]),
        ("Recall@10",    "recall_at_10",    s["recall_at_10"],    imp["recall_pct"]),
        ("NDCG@10",      "ndcg_at_10",      s["ndcg_at_10"],      imp["ndcg_pct"]),
    ]

    for label, key, val, d in metrics_q:
        colour, grade_lbl, desc = _grade(key, val)
        val_str = _c(f"{val:.4f}", colour)
        grade   = _c(f"  {grade_lbl:<10}", colour)
        d_str   = delta_str(d)
        print(f"  {_b(label):<20}  {val_str}   {grade}  vs prev: {d_str}")
        print(_c(f"  {'':20}  {desc}", _DIM))
        print()

    border("─")

    # ── Rating accuracy block ──────────────────────────────────────────────────
    print(_b("  RATING ACCURACY"))
    print(_c("  How precisely do we predict the star rating a user would give?\n", _DIM))

    rmse_val = s.get("rmse") or float('nan')
    if rmse_val == rmse_val:   # not NaN
        colour, grade_lbl, desc = _grade("rmse", rmse_val)
        val_str = _c(f"{rmse_val:.4f}", colour)
        grade   = _c(f"  {grade_lbl:<10}", colour)
        d_str   = delta_str(imp.get("rmse_pct"))
        print(f"  {'RMSE (1–5 stars)':<20}  {val_str}   {grade}  vs prev: {d_str}")
        print(_c(f"  {'':20}  {desc}", _DIM))
        print(_c(f"  {'':20}  Netflix Prize winner was 0.8563  |  Ours: {rmse_val:.4f}", _DIM))
    else:
        print(_c("  RMSE: not computed", _YELLOW))
    print()

    border("─")

    # ── Users & benchmark ──────────────────────────────────────────────────────
    print(f"  {_c('Users evaluated:', _DIM)} {entry['users_evaluated']}   "
          f"{_c('(power users ≥20 liked movies, known to SVD)', _DIM)}")
    print()
    print(f"  {_c('Global benchmark  (MovieLens-20M):', _DIM)}")
    print(f"  {_c('  SVD baseline ~0.08  |  TF-IDF hybrid ~0.12  |  NCF ~0.25  |  SOTA (GRU4Rec) ~0.26', _DIM)}")
    print()

    # ── Log paths ─────────────────────────────────────────────────────────────
    print(f"  {_c('Saved →', _DIM)}  {LOG_JSON}  |  {LOG_MD}")

    border()
    print()


# ─── Markdown log ─────────────────────────────────────────────────────────────
def _write_markdown(history):
    def fmt_pct(v):
        if v is None: return "baseline"
        return f"{'+'if v>=0 else''}{v}%"

    lines = ["# Evaluation Run History\n", "## Score Summary\n",
        "| Run | Date & Time | Commit | Precision@10 | Recall@10 | NDCG@10 | RMSE "
        "| P Δ% | R Δ% | N Δ% | RMSE Δ% | Users |",
        "|-----|-------------|--------|:-----------:|:---------:|:-------:|:----:"
        "|:----:|:----:|:----:|:-------:|:-----:|",
    ]

    for e in history:
        s, imp = e["scores"], e["improvement_vs_previous"]
        rmse_disp = f"{s['rmse']:.4f}" if s.get("rmse") else "—"
        lines.append(
            f"| {e['run']} | {e['timestamp']} | `{e['commit']}` "
            f"| **{s['precision_at_10']:.4f}** | {s['recall_at_10']:.4f} "
            f"| {s['ndcg_at_10']:.4f} | {rmse_disp} "
            f"| {fmt_pct(imp['precision_pct'])} | {fmt_pct(imp['recall_pct'])} "
            f"| {fmt_pct(imp['ndcg_pct'])} | {fmt_pct(imp.get('rmse_pct'))} "
            f"| {e['users_evaluated']} |"
        )

    lines.append("\n")
    lines.append("## Run Details\n")

    for e in history:
        s, imp = e["scores"], e["improvement_vs_previous"]
        rmse_disp = f"{s['rmse']:.4f}" if s.get("rmse") else "—"
        lines += [
            f"### Run #{e['run']} — {e['timestamp']}\n",
            f"**Commit:** `{e['commit']}`  |  **Users:** {e['users_evaluated']}\n",
            f"**Changes:** {e['changes']}\n",
            f"| Precision@10 | Recall@10 | NDCG@10 | RMSE |\n"
            f"|:---:|:---:|:---:|:---:|\n"
            f"| **{s['precision_at_10']:.4f}** | {s['recall_at_10']:.4f} "
            f"| {s['ndcg_at_10']:.4f} | {rmse_disp} |\n",
            f"| Δ Precision | Δ Recall | Δ NDCG | Δ RMSE |\n"
            f"|:---:|:---:|:---:|:---:|\n"
            f"| {fmt_pct(imp['precision_pct'])} | {fmt_pct(imp['recall_pct'])} "
            f"| {fmt_pct(imp['ndcg_pct'])} | {fmt_pct(imp.get('rmse_pct'))} |\n",
        ]
        if e.get("note"):
            lines.append(f"> **Analysis:** {e['note']}\n")
        lines.append("---\n")

    lines.append(f"_Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    with open(LOG_MD, "w") as f:
        f.write("\n".join(lines))
