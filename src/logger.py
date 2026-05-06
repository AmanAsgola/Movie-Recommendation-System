import json
import os
import subprocess
from datetime import datetime


LOG_JSON = "logs/evaluation_log.json"
LOG_MD   = "logs/evaluation_log.md"


# ─────────────────────────────────────────
# Git helpers
# ─────────────────────────────────────────
def _git(cmd):
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _get_git_info():
    commit_hash = _git(["git", "rev-parse", "--short", "HEAD"])
    commit_msg  = _git(["git", "log", "-1", "--pretty=%s"])
    return commit_hash, commit_msg


# ─────────────────────────────────────────
# Core log writer
# ─────────────────────────────────────────
def log_run(precision, recall, ndcg, n_users, changes_note=""):
    """Append one evaluation run to both the JSON and Markdown logs.

    Args:
        precision   : Precision@10 score
        recall      : Recall@10 score
        ndcg        : NDCG@10 score
        n_users     : number of users evaluated
        changes_note: optional manual description of what changed this run;
                      falls back to the latest git commit message
    """
    os.makedirs("logs", exist_ok=True)

    # ── Load history ──────────────────────────────────────────
    history = []
    if os.path.exists(LOG_JSON):
        with open(LOG_JSON, "r") as f:
            history = json.load(f)

    # ── Percentage change helper ───────────────────────────────
    def pct(new, old):
        if old is None or old == 0:
            return None
        return round((new - old) / abs(old) * 100, 2)

    prev = history[-1] if history else None
    prev_p = prev["scores"]["precision_at_10"] if prev else None
    prev_r = prev["scores"]["recall_at_10"]    if prev else None
    prev_n = prev["scores"]["ndcg_at_10"]      if prev else None

    commit_hash, commit_msg = _get_git_info()
    changes = changes_note.strip() if changes_note.strip() else commit_msg

    entry = {
        "run": len(history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit": commit_hash,
        "changes": changes,
        "users_evaluated": n_users,
        "scores": {
            "precision_at_10": round(precision, 4),
            "recall_at_10":    round(recall,    4),
            "ndcg_at_10":      round(ndcg,      4),
        },
        "improvement_vs_previous": {
            "precision_pct": pct(precision, prev_p),
            "recall_pct":    pct(recall,    prev_r),
            "ndcg_pct":      pct(ndcg,      prev_n),
        },
    }

    history.append(entry)

    # ── Write JSON ────────────────────────────────────────────
    with open(LOG_JSON, "w") as f:
        json.dump(history, f, indent=2)

    # ── Write / append Markdown ───────────────────────────────
    _write_markdown(history)

    # ── Console summary ───────────────────────────────────────
    _print_summary(entry)


# ─────────────────────────────────────────
# Markdown log (full history table)
# ─────────────────────────────────────────
def _write_markdown(history):
    lines = [
        "# Evaluation Run History\n",
        "| Run | Date & Time | Commit | Precision@10 | Recall@10 | NDCG@10 "
        "| P Δ% | R Δ% | N Δ% | Users | Changes |",
        "|-----|-------------|--------|-------------|-----------|---------|"
        "------|------|------|-------|---------|",
    ]

    for e in history:
        s   = e["scores"]
        imp = e["improvement_vs_previous"]

        def fmt_pct(v):
            if v is None:
                return "baseline"
            sign = "+" if v >= 0 else ""
            return f"{sign}{v}%"

        row = (
            f"| {e['run']} "
            f"| {e['timestamp']} "
            f"| `{e['commit']}` "
            f"| {s['precision_at_10']:.4f} "
            f"| {s['recall_at_10']:.4f} "
            f"| {s['ndcg_at_10']:.4f} "
            f"| {fmt_pct(imp['precision_pct'])} "
            f"| {fmt_pct(imp['recall_pct'])} "
            f"| {fmt_pct(imp['ndcg_pct'])} "
            f"| {e['users_evaluated']} "
            f"| {e['changes'][:80]} |"
        )
        lines.append(row)

    lines.append("\n")
    lines.append("---\n")
    lines.append(f"_Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    with open(LOG_MD, "w") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────
# Console pretty-printer
# ─────────────────────────────────────────
def _print_summary(entry):
    s   = entry["scores"]
    imp = entry["improvement_vs_previous"]

    def fmt(v):
        if v is None:
            return "baseline"
        arrow = "▲" if v >= 0 else "▼"
        return f"{arrow} {abs(v):.1f}%"

    w = 62
    print("\n" + "═" * w)
    print(f"  EVALUATION RUN #{entry['run']}   {entry['timestamp']}")
    print(f"  Commit : {entry['commit']}  —  {entry['changes'][:50]}")
    print(f"  Users evaluated : {entry['users_evaluated']}")
    print("─" * w)
    print(f"  {'Metric':<18} {'Score':>8}   {'vs Previous':>12}")
    print("─" * w)
    print(f"  {'Precision@10':<18} {s['precision_at_10']:>8.4f}   {fmt(imp['precision_pct']):>12}")
    print(f"  {'Recall@10':<18} {s['recall_at_10']:>8.4f}   {fmt(imp['recall_pct']):>12}")
    print(f"  {'NDCG@10':<18} {s['ndcg_at_10']:>8.4f}   {fmt(imp['ndcg_pct']):>12}")
    print("═" * w)
    print(f"  Log → logs/evaluation_log.json  |  logs/evaluation_log.md")
    print("═" * w + "\n")
