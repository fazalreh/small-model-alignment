"""Generate the judge-dependent parts of the report from the Kaggle judgments.

    python scripts/fill_report_results.py --judged_dir outputs/judged --repo https://github.com/<user>/small-model-alignment

Reads ``<comparison>_judged_summary.json`` / ``<comparison>_judged_final.jsonl`` (written by
scripts/judge_pairwise_kaggle.py) and writes two files that report/report.tex includes if present:

  report/results_auto.tex          macro definitions (\\RowUFA, \\ResUFwin, \\JudgeModel, \\RepoURL, ...)
  report/auto_interpretation.tex   the measured part of the interpretation section: win rates with CIs,
                                   per-category strengths/weaknesses, length analysis, persona-run
                                   selection rule, and short improvement / regression examples

report.tex itself is never edited. Comparisons that were not judged keep their red placeholders.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROWS = {  # macro -> comparison name
    "RowUFA": "dpo_uf_vs_sft_alpaca",
    "RowHHthreeH": "dpo_hh_lr3e-6_vs_sft_hh",
    "RowHHoneH": "dpo_hh_vs_sft_hh",
    "RowHHtenH": "dpo_hh_lr1e-5_vs_sft_hh",
    "RowHHthreeA": "dpo_hh_lr3e-6_vs_sft_alpaca",
    "RowUFH": "dpo_uf_vs_sft_hh",
    "RowSFTA": "sft_vs_qwen2.5-0.5b-base_alpaca",
}
LORA = {  # macro -> (comparison, label)
    "RowLoraA": ("lora_dpo_vs_smollm2-360m-instruct_persona", r"$5\!\times\!10^{-5}$, 3 ep"),
    "RowLoraB": ("lora_dpo_lr1e-4_ep3_vs_smollm2-360m-instruct_persona", r"$10^{-4}$, 3 ep"),
    "RowLoraC": ("lora_dpo_lr2e-4_ep5_vs_smollm2-360m-instruct_persona", r"$2\!\times\!10^{-4}$, 5 ep"),
}
NICE = {
    "dpo_uf_vs_sft_alpaca": "UF-DPO vs SFT on AlpacaEval",
    "dpo_hh_lr3e-6_vs_sft_hh": r"HH-DPO ($3\!\times\!10^{-6}$, selected) vs SFT on HH-eval",
    "dpo_hh_vs_sft_hh": r"HH-DPO ($10^{-6}$, guide) vs SFT on HH-eval",
    "dpo_hh_lr3e-6_vs_sft_alpaca": r"HH-DPO ($3\!\times\!10^{-6}$) vs SFT on AlpacaEval (alignment tax)",
}


def esc(text: str) -> str:
    """Escape free text (model outputs, prompts, judge reasons) for pdflatex."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
           "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}"}
    text = text.encode("ascii", "ignore").decode()
    text = "".join(rep.get(ch, ch) for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def short(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def pct(x, digits=1):
    return "--" if x is None else f"{100 * x:.{digits}f}\\%"


def load(judged_dir: Path, name: str):
    summ = judged_dir / f"{name}_judged_summary.json"
    if not summ.exists():
        return None
    final = judged_dir / f"{name}_judged_final.jsonl"
    rows = [json.loads(l) for l in final.read_text(encoding="utf-8").splitlines() if l.strip()] if final.exists() else []
    return json.loads(summ.read_text(encoding="utf-8")), rows


def crit(s, key="winner"):
    return s["criteria"].get(key) or next(iter(s["criteria"].values()))


def row_cells(s) -> str:
    c = crit(s)
    lo, hi = c["win_rate_ties_half_ci95"]
    return (f"{c['wins']} & {c['ties']} & {c['losses']} & {pct(c['win_rate'])} & {pct(c['win_rate_excl_ties'])} & "
            f"{pct(c['win_rate_ties_half'])} ({pct(lo, 0)}--{pct(hi, 0)}) & {c['failures']}/{c['unknown']} & "
            f"{pct(c['position_consistency'], 0)}")


def score_with_ci(s) -> str:
    c = crit(s)
    lo, hi = c["win_rate_ties_half_ci95"]
    return f"{pct(c['win_rate_ties_half'])} (95\\% CI {pct(lo, 0)}--{pct(hi, 0)})"


def category_extremes(c, k=2):
    cats = [(cat, b) for cat, b in c["per_category"].items() if b["win_rate_ties_half"] is not None]
    cats.sort(key=lambda kv: kv[1]["win_rate_ties_half"], reverse=True)
    fmt = lambda kv: f"{esc(kv[0])} ({pct(kv[1]['win_rate_ties_half'], 0)})"
    return ", ".join(map(fmt, cats[:k])), ", ".join(map(fmt, cats[-k:][::-1]))


def length_split(rows, key="outcome_winner"):
    score = {"win": 1.0, "tie": 0.5, "loss": 0.0}
    longer = [score[r[key]] for r in rows if r.get(key) in score and len(r["candidate_response"]) > len(r["baseline_response"])]
    other = [score[r[key]] for r in rows if r.get(key) in score and len(r["candidate_response"]) <= len(r["baseline_response"])]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return len(longer), mean(longer), len(other), mean(other)


def example(rows, outcome, key="outcome_winner", n=140):
    pool = sorted((r for r in rows if r.get(key) == outcome),
                  key=lambda r: len(r["candidate_response"]) + len(r["baseline_response"]))
    if not pool:
        return None
    r = pool[0]
    reason = next((x for x in r.get("judge_reasons", []) if x), "")
    text = (rf"\emph{{``{esc(short(r['instruction'], 90))}''}} ({esc(r.get('category') or '')}). "
            rf"DPO: ``{esc(short(r['candidate_response'], n))}'' SFT: ``{esc(short(r['baseline_response'], n))}''")
    return text + (rf" Judge: {esc(short(reason, 110))}" if reason else "")


def interpretation(found) -> str:
    items = []
    for name in ("dpo_uf_vs_sft_alpaca", "dpo_hh_lr3e-6_vs_sft_hh", "dpo_hh_vs_sft_hh", "dpo_hh_lr3e-6_vs_sft_alpaca"):
        if name not in found:
            continue
        s, rows = found[name]
        c = crit(s)
        lo, hi = c["win_rate_ties_half_ci95"]
        verdict = ("DPO is significantly better" if lo > 0.5 else
                   "SFT is significantly better" if hi < 0.5 else "the difference is not significant at the 95\\% level")
        best, worst = category_extremes(c)
        line = (rf"\item \textbf{{{NICE[name]}:}} W/T/L = {c['wins']}/{c['ties']}/{c['losses']}, score {score_with_ci(s)}; "
                rf"{verdict}. Best categories: {best}; worst: {worst}.")
        nl, ml, no, mo = length_split(rows)
        if ml is not None and mo is not None:
            line += (rf" When DPO's answer is longer ({nl} pairs) it scores {pct(ml, 0)}, otherwise ({no} pairs) {pct(mo, 0)}; "
                     rf"mean length {s['mean_chars_candidate']:.0f} vs.\ {s['mean_chars_baseline']:.0f} characters.")
        items.append(line)
        if name in ("dpo_uf_vs_sft_alpaca", "dpo_hh_lr3e-6_vs_sft_hh"):
            for outcome, word in (("win", "Improvement"), ("loss", "Regression")):
                ex = example(rows, outcome)
                if ex:
                    items.append(rf"\item[] \textit{{{word}.}} {ex}")
    lora = [(label, found[name][0]) for name, label in LORA.values() if name in found]
    if lora:
        parts, eligible = [], []
        for label, s in lora:
            st, qu = crit(s, "style_winner"), crit(s, "quality_winner")
            parts.append(rf"{label}: style {pct(st['win_rate_ties_half'], 0)}, quality {pct(qu['win_rate_ties_half'], 0)}")
            if qu["win_rate_ties_half_ci95"][1] >= 0.5:  # no significant quality loss vs. the base model
                eligible.append((st["win_rate_ties_half"] or 0.0, label))
        rule = (rf"highest judged style among runs with no significant quality loss (quality CI reaches 50\%) "
                rf"$\Rightarrow$ {max(eligible)[1]} (scores in Table~\ref{{tab:lora}})." if eligible else
                r"every run loses significantly on quality, so no run is selected (Table~\ref{tab:lora}).")
        items.append(r"\item \textbf{Persona selection (rule fixed before judging):} " + rule)
    return "\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}\n" if items else ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judged_dir", default="outputs/judged")
    p.add_argument("--report_dir", default="report")
    p.add_argument("--repo", default="", help="GitHub URL of your repository")
    args = p.parse_args()
    judged_dir, report_dir = Path(args.judged_dir), Path(args.report_dir)

    found = {}
    for name in list(ROWS.values()) + [n for n, _ in LORA.values()]:
        got = load(judged_dir, name)
        if got:
            found[name] = got
    defs = ["% generated by scripts/fill_report_results.py -- do not edit by hand"]
    if args.repo:
        defs.append(rf"\newcommand{{\RepoURL}}{{\url{{{args.repo}}}}}")
    if found:
        judge = esc(next(iter(found.values()))[0]["judge_model"])
        defs.append(rf"\newcommand{{\JudgeModel}}{{\texttt{{{judge}}}}}")
    if "dpo_uf_vs_sft_alpaca" in found:
        defs.append(rf"\newcommand{{\ResUFwin}}{{{score_with_ci(found['dpo_uf_vs_sft_alpaca'][0])}}}")
    if "dpo_hh_lr3e-6_vs_sft_hh" in found:
        defs.append(rf"\newcommand{{\ResHHwin}}{{{score_with_ci(found['dpo_hh_lr3e-6_vs_sft_hh'][0])}}}")
    for macro, name in ROWS.items():
        if name in found:
            defs.append(rf"\newcommand{{\{macro}}}{{{row_cells(found[name][0])}}}")
    for macro, (name, _) in LORA.items():
        if name in found:
            s = found[name][0]
            st, qu = crit(s, "style_winner"), crit(s, "quality_winner")
            cells = [st["wins"], st["ties"], st["losses"], qu["wins"], qu["ties"], qu["losses"]]
            defs.append(rf"\newcommand{{\{macro}}}{{{' & '.join(map(str, cells))}}}")
    (report_dir / "results_auto.tex").write_text("\n".join(defs) + "\n", encoding="utf-8")

    interp = interpretation(found)
    target = report_dir / "auto_interpretation.tex"
    if interp:
        target.write_text(interp, encoding="utf-8")
    elif target.exists():
        target.unlink()
    print(f"[fill] {len(found)} judged comparisons -> {report_dir / 'results_auto.tex'}"
          + (f", {target}" if interp else "") + f"; missing: {[n for n in ROWS.values() if n not in found]}")


if __name__ == "__main__":
    main()
