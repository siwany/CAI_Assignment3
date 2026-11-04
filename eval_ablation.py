"""
Retrieval ablation over query modes.

Scores every condition on the official dev set and the paraphrased (everyday-wording) set:
  - single-turn hit@k
  - multi-turn hit@k on every turn, on follow-up turns only (turn >= 2), and on the last turn
    (the last-turn number is what eval_dev.py reports)
Writes one row per scored turn, including the retrieval query actually used, to a CSV.

Usage:
    python eval_ablation.py --endpoint http://localhost:11434/v1 --model llama3.2:latest
    python eval_ablation.py --mock                                  # wiring check only
    python eval_ablation.py --endpoint ... --conditions current rewrite

Conditions are query modes (current / window / rrf / rewrite).
Only the rewrite mode's retrieval depends on the LLM, so the other modes run against the mock
endpoint unless --full is given: their retrieval results are identical and the run is much faster.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from mock_endpoint import start_mock_endpoint, stop_mock_endpoint  # noqa: E402
from ta_bot import TABot  # noqa: E402

DEFAULT_CONDITIONS = ["current", "rrf", "rewrite"]
SETS = {
    "dev": ("single_turn.jsonl", "multi_turn.jsonl"),
    "paraphrase": ("paraphrase_single_turn.jsonl", "paraphrase_multi_turn.jsonl"),
}


def load(name: str) -> list[dict]:
    with open(SCRIPT_DIR / "dev_eval" / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def hit(citations: list[str], gold: list[str]) -> float:
    return len(set(citations) & set(gold)) / len(gold) if gold else 0.0


def run_condition(cond: str, endpoint: str, model: str, sets: list[str], rows: list[dict]) -> dict:
    bot = TABot({
        "endpoint": endpoint, "model": model, "corpus_path": str(SCRIPT_DIR / "corpus"),
        "temperature": 0.0, "seed": 42, "query_mode": cond,
    })

    out = {}
    for set_name in sets:
        st_file, mt_file = SETS[set_name]
        st = []
        for item in load(st_file):
            res = bot.chat(f"{cond}-{item['id']}", "ablation_user", item["question"])
            h = hit(res["citations"], item["gold_paper_ids"])
            st.append(h)
            rows.append(dict(condition=cond, set=set_name, id=item["id"], turn=1, hit=h,
                             gold=" ".join(item["gold_paper_ids"]), cited=" ".join(res["citations"]),
                             query=bot.last_query))
        every, follow, last = [], [], []
        for item in load(mt_file):
            turns = item["turns"]
            for i, t in enumerate(turns, start=1):
                res = bot.chat(f"{cond}-{item['id']}", "ablation_user", t["content"])
                h = hit(res["citations"], t["gold_paper_ids"])
                every.append(h)
                if i >= 2:
                    follow.append(h)
                if i == len(turns):
                    last.append(h)
                rows.append(dict(condition=cond, set=set_name, id=item["id"], turn=i, hit=h,
                                 gold=" ".join(t["gold_paper_ids"]), cited=" ".join(res["citations"]),
                                 query=bot.last_query))
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        out[set_name] = dict(st=mean(st), n_st=len(st), mt_all=mean(every), n_all=len(every),
                             mt_follow=mean(follow), n_follow=len(follow), mt_last=mean(last), n_last=len(last))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="A3 retrieval ablation")
    p.add_argument("--endpoint", help="real endpoint for LLM-dependent conditions (rewrite)")
    p.add_argument("--model", default="mock-model")
    p.add_argument("--mock", action="store_true", help="use the mock endpoint for every condition")
    p.add_argument("--full", action="store_true", help="use the real endpoint for every condition")
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--out", default=str(SCRIPT_DIR / "ablation_logs" / "ablation_results.csv"))
    args = p.parse_args()
    if not args.mock and not args.endpoint:
        p.error("specify --endpoint <url> or --mock")

    mock_url, server = start_mock_endpoint()
    rows: list[dict] = []
    results = {}
    try:
        for cond in args.conditions:
            needs_llm = cond == "rewrite"
            use_real = not args.mock and (needs_llm or args.full)
            endpoint, model = (args.endpoint, args.model) if use_real else (mock_url, "mock-model")
            print(f"running {cond} ({'real endpoint' if use_real else 'mock endpoint'}) ...", flush=True)
            results[cond] = run_condition(cond, endpoint, model, args.sets, rows)
    finally:
        stop_mock_endpoint(server)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for set_name in args.sets:
        r0 = results[args.conditions[0]][set_name]
        print(f"\n[{set_name}]  single-turn n={r0['n_st']}, multi-turn turns n={r0['n_all']} "
              f"(follow-ups n={r0['n_follow']}, last turns n={r0['n_last']})")
        print(f"  {'condition':<16}{'single':>8}{'MT all':>8}{'MT >=2':>8}{'MT last':>9}")
        for cond in args.conditions:
            r = results[cond][set_name]
            print(f"  {cond:<16}{r['st']:>8.1%}{r['mt_all']:>8.1%}{r['mt_follow']:>8.1%}{r['mt_last']:>9.1%}")
    print(f"\nper-turn results: {args.out}")


if __name__ == "__main__":
    main()
