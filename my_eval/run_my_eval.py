#!/usr/bin/env python3
"""
run_my_eval.py — Extended self-made evaluation for A3 (mirrors eval_dev.py).

Runs your TABot against my_eval/ and reports:
  - Single-turn retrieval hit-rate (paper-level), overall and per question style
  - Multi-turn retrieval hit-rate on the last turn (as eval_dev.py) and on every turn
  - Memory probe pass/fail: positive, negative, isolation, reset

Scoring reuses eval_dev.py (hit_rate, preflight, single-turn), so numbers are
directly comparable with the dev set.

Usage
-----
    # With the mock endpoint (no GPU needed):
    python my_eval/run_my_eval.py --mock

    # With a real vLLM endpoint:
    python my_eval/run_my_eval.py --endpoint http://localhost:8000/v1 --model meta-llama/Llama-3.2-3B-Instruct

    # Retrieval only (skip memory probes):
    python my_eval/run_my_eval.py --endpoint ... --model ... --skip-memory
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from eval_dev import hit_rate, load_jsonl, run_preflight  # noqa: E402


def run_single_turn(bot, data: list[dict], session_prefix: str = "my-st") -> tuple[float, dict]:
    """Run single-turn questions, return (mean hit-rate, hit-rate per style)."""
    if not data:
        return 0.0, {}
    scores = []
    by_style = defaultdict(list)
    for item in data:
        qid = item["id"]
        style = item.get("style", "-")
        try:
            result = bot.chat(f"{session_prefix}-{qid}", "eval_user", item["question"])
            citations = result.get("citations", [])
            hr = hit_rate(citations, item.get("gold_paper_ids", []))
            gold = item.get("gold_paper_ids", [])
            status = "✓" if hr == 1.0 else f"partial ({hr:.0%})" if hr > 0 else "✗"
            print(f"  {qid} [{style}]: {status}  cited={citations}  gold={gold}")
        except Exception as exc:
            hr = 0.0
            print(f"  {qid} [{style}]: ERROR — {type(exc).__name__}: {exc}")
        scores.append(hr)
        by_style[style].append(hr)
    mean = sum(scores) / len(scores)
    return mean, {s: sum(v) / len(v) for s, v in by_style.items()}


def run_multi_turn(bot, data: list[dict], session_prefix: str = "my-mt") -> tuple[float, float]:
    """Run multi-turn dialogues, return (mean hit-rate on last turn, mean hit-rate on every turn)."""
    if not data:
        return 0.0, 0.0
    last_scores, all_scores = [], []
    for item in data:
        qid = item["id"]
        turn_marks = []
        hr = 0.0
        try:
            for turn in item["turns"]:
                result = bot.chat(f"{session_prefix}-{qid}", "eval_user", turn["content"])
                citations = result.get("citations", [])
                hr = hit_rate(citations, turn.get("gold_paper_ids", []))
                all_scores.append(hr)
                turn_marks.append("✓" if hr == 1.0 else "✗")
            # Gold papers come from the last turn (per-turn schema)
            last_gold = item["turns"][-1].get("gold_paper_ids", []) if item["turns"] else []
            status = "✓" if hr == 1.0 else f"partial ({hr:.0%})" if hr > 0 else "✗"
            print(f"  {qid} [{item.get('pattern', '-')}]: {status}  turns={''.join(turn_marks)}"
                  f"  cited={citations}  gold={last_gold}")
        except Exception as exc:
            hr = 0.0
            print(f"  {qid}: ERROR — {type(exc).__name__}: {exc}")
        last_scores.append(hr)
    last_mean = sum(last_scores) / len(last_scores)
    all_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return last_mean, all_mean


def run_memory_probes(bot, data: list[dict]) -> dict[str, tuple[int, int]]:
    """
    Run memory probes. Returns {probe_type: (passed, total)}.

    positive  : expected fact in the plant user's store AND in the test answer
    negative  : expected fact in neither
    isolation : plant and test users differ; fact must not be in the test user's store or answer
    reset     : reset_memory() runs after planting; fact must not come back
    """
    results: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item in data:
        pid = item["id"]
        probe_type = item.get("type", "positive")
        results[probe_type][1] += 1
        try:
            plant = item["plant_session"]
            test = item["test_session"]
            expected = item.get("expected_fact", "")

            # 1. Reset memory
            bot.reset_memory(plant["user_id"])
            bot.reset_memory(test["user_id"])

            # 2. Plant facts in session 1
            for turn in plant.get("turns", []):
                bot.chat(plant["session_id"], plant["user_id"], turn["content"])
            if probe_type == "reset":
                bot.reset_memory(plant["user_id"])

            # 3. Inspect store
            store_user = test["user_id"] if probe_type == "isolation" else plant["user_id"]
            memory = bot.get_memory(store_user)
            memory_text = " ".join(str(m) for m in memory).lower()

            # 4. Test in session 2
            answer = ""
            for t in test.get("turns", []):
                result = bot.chat(test["session_id"], test["user_id"], t["content"])
                answer = result.get("answer", "")
            answer = answer.lower()

            in_store = expected.lower() in memory_text
            in_output = expected.lower() in answer
            if probe_type == "positive":
                ok = in_store and in_output
                detail = f"store={'✓' if in_store else '✗'} output={'✓' if in_output else '✗'}"
            else:  # negative / isolation / reset
                ok = not in_store and not in_output
                detail = f"store={'✓' if not in_store else '✗'} output={'✓' if not in_output else '✗'}"

            if ok:
                results[probe_type][0] += 1
            print(f"  {pid} ({probe_type}): {'PASS' if ok else 'FAIL'} — {detail}")
            if not ok:
                print(f"      memory={memory}")
                print(f"      answer={answer[:200]!r}")
        except NotImplementedError:
            print(f"  {pid} ({probe_type}): SKIP — not yet implemented")
        except Exception as exc:
            print(f"  {pid} ({probe_type}): ERROR — {type(exc).__name__}: {exc}")
    return {k: (v[0], v[1]) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="A3 extended self-made evaluation")
    parser.add_argument(
        "--mock", action="store_true", help="Use the bundled mock endpoint"
    )
    parser.add_argument("--endpoint", type=str, default=None, help="vLLM endpoint URL")
    parser.add_argument("--model", type=str, default="mock-model", help="Model name")
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(ROOT_DIR / "corpus"),
        help="Path to corpus directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--query-mode", type=str, default=None, help="current | window | rrf | rewrite (default: TABot's)"
    )
    parser.add_argument(
        "--abstract-weight", type=float, default=None, help="weight of title+abstract ranking (0 disables)"
    )
    parser.add_argument("--skip-memory", action="store_true", help="Run retrieval only")
    args = parser.parse_args()

    # Resolve endpoint
    server = None
    if args.mock:
        from mock_endpoint import start_mock_endpoint

        endpoint, server = start_mock_endpoint()
        print(f"Mock endpoint started at {endpoint}\n")
    elif args.endpoint:
        endpoint = args.endpoint
    else:
        print("ERROR: specify --mock or --endpoint <url>", file=sys.stderr)
        sys.exit(1)

    config = {
        "endpoint": endpoint,
        "model": args.model,
        "corpus_path": args.corpus,
        "temperature": 0.0,
        "seed": args.seed,
    }
    if args.query_mode:
        config["query_mode"] = args.query_mode
    if args.abstract_weight is not None:
        config["abstract_weight"] = args.abstract_weight

    # Import and instantiate
    try:
        from ta_bot import TABot

        bot = TABot(config)
    except NotImplementedError:
        print("TABot.__init__ raises NotImplementedError — implement the stub first.")
        sys.exit(1)

    # Preflight checks
    run_preflight(bot)

    # Load eval data
    st_data = load_jsonl(SCRIPT_DIR / "single_turn.jsonl")
    mt_data = load_jsonl(SCRIPT_DIR / "multi_turn.jsonl")
    mp_data = [] if args.skip_memory else load_jsonl(SCRIPT_DIR / "memory_probes.jsonl")

    # Run
    print("=" * 60)
    print("SINGLE-TURN RETRIEVAL")
    print("-" * 60)
    st_hr, st_by_style = run_single_turn(bot, st_data)

    print()
    print("=" * 60)
    print("MULTI-TURN RETRIEVAL")
    print("-" * 60)
    mt_hr, mt_all_hr = run_multi_turn(bot, mt_data)

    mp_results = {}
    if mp_data:
        print()
        print("=" * 60)
        print("MEMORY PROBES")
        print("-" * 60)
        if args.mock:
            print("  (mock: answers are canned, so positive probes fail on output — check store only)")
        mp_results = run_memory_probes(bot, mp_data)

    # Summary
    print()
    print("=" * 60)
    print(f"SUMMARY (my_eval — query_mode={bot.query_mode}, abstract_weight={bot.abstract_weight})")
    print("-" * 60)
    print(f"  Single-turn hit-rate:  {st_hr:.1%}  ({len(st_data)} questions)")
    for style, hr in sorted(st_by_style.items()):
        n = sum(1 for d in st_data if d.get("style", "-") == style)
        print(f"    {style:<12} {hr:.1%}  ({n})")
    n_turns = sum(len(d["turns"]) for d in mt_data)
    print(f"  Multi-turn hit-rate:   {mt_hr:.1%}  ({len(mt_data)} dialogues, last turn)")
    print(f"                         {mt_all_hr:.1%}  ({n_turns} turns, every turn)")
    for probe_type, (passed, total) in mp_results.items():
        print(f"  Memory {probe_type + ':':<15}{passed}/{total} passed")
    print("=" * 60)

    if server:
        from mock_endpoint import stop_mock_endpoint

        stop_mock_endpoint(server)


if __name__ == "__main__":
    main()
