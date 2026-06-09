"""
RulesBot retrieval evaluation harness (Optional Challenge 3).

A lightweight, automated alternative to manually spot-checking answers. We define
a set of question/answer pairs where we know which GAME the answer should come
from, run each question through retrieve(), and score whether the correct game
shows up in the retrieved results.

This is a simplified version of how production RAG systems are evaluated:
instead of eyeballing outputs, you measure retrieval quality against a fixed set
of known-good examples, so you can tell whether a change (chunk size, n_results,
embedding model) made retrieval better or worse.

Two metrics are reported:
  - top-1 accuracy : did the #1 result come from the correct game?
  - top-k accuracy : did the correct game appear anywhere in the top n_results?

Run:  .venv/Scripts/python.exe eval.py
"""

from retriever import retrieve
from config import N_RESULTS

# Each case: a question, the game it should retrieve from, and a keyword we'd
# expect to see in a relevant chunk (a loose sanity check on content, not an
# exact-match requirement). Keep these grounded in rules that actually exist in
# the loaded /docs files.
EVAL_CASES = [
    {
        "query": "What happens when you roll a 7?",
        "expected_game": "Catan",
        "expect_keyword": "robber",
    },
    {
        "query": "How many resource cards can I hold before the robber forces a discard?",
        "expected_game": "Catan",
        "expect_keyword": "7",
    },
    {
        "query": "How do you get out of Jail in Monopoly?",
        "expected_game": "Monopoly",
        "expect_keyword": "Jail",
    },
    {
        "query": "What happens when you run out of disease cubes in Pandemic?",
        "expected_game": "Pandemic",
        "expect_keyword": "cube",
    },
    {
        "query": "How does an outbreak spread in Pandemic?",
        "expected_game": "Pandemic",
        "expect_keyword": "outbreak",
    },
    {
        "query": "How does the Spymaster give clues in Codenames?",
        "expected_game": "Codenames",
        "expect_keyword": "clue",
    },
    {
        "query": "Can two players claim the same route in Ticket to Ride?",
        "expected_game": "Ticket To Ride",
        "expect_keyword": "route",
    },
    {
        "query": "How does attacking work in Risk?",
        "expected_game": "Risk",
        "expect_keyword": "dice",
    },
    {
        "query": "When can you play a Wild Draw Four in Uno?",
        "expected_game": "Uno",
        "expect_keyword": "Wild",
    },
    {
        "query": "How does making a Suggestion work in Clue?",
        "expected_game": "Clue",
        "expect_keyword": "suggestion",
    },
]


def _norm(game):
    return game.strip().lower()


def evaluate(cases=EVAL_CASES, n_results=N_RESULTS, verbose=True):
    """Run every eval case through retrieve() and report accuracy."""
    top1_hits = 0
    topk_hits = 0
    keyword_hits = 0
    failures = []

    for case in cases:
        results = retrieve(case["query"], n_results=n_results)
        games = [_norm(r["game"]) for r in results]
        expected = _norm(case["expected_game"])

        top1 = bool(results) and games[0] == expected
        topk = expected in games
        # keyword check against any chunk from the correct game
        correct_game_text = " ".join(
            r["text"].lower() for r in results if _norm(r["game"]) == expected
        )
        kw = case["expect_keyword"].lower() in correct_game_text

        top1_hits += top1
        topk_hits += topk
        keyword_hits += kw

        if verbose:
            mark = "PASS" if top1 else ("topk" if topk else "FAIL")
            top = f"{results[0]['game']} ({results[0]['distance']:.3f})" if results else "<none>"
            print(f"[{mark:4}] {case['query'][:52]:54} -> {top}")

        if not topk:
            failures.append({
                "query": case["query"],
                "expected_game": case["expected_game"],
                "got": [(r["game"], round(r["distance"], 3)) for r in results],
            })

    n = len(cases)
    print("\n" + "=" * 60)
    print(f"Cases:            {n}")
    print(f"Top-1 accuracy:   {top1_hits}/{n} = {top1_hits / n:.0%}  (correct game ranked #1)")
    print(f"Top-{n_results} accuracy:   {topk_hits}/{n} = {topk_hits / n:.0%}  (correct game in top {n_results})")
    print(f"Keyword present:  {keyword_hits}/{n} = {keyword_hits / n:.0%}  (expected term in a correct-game chunk)")
    print("=" * 60)

    if failures:
        print("\nFAILURES (correct game not retrieved at all) — logged for review:")
        for f in failures:
            print(f"  Q: {f['query']}")
            print(f"     expected: {f['expected_game']}  |  got: {f['got']}")
    else:
        print("\nNo retrieval failures — the correct game appeared for every case.")

    return {
        "n": n,
        "top1": top1_hits,
        "topk": topk_hits,
        "keyword": keyword_hits,
        "failures": failures,
    }


if __name__ == "__main__":
    evaluate()
