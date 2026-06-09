# RulesBot — Planning Doc

Use this file to record your design decisions as you work through the lab.
There are no wrong answers — write enough that you could explain your reasoning to another group.

---

## Chunking Strategy

**Chunk size:** 300 characters.

Long enough to hold one complete rule (a rule is usually 1–3 sentences), short
enough that retrieval returns a targeted result instead of a wall of mixed
rules. Confirmed by experiment: 300 gives clean per-game separation, while 1200
dilutes the embedding and 80 fragments rules (see "Design Experiments" below).

**Overlap:** 50 characters.

Roughly one short sentence duplicated at each boundary, so a rule that lands on
a chunk edge still appears intact in at least one chunk. Enough to protect
boundary content without meaningfully bloating the DB.

**Why this strategy fits rule book text:**

Rule text is semantically dense — each sentence is a distinct constraint. That
pushes toward small chunks (one rule = one retrievable unit) rather than the
paragraph-level chunks you'd use for prose. The character-based sliding window
is simple and produces uniform chunk sizes; its cost is that it ignores
sentence boundaries (chunks can start mid-sentence), which inflates distances
slightly but doesn't hurt ranking.

---

## Retrieval Observations

After implementing retrieval, try these test queries and record what comes back:

| Query | Top result game | Does it make sense? |
|-------|----------------|---------------------|
| "How do you win?" | Monopoly (0.507), then Risk (0.509), Ticket to Ride (0.522) | Yes — a broad question legitimately matches the victory rules of several games; distances are close because all are equally relevant. |
| "What happens when you roll a 7?" | Catan (0.466) | Yes — Catan is the only loaded game where rolling a 7 is a rule (robber). Ranks 2–3 are Risk dice chunks, clearly farther (0.597+). |
| "Can two players share a route?" | Ticket to Ride (0.365) | Yes — all three top results are Ticket to Ride, the only game with "routes". Tight, correct cluster. |

**Anything surprising?**

Distances run higher than the lab's example numbers (best real matches ~0.37,
not ~0.14). Cause: character-based chunks start/end mid-sentence, so even a
correct chunk isn't a clean semantic unit. Ranking is unaffected — but it shows
why a fixed "0.5 = irrelevant" cutoff would be wrong here: the legitimate "how
do you win?" Monopoly answer sits at 0.507 and would be wrongly discarded.

---

## Response Quality

After implementing generation, try 2–3 questions and assess the answers:

| Query | Answer accurate? | Properly grounded? | Cited the right game? |
|-------|-----------------|-------------------|----------------------|
| "How do you get out of Jail in Monopoly?" | Yes ($50 fine / doubles / Get Out of Jail Free card) | Yes — every detail is from the loaded Monopoly chunk | Yes — "According to the Monopoly rules…" |
| "Can I trade resources with the bank?" | Yes (Catan 4-for-1) | Yes — and it noted Monopoly's chunk doesn't cover resource trading | Yes — answered Catan, labeled the Monopoly aside |
| "What are the rules of chess?" (not loaded) | Correctly refused | Yes — returned the exact fallback despite weak chunks in context | N/A — admitted it isn't in the books |

**What would you change about the prompt to improve grounding?**

The prompt already prohibits a specific behavior ("do not use outside knowledge,
do not guess… even if it sounds plausible") and prescribes an exact fallback
sentence, which held up even on the out-of-corpus chess query. The next
improvement would be requiring an inline source tag per claim (e.g. a
`[Catan]` marker on each sentence) so multi-game answers are auditable
sentence-by-sentence, not just per answer.

---

# Design Experiments

Reproduce with `.venv/Scripts/python.exe chunking_experiment.py` and `eval.py`.

## Experiment 1 — Chunk size: tiny vs. base vs. huge

**Setup:** Same 8 documents, same embedding model, same 4 queries, `n_results=3`.
Only the chunking config differs; each config gets its own in-memory ChromaDB
collection so the persistent store is untouched.

| Config | chunk_size | overlap | total chunks |
|--------|-----------:|--------:|-------------:|
| tiny   | 80         | 15      | **563**      |
| base   | 300        | 50      | **149**      |
| huge   | 1200       | 100     | **36**       |

### Top result per query (distance = cosine, lower is better)

| Query | tiny | base | huge |
|-------|------|------|------|
| roll a 7? | Catan **0.358** | Catan 0.466 | Catan 0.638 ⚠️ wrong rule ("TURN STRUCTURE") |
| out of Jail (Monopoly)? | Monopoly 0.427 | Monopoly **0.367** | Monopoly 0.434 |
| run out of disease cubes? | Pandemic **0.242** | Pandemic 0.407 | Pandemic 0.488 |
| attacking in Risk? | Risk 0.453 | Risk **0.383** | Risk 0.529 |

### Where each config fails, and why

**tiny (80 chars) — lowest distances, but two real problems.**
- *Cross-game contamination below #1.* "roll a 7?" top-3 was Catan (0.358),
  **Monopoly (0.476), Pandemic (0.481)** — fragments containing "7"/"turns"
  from other games leak in, because an 80-char fragment lacks the context that
  keeps it inside its own game.
- *Clipped content hurts the generator even when ranking is right.* The top
  Catan chunk was `"he number rolled. ROLLING A 7 When a 7 is rolled, no
  resources are produce"` — starts and ends mid-word. Retrieval "worked" but
  the chunk is too truncated to ground a complete answer.
- Counter-intuitively, tiny had the **best** top-1 distances on precise queries
  (0.242, 0.358): a tight fragment matches a tight question closely. So
  "smaller = worse" isn't true at the top-1 level for these short clean docs —
  its real cost is contamination and clipped context.

**base (300 chars) — the balance, and what ships.**
- Cleanest separation: "roll a 7?" top-3 = Catan, Risk, Risk, **no
  Monopoly/Pandemic noise**. Each chunk ≈ one whole rule, so its embedding stays
  distinctively about that game's mechanic.
- Distances higher than tiny only because character splitting starts chunks
  mid-sentence. Ranking and game attribution correct everywhere.

**huge (1200 chars) — worst retrieval, clearest failure mode.**
- Highest distances, and the **only config that returned the wrong rule at #1**:
  "roll a 7?" surfaced a Catan "TURN STRUCTURE" chunk, plus Uno and Risk in the
  top 3.
- *Why:* a 1200-char chunk spans many rules, so its embedding is an **average**
  of all of them and the specific fact you asked about gets washed out. Big
  chunks blur meaning.

### Takeaway
Precision peaks at small–medium chunks for dense rule text. Too large dilutes
(wrong rule retrieved); too small fragments (contamination + clipped context
that weakens the answer). `chunk_size=300` is a sound default. Best likely
upgrade: **sentence-aware splitting at ~300 chars** — keeps base's clean
separation while fixing the mid-sentence clipping.

## Experiment 2 — Automated retrieval eval (`eval.py`)

10 question/expected-game pairs across all 8 games. For each: is the correct
game ranked #1, in the top 3, and does an expected keyword appear in a
correct-game chunk?

**Result (base config, the shipped system):**

```
Top-1 accuracy:   10/10 = 100%   (correct game ranked #1)
Top-3 accuracy:   10/10 = 100%   (correct game in top 3)
Keyword present:  10/10 = 100%   (expected term in a correct-game chunk)
No retrieval failures.
```

A miniature of the harness production RAG teams use: a fixed eval set you re-run
after any change to see immediately whether retrieval improved or regressed,
instead of trusting spot checks. `eval.py` logs any failure with the query and
what came back, ready for review.

## Experiment 3 — Adding a ninth game (Scrabble)

Wrote `docs/scrabble.txt` in the house style and ingested it incrementally
(20 chunks → 169 total). Re-running `eval.py` still scored 10/10 top-1 on the
original 8 games — no regression from adding a game.

**Retrieval quality on the new game is good but phrasing-sensitive:**

| Query | Top result |
|-------|-----------|
| "How many points is the bingo bonus in Scrabble?" | Scrabble **0.236** ✅ exact bingo chunk |
| "What does a blank tile score?" | Scrabble 0.342 ✅ |
| "What is the bonus for using all seven tiles in Scrabble?" | Scrabble 0.379, but **wrong chunk** — generator correctly said "I couldn't find that" |

**The instructive failure:** "bonus for using all seven tiles" did NOT retrieve
the bingo rule — the sentence "If a player uses all 7 tiles… they score a
50-point bonus" never reached the top 6. Rank 3 was only the *tail* of that
section ("…is bonus is added after any premium squares are applied"), because
character chunking split the key sentence from its qualifier into separate
chunks. Two lessons:
  1. **Retrieval failure, not generation failure** — the answer exists in the
     corpus, but the chunk holding it wasn't retrieved for this phrasing. The
     near-identical query with the word "bingo" retrieved it at 0.236. This is
     exactly the retrieval-vs-generation distinction from the discussion prompts.
  2. **Grounding held.** The model surely "knows" Scrabble's 50-point bonus from
     training, but it refused to state it because the retrieved context didn't
     contain it. Honest "I don't know" over confident-but-ungrounded — the system
     behaving as designed, even when it costs a recoverable answer.

Mitigations: sentence-aware chunking (keeps the rule whole), a higher
`n_results`, or query expansion ("seven tiles" → "bingo"). Document quality also
matters — the answer was present and correctly written; the limitation was
purely how it got split.

## Ideas not yet tried
- Loose safety cutoff in `generate_response()` (drop chunks above ~0.85, the
  observed nonsense band) — discards pure noise without risking real answers.
- Sentence-aware chunking at ~300 chars (Experiment 1 takeaway).
- Negative eval cases (questions with no answer in any rulebook) to measure the
  grounding fallback rate, not just retrieval accuracy.
