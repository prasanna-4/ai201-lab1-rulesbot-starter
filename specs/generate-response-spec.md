# Spec: `generate_response()`

**File:** `generator.py`
**Status:** Spec incomplete — fill in all blank fields before implementing

---

## Purpose

Given a user query and a list of retrieved rule chunks, generate a response that directly answers the question using only the retrieved text as context. The response must be grounded — it should not draw on the model's general knowledge of board games, only on what was retrieved.

---

## Input / Output Contract

**Inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | The user's original question |
| `retrieved_chunks` | `list[dict]` | Ranked list of chunks from `retrieve()`, each with `"text"`, `"game"`, and `"distance"` |

**Output:** `str`

A plain string containing the response to show the user. The response should:
- Answer the question using only the retrieved rule text
- Identify which game the answer comes from
- Acknowledge clearly when the answer is not found in the loaded rules

Returns a fallback string (not an error) when `retrieved_chunks` is empty.

---

## Design Decisions

*Complete the fields below before writing any code. Use your AI tool in Plan or Ask mode to help you reason through what belongs here — but the decisions are yours.*

---

### Context formatting

*How will you format the retrieved chunks before passing them to the LLM? Describe the structure — not the code. Consider: will you label chunks by game? Include distance scores? Separate chunks with delimiters?*

```
Each chunk is rendered as a numbered, game-labelled block separated by blank
lines, e.g.:

  [Source 1 — Catan]
  <chunk text>

  [Source 2 — Pandemic]
  <chunk text>

Decisions:
  - Label every chunk with its game. This is what lets the model cite the
    correct source and is essential when results span multiple games.
  - Number the sources. Research on multi-document prompting shows that
    explicit, delimited source boundaries help the model attribute facts to the
    right document and reduce blending of unrelated chunks.
  - Do NOT include distance scores. They are an internal retrieval signal, not
    meaning the model should reason about; exposing them invites the model to
    editorialize about confidence instead of answering from the text.
  - Blank-line + bracket delimiters keep boundaries unambiguous so a rule from
    one game can't bleed into another in the model's reading.
```

---

### System prompt — grounding instruction

*Write the exact system prompt instruction you will use to prevent the model from answering beyond the retrieved text. This is the most important design decision in this function.*

```
Exact text used in the system prompt:

"You are RulesBot, a board game rules assistant. Answer the user's question
using ONLY the rule text provided in the context below. Do not use any outside
knowledge about board games, and do not guess or fill in gaps from what you
already know. If the answer is not contained in the provided context, reply
exactly: \"I couldn't find that in the loaded rule books.\" Do not add
information that is not stated in the context, even if it sounds plausible."

Why this wording:
  - It prohibits a SPECIFIC behavior ("do not use outside knowledge / do not
    guess") rather than requesting a vague outcome ("be accurate") — the latter
    gives the model room to sidestep.
  - It prescribes an EXACT fallback sentence, so "I don't know" is a defined,
    detectable output rather than left to improvisation.
  - "even if it sounds plausible" closes the most common loophole: a confident,
    well-written answer that happens to come from training data.
```

---

### System prompt — citation instruction

*Write the exact instruction you will use to tell the model to identify which game its answer comes from.*

```
Exact text used in the system prompt:

"Always state which game your answer is about, naming the game explicitly (e.g.
\"According to the Catan rules, ...\"). If the relevant rules come from more
than one game, answer each game separately and label each part with its game."

Why: the citation makes grounding VERIFIABLE — a user can check the answer
against the named rulebook. It also forces the model to commit to a source
rather than producing a generic, game-agnostic answer, and the multi-game
clause handles broad queries like "how do you win?" cleanly.
```

---

### Fallback behavior

*What should the response say when the answer isn't found in the loaded rule books? Write the exact fallback message.*

```
Two distinct cases:

1. retrieve() returned NOTHING (empty list — e.g. empty collection). Return,
   without calling the LLM, the existing message:
   "I couldn't find anything relevant in the loaded rule books. Try rephrasing
   your question — or check that your ingestion pipeline is working."

2. Chunks were retrieved but none actually contain the answer. We don't hard-
   filter this case in code; instead the grounding prompt instructs the model
   to reply exactly:
   "I couldn't find that in the loaded rule books."

Keeping the two messages distinct separates a pipeline problem (case 1) from an
honest "not in the rules" (case 2).
```

---

### Handling low-relevance chunks

*`retrieved_chunks` may include chunks with high distance scores (weak relevance). Will you filter these out before building context, pass them all in, or handle them another way? What are the tradeoffs?*

```
Primary defense is the grounding prompt, not a distance filter. We pass the
retrieved chunks into context and rely on the prompt to make the model ignore
chunks that don't contain the answer and emit the fallback sentence instead.

We DON'T apply a tight numeric cutoff because measured distances make it unsafe:
real answers sit at 0.37–0.51 while nonsense sits at 0.87+, with no clean gap —
a 0.5–0.6 threshold would silently drop valid answers (e.g. the 0.507 "how do
you win?" Monopoly chunk).

Tradeoffs:
  - Hard filter: cleaner context, but risks discarding the only relevant chunk
    and producing false "I don't know" answers.
  - Prompt-only (chosen): robust to threshold-picking, and a 70B model reliably
    follows a strong grounding instruction. Cost: a few weak chunks occupy the
    context window — acceptable at n_results=3.
A very loose safety cutoff (e.g. drop chunks above ~0.85, the nonsense band)
could be added later without risking real answers, but isn't required.
```

---

### Message structure

*Describe how you will structure the messages list for the API call — what goes in the system message vs. the user message?*

```
Two messages:

  system: the fixed grounding + citation + fallback instructions (who the model
          is and the rules it must obey). Stable across every request.

  user:   the per-request payload — the formatted context block followed by the
          user's question, e.g.:

            CONTEXT:
            [Source 1 — Catan]
            ...

            QUESTION: <the user's query>

Why split this way: instructions belong in the system role where the model
weights them most heavily as standing policy; the variable context+question
belong in the user role as the thing to act on. Keeping the grounding rules out
of the user message also means user input can't dilute or override them.

API call: _client.chat.completions.create(model=LLM_MODEL, messages=[...],
temperature low (~0) so answers stay faithful to the text rather than creative.
```

---

## Implementation Notes

*Fill this in after implementing and testing.*

**Test query and response:**

```
Query: How do you get out of Jail in Monopoly?
Response: "According to the Monopoly rules, to get out of Jail you can: pay a
  $50 fine before rolling on any of your next three turns, use a Get Out of Jail
  Free card, or roll doubles..." (abbreviated)
Correctly grounded? Yes — every detail (incl. the "three turns" wording) is from
  the loaded Monopoly chunk, not generic knowledge; cited the source explicitly.
Cited the right game? Yes — opened with "According to the Monopoly rules".

Grounding stress test — Query: "What are the rules of chess?" (not loaded)
Response: "I couldn't find that in the loaded rule books." — exact fallback,
  even though weak Monopoly/Catan chunks were in context. No improvisation.
```

**One thing you changed from your original spec after seeing the actual output:**

```
I kept the prompt-only grounding (no distance filter) and it held up better than
expected — the model emitted the exact fallback for the out-of-corpus chess query
despite being handed loosely-related chunks. The one behavior I had NOT fully
anticipated was the multi-game answer: for "Can I trade with the bank?" the model
answered Catan's 4-for-1 rule AND volunteered that the Monopoly chunk doesn't
cover resource trading. That's correct and useful, but it means responses can be
longer/multi-part than the single-game answer I first pictured — the citation
instruction's multi-game clause is doing real work, so I left it in.
```
