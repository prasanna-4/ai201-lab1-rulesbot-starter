# Spec: `retrieve()`

**File:** `retriever.py`
**Status:** Spec incomplete — fill in all blank fields before implementing

---

## Purpose

Given a user's natural language query, find the most relevant chunks from the vector store using semantic similarity search. Return them ranked by relevance so that `generate_response()` can use them as context.

---

## Input / Output Contract

**Inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | The user's natural language question |
| `n_results` | `int` | Maximum number of chunks to return (default: `N_RESULTS` from `config.py`) |

**Output:** `list[dict]`

Each dict in the returned list must contain exactly these keys:

| Key | Type | Description |
|-----|------|-------------|
| `"text"` | `str` | The chunk text |
| `"game"` | `str` | The game name this chunk came from |
| `"distance"` | `float` | Cosine distance score — lower means more similar to the query |

Results should be ordered from most to least relevant (lowest to highest distance). Returns an empty list `[]` if the collection contains no documents.

---

## Design Decisions

*Complete the fields below before writing any code. Use your AI tool in Plan or Ask mode to help you reason through what belongs here — but the decisions are yours.*

---

### Query approach

*Describe how you will use `_collection.query()` to find relevant chunks. What arguments will you pass, and why?*

```
Call _collection.query() with three arguments:
  - query_texts=[query]  — a list holding the single user question. ChromaDB
    embeds it with the SAME model used for the stored chunks (all-MiniLM-L6-v2),
    so the query vector lives in the same 384-dim space as the chunks.
  - n_results=n_results  — how many of the closest chunks to return (default 3
    from config.N_RESULTS).
  - include=["documents", "metadatas", "distances"]  — we need all three:
    documents (the chunk text), metadatas (to recover the game name), and
    distances (the cosine score we expose so generate_response can judge
    relevance). ids/embeddings are not needed in the return contract.

The embedding function is attached to the collection at creation, so we hand
over raw text and ChromaDB does the vector math and nearest-neighbour search.
```

---

### Return structure

*Sketch out what one item in your return list looks like as a concrete example. Where does each field come from in the query results?*

```
A flat list of dicts, ordered closest-first. One item looks like:

  {
    "text":     "x, that hex produces no resources that turn, ...",  # results["documents"][0][i]
    "game":     "Catan",                                             # results["metadatas"][0][i]["game"]
    "distance": 0.466,                                               # results["distances"][0][i]
  }

Each field is pulled from the matching position i in the three parallel inner
lists. "text" comes from documents, "game" is read out of the metadata dict we
stored in embed_and_store ({"game": ...}), and "distance" is the cosine distance.
The lists are already returned in ascending-distance order, so iterating in order
gives most-relevant-first for free.
```

---

### Handling the nested result structure

*`_collection.query()` returns nested lists. Describe what index you need to access to get the actual list of results for a single query, and why the nesting exists.*

```
query() supports BATCH queries — you can pass many query strings at once, so
every field is a list-of-lists: one inner list per query string. The outer
length equals the number of queries.

We pass exactly one query, so our results live at index [0] of each field:
  results["documents"][0]   -> list of chunk texts
  results["metadatas"][0]   -> list of metadata dicts
  results["distances"][0]   -> list of distances

Verified: with one query and n_results=3, outer len == 1 and inner len == 3.
Forgetting the [0] (e.g. iterating results["documents"] directly) gives you the
single inner list as one element, which is the #1 bug in this milestone.
```

---

### Relevance threshold

*Will you filter out results above a certain distance score, or return all `n_results` regardless of how relevant they are? What are the tradeoffs of each approach?*

```
retrieve() returns all n_results unfiltered, ranked closest-first. It stays a
pure search primitive; relevance judgement is left to generate_response().

Why no hard threshold HERE: measured distances on this corpus show relevant
answers land around 0.37–0.51 (Pandemic disease-cubes 0.373, roll-a-7 0.466,
"how do you win?" Monopoly 0.507), borderline 0.5–0.62, and total nonsense
("airspeed of a swallow") at 0.87+. A naive 0.5 cutoff — which the system-design
doc loosely suggests — would wrongly discard the legitimate 0.507 Monopoly
answer. Because character-based chunking inflates distances (chunks start/end
mid-sentence), there is no single clean threshold that separates good from bad
without losing real answers.

Tradeoffs:
  - Filtering in retrieve(): risks returning [] for valid questions and forces
    the threshold decision into a low-level function that can't see the prompt.
  - Returning all n_results: simpler, predictable, and lets the grounding prompt
    be the real defense — if the chunks don't contain the answer, the LLM says
    so. Cost: weak chunks still reach the generator, mitigated by grounding.
We chose the second. Distance is surfaced so a threshold CAN be applied later.
```

---

### Edge cases

*How does your implementation behave when: (a) the collection is empty, (b) the query matches no chunks well, (c) the query matches chunks from multiple games?*

```
(a) Empty collection: guard with `if _collection.count() == 0: return []`
    before querying. generate_response() turns [] into its fallback message.

(b) No good match: query() ALWAYS returns the n_results nearest chunks, even
    if they're far away — there is no "nothing matched". We still return them,
    but their high distances (e.g. 0.87+) signal weak relevance, and the
    grounding prompt makes the LLM admit the answer isn't in the rules.

(c) Multiple games: expected and correct. A broad question like "how do you
    win?" legitimately matches several games' victory rules; we return the
    closest across the whole corpus regardless of game. The "game" field on
    each result lets the generator attribute each answer to its source.
```

---

## Implementation Notes

*Fill this in after implementing, before moving to Milestone 3.*

**Test query and top result returned:**

```
Query: What happens when you run out of disease cubes in Pandemic?
Top result game: Pandemic
Distance score: 0.373
Does it make sense? Yes — top 3 results were all Pandemic, and the #1 chunk
contains the exact rule (lose immediately if a color of cubes runs out). The
right game and the right rule surfaced first.
```

**One thing about the query results that surprised you:**

```
Distance scores were noticeably HIGHER than the lab's example numbers. The
lab shows ~0.14 for a great "roll a 7" match; our best real matches sit around
0.37–0.47. The cause is character-based chunking: chunks start and end mid-
sentence (e.g. the top "roll a 7" chunk begins "x, that hex produces no
resources..."), so even a correct chunk isn't a clean semantic unit and its
distance is inflated. Ranking and game attribution are still correct — but it
showed clearly why a fixed "0.5 = irrelevant" threshold would be wrong here.
```
