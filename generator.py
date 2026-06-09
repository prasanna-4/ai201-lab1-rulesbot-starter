from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)


def generate_response(query, retrieved_chunks):
    """
    Generate a grounded answer from retrieved rule chunks.

    TODO — Milestone 3:

    `retrieved_chunks` is the list returned by retrieve(). Each item is a dict:
      - "text"     : the chunk text
      - "game"     : the game name
      - "distance" : similarity score (you can use this to filter weak matches)

    Before writing code, talk through these with your group:
      - How will you format the chunks into a context block for the prompt?
      - What instructions will stop the model from answering beyond what the
        rules say? (Grounding is the whole point — a confident wrong answer
        is worse than an honest "I don't know.")
      - How will you surface which game each answer comes from?

    Your response should:
      1. Answer using only the retrieved context — not the model's general knowledge
      2. Make clear which game the answer comes from
      3. Say so clearly when the answer isn't in the loaded rules

    Return the response as a plain string.
    """
    if not retrieved_chunks:
        return (
            "I couldn't find anything relevant in the loaded rule books. "
            "Try rephrasing your question — or check that your ingestion pipeline is working."
        )

    # Grounding is enforced here. We prohibit specific behaviors (no outside
    # knowledge, no guessing) and prescribe an exact fallback sentence so an
    # honest "I don't know" is a defined, detectable output.
    system_prompt = (
        "You are RulesBot, a board game rules assistant. Answer the user's "
        "question using ONLY the rule text provided in the context below. Do not "
        "use any outside knowledge about board games, and do not guess or fill in "
        "gaps from what you already know. If the answer is not contained in the "
        "provided context, reply exactly: \"I couldn't find that in the loaded "
        "rule books.\" Do not add information that is not stated in the context, "
        "even if it sounds plausible.\n\n"
        "Always state which game your answer is about, naming the game explicitly "
        "(e.g. \"According to the Catan rules, ...\"). If the relevant rules come "
        "from more than one game, answer each game separately and label each part "
        "with its game."
    )

    # Format each chunk as a numbered, game-labelled block. Explicit source
    # boundaries help the model attribute facts to the right document and keep
    # rules from different games from blending together.
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[Source {i} — {chunk['game']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    return response.choices[0].message.content
