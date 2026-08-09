from typing import Dict, List
from data_model import SemanticFact, FactCategory
import json                              # Serialisation and JSON response parsing.
import os                                # Environment variables.
import time                              # Rate-limit delays.
from datetime import datetime, timezone
DISTILLATION_PROMPT = """You are a semantic memory distiller for FinCoach.
Your job is to read a set of past session summaries and extract GENERAL TRUTHS about the user.

These are not facts about specific sessions (that is episodic memory).
These are patterns that hold across multiple sessions — reusable generalisations.

Return a JSON object with this structure:
{
  "facts": [
    {
      "statement": "A single general truth about the user — no specific dates or sessions",
      "category": "behavioural | financial | risk | preference | communication | general",
      "confidence": 0.6 to 0.9 (based on how consistently this pattern appears)
    }
  ]
}

CATEGORIES:
- behavioural : how the user typically acts and makes decisions
- financial   : general financial patterns and tendencies
- risk        : risk tolerance and risk-related patterns
- preference  : what the user consistently prefers or avoids
- communication: how the user likes to receive and process advice
- general     : anything that does not fit the above

CONFIDENCE GUIDANCE:
- 0.6 : pattern observed in 1 session — possible but uncertain
- 0.7 : pattern in 2 sessions — likely
- 0.8 : pattern in 3+ sessions — reliable
- 0.9 : consistent across all sessions — strong generalisation

STRICT RULES:
1. Return ONLY valid JSON — no markdown, no explanation.
2. Each statement must be general — no specific dates, session numbers, or amounts.
3. Extract only patterns that genuinely appear across episodes — do not invent.
4. Maximum 8 facts — choose the most useful and reliable ones.
5. Each fact must be actionable — it should change how FinCoach advises this user.
6. Write in third person: 'The user...' or 'Chiru...'
"""


def distil_episodes_to_facts(
    episode_summaries: List[Dict],
    user_id: str,
) -> List[SemanticFact]:
    """
    Extract semantic facts from a list of episode summaries.
    This is the distillation step: many episodes → compact general truths.

    Args:
        episode_summaries: List of dicts with 'summary', 'date', 'episode_id'.
        user_id:           The user these episodes belong to.

    Returns:
        List of SemanticFact objects ready for merging into the semantic store.
    """

    # Format episode summaries for the distillation prompt.
    episodes_text = "\n\n".join(
        f"[Session {i+1} — {ep.get('date', 'unknown')}]:\n{ep['summary']}"
        for i, ep in enumerate(episode_summaries)
    )
    # Number the sessions to help the model assess pattern frequency.

    response = client.chat.completions.create(
        model=DISTILLATION_MODEL,
        # gpt-4o-mini: pattern extraction task — cheaper model is sufficient.
        max_tokens=800,
        # 8 facts at ~50 tokens each = 400 tokens max — 800 gives headroom.
        temperature=0.0,
        # Deterministic — fact extraction should not vary.
        response_format={"type": "json_object"},
        # Guaranteed valid JSON — no parsing issues.
        messages=[
            {"role": "system", "content": DISTILLATION_PROMPT},
            {"role": "user",   "content":
             f"Extract general semantic facts from these {len(episode_summaries)} sessions:\n\n"
             f"{episodes_text}"
            },
        ]
    )

    raw = json.loads(response.choices[0].message.content)
    raw_facts = raw.get("facts", [])

    # Convert raw dicts to SemanticFact objects.
    facts = []
    episode_ids = [ep.get("episode_id", "") for ep in episode_summaries]
    now = datetime.now(timezone.utc).isoformat()

    for raw_fact in raw_facts:
        statement  = raw_fact.get("statement", "").strip()
        if not statement:
            continue
            # Skip empty statements — safety guard.

        # Generate a short hash as the fact_id for deduplication.
        import hashlib
        fact_id = hashlib.md5(
            f"{user_id}:{statement[:80]}".encode()
        ).hexdigest()[:12]
        # MD5 of (user_id + first 80 chars of statement).
        # Same statement for same user → same ID → natural deduplication.

        fact = SemanticFact(
            fact_id=fact_id,
            user_id=user_id,
            statement=statement,
            category=raw_fact.get("category", FactCategory.GENERAL),
            confidence=float(raw_fact.get("confidence", 0.6)),
            observation_count=len(episode_summaries),
            # All provided episodes contributed to each extracted fact.
            source_episode_ids=episode_ids,
            first_observed=now,
            last_confirmed=now,
        )
        facts.append(fact)

    tokens_used = response.usage.total_tokens
    print(f"  [DISTIL] {len(episode_summaries)} episodes → "
          f"{len(facts)} semantic facts | tokens: {tokens_used}")

    return facts


print("Distillation engine defined.")