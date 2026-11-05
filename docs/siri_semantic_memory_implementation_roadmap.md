# Implementation Roadmap: Custom Semantic Memory for `siri`

This document serves as the step-by-step developer guide for adding custom, production-grade **Semantic Memory** to the `siri` LangGraph project.

---

## 📌 Architectural Principles
1. **Realtime Path (Fast Fetch)**: Chat queries execute in $< 2$ seconds by fetching reliable pre-stored facts ($\ge 0.65$) from `SemanticMemoryStore` and injecting them into `LangGraph State`.
2. **Async Background Path (Distillation)**: Heavy LLM fact extraction (`gpt-4o-mini`) runs in the background after sessions finish when $\ge 2$ session summaries accumulate.
3. **Fact Evolution**: Facts grow stronger ($+0.1$) on confirmation and weaker ($-0.2$) on contradiction.

---

## 🗺️ Implementation Roadmap & Checklist

### Phase 1: Create Memory Core Module (`memory/`)
- [x] Create `c:\Flutter Projects\siri\memory\` folder.
- [ ] Create `memory/__init__.py`
- [ ] Create `memory/data_model.py` (Part 1 - `SemanticFact` dataclass & `FactCategory`)
- [ ] Create `memory/distiller.py` (Part 2 - `distil_episodes_to_facts` using `gpt-4o-mini` & `hashlib.md5`)
- [ ] Create `memory/store.py` (Part 3 - `SemanticMemoryStore` with `merge_facts`, `get_reliable_facts`, and `format_for_injection`)

### Phase 2: Extend LangGraph State Schema
- [ ] Modify `c:\Flutter Projects\siri\state\message_state.py` to add `semantic_profile: str` to `State(MessagesState)`.

### Phase 3: Update Worker & Supervisor Agents
- [ ] Update System Prompts in `agents/supervisor_agent.py`, `agents/research_agent.py`, `agents/weather_agent.py`, `agents/actionagent.py` to incorporate `{semantic_profile}` from State.

### Phase 4: Wire Realtime Retrieval & Async Loop in `main.py`
- [ ] Import memory store in `c:\Flutter Projects\siri\main.py`.
- [ ] Inject reliable profile into `initial_state` before running graph stream.
- [ ] Summarize finished session messages into `episode_summary`.
- [ ] Queue summary and auto-trigger distillation when pending queue $\ge 2$.
- [ ] Persist memory store to JSON disk storage (`siri_memory.json`).

---

## 📄 File Specifications & Code Drafts

### 1. `memory/data_model.py`
```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict

class FactCategory(str, Enum):
    BEHAVIOURAL   = "behavioural"
    FINANCIAL     = "financial"
    RISK          = "risk"
    PREFERENCE    = "preference"
    COMMUNICATION = "communication"
    GENERAL       = "general"

@dataclass
class SemanticFact:
    fact_id: str
    user_id: str
    statement: str
    category: str
    confidence: float
    observation_count: int
    source_episode_ids: List[str]
    first_observed: str
    last_confirmed: str
    evolution_log: List[Dict] = field(default_factory=list)

    def is_reliable(self, threshold: float = 0.65) -> bool:
        return self.confidence >= threshold

    def strengthen(self, episode_id: str) -> None:
        old_confidence = self.confidence
        self.confidence = min(1.0, self.confidence + 0.1)
        self.observation_count += 1
        if episode_id not in self.source_episode_ids:
            self.source_episode_ids.append(episode_id)
        self.last_confirmed = datetime.now(timezone.utc).isoformat()
        self.evolution_log.append({
            "action": "confirmed",
            "old_confidence": round(old_confidence, 3),
            "new_confidence": round(self.confidence, 3),
            "episode_id": episode_id,
            "timestamp": self.last_confirmed,
        })

    def weaken(self, reason: str = "") -> None:
        old_confidence = self.confidence
        self.confidence = max(0.0, self.confidence - 0.2)
        now = datetime.now(timezone.utc).isoformat()
        self.evolution_log.append({
            "action": "weakened",
            "old_confidence": round(old_confidence, 3),
            "new_confidence": round(self.confidence, 3),
            "reason": reason,
            "timestamp": now,
        })

    def format_for_injection(self) -> str:
        conf_label = "[high confidence]" if self.confidence >= 0.85 else "[medium confidence]"
        return f"- {self.statement} {conf_label}"
```

### 2. `state/message_state.py` (Extension)
```python
from langgraph.graph import MessagesState
from pydantic import BaseModel
from typing import Literal

class State(MessagesState):
    next: str
    semantic_profile: str  # <--- Added for semantic memory injection
```

### 3. `main.py` Integration Snippet
```python
# Realtime Chat Path:
profile = memory_store.format_for_injection(categories=None) # Filters confidence >= 0.65
initial_state = {
    "messages": [{"role": "user", "content": user_question}],
    "semantic_profile": profile
}

# Run Graph...
# ...

# Post-Session Async Summary & Distillation:
full_text = "\n".join([f"{m.type}: {m.content}" for m in final_state["messages"]])
episode_summary = summarize_llm(full_text) # 1-call gpt-4o-mini

pending_episodes.append({
    "summary": episode_summary,
    "episode_id": f"ep_{session_id}",
    "date": datetime.now(timezone.utc).isoformat()
})

if len(pending_episodes) >= 2:
    new_facts = distil_episodes_to_facts(pending_episodes, user_id="siri_user")
    memory_store.merge_facts(new_facts)
    memory_store.persist("siri_memory.json")
    pending_episodes.clear()
```

---

## 🧪 Verification Plan
1. **Run Turn 1**: Ask a query where user expresses a specific preference (e.g. *"I prefer metric units and short answers"*). Verify summary is generated.
2. **Run Turn 2**: Ask a second query to trigger distillation threshold ($\ge 2$). Check that `siri_memory.json` is created with distilled facts.
3. **Run Turn 3**: Ask a new question. Inspect logs to confirm `semantic_profile` with `confidence >= 0.65` is injected into initial LangGraph State!
