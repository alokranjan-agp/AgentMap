
from __future__ import annotations
from typing import List, Dict, Any, Optional
import hashlib
try:
    from any_llm import get_client
except Exception:
    get_client = None  # type: ignore

def stable_tiebreak_key(goal: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{goal}|{candidate_id}".encode("utf-8")).hexdigest()

class LLMRanker:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.0):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.client = get_client(provider) if (get_client and provider) else None

    def rank(self, goal: str, candidates: List[Dict[str, Any]]) -> List[str]:
        ids = [c["id"] for c in candidates]
        if not self.client:
            return sorted(ids, key=lambda c: (0, c))
        prompt = {"role":"system","content":"Rank candidates by relevance. Return a JSON list of IDs only."}
        user = {"role":"user","content": f"Goal: {goal}\nCandidates:\n" + "\n".join([f"- {c['id']}: {c.get('desc','')}" for c in candidates])}
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=[prompt, user], temperature=self.temperature)
            text = resp.choices[0].message["content"] if hasattr(resp.choices[0], "message") else resp.choices[0].message.content
            import json
            ranked = json.loads(text)
            ranked = [r for r in ranked if r in ids]
            rest = [i for i in ids if i not in ranked]
            rest = sorted(rest, key=lambda c: stable_tiebreak_key(goal, c))
            return ranked + rest
        except Exception:
            return sorted(ids, key=lambda c: stable_tiebreak_key(goal, c))
