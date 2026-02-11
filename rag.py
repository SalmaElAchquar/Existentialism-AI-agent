import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests

INDEX_DIR = Path("index")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

import re
from typing import List, Dict, Any
REFUSAL_TEXT = (
    "I cannot answer this question within the constraints of this agent. "
    "The answer would require concepts or frameworks not contained in the provided corpus."
)
ALLOWED_TOPIC_PATTERNS = [
    # core movement / era
    r"\bexistential(ism|ist)?\b",
    r"\bsartre\b", r"\bjean[- ]paul\b",
    r"\bbeauvoir\b", r"\bsimone\b",
    r"\bcamus\b", r"\bheidegger\b",
    r"\bhusserl\b", r"\bmerleau[- ]ponty\b",
    r"\bjaspers\b",

   # core existential vocabulary (general philosophical terms used in the corpus)
    r"\bexistence\b", r"\bexist\b", r"\bto exist\b",
    r"\bbeing\b", r"\bessence\b",


    
    # core existentialist concepts (Sartre-heavy)
    r"\bbad faith\b", r"\bmauvaise foi\b",
    r"\bexistence precedes essence\b",
    r"\bfreedom\b", r"\bresponsibilit(y|ies)\b",
    r"\bauthentic(ity)?\b",
    r"\banguish\b", r"\babandonment\b",
    r"\bfor[- ]itself\b", r"\bin[- ]itself\b",
]
def in_allowed_domain(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in ALLOWED_TOPIC_PATTERNS)


BANNED_QUERY_PATTERNS = [
    # clinical / therapy / self-help
    r"\bdepression\b", r"\banxiety\b", r"\btherapy\b", r"\btreat(ment)?\b",
    r"\bpsychiatr(y|ic)\b", r"\bpsycholog(y|ical)\b", r"\bdiagnos(e|is)\b",
    r"\bcope\b", r"\bheal(ing)?\b", r"\bwell[- ]?being\b",

    # practical advice framing
    r"\bwhat should i do\b", r"\bshould i\b", r"\bhelp me\b", r"\bovercome\b",
    r"\bsteps\b", r"\badvice\b",

    # comparison / outside-corpus traditions
    r"\bcompare\b", r"\bvs\b", r"\bversus\b",
    r"\bstoic(ism)?\b", r"\bbuddh(ism|ist)\b",
    r"\bchristianity\b", r"\bislam\b", r"\bhindu(ism)?\b",

    # modern topics likely outside corpus
    r"\bai\b", r"\bartificial intelligence\b", r"\bsocial media\b", r"\bclimate change\b",
    r"\btwitter\b", r"\btiktok\b", r"\binstagram\b", r"\bchatgpt\b",
]

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were",
    "what","why","how","does","do","did","should","can","could","would","i","you","we",
    "my","your","our","me","it","this","that","these","those"
}

def has_banned_terms(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in BANNED_QUERY_PATTERNS)

def _keywords(text: str):
    toks = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in toks if t not in STOPWORDS]

def query_supported_by_context(query: str, passages: List[Dict[str, Any]]) -> bool:
    q_terms = set(_keywords(query))
    if not q_terms:
        return False

    ctx = " ".join(p.get("chunk","") for p in passages).lower()
    hits = sum(1 for t in q_terms if t in ctx)

    required = 1 if len(q_terms) <= 4 else 2
    return hits >= required

def should_refuse_query(query: str, passages: List[Dict[str, Any]] = None) -> bool:
    # Gate A: hard banned categories
    if has_banned_terms(query):
        return True

    # Gate B: enforce literal support from retrieved context
    if passages is not None and not query_supported_by_context(query, passages):
        return True
    if not in_allowed_domain(query):
         return True
    return False

# --- RAG Settings ---
TOP_K = 8               # was 5 (faster)
MIN_SCORE = 0.25          # was 0.25 (stricter; refuse more)
MIN_PASSAGES = 2
MAX_CONTEXT_CHARS = 3000  # was 8000 (faster)

def load_index():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return index, chunks, model

def retrieve(query: str, index, chunks, model):
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, TOP_K)

    results = []
    best = float(scores[0][0]) if len(scores[0]) else 0.0

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        if float(score) < MIN_SCORE:   # filter weak passages
            continue
        item = chunks[int(idx)]
        results.append({
            "score": float(score),
            "chunk": item["chunk"],
            "source": item["source"],
            "page": item["page"],
        })

    return results, best


def build_context(passages: List[Dict[str, Any]]) -> str:
    context_parts = []
    total = 0
    for p in passages:
        snippet = f"[{p['source']} p.{p['page']}] {p['chunk']}"
        if total + len(snippet) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(context_parts)

def refuse() -> str:
    return REFUSAL_TEXT
# ----- Local LLM via Ollama (simple) -----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

SYSTEM_RULES = """
You are a constrained philosophical agent representing Existentialism, and you are STRICTLY LIMITED to the provided context passages.
You MUST NOT use any external knowledge, biography, dates, history, psychology/therapy, clinical language, or modern frameworks.
If the context does not directly support the answer, you MUST refuse.

REFUSAL RULE:
If you cannot point to the provided context as evidence for your answer, respond ONLY with:
"I cannot answer this question within the constraints of this agent. The answer is not supported by the provided corpus."

STYLE:
Be concise, intellectually playful (dry existential humor is allowed), and Socratic.
You are not a therapist. No advice.

MANDATORY OUTPUT STRUCTURE (NON-NEGOTIABLE):

If you answer (i.e., do not refuse), your response MUST contain EXACTLY TWO sections:

[SECTION 1] Explanation
- Explain using ONLY the context passages.
- Do NOT import external definitions or examples.
- Include EXACTLY ONE short quote (≤ 25 words) taken directly from the context passages.
- If a source tag (filename/page) is present, include it.

[SECTION 2] Question:
- Write EXACTLY ONE reflective question.
- The question must end with a question mark (?).
- The question must make the user think.
- The question must relate to the concept explained in Section 1.
- Do NOT give advice, therapy, or prescriptions.

If this structure is violated, the response is INVALID and must be regenerated.
"""


def generate_answer(query: str, context: str) -> str:
    prompt = f"""{SYSTEM_RULES}

User question: {query}

Context passages (ONLY allowed source):
{context}

Now respond following the mandatory structure."""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()
