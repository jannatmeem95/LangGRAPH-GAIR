import json
import re
import time
import random
import threading
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from string import Template
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import trafilatura
from bs4 import BeautifulSoup
from openai import OpenAI
from tqdm import tqdm


# =========================
# CONFIG
# =========================

MODEL = "gpt-4o-mini"

# Your current settings are expensive/slow; keep if you want,
# but recommended: MAX_INPUT_TOKENS=2500, MAX_OUTPUT_TOKENS=900, MAX_FACTS_PER_CHUNK=6
MAX_INPUT_TOKENS = 1500
MAX_OUTPUT_TOKENS = 1500
MAX_FACTS_PER_CHUNK = 8

FETCH_TIMEOUT = 25
USER_AGENT = "temporal-fact-extractor/1.0"
OPENAI_RETRIES = 2

# Parallelism
MAX_WORKERS_URL = 8           # URLs per query
OPENAI_CONCURRENCY = 12       # max concurrent OpenAI calls across ALL threads

# Cache (single JSON file)
CACHE_PATH = Path("cache/parallel_url_cache_all.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Output
OUT_JSONL = Path("output/url_to_facts_multihop.jsonl")
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

# Cache batching
CACHE_FLUSH_EVERY = 25  # flush cache after N new URLs


# =========================
# GLOBALS (thread-safe)
# =========================

client = OpenAI()

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

_cache_lock = threading.Lock()
_cache_new_writes = 0

_openai_sem = threading.BoundedSemaphore(OPENAI_CONCURRENCY)


# =========================
# TOKEN CHUNKING
# =========================

def get_encoder():
    try:
        import tiktoken
        return tiktoken.encoding_for_model(MODEL)
    except Exception:
        return None

ENCODER = get_encoder()

def chunk_text(text: str, max_tokens: int) -> List[str]:
    if not text.strip():
        return []

    if ENCODER is None:
        approx_chars = max_tokens * 3
        return [text[i:i + approx_chars] for i in range(0, len(text), approx_chars)]

    tokens = ENCODER.encode(text)
    return [ENCODER.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


# =========================
# FETCH + CLEAN
# =========================

def fetch_html(url: str) -> str:
    r = session.get(url, timeout=FETCH_TIMEOUT)
    r.raise_for_status()
    return r.text

def normalize_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def extract_main_text(html: str, url: Optional[str] = None) -> str:
    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if extracted and extracted.strip():
            return normalize_text(extracted)
    except Exception:
        pass

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return normalize_text(soup.get_text(separator="\n"))


# =========================
# SINGLE-FILE CACHE (JSON)
# =========================

def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _atomic_write(data: dict):
    tmp = CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(CACHE_PATH)

def load_cached_url(cache: dict, url: str):
    return cache.get(url)

def save_cached_url(cache: dict, url: str, text: str, chunks: list[str]):
    cache[url] = {"cached_at": time.time(), "text": text, "chunks": chunks}

def maybe_persist_cache(cache: dict):
    global _cache_new_writes
    with _cache_lock:
        _cache_new_writes += 1
        if _cache_new_writes >= CACHE_FLUSH_EVERY:
            _atomic_write(cache)
            _cache_new_writes = 0

def force_persist_cache(cache: dict):
    with _cache_lock:
        _atomic_write(cache)


# =========================
# JSON ROBUSTNESS
# =========================

def extract_first_json_object(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    start = s.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output")

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]

    raise ValueError("No complete JSON object found (truncated output?)")


# =========================
# PROMPT
# =========================

PROMPT = Template(
"""Extract ONLY $maxfacts time-sensitive (temporal) facts from the text.

Return JSON ONLY in this format:

{
  "facts": [
    {
      "subject": "...",
      "relation": "...",
      "object": "...",
      "start_date": "YYYY or YYYY-MM-DD or null",
      "end_date": "YYYY or YYYY-MM-DD or null",
      "confidence": 0.0,
      "justification": "short explanation",
      "evidence_hint": "short phrase from text (NO quotes, NO newlines)"
    }
  ]
}

Rules:
- Do NOT fabricate dates
- confidence ∈ [0,1]
- evidence_hint must appear in the text
- evidence_hint must NOT contain quotation marks
- Output JSON only. No explanation.

Text:
$chunk
"""
)


# =========================
# FACT POSTPROCESS
# =========================

def fact_key(f: Dict[str, Any]) -> tuple:
    return (
        (f.get("subject") or "").lower(),
        (f.get("relation") or "").lower(),
        (f.get("object") or "").lower(),
        f.get("start_date"),
        f.get("end_date"),
    )

def dedupe_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for f in facts:
        if not isinstance(f, dict):
            continue
        k = fact_key(f)
        try:
            c_new = float(f.get("confidence", 0.0) or 0.0)
        except Exception:
            c_new = 0.0

        if k not in best:
            best[k] = f
        else:
            try:
                c_old = float(best[k].get("confidence", 0.0) or 0.0)
            except Exception:
                c_old = 0.0
            if c_new > c_old:
                best[k] = f
    return list(best.values())

def locate_evidence(text: str, hint: str) -> Dict[str, Optional[int]]:
    if not hint:
        return {"start_char": None, "end_char": None}
    idx = text.lower().find(hint.lower())
    if idx == -1:
        return {"start_char": None, "end_char": None}
    return {"start_char": idx, "end_char": idx + len(hint)}


# =========================
# GPT CALL (parallel-safe)
# =========================

def _call_openai_json(prompt: str, max_tokens: int) -> dict:
    # Global semaphore to keep concurrency bounded
    with _openai_sem:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        first = extract_first_json_object(content)
        return json.loads(first)

def extract_facts_from_chunk(chunk: str) -> List[Dict[str, Any]]:
    """
    Retry-shrinks output if needed to avoid truncation.
    """
    maxfacts = MAX_FACTS_PER_CHUNK
    max_tokens = MAX_OUTPUT_TOKENS

    for attempt in range(OPENAI_RETRIES):
        try:
            prompt = PROMPT.substitute(maxfacts=maxfacts, chunk=chunk)
            obj = _call_openai_json(prompt, max_tokens=max_tokens)
            facts = obj.get("facts", [])
            return facts if isinstance(facts, list) else []
        except ValueError as e:
            # truncated JSON: shrink
            if "No complete JSON object found" in str(e) and maxfacts > 2:
                maxfacts = max(2, maxfacts // 2)
                max_tokens = max(400, int(max_tokens * 0.75))
                continue
            raise
        except Exception:
            if attempt == OPENAI_RETRIES - 1:
                raise
            time.sleep((2 ** attempt) + random.random())

    return []


# =========================
# URL PIPELINE (one URL)
# =========================

def process_url(cache: dict, url: str) -> Tuple[str, List[Dict[str, Any]]]:
    cached = load_cached_url(cache, url)

    if cached:
        text = cached["text"]
        chunks = cached["chunks"]
    else:
        html = fetch_html(url)
        text = extract_main_text(html, url)
        chunks = chunk_text(text, MAX_INPUT_TOKENS)
        with _cache_lock:
            save_cached_url(cache, url, text, chunks)
        maybe_persist_cache(cache)

    # Parallelize chunks via shared pool in outer layer? (We’ll do per-URL pool here safely.)
    all_facts: List[Dict[str, Any]] = []

    # Chunk-level parallel calls
    with ThreadPoolExecutor(max_workers=min(OPENAI_CONCURRENCY, 16)) as ex:
        futures = [ex.submit(extract_facts_from_chunk, ch) for ch in chunks]
        for fut in as_completed(futures):
            facts = fut.result()
            for f in facts:
                if isinstance(f, dict):
                    f["evidence"] = locate_evidence(text, f.get("evidence_hint", ""))
                    all_facts.append(f)

    return url, dedupe_facts(all_facts)



# =========================
# PROGRESSIVE OUTPUT (thread-safe)
# =========================
_out_lock = threading.Lock()

def append_jsonl_threadsafe(path: Path, record: dict, do_fsync: bool = False):
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _out_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            if do_fsync:
                os.fsync(f.fileno())


# =========================
# QUERY PIPELINE (parallel URLs)
# =========================
def extract_temporal_facts_parallel(cache: dict, urls: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}

    # URL-level parallelism
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_URL) as ex:
        fut_to_url = {ex.submit(process_url, cache, url): url for url in urls}
        for fut in as_completed(fut_to_url):
            url = fut_to_url[fut]
            try:
                u, facts = fut.result()
                out[u] = facts

                # write progress immediately (one line per URL)
                append_jsonl_threadsafe(
                    OUT_JSONL,
                    {u: facts},
                    do_fsync=False,  # set True if you want max safety
                )
                
            except Exception as e:
                print(f"[ERROR] {url}: {e}")
                out[url] = []
                # ✅ still write error line so you know it failed
                append_jsonl_threadsafe(
                    OUT_JSONL,
                    {u: facts},
                    do_fsync=False,
                )

    return out


# =========================
# MAIN
# =========================

def append_jsonl(path: Path, record: dict):
    # Proper JSONL: one JSON object per line, no indent
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    cache = load_cache()
    results = {}

    # Process queries sequentially (safe), but URLs inside each query are parallel
    with open('/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/cache/parallel_url_cache_multihop.json','r') as f:
        cached_urls = json.load(f)

    urls = [u for u in cached_urls if "wikipedia.org" in u]

    facts_by_url = extract_temporal_facts_parallel(cache, urls)

    # attach to results
    # results[query] = payload
    # results[query]["temporal_facts"] = facts_by_url

    # JSONL record per query (recommended)
    # append_jsonl(OUT_JSONL, {"temporal_facts": facts_by_url})

    # Final cache flush
    force_persist_cache(cache)

    # Final all-in-one JSON (optional)
    with open("output/url_to_facts_multihop.json", "w", encoding="utf-8") as f:
        json.dump(facts_by_url, f, ensure_ascii=False, indent=2)

    print("[DONE] wrote:", str(OUT_JSONL), "and output/url_to_facts_multihop.json")
