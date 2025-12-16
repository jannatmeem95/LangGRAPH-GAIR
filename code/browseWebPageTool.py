import json
import os
import re
import time
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import requests
import trafilatura
from bs4 import BeautifulSoup
from openai import OpenAI


# =========================
# CONFIG
# =========================

MODEL = "gpt-4o-mini"

MAX_INPUT_TOKENS = 1500
MAX_OUTPUT_TOKENS = 1500

FETCH_TIMEOUT = 25
USER_AGENT = "temporal-fact-extractor/1.0"
OPENAI_RETRIES = 2

OPENAI_CONCURRENCY = 12  # global cap across threads/tools

CACHE_PATH = Path("cache/parallel_url_cache_all.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_FLUSH_EVERY = 25


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
# XML TAG PARSING
# =========================

def extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None

def normalize_yes_no(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.strip().lower()
    return x if x in ("yes", "no") else None


# =========================
# GAIR PROMPTS
# =========================

QUICK_SUMMARY_PROMPT = """You are an AI assistant analyzing webpage content to determine if it's helpful for answering a user's question. Given:

1. User query: {user_query}
2. Search query: {search_query}
3. Webpage content: {first_page_fetch_res}

Evaluate if this webpage contains useful information for answering the user's question or search query.

Provide your analysis in this format:
<helpful>yes/no</helpful>
<summary>If helpful: Concise summary of relevant information that helps answer the query</summary>"""

EXTRACT_NEW_INFO_PROMPT = """You are a helpful AI research assistant. I will provide you:
* The user's main question. This is a complex question that requires a deep research to answer.
* A sub-question. The main question has been broken down into a set of sub-questions to help you focus on specific aspects of the main question, and this sub-question is the current focus.
* The context so far. This includes all the information that has been gathered from previous turns, including the sub-questions and the information gathered from other resources for them.
* One page of a webpage content as well as the page index.

Your task is to read the webpage content carefully and extract all *new* information (compared to the context so far) that could help answer either the main question or the sub-question.

Also decide whether we need to read more content from this webpage by paging down.

Your answer should follow the following format:
<extracted_info>...</extracted_info>
<page_down>yes/no</page_down>
<short_summary>...</short_summary>

<main_question>
{main_question}
</main_question>

<context_so_far>
{context_so_far}
</context_so_far>

<current_sub_question>
{sub_question}
<current_sub_question>

<webpage_content>
    <page_index>{page_index}</page_index>
    <total_page_number>{total_pages}</total_page_number>
    <current_page_content>{page_content}</current_page_content>
</webpage_content>
"""


# =========================
# OPENAI CALLS (bounded)
# =========================

def _call_openai_text(system: str, user: str, max_tokens: int) -> str:
    with _openai_sem:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    return (resp.choices[0].message.content or "").strip()


def call_quick_summary(user_query: str, search_query: str, page_chunk: str) -> Tuple[str, str]:
    prompt = QUICK_SUMMARY_PROMPT.format(
        user_query=user_query,
        search_query=search_query,
        first_page_fetch_res=page_chunk[:12000],
    )
    out = _call_openai_text("Follow format strictly.", prompt, max_tokens=350)
    helpful = normalize_yes_no(extract_tag(out, "helpful")) or "no"
    summary = extract_tag(out, "summary") or ""
    return helpful, summary


def call_extract_new_info(
    main_question: str,
    sub_question: str,
    context_so_far: str,
    page_content: str,
    page_index: int,
    total_pages: int,
) -> Tuple[str, str, str]:
    prompt = EXTRACT_NEW_INFO_PROMPT.format(
        main_question=main_question,
        sub_question=sub_question,
        context_so_far=context_so_far[-20000:],  # bound
        page_content=page_content[:20000],
        page_index=page_index,
        total_pages=total_pages,
    )
    out = _call_openai_text("Follow format strictly.", prompt, max_tokens=MAX_OUTPUT_TOKENS)
    extracted = extract_tag(out, "extracted_info") or ""
    page_down = normalize_yes_no(extract_tag(out, "page_down")) or "no"
    short_summary = extract_tag(out, "short_summary") or ""
    return extracted, page_down, short_summary


# =========================
# BROWSE TOOL (chunk loop)
# =========================

@dataclass
class BrowseResult:
    url: str
    used: bool                      # quick_summary helpful?
    total_pages: int
    last_page_index: int
    appended_context: str           # what we added to context
    short_summaries: List[str]


class BrowseWebpageTool:
    def __init__(self, cache: dict):
        self.cache = cache

    def _get_text_and_chunks(self, url: str) -> Tuple[str, List[str]]:
        cached = load_cached_url(self.cache, url)
        if cached:
            return cached["text"], cached["chunks"]

        html = fetch_html(url)
        text = extract_main_text(html, url)
        chunks = chunk_text(text, MAX_INPUT_TOKENS)

        with _cache_lock:
            save_cached_url(self.cache, url, text, chunks)
        maybe_persist_cache(self.cache)

        return text, chunks

    def browse(
        self,
        main_question: str,
        sub_question: str,
        context_so_far: str,
        url: str,
        search_query: str,
        start_page_index: int = 0,
        max_pages_per_url: int = 8,
    ) -> BrowseResult:
        _, chunks = self._get_text_and_chunks(url)
        total_pages = len(chunks)

        appended = []
        short_summaries: List[str] = []

        if total_pages == 0:
            return BrowseResult(url=url, used=False, total_pages=0, last_page_index=start_page_index,
                               appended_context="\n\n[Empty page] No text extracted.", short_summaries=[])

        # GAIR: quick helpfulness on first page we read for this URL
        first_idx = min(max(start_page_index, 0), total_pages - 1)
        helpful, summary = call_quick_summary(main_question, search_query, chunks[first_idx])
        short_summaries.append(f"[QuickSummary helpful={helpful}] {summary}".strip())

        if helpful == "no":
            appended.append(f"\n\n[Unhelpful page] url={url}\n{summary}".strip())
            return BrowseResult(url=url, used=False, total_pages=total_pages, last_page_index=first_idx,
                               appended_context="\n".join(appended), short_summaries=short_summaries)

        # sequential chunk processing (stateful)
        page_index = first_idx
        pages_read = 0
        cur_context = context_so_far

        while True:
            if pages_read >= max_pages_per_url:
                appended.append(f"\n\n[Stop] max_pages_per_url reached for url={url}")
                break
            if page_index >= total_pages:
                break

            extracted, page_down, ss = call_extract_new_info(
                main_question=main_question,
                sub_question=sub_question,
                context_so_far=cur_context,
                page_content=chunks[page_index],
                page_index=page_index,
                total_pages=total_pages,
            )

            pages_read += 1
            short_summaries.append(ss.strip() if ss.strip() else "No short summary.")

            if extracted.strip():
                block = f"\n\n[Extracted]\nurl={url}\npage_index={page_index}/{total_pages-1}\n{extracted.strip()}"
            else:
                block = f"\n\n[No new info]\nurl={url}\npage_index={page_index}/{total_pages-1}"
            appended.append(block)

            # Update context incrementally so the next chunk extraction can be "new vs context"
            cur_context = (cur_context + "\n" + block)[-50000:]  # keep bounded

            # last page guard
            if page_index >= total_pages - 1:
                break
            if page_down != "yes":
                break

            page_index += 1

        return BrowseResult(
            url=url,
            used=True,
            total_pages=total_pages,
            last_page_index=page_index,
            appended_context="\n".join(appended),
            short_summaries=short_summaries,
        )
