"""
deepresearch_langgraph_full.py

GAIR-style DeepResearcher in LangGraph (planner-only router)
- State schema matches your ResearchState exactly.
- Search uses Brave web_search() adapter (cached).
- Browse uses your fetch/trafilatura/BeautifulSoup + token chunking + disk cache.
- browse chunk loop is SEQUENTIAL (context-dependent) and uses EXTRACT_NEW_INFO_PROMPT.

Install:
  pip install langgraph langchain-openai requests trafilatura beautifulsoup4 tqdm

Env:
  export OPENAI_API_KEY=...
  export BRAVE_API_KEY=...

Run:
  python deepresearch_langgraph_full.py
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Literal

import requests
import trafilatura
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from tqdm import tqdm

# ============================================================
# 0) TYPES
# ============================================================

ActionType = Literal["search", "browse", "final"]


class ResearchState(TypedDict, total=False):
    main_question: str
    context_so_far: str

    # search
    last_search_queries: List[str]
    last_search_results_by_query: List[Dict[str, Any]]  # [{"query": q, "results": [..]}]
    last_search_results_flat: List[Dict[str, Any]]      # flattened list of results dicts
    last_search_results_text: str

    # planner decision
    action: ActionType
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    final_answer: Optional[str]

    # browsing
    url_page_index: Dict[str, int]
    visited_pages: List[str]
    max_pages_per_url: int
    max_chunks_per_browse_call: int

    # settings
    brave_api_key: str
    search_top_k: int
    search_country: Optional[str]
    search_lang: Optional[str]
    search_as_of: Optional[str]

    # bookkeeping
    step_count: int
    max_steps: int
    trace: List[Dict[str, Any]]


# ============================================================
# 1) PROMPTS (GAIR-aligned)
# ============================================================

RESEARCH_PLANNER_PROMPT = """## Background information
* Today is {today}
* You are Deep AI Research Assistant
The question I give you is a complex question that requires a *deep research* to answer.
I will provide you with two tools to help you answer the question:
* A web search tool to help you perform google search.
* A webpage browsing tool to help you get new page content.
You don’t have to answer the question now, but you should first think about the research
plan or what to search next.
Your output format should be one of the following two formats:
<think>
YOUR THINKING PROCESS
</think>
<answer>
YOUR ANSWER AFTER GETTING ENOUGH INFORMATION
</answer>
or
<think>
YOUR THINKING PROCESS
</think>
<tool call>
YOUR TOOL CALL WITH CORRECT FORMAT
</tool call>
You should always follow the above two formats strictly.
Only output the final answer (in words, numbers or phrase) inside the <answer></answer> tag,
without any explanations or extra information. If this is a yes-or-no question, you should only answer
yes or no.
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type":"function","function":{{"name":"web search","description":"Search the web for relevant information from google. You should use this tool if the historical page content is not enough to answer the question. Or last search result is not relevant to the question.","parameters":{{"type":"object","properties":{{"query":{{"type":"array","items":{{"type":"string","description":"The query to search, which helps answer the question"}},"description":"The queries to search"}}}},"required":["query"],"minItems":1,"uniqueItems":true}}}}}}
{{"type":"function","function":{{"name":"browse webpage","description":"Browse the webpage and return the content that not appeared in the conversation history. You should use this tool if the last action is search and the search result maybe relevant to the question.","parameters":{{"type":"object","properties":{{"url list":{{"type":"array","items":{{"type":"string","description":"The chosen url from the search result, do not use url that not appeared in the search result"}},"description":"The chosen urls from the search result."}}}},"required":["url list"]}}}}}}
</tools>
For each function call, return a json object with function name and arguments within
<tool call></tool call> XML tags:
<tool call>
{{"name": "<function-name>", "arguments": <args-json-object>}}
</tool call>

<main_question>
{main_question}
</main_question>

<context_so_far>
{context_so_far}
</context_so_far>

<last_search_results>
{last_search_results}
</last_search_results>
"""

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

In addition to the extracted information, decide whether we need to read more content from this webpage by paging down.

Your answer should follow the following format:
<extracted_info>...</extracted_info>
<page_down>yes/no</page_down>
<short_summary>...</short_summary>

Important note: Use the same language as the user's main question for the short summary.

<main_question>
{main_question}
</main_question>

<context_so_far>
{context_so_far}
</context_so_far>

<current_sub_question>
{sub_question}
</current_sub_question>

<webpage_content>
    <page_index>{page_index}</page_index>
    <total_page_number>{total_pages}</total_page_number>
    <current_page_content>{page_content}</current_page_content>
</webpage_content>
"""


# ============================================================
# 2) LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=900)


# ============================================================
# 3) PARSING UTILS
# ============================================================

def now_ymd() -> str:
    return time.strftime("%Y-%m-%d")

def extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None

def normalize_yes_no(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.strip().lower()
    return x if x in ("yes", "no") else None

def parse_tool_call(tool_call_text: str) -> Tuple[str, Dict[str, Any]]:
    # Expect JSON object in <tool call> like {"name": "...", "arguments": {...}}
    # Robustly extract first {...} if extra text sneaks in.
    m = re.search(r"\{.*\}", tool_call_text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in <tool call> block.")
    obj = json.loads(m.group(0))
    name = obj["name"]
    args = obj.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    return name, args


# ============================================================
# 4) BRAVE SEARCH (your code + tiny helpers)
# ============================================================

SEARCH_CACHE_DIR = Path("/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/temporal_cache_store/search")
SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def make_query_cache_key(
    query: str, engine: str, top_k: int, country: Optional[str], lang: Optional[str], as_of: Optional[str]
) -> str:
    import hashlib
    base = json.dumps(
        {"q": query, "engine": engine, "top_k": top_k, "country": country, "lang": lang, "as_of": as_of},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def load_search_cache(engine: str, key: str) -> Optional[List[Dict[str, Any]]]:
    p = SEARCH_CACHE_DIR / engine / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_search_cache(engine: str, key: str, results: List[Dict[str, Any]]) -> None:
    d = SEARCH_CACHE_DIR / engine
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{key}.json"
    p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

def brave_search(
    query: str,
    brave_api_key: str,
    top_k: int = 5,
    country: Optional[str] = None,
    lang: Optional[str] = None,
    depth: int = 0,
) -> List[Dict[str, Any]]:
    if not brave_api_key:
        raise ValueError("Missing BRAVE_API_KEY")
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": brave_api_key}
    params: Dict[str, Any] = {"q": query, "count": min(int(top_k or 5), 20)}
    if country:
        params["country"] = country
    if lang:
        params["lang"] = lang

    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        out: List[Dict[str, Any]] = []
        for e in (data.get("web", {}) or {}).get("results", []):
            out.append(
                {
                    "title": e.get("title", "") or "",
                    "link": e.get("url", "") or "",
                    "snippet": e.get("description", "") or "",
                    "age": e.get("age", None),
                }
            )
        return out[:top_k]
    except Exception as ex:
        if depth < 3:
            time.sleep(0.4 + 0.2 * depth + random.random() * 0.2)
            return brave_search(query, brave_api_key, top_k, country, lang, depth + 1)
        print(f"Brave search API error: {ex}")
        return []

def web_search(
    queries: list[str],
    brave_api_key: str,
    top_k: int = 5,
    country: str | None = None,
    lang: str | None = "en",
    as_of: str | None = "2025-11-11",
):
    engine = "brave"
    all_results = []
    for q in queries:
        key = make_query_cache_key(q, engine, top_k, country, lang, as_of)
        cached = load_search_cache(engine, key)
        if cached is not None:
            all_results.append({"query": q, "results": cached})
            continue

        hits = brave_search(q, brave_api_key, top_k, country, lang)
        results = [
            {"title": h.get("title", ""), "url": h.get("link", ""), "snippet": h.get("snippet", ""), "published": h.get("age", None)}
            for h in hits
            if h.get("link")
        ]
        save_search_cache(engine, key, results)
        all_results.append({"query": q, "results": results})
    return all_results

def flatten_search_results(by_query: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for block in by_query or []:
        for r in (block.get("results") or []):
            if r.get("url"):
                flat.append(r)
    seen = set()
    uniq = []
    for r in flat:
        u = r.get("url")
        if u in seen:
            continue
        seen.add(u)
        uniq.append(r)
    return uniq

def render_search_results(by_query: List[Dict[str, Any]], max_items_per_query: int = 5) -> str:
    if not by_query:
        return "None"
    chunks = []
    for block in by_query:
        q = block.get("query", "")
        rs = (block.get("results") or [])[:max_items_per_query]
        lines = [f"Query: {q}"]
        if not rs:
            lines.append("  (no results)")
        for i, r in enumerate(rs, 1):
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            published = r.get("published", None)
            pub_s = f" | published={published}" if published else ""
            lines.append(f"  {i}. {title}\n     {url}{pub_s}\n     {snippet}".rstrip())
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)

def url_in_flat_results(url: str, flat_results: List[Dict[str, Any]]) -> bool:
    return any((r.get("url") == url) for r in (flat_results or []))


# ============================================================
# 5) BROWSE PIPELINE (your caching + trafilatura + chunking)
#    - NOTE: sequential chunk loop (stateful); no parallel chunks here.
# ============================================================

MODEL_FOR_ENCODER = "gpt-4o-mini"
MAX_INPUT_TOKENS = 1500
MAX_OUTPUT_TOKENS = 1500

FETCH_TIMEOUT = 25
USER_AGENT = "temporal-fact-extractor/1.0"

URL_CACHE_PATH = Path("/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/cache/parallel_url_cache_all.json")
URL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_FLUSH_EVERY = 25

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

_cache_lock = threading.Lock()
_cache_new_writes = 0

def get_encoder():
    try:
        import tiktoken
        return tiktoken.encoding_for_model(MODEL_FOR_ENCODER)
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

def load_url_cache() -> dict:
    if not URL_CACHE_PATH.exists():
        return {}
    with open(URL_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _atomic_write_url_cache(data: dict):
    tmp = URL_CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(URL_CACHE_PATH)

def load_cached_url(cache: dict, url: str):
    return cache.get(url)

def save_cached_url(cache: dict, url: str, text: str, chunks: list[str]):
    cache[url] = {"cached_at": time.time(), "text": text, "chunks": chunks}

def maybe_persist_url_cache(cache: dict):
    global _cache_new_writes
    with _cache_lock:
        _cache_new_writes += 1
        if _cache_new_writes >= CACHE_FLUSH_EVERY:
            _atomic_write_url_cache(cache)
            _cache_new_writes = 0

def force_persist_url_cache(cache: dict):
    with _cache_lock:
        _atomic_write_url_cache(cache)

def call_llm_text(prompt: str, max_tokens: int) -> str:
    resp = llm.invoke([HumanMessage(content=prompt)])
    # LangChain already enforces model + max_tokens at construction
    return (resp.content or "").strip()

def quick_summary(main_question: str, search_query: str, first_chunk: str) -> Tuple[str, str]:
    prompt = QUICK_SUMMARY_PROMPT.format(
        user_query=main_question,
        search_query=search_query,
        first_page_fetch_res=first_chunk[:12000],
    )
    out = llm.invoke([HumanMessage(content=prompt)]).content
    helpful = normalize_yes_no(extract_tag(out, "helpful")) or "no"
    summary = extract_tag(out, "summary") or ""
    return helpful, summary

def extract_new_info(
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
        context_so_far=context_so_far[-20000:],
        page_content=page_content[:20000],
        page_index=page_index,
        total_pages=total_pages,
    )
    out = llm.invoke([HumanMessage(content=prompt)]).content
    extracted = extract_tag(out, "extracted_info") or ""
    page_down = normalize_yes_no(extract_tag(out, "page_down")) or "no"
    short_summary = extract_tag(out, "short_summary") or ""
    return extracted, page_down, short_summary

TRACE_DIR = Path("/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/output_multihop/traces")
TRACE_DIR.mkdir(parents=True, exist_ok=True)

def persist_trace(state: ResearchState, run_id: Optional[str] = None):
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    path = TRACE_DIR / f"trace_run.json"
    # with open(path, "w", encoding="utf-8") as f:
    #     json.dump(
    #         {
    #             "main_question": state.get("main_question"),
    #             "final_answer": state.get("final_answer"),
    #             "trace": state.get("trace", []),
    #         },
    #         f,
    #         ensure_ascii=False,
    #         indent=2,
    #     )
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "main_question": state.get("main_question"),
        "final_answer": state.get("final_answer"),
        "trace": state.get("trace", []),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({state.get("main_question"):record}, ensure_ascii=False) + "\n")

    return record, path


TRACE_JSONL = Path("/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/output_multihop/trace_steps.jsonl")
TRACE_JSONL.parent.mkdir(parents=True, exist_ok=True)

_trace_lock = threading.Lock()

def append_trace_step(step: dict):
    with _trace_lock:
        with open(TRACE_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")


# ============================================================
# 6) NODES
# ============================================================

def planner_node(state: ResearchState) -> ResearchState:
    state = dict(state)
    state.setdefault("trace", [])
    state.setdefault("context_so_far", "None")
    state["step_count"] = state.get("step_count", 0) + 1

    ctx = state["context_so_far"]
    if state["step_count"] >= state.get("max_steps", 15):
        forced_ctx = ctx + "\n\n[System note] Step budget reached. You MUST output ONLY a final <answer> now. Do NOT call tools."
        prompt = RESEARCH_PLANNER_PROMPT.format(
            today=now_ymd(),
            main_question=state["main_question"],
            context_so_far=forced_ctx,
            last_search_results=state.get("last_search_results_text", "None"),
        )
        out = llm.invoke([HumanMessage(content=prompt)]).content
        answer = extract_tag(out, "answer")

        # If the model still refuses to comply, fall back to a safe best-effort
        if not answer or not answer.strip():
            answer = "Unknown"

        state["action"] = "final"
        state["final_answer"] = answer.strip()
        state["tool_name"] = None
        state["tool_args"] = {}
        state["trace"].append({"node": "planner", "action": "forced_final_max_steps", "raw": out})

        append_trace_step({
            "timestamp": time.time(),
            "node": "planner",
            "step": state["step_count"],
            "event": state["trace"][-1],
            "question": state["main_question"],
        })
        return state
    
    prompt = RESEARCH_PLANNER_PROMPT.format(
        today=now_ymd(),
        main_question=state["main_question"],
        context_so_far=ctx,
        last_search_results=state.get("last_search_results_text", "None"),
    )
    out = llm.invoke([HumanMessage(content=prompt)]).content

    answer = extract_tag(out, "answer")
    tool_call = extract_tag(out, "tool call")

    if answer is not None:
        state["action"] = "final"
        state["final_answer"] = answer.strip()
        state["tool_name"] = None
        state["tool_args"] = {}
        state["trace"].append({"node": "planner", "action": "final", "raw": out})
        append_trace_step({
            "timestamp": time.time(),
            "node": "planner",
            "step": state["step_count"],
            "event": state["trace"][-1],
            "question": state["main_question"],
        })

        return state

    if tool_call is None:
        # fallback
        state["action"] = "search"
        state["tool_name"] = "web search"
        state["tool_args"] = {"query": [state["main_question"]]}
        state["trace"].append({"node": "planner", "action": "fallback_search", "raw": out})
        append_trace_step({
            "timestamp": time.time(),
            "node": "planner",
            "step": state["step_count"],
            "event": state["trace"][-1],
            "question": state["main_question"],
        })

        return state

    try:
        name, args = parse_tool_call(tool_call)
    except Exception:
        name, args = "web search", {"query": [state["main_question"]]}

    if name == "web search":
        state["action"] = "search"
    elif name == "browse webpage":
        state["action"] = "browse"
    else:
        state["action"] = "search"
        name, args = "web search", {"query": [state["main_question"]]}

    state["tool_name"] = name
    state["tool_args"] = args if isinstance(args, dict) else {}
    state["trace"].append({"node": "planner", "action": state["action"], "tool_name": name, "tool_args": state["tool_args"], "raw": out})
    append_trace_step({
            "timestamp": time.time(),
            "node": "planner",
            "step": state["step_count"],
            "event": state["trace"][-1],
            "question": state["main_question"],
        })

    return state


def search_node(state: ResearchState) -> ResearchState:
    state = dict(state)
    state.setdefault("trace", [])

    queries = (state.get("tool_args") or {}).get("query") or [state["main_question"]]
    if not isinstance(queries, list) or not queries:
        queries = [state["main_question"]]
    queries = [str(q).strip() for q in queries if str(q).strip()]
    if not queries:
        queries = [state["main_question"]]

    brave_api_key = state.get("brave_api_key") or os.environ.get("BRAVE_API_KEY", "")
    if not brave_api_key:
        raise ValueError("Missing BRAVE_API_KEY")

    by_query = web_search(
        queries=queries,
        brave_api_key=brave_api_key,
        top_k=int(state.get("search_top_k", 5)),
        country=state.get("search_country", None),
        lang=state.get("search_lang", "en"),
        as_of=state.get("search_as_of", "2025-11-11"),
    )

    flat = flatten_search_results(by_query)
    state["last_search_queries"] = queries
    state["last_search_results_by_query"] = by_query
    state["last_search_results_flat"] = flat
    state["last_search_results_text"] = render_search_results(by_query)

    state["trace"].append({"node": "search", "queries": queries, "num_results": len(flat)})
    append_trace_step({
            "timestamp": time.time(),
            "node": "search",
            "step": state["step_count"],
            "event": state["trace"][-1],
            "question": state["main_question"],
        })

    return state


def browse_node(state: ResearchState) -> ResearchState:
    """
    Uses your fetch+extract+cache+chunk_text pipeline,
    and sequentially applies QUICK_SUMMARY then EXTRACT_NEW_INFO_PROMPT with page_down control.
    """
    state = dict(state)
    state.setdefault("trace", [])
    state.setdefault("context_so_far", "None")
    state.setdefault("url_page_index", {})
    state.setdefault("visited_pages", [])

    url_list = (state.get("tool_args") or {}).get("url list") or []
    # url_list = [url for url in url_list if "wikipedia.org" in url]
    if not isinstance(url_list, list):
        url_list = []
    url_list = [str(u).strip() for u in url_list if str(u).strip()]
    if not url_list:
        state["trace"].append({"node": "browse", "error": "empty_url_list"})
        append_trace_step({
            "timestamp": time.time(),
            "node": "browse",
            "step": state["step_count"],
            "event": state["trace"][-1],
            "question": state["main_question"],
        })

        return state

    # enforce GAIR constraint: must be from last search results
    flat = state.get("last_search_results_flat", [])
    url_list = [u for u in url_list if url_in_flat_results(u, flat)]
    if not url_list:
        state["context_so_far"] += "\n\n[Browse skipped] URL not in last search results."
        state["trace"].append({"node": "browse", "error": "urls_not_in_last_results"})
        append_trace_step({
            "timestamp": time.time(),
            "node": "browse",
            "step": state["step_count"],
            "event": state["trace"][-1],
            "question": state["main_question"],
        })

        return state

    # load url cache once per call
    cache = load_url_cache()

    max_pages_per_url = int(state.get("max_pages_per_url", 8))
    max_chunks_total = int(state.get("max_chunks_per_browse_call", 8))
    chunks_used = 0

    search_query_str = "; ".join((state.get("last_search_queries") or [])[:3])
    sub_question = "Extract incremental evidence to answer the main question."

    for url in url_list:
        start_page = int(state["url_page_index"].get(url, 0))

        # get cached text+chunks
        cached = load_cached_url(cache, url)
        if cached and isinstance(cached.get("chunks"), list) and cached.get("text"):
            text = cached["text"]
            chunks = cached["chunks"]
        else:
            try:
                html = fetch_html(url)
                text = extract_main_text(html, url)
                chunks = chunk_text(text, MAX_INPUT_TOKENS)
                with _cache_lock:
                    save_cached_url(cache, url, text, chunks)
                maybe_persist_url_cache(cache)
            except Exception as e:
                state["context_so_far"] += f"\n\n[Browse error] url={url}\n{e}"
                state["trace"].append({"node": "browse", "url": url, "error": str(e)})
                append_trace_step({
                    "timestamp": time.time(),
                    "node": "browse",
                    "step": state["step_count"],
                    "event": state["trace"][-1],
                    "question": state["main_question"],
                })
                
                continue

        total_pages = len(chunks)
        if total_pages == 0:
            state["context_so_far"] += f"\n\n[Empty page] url={url}"
            state["trace"].append({"node": "browse", "url": url, "empty": True})
            append_trace_step({
                    "timestamp": time.time(),
                    "node": "browse",
                    "step": state["step_count"],
                    "event": state["trace"][-1],
                    "question": state["main_question"],
                })
            continue

        page_index = min(max(start_page, 0), total_pages - 1)
        pages_read_for_url = 0

        # quick summary gate (on first read for this URL in this call)
        first_visit_key = f"{url}|{page_index}"
        if first_visit_key not in state["visited_pages"]:
            helpful, summary = quick_summary(state["main_question"], search_query_str, chunks[page_index])
            state["trace"].append({"node": "quick_summary", "url": url, "helpful": helpful, "summary": summary[:300]})
            append_trace_step({
                    "timestamp": time.time(),
                    "node": "browse",
                    "step": state["step_count"],
                    "event": state["trace"][-1],
                    "question": state["main_question"],
                })
            if helpful == "no":
                state["context_so_far"] += f"\n\n[Unhelpful page] url={url}\n{summary}"
                # try next chunk ONCE instead of poisoning the URL
                if page_index + 1 < total_pages:
                    page_index += 1
                else:
                    state["visited_pages"].append(first_visit_key)
                    continue
                # state["context_so_far"] += f"\n\n[Unhelpful page] url={url}\n{summary}"
                # # mark as visited so we don't retry same first page forever
                # state["visited_pages"].append(first_visit_key)
                # continue

        # sequential paging loop
        
        while True:
            if chunks_used >= max_chunks_total:
                state["context_so_far"] += f"\n\n[Stop] max_chunks_per_browse_call reached ({max_chunks_total})."
                state["trace"].append({"node": "browse", "url": url, "stop": "max_chunks_per_browse_call"})
                append_trace_step({
                    "timestamp": time.time(),
                    "node": "browse",
                    "step": state["step_count"],
                    "event": state["trace"][-1],
                    "question": state["main_question"],
                })
                break

            visit_key = f"{url}|{page_index}"
            if visit_key in state["visited_pages"]:
                state["trace"].append({"node": "browse", "url": url, "page_index": page_index, "skip": "already_visited"})
                append_trace_step({
                    "timestamp": time.time(),
                    "node": "browse",
                    "step": state["step_count"],
                    "event": state["trace"][-1],
                    "question": state["main_question"],
                })
                break
            state["visited_pages"].append(visit_key)

            extracted, page_down, short_summary = extract_new_info(
                main_question=state["main_question"],
                sub_question=sub_question,
                context_so_far=state["context_so_far"],
                page_content=chunks[page_index],
                page_index=page_index,
                total_pages=total_pages,
            )
            chunks_used += 1
            pages_read_for_url += 1

            if extracted.strip():
                state["context_so_far"] += f"\n\n[Extracted]\nurl={url}\npage_index={page_index}/{total_pages-1}\n{extracted.strip()}"
            else:
                state["context_so_far"] += f"\n\n[No new info]\nurl={url}\npage_index={page_index}/{total_pages-1}"

            if short_summary.strip():
                state["context_so_far"] += f"\n[Progress] {short_summary.strip()}"

            state["trace"].append({
                "node": "extract",
                "url": url,
                "page_index": page_index,
                "total_pages": total_pages,
                "page_down": page_down,
                "short_summary": short_summary[:300],
            })
            append_trace_step({
                "timestamp": time.time(),
                "node": "browse",
                "step": state["step_count"],
                "event": state["trace"][-1],
                "question": state["main_question"],
            })


            # last page guard
            if page_index >= total_pages - 1:
                page_down = "no"

            # if page_down == "yes" and (page_index + 1) < total_pages and (page_index + 1) < max_pages_per_url:
            #     page_index += 1
            #     state["url_page_index"][url] = page_index
            #     continue
            if (
                page_down == "yes"
                and (page_index + 1) < total_pages
                and pages_read_for_url < max_pages_per_url
            ):
                page_index += 1
                state["url_page_index"][url] = page_index
                continue

            # next browse starts after last processed page
            state["url_page_index"][url] = page_index + 1
            break

    force_persist_url_cache(cache)
    return state


def final_node(state: ResearchState) -> ResearchState:
    return state


# ============================================================
# 7) ROUTING + GRAPH
# ============================================================

def route_from_planner(state: ResearchState) -> str:
    a = state.get("action")
    if a == "search":
        return "search"
    if a == "browse":
        return "browse"
    return "final"

def build_app():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("search", search_node)
    g.add_node("browse", browse_node)
    g.add_node("final", final_node)

    g.set_entry_point("planner")
    g.add_conditional_edges("planner", route_from_planner, {
        "search": "search",
        "browse": "browse",
        "final": "final",
    })
    g.add_edge("search", "planner")
    g.add_edge("browse", "planner")
    g.add_edge("final", END)
    return g.compile()


# ============================================================
# 8) EXAMPLE RUN
# ============================================================

if __name__ == "__main__":
    app = build_app()

    with open("/rhome/jmeem001/PAT-data/November2025/PAT-multihop_with_date.json", 'r') as f:
        data = json.load(f)

    # lst = list(data.keys())[0]

    # data = { lst: data[lst]}

    traces = []
    answers = {}
    for q in tqdm(data):
        init: ResearchState = {
            "main_question": q,
            "context_so_far": "None",

            "last_search_queries": [],
            "last_search_results_by_query": [],
            "last_search_results_flat": [],
            "last_search_results_text": "None",

            "action": "search",
            "tool_name": None,
            "tool_args": {},
            "final_answer": None,

            "url_page_index": {},
            "visited_pages": [],
            "max_pages_per_url": 8,
            "max_chunks_per_browse_call": 6,

            "brave_api_key": os.environ.get("BRAVE_API_KEY", ""),
            "search_top_k": 5,
            "search_country": "us",
            "search_lang": "en",
            "search_as_of": "2025-11-11",

            "step_count": 0,
            "max_steps": 10,
            "trace": [],
        }

        out = app.invoke(init, config={"recursion_limit": 100})
        trace_out, trace_path = persist_trace(out)
        traces.append(trace_out)
        print("[TRACE SAVED]", trace_path)
        
        print("\n===== FINAL ANSWER =====")
        # print(out.get("final_answer"))
        answers[q] = out.get("final_answer")
        
        with open('/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/output_multihop/final_answers.jsonl','a') as f:
            f.write(json.dumps({q:answers[q]}, ensure_ascii=False) + "\n")
        # if len(answers) == 50:
        #     break

    with open('/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/output_multihop/trace_out.json','w') as f:
        json.dump(traces, f, indent = 4)
    with open('/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/output_multihop/final_answers.json','w') as f:
        json.dump(answers,f,indent =4)
         
    
    # print(json.dumps(out.get("trace", []), indent=2))
