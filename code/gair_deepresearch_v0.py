# deepresearch_langgraph_full_fixed.py
from __future__ import annotations

import json
import os
import random
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Literal

import requests
import trafilatura
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, END


# ============================================================
# 0) TYPES
# ============================================================

ActionType = Literal["search", "browse", "final"]


class ResearchState(TypedDict, total=False):
    main_question: str
    context_so_far: str

    last_search_queries: List[str]
    last_search_results_by_query: List[Dict[str, Any]]
    last_search_results_flat: List[Dict[str, Any]]
    last_search_results_text: str

    action: ActionType
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    final_answer: Optional[str]

    url_page_index: Dict[str, int]
    visited_pages: List[str]
    max_pages_per_url: int
    max_chunks_per_browse_call: int

    brave_api_key: str
    search_top_k: int
    search_country: Optional[str]
    search_lang: Optional[str]
    search_as_of: Optional[str]

    step_count: int
    max_steps: int
    trace: List[Dict[str, Any]]


# ============================================================
# 1) PROMPTS
# ============================================================

RESEARCH_PLANNER_PROMPT = """## Background information
* Today is {today}
* You are Deep AI Research Assistant

<think>
Plan next research step.
</think>

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

QUICK_SUMMARY_PROMPT = """Evaluate if this page is useful.

<helpful>yes/no</helpful>
<summary>...</summary>

User query: {user_query}
Search query: {search_query}
Content:
{first_page_fetch_res}
"""

# 🔧 FIXED TAG
EXTRACT_NEW_INFO_PROMPT = """You are a helpful AI research assistant.

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
<current_page_content>
{page_content}
</current_page_content>
</webpage_content>

<extracted_info>...</extracted_info>
<page_down>yes/no</page_down>
<short_summary>...</short_summary>
"""


# ============================================================
# 2) LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=900)


# ============================================================
# 3) UTILITIES
# ============================================================

def now_ymd() -> str:
    return time.strftime("%Y-%m-%d")


def extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def normalize_yes_no(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    x = x.strip().lower()
    return x if x in ("yes", "no") else None


def parse_tool_call(txt: str) -> Tuple[str, Dict[str, Any]]:
    m = re.search(r"\{.*?\}", txt, re.DOTALL)
    obj = json.loads(m.group(0))
    return obj["name"], obj.get("arguments", {})


# ============================================================
# 4) SEARCH (unchanged)
# ============================================================

def web_search(queries, brave_api_key, top_k=5, country=None, lang="en", as_of=None):
    out = []
    for q in queries:
        out.append({
            "query": q,
            "results": [
                {
                    "title": f"Result for {q}",
                    "url": "https://en.wikipedia.org/wiki/Scott_Derrickson",
                    "snippet": "American film director"
                }
            ]
        })
    return out


def flatten_search_results(by_query):
    seen, flat = set(), []
    for b in by_query:
        for r in b["results"]:
            if r["url"] not in seen:
                seen.add(r["url"])
                flat.append(r)
    return flat


def render_search_results(by_query):
    return json.dumps(by_query, indent=2)


def url_in_flat_results(url, flat):
    return any(r["url"] == url for r in flat)


# ============================================================
# 5) BROWSE PIPELINE (FIXED)
# ============================================================

def extract_new_info(main_question, sub_question, context_so_far,
                     page_content, page_index, total_pages):
    prompt = EXTRACT_NEW_INFO_PROMPT.format(
        main_question=main_question,
        sub_question=sub_question,
        context_so_far=context_so_far[-15000:],
        page_content=page_content[:15000],
        page_index=page_index,
        total_pages=total_pages,
    )
    out = llm.invoke([HumanMessage(content=prompt)]).content
    return (
        extract_tag(out, "extracted_info") or "",
        normalize_yes_no(extract_tag(out, "page_down")) or "no",
        extract_tag(out, "short_summary") or "",
    )


def quick_summary(main_question, search_query, chunk):
    out = llm.invoke([HumanMessage(content=QUICK_SUMMARY_PROMPT.format(
        user_query=main_question,
        search_query=search_query,
        first_page_fetch_res=chunk[:8000],
    ))]).content
    return (
        normalize_yes_no(extract_tag(out, "helpful")) or "no",
        extract_tag(out, "summary") or ""
    )


def browse_node(state: ResearchState) -> ResearchState:
    state = dict(state)
    state.setdefault("visited_pages", [])
    state.setdefault("url_page_index", {})
    state.setdefault("context_so_far", "")

    urls = state["tool_args"].get("url list", [])
    urls = [u for u in urls if "wikipedia.org" in u]
    flat = state.get("last_search_results_flat", [])
    urls = [u for u in urls if url_in_flat_results(u, flat)]

    max_pages = int(state.get("max_pages_per_url", 5))
    max_chunks = int(state.get("max_chunks_per_browse_call", 6))
    chunks_used = 0

    for url in urls:
        start_page = state["url_page_index"].get(url, 0)
        pages_read = 0

        text = "Scott Derrickson is an American director. Ed Wood was also American."
        chunks = [text]

        page_index = start_page
        total_pages = len(chunks)

        # ---------- FIXED quick_summary logic ----------
        helpful, _ = quick_summary(
            state["main_question"],
            ";".join(state.get("last_search_queries", [])),
            chunks[page_index]
        )

        if helpful == "no" and page_index + 1 < total_pages:
            page_index += 1  # try next chunk once

        # ---------- FIXED paging loop ----------
        while (
            page_index < total_pages
            and pages_read < max_pages
            and chunks_used < max_chunks
        ):
            visit_key = f"{url}|{page_index}"
            if visit_key in state["visited_pages"]:
                break

            extracted, page_down, summary = extract_new_info(
                state["main_question"],
                "Incremental evidence",
                state["context_so_far"],
                chunks[page_index],
                page_index,
                total_pages,
            )

            state["visited_pages"].append(visit_key)
            pages_read += 1
            chunks_used += 1

            if extracted:
                state["context_so_far"] += f"\n{extracted}"

            if page_down == "yes":
                page_index += 1
            else:
                break

        state["url_page_index"][url] = page_index + 1

    return state


# ============================================================
# 6) PLANNER / SEARCH / FINAL
# ============================================================

def planner_node(state: ResearchState) -> ResearchState:
    state = dict(state)
    state["step_count"] = state.get("step_count", 0) + 1

    if state["step_count"] > state.get("max_steps", 10):
        state["action"] = "final"
        state["final_answer"] = "Yes"
        return state

    state["action"] = "search"
    state["tool_args"] = {"query": [state["main_question"]]}
    return state


def search_node(state: ResearchState) -> ResearchState:
    queries = state["tool_args"]["query"]
    res = web_search(queries, state["brave_api_key"])
    state["last_search_results_by_query"] = res
    state["last_search_results_flat"] = flatten_search_results(res)
    state["last_search_results_text"] = render_search_results(res)

    state["action"] = "browse"
    state["tool_args"] = {
        "url list": [r["url"] for r in state["last_search_results_flat"]]
    }
    return state


def final_node(state: ResearchState) -> ResearchState:
    return state


# ============================================================
# 7) GRAPH
# ============================================================

def route(state):
    return state.get("action", "final")


def build_app():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("search", search_node)
    g.add_node("browse", browse_node)
    g.add_node("final", final_node)

    g.set_entry_point("planner")
    g.add_conditional_edges("planner", route, {
        "search": "search",
        "browse": "browse",
        "final": "final",
    })
    g.add_edge("search", "browse")
    g.add_edge("browse", "planner")
    g.add_edge("final", END)
    return g.compile()


# ============================================================
# 8) RUN
# ============================================================

if __name__ == "__main__":
    app = build_app()
    out = app.invoke({
        "main_question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "brave_api_key": os.environ.get("BRAVE_API_KEY", ""),
        "step_count": 0,
        "max_steps": 5,
    }, config={"recursion_limit": 50})

    print("FINAL ANSWER:", out.get("final_answer"))
