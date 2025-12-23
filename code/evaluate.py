#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a QA JSON/JSONL file using:
  - Reference-based Exact Match (EM) with robust normalization
  - Model-as-a-Judge (Qwen via vLLM OpenAI-compatible server)

Example:
  python eval_with_qwen_judge.py data.jsonl \
      --base http://localhost:8000/v1 --model Qwen --api-key abc

Input format (per item): flexible keys supported
{
  "qid": "379@2024-04-15",
  "question": "...",             # or "q"
  "answers": ["Al-Nassr", ...],  # or "gold"
  "agent_answer": "Al Nassr",    # or "prediction"
  "t": "2024-04-15"              # optional present-anchored time
}

Outputs:
  - Prints EM and Judge tallies.
  - Optionally dumps per-item judge results via --dump path.jsonl
"""

import os
import re
import sys
import json
import argparse
import string
import unicodedata
import requests
from json import JSONDecoder

try:
    from unidecode import unidecode
except ImportError:
    # Minimal fallback if unidecode isn't available
    def unidecode(s: str) -> str:
        return s

# -------------------------
# Normalization utilities
# -------------------------

def normalize_answer2(s: str) -> str:
    # 1) Unicode normalize & strip accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # 2) lowercase
    s = s.lower()

    # 3) join dotted initials like "F.C.", "S.C.", "A.C." -> "fc", "sc", "ac"
    s = re.sub(
        r"\b(?:[a-z]\.){1,}[a-z]\.?\b",
        lambda m: m.group(0).replace(".", ""),
        s,
    )

    # 4) replace any non-alphanumeric with spaces
    s = re.sub(r"[^0-9a-z]+", " ", s)

    return s

def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

def white_space_fix(text: str) -> str:
    return " ".join(text.split())

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation and extra whitespace, after robust base normalization."""
    return white_space_fix(remove_punc(normalize_answer2(s)))

# -------------------------
# Flexible JSON reader
# -------------------------

def iter_json_objects(path: str):
    # Try whole-file JSON (array or single object)
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        val = json.loads(data)
        if isinstance(val, list):
            for o in val:
                yield o
        elif isinstance(val, dict):
            yield val
        return
    except json.JSONDecodeError:
        pass

    # Try JSONL fallback
    lines = data.splitlines()
    jsonl_ok = True
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        try:
            yield json.loads(s)
        except json.JSONDecodeError:
            jsonl_ok = False
            break
    if jsonl_ok:
        return

    # Concatenated objects fallback
    dec = JSONDecoder()
    i, n = 0, len(data)
    while i < n:
        while i < n and data[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(data, i)
        yield obj
        i = j

# -------------------------
# EM scoring
# -------------------------

def exact_match(agent_ans: str, gold_list) -> bool:
    agent_n = normalize_answer(unidecode(agent_ans or ""))
    for g in gold_list or []:
        if not isinstance(g, str):
            continue
        if normalize_answer(unidecode(g)) in agent_n:
            return True
        if agent_n != "" and agent_n in normalize_answer(unidecode(g)):
            return True
    return False

# -------------------------
# Qwen Judge
# -------------------------


RUBRIC_INSTRUCTIONS = (
    "You are a strict grader for open-domain QA.\n\n"
    "You will receive:\n"
    "- question q\n"
    "- answer candidate a_model (produced by a system)\n"
    "- list of acceptable gold answers gold[] (strings)\n"
    "- OPTIONAL: time_anchor t (ISO date) meaning answers must be correct as of t\n\n"
    "Grading rules:\n"
    "1) If a_model semantically states any gold[] with the same meaning (allowing paraphrases, aliases, "
    "accent/diacritic differences, initials expansions, hyphen spacing), mark \"CORRECT\".\n"
    "2) If a_model is close but incomplete or hedged (e.g., multiple guesses with one correct, or missing crucial qualifier/time), mark \"PARTIAL\".\n"
    "3) If a_model is wrong, contradictory, or unverifiable, mark \"INCORRECT\".\n"
    "4) If t is provided, prefer answers that are true as of t; if a_model asserts a state after t, mark \"INCORRECT\".\n"
    "5) Ignore formatting, punctuation, and casing.\n\n"
    "Return STRICT JSON ONLY in the following schema:\n"
    "{\"verdict\":\"CORRECT|PARTIAL|INCORRECT\",\"rationale\":\"<one short sentence>\", \"matched_gold\": \"<string or empty>\"}\n"
    "Do not add anything else."
)

def judge_messages(q: str, a_model: str, gold_list, t=None):
    PROMPT = '''You will be given a question and its ground truth answer list where each item can be a ground truth answer. Provided a pred_answer, you need to judge if the pred_answer correctly answers the question based on the ground truth answer list.
    You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).

    Here is the criteria for the judgement:
    1. The pred_answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.
    2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, and the pred_answer should be semantically same to at least one of them.

    question: {q}
    ground truth answers: {gold_list}
    pred_answer: {a_model}

    Follow the rubric exactly and return strict JSON only.

    The output should in the following json format:
    {\"verdict\":\"CORRECT|PARTIAL|INCORRECT\",\"rationale\":\"<your rationale for the judgement, as a text>\", \"matched_gold\": \"<string or empty>\"}\n"

    Your output:
    '''
    prompt = PROMPT.replace("{q}",q).replace("{gold_list}",str(gold_list)).replace("{a_model}",a_model)
    user = {
        "rubric": RUBRIC_INSTRUCTIONS,
        "q": q or "",
        "a_model": a_model or "",
        "gold": gold_list or [],
        # "t": t or ""
    }
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]

def call_qwen_judge(
    base_url: str,
    api_key: str,
    model_name: str,
    q: str,
    a_model: str,
    gold_list,
    t=None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    attempts: int = 2,
    timeout: int = 60
):
    payload = {
        "model": model_name,
        "messages": judge_messages(q, a_model, gold_list, t),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    last_err = None

    for _ in range(attempts):
        try:
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )
        except Exception as e:
            last_err = e
            print("Request error talking to vLLM:", repr(e), file=sys.stderr)
            continue

        if resp.status_code != 200:
            # <<< ADD THIS DEBUG PRINT >>>
            print(
                "vLLM HTTP error:",
                resp.status_code,
                resp.text[:2000],  # print first 2k chars
                file=sys.stderr,
            )
            last_err = RuntimeError(f"HTTP {resp.status_code}")
            continue

        # from here on we know status_code == 200
        text = resp.json()["choices"][0]["message"]["content"].strip()

        try:
            verdict = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                continue
            try:
                verdict = json.loads(m.group(0))
            except Exception:
                continue

        if isinstance(verdict, dict) and "verdict" in verdict:
            v = str(verdict.get("verdict", "")).upper()
            if v in {"CORRECT", "PARTIAL", "INCORRECT"}:
                return {
                    "verdict": v,
                    "rationale": verdict.get("rationale", ""),
                    "matched_gold": verdict.get("matched_gold", "")
                }

    # Fallback
    print("Judge JSON invalid after retries; last_err:", repr(last_err), file=sys.stderr)
    return {
        "verdict": "INCORRECT",
        "rationale": "Judge JSON invalid after retries",
        "matched_gold": ""
    }

# Evaluation loop
# -------------------------

def evaluate_with_judge(
    path: str,
    base_url: str,
    model_name: str,
    api_key: str,
    time_key: str = "t",
    temperature: float = 0.0,
    max_tokens: int = 256,
    attempts: int = 2,
    dump_path: str = None
):
    total = 0
    em_correct = 0
    em_correct_trace = 0
    judge_counts = {"CORRECT": 0, "PARTIAL": 0, "INCORRECT": 0}

    # Optional dump
    dump_f = open(dump_path, "w", encoding="utf-8") if dump_path else None

    with open('/rhome/jmeem001/PAT-data/November2025/PAT-singlehop_with_date.json', 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    gold_data ={unidecode(k): v for k, v in gold_data.items()}

    xx = 0
    tp = 0
   
    with open('/rhome/jmeem001/langgraph-scratch/dec16-LangGRAPH-GAIR/LangGRAPH-GAIR/output_temporal_verify_multihop/final_answers.json','r') as f:
        d = json.load(f)
    # for obj in iter_json_objects(path):
    for q,v in d.items():
        
        # q = obj.get("question") or obj.get("q") or ""
        q = unidecode(q)
        # if q in base:
        # ss = obj.copy()
        # ss['gold answers'] = gold_data[q]['text answers']
        # limpsum.append(ss)
            # continue
        total += 1
        # # trace = obj.get("trace")[-2:]
        # tr = ""
        # for t in trace:
        #     if "tool_result_preview" in t:
        #         tr += t["think"] + "\n" + t["tool_result_preview"] + "\n"
        #     else:
        #         tr+= t["think"] + "\n"
        # trace = '.'.join([t["think"] + "\n"+ t["tool_result_preview"] for t in trace])
        # trace = tr.strip()
        # a_model = obj.get("final_answer") or obj.get("prediction") or ""
        gold = gold_data[q]["text answers"] 
        # t = obj.get(time_key)
        t = "2025-11-11"

        # EM
        if exact_match(v, gold):# or exact_match(trace, gold):
            em_correct += 1


    if dump_f:
        dump_f.close()

    print(f"Items: {total}")
    if total > 0:
        em = em_correct / total
        corr = judge_counts["CORRECT"]
        part = judge_counts["PARTIAL"]
        inc = judge_counts["INCORRECT"]
        judge_acc = (corr + 0.5 * part) / total
        print(f"EM: {em_correct}/{total} = {em:.4f}")
        
        print(f"EM Correct Trace: {em_correct_trace}")

    else:
        print("0 items. EM: 0.0000 | Judge: CORRECT=0 PARTIAL=0 INCORRECT=0 | JudgeAcc: 0.0000")

# -------------------------
# CLI
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate with EM + Qwen Judge")
    ap.add_argument("path", help="Path to JSON/JSONL file")
    ap.add_argument("--base", default=os.environ.get("OPENAI_BASE", "http://localhost:8000/v1"),
                    help="OpenAI-compatible base URL (default: http://localhost:8000/v1)")
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen"),
                    help="Served model name (default: Qwen)")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "abc"),
                    help="API key string (default: 'abc')")
    ap.add_argument("--time-key", default="t",
                    help="Field name for present-anchored time (default: t)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Judge temperature (default: 0.0)")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="Judge max tokens (default: 256)")
    ap.add_argument("--attempts", type=int, default=2,
                    help="Judge JSON parse attempts (default: 2)")
    ap.add_argument("--dump", default=None,
                    help="Optional path to dump per-item results as JSONL")
    return ap.parse_args()

def main():
    args = parse_args()
    evaluate_with_judge(
        path=args.path,
        base_url=args.base,
        model_name=args.model,
        api_key=args.api_key,
        time_key=args.time_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        attempts=args.attempts,
        dump_path=args.dump
    )

if __name__ == "__main__":
    main()


# python my_llm_as_judge.py "/home/csgrad/jmeem001/gair-deepresearcher/DeepResearcher/nov19/rollout/rollout_log_3_pat_145.json" --base http://localhost:8000/v1 --model deepresearcher-7b --api-key abc


"""

Items: 145
EM: 54/145 = 0.3724
Judge: CORRECT=56 PARTIAL=5 INCORRECT=84
Judge Accuracy (PARTIAL=0.5): 0.4034


Items: 145
With Trace EM: 69/145 = 0.4759
Judge: CORRECT=60 PARTIAL=11 INCORRECT=74
Judge Accuracy (PARTIAL=0.5): 0.4517
EM Correct Trace: 16

#python my_llm_as_judge.py "/home/csgrad/jmeem001/gair-deepresearcher/DeepResearcher/nov19/rollout/rollout_log_5_tqr_pat_145.json" --base http://localhost:8000/v1 --model deepresearcher-7b --api-key abc
# """
# Items: 145
# EM: 91/145 = 0.6276
# Number of questions: 145
# Average Search time per question: 2.747867 seconds
# Average crawling/reading time per question: 5.198966 seconds

# === Average PLANNING tokens per question ===
# Prompt tokens:     4011.58
# Completion tokens: 314.92
# Total tokens:      4326.50

# === Average READING tokens per question ===
# Prompt tokens:     6026.39
# Completion tokens: 397.03
# Total tokens:      6423.42