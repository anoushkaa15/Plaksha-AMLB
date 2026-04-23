"""
agent.py — submission file.

Hybrid Wikipedia navigator for the HW4 scorer.
"""

import json
import os
import random
import re
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Set, Tuple

from wiki_tool import get_links

# ── API key — read from file, never hardcode ──────────────────────────────────
_API_KEY_PATH = os.path.expanduser("~/gemini_api_key.txt")
with open(_API_KEY_PATH) as f:
    _API_KEY = f.read().strip()
if not _API_KEY:
    raise RuntimeError("~/gemini_api_key.txt is empty. Put your Gemini key on one line.")

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key=" + _API_KEY
)

# ── Global caches (shared across all pairs in one run) ───────────────────────
_LINK_CACHE: Dict[str, List[Dict[str, str]]] = {}
_TITLE_TOKEN_CACHE: Dict[str, Set[str]] = {}

_STOP = {
    "the", "of", "and", "in", "to", "a", "an", "on", "for", "by", "with",
    "from", "at", "is", "as", "or", "new", "list", "history", "united", "state",
}

_BAD_HINTS = {
    "album", "song", "episode", "season", "film", "novel", "character",
    "disambiguation", "unicode", "award", "awards", "soundtrack",
}

_HUB_HINTS = {
    "history", "science", "technology", "mathematics", "physics", "biology",
    "chemistry", "geography", "economics", "philosophy", "religion", "culture",
    "politics", "engineering", "education", "society", "renaissance", "war",
}


def _ask_gemini(prompt: str) -> str:
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    req = urllib.request.Request(
        _GEMINI_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _url_title(url: str) -> str:
    return url.split("/wiki/")[-1].split("#")[0].replace("_", " ").strip()


def _tok(title: str) -> Set[str]:
    k = title.lower()
    if k in _TITLE_TOKEN_CACHE:
        return _TITLE_TOKEN_CACHE[k]
    t = {
        w
        for w in re.findall(r"[a-z0-9]+", k)
        if len(w) > 2 and w not in _STOP
    }
    _TITLE_TOKEN_CACHE[k] = t
    return t


def _sim(a: str, b: str) -> float:
    sa, sb = _tok(a), _tok(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / max(1, len(sb)) + 0.35 * inter / max(1, len(sa))


def _quality_penalty(title: str) -> float:
    t = title.lower()
    p = 0.0
    if any(h in t for h in _BAD_HINTS):
        p += 1.6
    if re.search(r"\b(19|20)\d{2}\b", t):
        p += 0.8
    if len(t) > 45:
        p += 0.5
    if "(" in t and ")" in t:
        p += 0.6
    return p


def _hub_bonus(title: str, target_title: str) -> float:
    t = title.lower()
    target = target_title.lower()
    b = 0.0
    if any(h in t for h in _HUB_HINTS):
        b += 0.45
    if any(h in target for h in _HUB_HINTS) and any(h in t for h in _HUB_HINTS):
        b += 0.4
    return b


def _score_title(title: str, target_title: str) -> float:
    s = 1.7 * _sim(title, target_title)
    tl, gl = title.lower(), target_title.lower()
    if tl in gl or gl in tl:
        s += 1.4
    s += _hub_bonus(title, target_title)
    s -= _quality_penalty(title)
    return s


def _get_links_cached(url: str) -> List[Dict[str, str]]:
    if url in _LINK_CACHE:
        return _LINK_CACHE[url]
    links = get_links(url)
    _LINK_CACHE[url] = links
    return links


def _empty_result(pair: dict) -> dict:
    s = pair["start"]
    return {
        "pair_id": pair["pair_id"],
        "path": [s, s],
        "steps": 1,
        "llm_calls": 0,
        "success": False,
        "link_counts": [],
    }


def _result(pair_id: str, path: List[str], llm_calls: int, link_counts: List[int], success: bool) -> dict:
    return {
        "pair_id": pair_id,
        "path": path,
        "steps": len(path) - 1,
        "llm_calls": llm_calls,
        "success": success,
        "link_counts": link_counts,
    }


def _rank_candidates(
    links: Sequence[Dict[str, str]],
    target_title: str,
    visited: Set[str],
    k: int = 16,
) -> List[Tuple[float, Dict[str, str]]]:
    scored: List[Tuple[float, Dict[str, str]]] = []
    for l in links:
        u, t = l["url"], l["text"]
        if u in visited:
            continue
        scored.append((_score_title(t, target_title), l))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def _llm_pick(current_title: str, target_title: str, candidates: Sequence[Dict[str, str]]) -> Optional[int]:
    if not candidates:
        return None
    numbered = "\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(candidates))
    prompt = (
        "You are playing the Wikipedia navigation game.\n"
        f"Current page: {current_title}\n"
        f"Goal page: {target_title}\n\n"
        "Pick the best NEXT link to reach the goal quickly.\n"
        "Prefer broad conceptual bridges over niche pages.\n"
        "Reply with ONLY one integer index.\n\n"
        f"Options:\n{numbered}"
    )
    out = _ask_gemini(prompt).strip()
    m = re.search(r"\d+", out)
    if not m:
        return None
    idx = int(m.group(0)) - 1
    if 0 <= idx < len(candidates):
        return idx
    return None


def _two_hop_check(current_url: str, target_url: str, ranked: Sequence[Tuple[float, Dict[str, str]]], budget_expand: int = 10) -> Optional[List[str]]:
    """Search current -> mid -> target over strongest mids first."""
    mids = [l for _, l in ranked[:budget_expand]]
    for m in mids:
        mid_url = m["url"]
        try:
            links2 = _get_links_cached(mid_url)
        except Exception:
            continue
        if any(x["url"] == target_url for x in links2):
            return [current_url, mid_url, target_url]
    return None


def _solve_one(pair: dict, pair_deadline: float, hard_deadline: float) -> dict:
    pair_id = pair["pair_id"]
    start_url = pair["start"]
    target_url = pair["target"]
    target_title = _url_title(target_url)
    difficulty = pair.get("difficulty", "medium")

    # step cap tuned for score/time tradeoff
    max_steps = {"easy": 8, "medium": 10, "hard": 8}.get(difficulty, 9)
    llm_budget = {"easy": 4, "medium": 6, "hard": 3}.get(difficulty, 4)

    path = [start_url]
    visited = {start_url}
    llm_calls = 0
    link_counts: List[int] = []
    current = start_url

    for step in range(max_steps):
        now = time.time()
        if now >= pair_deadline or now >= hard_deadline - 0.35:
            break

        if current == target_url or _url_title(current).lower() == target_title.lower():
            return _result(pair_id, path, llm_calls, link_counts, True)

        try:
            links = _get_links_cached(current)
        except Exception:
            break

        link_counts.append(len(links))
        if not links:
            break

        direct = next((l for l in links if l["url"] == target_url), None)
        if direct:
            path.append(target_url)
            return _result(pair_id, path, llm_calls, link_counts, True)

        ranked = _rank_candidates(links, target_title, visited, k=16)
        if not ranked:
            fallback = next((l for l in links if l["url"] not in visited), links[0])
            current = fallback["url"]
            path.append(current)
            visited.add(current)
            continue

        # Early 2-hop check over strong candidates only
        if step <= 2:
            burst = _two_hop_check(current, target_url, ranked, budget_expand=8)
            if burst is not None:
                path.extend(burst[1:])
                return _result(pair_id, path, llm_calls, link_counts, True)

        shortlist = [l for _, l in ranked[:10]]
        picked = None

        if llm_calls < llm_budget and len(shortlist) >= 4:
            try:
                idx = _llm_pick(_url_title(current), target_title, shortlist)
                llm_calls += 1
                if idx is not None:
                    picked = shortlist[idx]
            except Exception:
                picked = None

        if picked is None:
            # mild randomness in top 3 prevents deterministic local traps
            top = [l for _, l in ranked[:3]]
            picked = random.choice(top) if len(top) > 1 and step >= 3 else top[0]

        current = picked["url"]
        path.append(current)
        visited.add(current)

    success = _url_title(path[-1]).lower() == target_title.lower()
    if len(path) < 2:
        path = [start_url, start_url]
    return _result(pair_id, path, llm_calls, link_counts, success)


def solve_all(pairs: list, deadline: float) -> list:
    """Solve all pairs within shared deadline."""
    if not pairs:
        return []

    difficulty_rank = {"easy": 0, "medium": 1, "hard": 2}
    indexed = list(enumerate(pairs))
    ordered = sorted(indexed, key=lambda x: (difficulty_rank.get(x[1].get("difficulty", "hard"), 3), x[0]))

    out: Dict[int, dict] = {}

    for j, (orig_idx, pair) in enumerate(ordered):
        now = time.time()
        if now >= deadline - 0.6:
            out[orig_idx] = _empty_result(pair)
            continue

        remaining = len(ordered) - j
        rem_time = max(0.0, deadline - now - 0.5)
        per_pair = max(2.0, rem_time / max(1, remaining))

        d = pair.get("difficulty", "medium")
        # Invest more in easy/medium where score probability is higher.
        if d == "easy":
            pair_budget = min(18.0, per_pair * 1.45)
        elif d == "medium":
            pair_budget = min(14.0, per_pair * 1.10)
        else:
            pair_budget = min(8.0, per_pair * 0.80)

        pair_deadline = min(deadline - 0.4, now + pair_budget)

        try:
            out[orig_idx] = _solve_one(pair, pair_deadline, deadline)
        except Exception:
            out[orig_idx] = _empty_result(pair)

    return [out[i] for i in range(len(pairs))]
