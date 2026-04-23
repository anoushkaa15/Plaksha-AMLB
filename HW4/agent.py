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

_PLAN_CONCEPTS = [
    "History", "Science", "Technology", "Mathematics", "Physics", "Biology",
    "Chemistry", "Geology", "Law", "Philosophy", "Engineering", "Economics",
    "Society", "Culture", "Religion", "Geography",
]



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






def _llm_plan_bridges(start_title: str, target_title: str) -> List[str]:
    opts = ", ".join(_PLAN_CONCEPTS)
    prompt = (
        "Pick up to TWO bridge Wikipedia concepts to navigate from START to GOAL.\n"
        f"START: {start_title}\n"
        f"GOAL: {target_title}\n"
        f"Allowed concepts only: {opts}\n"
        "Return ONLY JSON array, e.g. [\"History\",\"Science\"]"
    )
    out = _ask_gemini(prompt)
    m = re.search(r"\[[\s\S]*\]", out)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    picked = []
    allowed = {x.lower(): x for x in _PLAN_CONCEPTS}
    for v in arr[:2]:
        if not isinstance(v, str):
            continue
        key = v.strip().lower()
        if key in allowed and allowed[key] not in picked:
            picked.append(allowed[key])
    return picked


def _choose_pivot(target_title: str) -> Optional[str]:
    t = target_title.lower()
    # Lightweight hand-tuned pivots for common "hard jump" targets.
    if "roman law" in t:
        return "Law"
    if "nuclear physics" in t:
        return "Physics"
    if "plate tectonics" in t:
        return "Geology"
    if "mathematics" in t:
        return "Mathematics"
    if "computer science" in t:
        return "Science"
    return None

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




def _bounded_bfs(
    start_url: str,
    target_url: str,
    target_title: str,
    visited_hint: Set[str],
    depth_limit: int,
    node_limit: int,
    local_deadline: float,
) -> Optional[List[str]]:
    """Small best-first BFS burst to recover from local traps."""
    if start_url == target_url:
        return [start_url]

    q: List[Tuple[List[str], int]] = [([start_url], 0)]
    seen = {start_url, *visited_hint}
    expanded = 0

    while q and expanded < node_limit and time.time() < local_deadline:
        path, depth = q.pop(0)
        u = path[-1]
        if depth >= depth_limit:
            continue
        try:
            links = _get_links_cached(u)
        except Exception:
            continue
        expanded += 1

        direct = next((l for l in links if l["url"] == target_url), None)
        if direct is not None:
            return path + [target_url]

        ranked = _rank_candidates(links, target_title, seen, k=18)
        for _, l in ranked:
            v = l["url"]
            if v in seen:
                continue
            seen.add(v)
            q.append((path + [v], depth + 1))

        # Keep frontier focused on promising nodes.
        if len(q) > 120:
            q.sort(key=lambda item: _score_title(_url_title(item[0][-1]), target_title), reverse=True)
            q = q[:120]

    return None

def _solve_one(pair: dict, pair_deadline: float, hard_deadline: float) -> dict:
    pair_id = pair["pair_id"]
    start_url = pair["start"]
    target_url = pair["target"]
    target_title = _url_title(target_url)
    difficulty = pair.get("difficulty", "medium")
    pivot_title = _choose_pivot(target_title)
    llm_plan: List[str] = []
    llm_calls = 0
    try:
        llm_plan = _llm_plan_bridges(_url_title(start_url), target_title)
        if llm_plan:
            llm_calls += 1
    except Exception:
        llm_plan = []

    phase_targets = llm_plan[:]
    if pivot_title and pivot_title not in phase_targets:
        phase_targets.append(pivot_title)
    phase_targets.append(target_title)
    phase_i = 0
    phase_target = phase_targets[phase_i]

    # step cap tuned for score/time tradeoff
    max_steps = {"easy": 8, "medium": 10, "hard": 8}.get(difficulty, 9)
    llm_budget = {"easy": 4, "medium": 6, "hard": 3}.get(difficulty, 4)
    bfs_node_budget = {"easy": 60, "medium": 120, "hard": 180}.get(difficulty, 100)
    bfs_used = False

    path = [start_url]
    visited = {start_url}
    link_counts: List[int] = []
    current = start_url

    for step in range(max_steps):
        now = time.time()
        if now >= pair_deadline or now >= hard_deadline - 0.35:
            break

        curr_title = _url_title(current)
        if current == target_url or curr_title.lower() == target_title.lower():
            return _result(pair_id, path, llm_calls, link_counts, True)

        ct = curr_title.lower()
        if phase_i < len(phase_targets) - 1:
            goal_l = phase_target.lower()
            if ct == goal_l or goal_l in ct:
                phase_i += 1
                phase_target = phase_targets[phase_i]

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

        if phase_target != target_title:
            phase_link = next((l for l in links if _url_title(l["url"]).lower() == phase_target.lower()), None)
            if phase_link is not None:
                current = phase_link["url"]
                path.append(current)
                visited.add(current)
                continue

        ranked = _rank_candidates(links, phase_target, visited, k=16)
        if not ranked:
            fallback = next((l for l in links if l["url"] not in visited), links[0])
            current = fallback["url"]
            path.append(current)
            visited.add(current)
            continue

        # One focused BFS burst per pair can recover from near-target traps.
        if (not bfs_used) and step >= 2 and (pair_deadline - time.time()) > 2.4:
            bfs_used = True
            bfs_path = _bounded_bfs(
                current,
                target_url,
                target_title,
                visited,
                depth_limit=3,
                node_limit=bfs_node_budget,
                local_deadline=min(pair_deadline - 0.3, time.time() + 2.2),
            )
            if bfs_path and len(bfs_path) >= 2:
                path.extend(bfs_path[1:])
                return _result(pair_id, path, llm_calls, link_counts, True)

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
                idx = _llm_pick(_url_title(current), phase_target, shortlist)
                llm_calls += 1
                if idx is not None:
                    picked = shortlist[idx]
            except Exception:
                picked = None

        if picked is None:
            # Escape local traps: if target similarity is still tiny after a few steps, prefer broad hub pages.
            if step >= 3 and _score_title(_url_title(current), target_title) < 0.25:
                hub = next((l for _, l in ranked if any(h in l["text"].lower() for h in _HUB_HINTS)), None)
                if hub is not None:
                    picked = hub

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
