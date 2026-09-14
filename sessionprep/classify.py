# Session Prep — audio toolkit.
# Copyright (C) 2026 Sandro Giacometti
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import re
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path

from . import classify_rules

PACKAGE_DIR = Path(__file__).parent
RULES_PATH = PACKAGE_DIR / "config.yaml"

VERSIONISH = re.compile(r"^(V\d+|(19|20)\d{2}|\d{7,})$", re.I)
LEADING_INDEX = re.compile(r"^\d{1,2}$")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
WORD_DIGIT_BOUNDARY = re.compile(r"(?<=[A-Za-z]{2})(?=\d)")
NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")

CAPTURE_NEUTRAL = -1

@dataclass
class Rule:
    group: str
    role: str
    name: str
    need: list
    _: KW_ONLY
    avoid: tuple = ()
    keep_extra: bool = False
    keep_anchor: bool = False
    drop: tuple = ()
    numbering: str = ""
    kind: str = "instrument"

def reload_rules() -> None:
    global TABLES, RULES, ROLE_ORDER, GROUP_ORDER, NOISE, WEAK_EXTRA, SIDE_TOKENS, CAPTURE_RANK, _ROLE_RANK, UNRANKED_ROLE
    TABLES = classify_rules.load(RULES_PATH, Rule)
    RULES = TABLES.rules
    ROLE_ORDER = TABLES.role_order
    GROUP_ORDER = TABLES.group_order
    NOISE = TABLES.noise
    WEAK_EXTRA = TABLES.weak_extra
    SIDE_TOKENS = TABLES.side_tokens
    CAPTURE_RANK = TABLES.capture_rank
    _ROLE_RANK = {role: index for index, role in enumerate(ROLE_ORDER)}
    UNRANKED_ROLE = len(ROLE_ORDER)

def append_learned_rules_to_config(entries: list[dict]) -> None:
    if not entries:
        return

    with RULES_PATH.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    import yaml
    try:
        parsed = yaml.safe_load(raw_text)
    except Exception:
        return

    rules = parsed.get("rules", [])
    new_rules_text = ""

    for entry in entries:
        token = entry.get("token", "").strip().upper()
        group = entry.get("group", "").strip()
        name = entry.get("name", "").strip() or token

        if not token or not group:
            continue

        target_rule = next((r for r in rules if r.get("name") == name and r.get("tag") == group), None)

        inserted = False
        if target_rule:
            when = target_rule.get("when") or target_rule.get("match")
            if when and isinstance(when, list) and len(when) == 1 \
                    and isinstance(when[0], str):
                word_group = when[0]
                pattern = re.compile(
                    rf"^(\s*{re.escape(word_group)}\s*:\s*\[)(.*?)(\].*)$",
                    re.MULTILINE)

                def replacer(match):
                    existing = match.group(2)
                    items = [t.strip().strip('"\'').upper()
                             for t in existing.split(',')]
                    if token in items:
                        return match.group(0)
                    separator = ", " if existing.strip() else ""
                    return f"{match.group(1)}{existing}{separator}{token}{match.group(3)}"

                new_text, subs = pattern.subn(replacer, raw_text)
                if subs == 1:
                    raw_text = new_text
                    inserted = True

        if not inserted:
            new_rules_text += (
                f"  - name: \"{name}\"\n"
                f"    tag: {group}\n"
                f"    when:\n"
                f"      - [{token}]\n"
                f"    extra: yes\n"
                f"    number: spaced\n\n"
            )

    if new_rules_text:
        marker = re.search(r"^order:\s*$", raw_text, re.MULTILINE)
        if marker:
            cut = marker.start()
            raw_text = (raw_text[:cut].rstrip() + "\n\n" + new_rules_text
                        + raw_text[cut:])
        else:
            raw_text = raw_text.rstrip() + "\n\n" + new_rules_text

    with RULES_PATH.open("w", encoding="utf-8") as f:
        f.write(raw_text)

    reload_rules()

reload_rules()

@dataclass(kw_only=True)
class Classification:
    group: str
    role: str
    name: str
    rank: tuple
    confidence: str = "high"
    side: str = ""
    extra: str = ""
    matched_number: int | None = None
    tokens: list = field(default_factory=list)
    part: str = ""
    capture_rank: int = CAPTURE_NEUTRAL
    capture_number: int = 0

def tokenise(stem: str) -> list[str]:
    text = stem.replace("&", " ")
    text = CAMEL_BOUNDARY.sub(" ", text)
    text = WORD_DIGIT_BOUNDARY.sub(" ", text)
    text = NON_ALNUM.sub(" ", text)
    return [t for t in text.split() if t]

def _clean(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in NOISE and not VERSIONISH.match(t)]

def _matches(rule: Rule, tokens: set[str]) -> bool:
    if any(bad in tokens for bad in rule.avoid):
        return False
    return all(group & tokens for group in rule.need)

def _score(rule: Rule, index: int) -> tuple:
    return (1 if rule.kind == "instrument" else 0, len(rule.need), -index)

def _first_number(tokens: list[str]) -> int | None:
    for token in tokens:
        if token.isdigit():
            value = int(token)
            if 1 <= value <= 32:
                return value
    return None

def sanitise_tag_like(text: str) -> str:
    cleaned = "".join(c for c in str(text).strip() if c.isalnum() or c in " _-")
    return " ".join(cleaned.split()).replace(" ", "_")

def known_tokens() -> set[str]:
    words = set()
    for rule in RULES:
        for group in rule.need:
            words |= group
    return words

def common_prefix(token_lists: list[list[str]], threshold: float = 0.7):
    if len(token_lists) < 5:
        return []

    instrument_words = known_tokens()
    needed = max(3, int(len(token_lists) * threshold))
    best: list[str] = []

    for length in range(1, 4):
        counts: dict[tuple, int] = {}
        for tokens in token_lists:
            if len(tokens) > length:
                key = tuple(t.upper() for t in tokens[:length])
                counts[key] = counts.get(key, 0) + 1
        if not counts:
            break
        candidate, hits = max(counts.items(), key=lambda kv: kv[1])
        if hits < needed:
            continue
        if any(word in instrument_words for word in candidate):
            continue
        if not _prefix_is_safe(token_lists, list(candidate)):
            continue
        best = list(candidate)
    return best

def _prefix_is_safe(token_lists: list[list[str]], prefix: list[str]) -> bool:
    for tokens in token_lists:
        head = [t.upper() for t in tokens[:len(prefix)]]
        if head != prefix:
            continue
        rest = [t for t in tokens[len(prefix):]
                if t.upper() not in NOISE and not VERSIONISH.match(t)]
        if not rest:
            return False
    return True

def split_capture(tokens: list[str]) -> tuple[str, int, int]:
    part: list[str] = []
    rank: int | None = None
    number = 0
    index = 0
    while index < len(tokens):
        word = tokens[index].upper()
        if word in CAPTURE_RANK:
            found = CAPTURE_RANK[word]
            rank = found if rank is None else max(rank, found)
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            if following.isdigit() and len(following) <= 2:
                number = int(following)
                index += 2
                continue
            index += 1
            continue
        part.append(tokens[index].lower())
        index += 1
    return " ".join(part), (CAPTURE_NEUTRAL if rank is None else rank), number

def classify_tokens(tokens: list[str], rules: list[Rule] | None = None) -> Classification | None:
    if not tokens:
        return None

    rules = rules or RULES
    side = ""
    upper = [t.upper() for t in tokens]
    if upper and upper[-1] in SIDE_TOKENS:
        side = SIDE_TOKENS[upper[-1]]

    kept = [t for index, t in enumerate(tokens)
            if t.upper() not in NOISE
            and not VERSIONISH.match(t)
            and not (index == 0 and LEADING_INDEX.match(t))]
    token_set = {t.upper() for t in kept}
    if not token_set:
        return None

    best = None
    for index, rule in enumerate(rules):
        if not _matches(rule, token_set):
            continue
        score = _score(rule, index)
        if best is None or score > best[0]:
            best = (score, rule)
    if best is None:
        return None
    rule = best[1]

    consumed = set()
    for group in rule.need:
        consumed |= (group & token_set)
    if rule.keep_anchor:
        words = rule.name.upper().split()
        redundant = lambda t: any(t.startswith(w) or w.startswith(t)
                                  for w in words)
        consumed -= {t for t in consumed if not redundant(t)}

    extra_tokens = [
        t for t in kept
        if t.upper() not in consumed
        and t.upper() not in WEAK_EXTRA
        and t.upper() not in rule.drop
        and not (side and t.upper() in SIDE_TOKENS)
    ]
    extra = " ".join(t.lower() for t in extra_tokens) if (rule.keep_extra or rule.keep_anchor) else ""

    name = rule.name
    if "{n}" in name:
        name = name.replace("{n}", str(_first_number(tokens) or 1))
    if extra:
        name = f"{name} {extra}"

    part, capture_rank, capture_number = split_capture(extra_tokens)

    return Classification(
        group=rule.group,
        role=rule.role,
        name=name,
        rank=(GROUP_ORDER.index(rule.group),
              _ROLE_RANK.get(rule.role, UNRANKED_ROLE)),
        confidence="high" if len(rule.need) > 1 else "low",
        side=side,
        extra=extra,
        matched_number=_first_number(tokens),
        tokens=[t.upper() for t in kept],
        part=part,
        capture_rank=capture_rank,
        capture_number=capture_number,
    )

def classify_one(stem: str) -> Classification | None:
    return classify_tokens(tokenise(stem), RULES)

def _rule_for(role: str, rules: list[Rule] | None = None) -> Rule:
    for rule in (rules or RULES):
        if rule.role == role:
            return rule
    raise KeyError(role)

def classify_many(stems: list[str]) -> list[Classification | None]:
    token_lists = [tokenise(stem) for stem in stems]

    prefix = common_prefix(token_lists)
    if prefix:
        stripped = []
        for tokens in token_lists:
            head = [t.upper() for t in tokens[:len(prefix)]]
            stripped.append(tokens[len(prefix):] if head == prefix else tokens)
        token_lists = stripped

    results = [classify_tokens(tokens, RULES) for tokens in token_lists]

    by_role: dict[str, list[Classification]] = {}
    for result in results:
        if result is not None:
            by_role.setdefault(result.role, []).append(result)

    for role, group in by_role.items():
        rule = _rule_for(role, RULES)

        buckets: dict[str, tuple[str, list[Classification]]] = {}
        for item in group:
            display_base = rule.name
            if item.extra:
                display_base = f"{display_base} {item.extra}"

            match_key = display_base.upper()
            if match_key not in buckets:
                buckets[match_key] = (display_base, [])
            buckets[match_key][1].append(item)

        for match_key, (base, items) in buckets.items():
            needs_number = len(items) > 1 or "{n}" in rule.name
            if not needs_number:
                items[0].name = base
                continue

            sides = [item.side for item in items]
            if len(items) == 2 and set(sides) == {"L", "R"}:
                for item in items:
                    item.name = f"{base} {item.side}"
                continue

            used: set[int] = set()
            running = 0
            for item in items:
                number = item.matched_number
                if number is None or number in used:
                    running += 1
                    while running in used:
                        running += 1
                    number = running
                used.add(number)

                if "{n}" in rule.name:
                    item.name = rule.name.replace("{n}", str(number))
                    if item.extra:
                        item.name = f"{item.name} {item.extra}"
                else:
                    item.name = f"{base}{rule.numbering.format(n=number)}"

    return results

def teachable_tokens(stem: str, prefix: list[str] | None = None) -> list[str]:
    tokens = tokenise(stem)
    if prefix:
        head = [t.upper() for t in tokens[:len(prefix)]]
        if head == prefix:
            tokens = tokens[len(prefix):]
    known = known_tokens()
    candidates = [
        t for t in tokens
        if t.upper() not in known
        and t.upper() not in NOISE
        and not VERSIONISH.match(t)
        and not t.isdigit()
        and t.upper() not in WEAK_EXTRA
        and t.upper() not in SIDE_TOKENS
        and len(t) > 1
    ]
    seen, unique = set(), []
    for token in candidates:
        if token.upper() not in seen:
            seen.add(token.upper())
            unique.append(token)
    return sorted(unique, key=len, reverse=True)

def sort_key(result: Classification | None, fallback: int) -> tuple:
    if result is None:
        return (len(GROUP_ORDER), 0, fallback)
    return (result.rank[0], result.rank[1], fallback)
