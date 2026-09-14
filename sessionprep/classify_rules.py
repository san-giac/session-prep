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

from dataclasses import dataclass, field
from pathlib import Path
import yaml

NUMBERING = {"none": "", "spaced": " {n}", "tight": "{n}"}

RULE_KEYS = {"name", "tag", "role", "when", "match", "unless", "number",
             "extra", "keep_word", "ignore", "position"}
REQUIRED_KEYS = ("name", "tag")

class RulesError(Exception):
    pass

@dataclass
class Tables:
    rules: list = field(default_factory=list)
    role_order: list[str] = field(default_factory=list)
    group_order: list[str] = field(default_factory=list)
    noise: set = field(default_factory=set)
    weak_extra: set = field(default_factory=set)
    side_tokens: dict = field(default_factory=dict)
    capture_rank: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

def _upper_set(values, where: str) -> set:
    if isinstance(values, str):
        values = values.split()
    if not isinstance(values, list):
        raise RulesError(f"{where}: expected a list of words, got {type(values).__name__}")
    return {str(v).upper() for v in values}

def _flatten_words(raw_words: dict, source: str) -> dict[str, set[str]]:
    tokens = {}

    def _walk(data, path=""):
        if isinstance(data, dict):
            for k, v in data.items():
                _walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(data, list):
            key = path.split(".")[-1].lower()
            if key in tokens:
                raise RulesError(
                    f"{source}: two word groups are both called {key!r} "
                    f"(the second is words.{path}). Group names have to be "
                    f"unique across all the categories, since rules refer to "
                    f"them by the last part of the name only")
            tokens[key] = _upper_set(data, f"{source}: words.{path}")

    _walk(raw_words)
    return tokens

def _resolve_groups(entries, tokens: dict, where: str) -> list:
    resolved = []
    if not isinstance(entries, list):
        raise RulesError(f"{where}: expected a list, got {type(entries).__name__}")
    for entry in entries:
        if isinstance(entry, str):
            key = entry.lower()
            if key not in tokens:
                known = ", ".join(sorted(tokens)) or "(none)"
                raise RulesError(
                    f"{where}: no word group called {entry!r}. Defined in `words`: {known}")
            resolved.append(set(tokens[key]))
        else:
            resolved.append(_upper_set(entry, where))
    return resolved

def load(config_path: Path | str, rule_cls) -> Tables:
    path = Path(config_path)
    if not path.exists():
        raise RulesError(f"Configuration file not found: {path}")

    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise RulesError(f"not valid YAML — {exc}") from exc

    tokens = _flatten_words(raw.get("words") or {}, path.name)

    tables = Tables()
    tables.group_order = [str(g) for g in (raw.get("groups") or [])]
    tables.role_order = [str(r) for r in (raw.get("order") or [])]
    tables.noise = _upper_set(raw.get("noise") or [], f"{path.name}: noise")
    tables.weak_extra = _upper_set(raw.get("weak_extra") or [], f"{path.name}: weak_extra")
    tables.side_tokens = {str(k).upper(): str(v).upper()
                          for k, v in (raw.get("sides") or {}).items()}

    for rank, label in enumerate(("source", "amp", "processed", "distant")):
        for word in _upper_set((raw.get("capture") or {}).get(label) or [],
                               f"{path.name}: capture.{label}"):
            tables.capture_rank.setdefault(word, rank)

    entries = raw.get("rules") or []
    if not isinstance(entries, list):
        raise RulesError(f"{path.name}: `rules` must be a list")

    for index, entry in enumerate(entries):
        where = f"{path.name}: rule {index + 1}"
        if not isinstance(entry, dict):
            raise RulesError(f"{where}: expected a mapping")

        unknown = set(entry) - RULE_KEYS
        if unknown:
            raise RulesError(
                f"{where}: unknown field(s) {sorted(unknown)}. Allowed: "
                f"{sorted(RULE_KEYS)}. A misspelt optional field would "
                f"otherwise be ignored in silence")
        for required in REQUIRED_KEYS:
            if required not in entry:
                raise RulesError(f"{where}: missing `{required}`")

        match_entries = entry.get("when") or entry.get("match")
        if not match_entries:
            raise RulesError(f"{where}: missing `when` match conditions")

        name_str = str(entry["name"])
        role_str = str(entry.get("role", name_str))

        number = str(entry.get("number", "none"))
        if number not in NUMBERING:
            raise RulesError(f"{where}: invalid `number` value {number!r}")

        avoid = set().union(*_resolve_groups(entry.get("unless") or [], tokens, where)) \
            if entry.get("unless") else set()
        ignore = set().union(*_resolve_groups(entry.get("ignore") or [], tokens, where)) \
            if entry.get("ignore") else set()

        tables.rules.append(rule_cls(
            str(entry["tag"]),
            role_str,
            name_str,
            _resolve_groups(match_entries, tokens, where),
            avoid=tuple(sorted(avoid)),
            keep_extra=bool(entry.get("extra", False)),
            keep_anchor=bool(entry.get("keep_word", False)),
            drop=tuple(sorted(ignore)),
            numbering=NUMBERING[number],
            kind="mic" if entry.get("position", False) else "instrument",
        ))

    roles = {rule.role for rule in tables.rules}
    for listed in tables.role_order:
        if listed not in roles:
            tables.warnings.append(
                f"`order` lists {listed!r}, which no rule produces — "
                f"check for a renamed or deleted rule")
    for rule in tables.rules:
        if rule.role not in tables.role_order:
            tables.role_order.append(rule.role)
            tables.warnings.append(
                f"{rule.role!r} is missing from `order`; those tracks sort "
                f"last within the {rule.group!r} tag")

    return tables
