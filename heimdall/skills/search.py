"""Поиск скиллов: то, что в бою делает ``find_skills``.

Что именно индексирует боевой сервис для ``relevance`` — открытый вопрос к
команде (CLAUDE.md §9). Здесь индексируются ``title`` и ``description``, потому
что именно их агент видит в выдаче ``SkillSummary``: ни ``tags``, ни ``purpose``
в карточке нет, и полагаться на них при выборе он не может.

Отсюда прямое следствие для генератора: ключевые слова обязаны быть внутри
``description``, а не в ``tags``. Метрика recall@3, посчитанная на этом индексе,
предсказывает боевое поведение ровно настолько, насколько верна эта модель, —
поэтому она и публикуется с пометкой «на локальном индексе».
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

#: Минимальная длина токена. Русские окончания режутся грубым стеммингом:
#: «должности» и «должность» должны попадать в один токен.
_MIN_TOKEN = 3
_WORD = re.compile(r"[\w-]+", re.UNICODE)

_ENDINGS = ("ями", "ами", "ого", "ому", "ыми", "ими", "ей", "ой", "ые", "ый", "ая",
            "ое", "ом", "ах", "ях", "ов", "ев", "ий", "ия", "ию", "ам", "ям", "ах",
            "и", "ы", "а", "о", "е", "у", "ю", "я", "ь")


def tokenize(text: str) -> list[str]:
    """Разбить текст на нормализованные токены."""
    out = []
    for word in _WORD.findall((text or "").lower()):
        if len(word) < _MIN_TOKEN:
            continue
        out.append(_stem(word))
    return out


def _stem(word: str) -> str:
    """Грубый стемминг: снять одно русское окончание, если слово останется длинным."""
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            return word[: -len(ending)]
    return word


class Index:
    """BM25-подобный индекс по полям, которые видит агент."""

    K1 = 1.5
    B = 0.75

    def __init__(self, documents: dict[str, str]) -> None:
        self._tokens = {name: tokenize(text) for name, text in documents.items()}
        self._freq = {name: Counter(tokens) for name, tokens in self._tokens.items()}
        self._len = {name: len(tokens) or 1 for name, tokens in self._tokens.items()}
        self._avg = (sum(self._len.values()) / len(self._len)) if self._len else 1.0
        self._df: Counter = Counter()
        for tokens in self._tokens.values():
            self._df.update(set(tokens))
        self._n = max(1, len(self._tokens))

    def score(self, query: str, names: Iterable[str]) -> dict[str, float]:
        """Оценить релевантность и нормировать её в 0..1."""
        terms = tokenize(query)
        raw: dict[str, float] = {}
        for name in names:
            freq = self._freq.get(name)
            if freq is None:
                raw[name] = 0.0
                continue
            total = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (self._n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                norm = tf * (self.K1 + 1) / (
                    tf + self.K1 * (1 - self.B + self.B * self._len[name] / self._avg))
                total += idf * norm
            raw[name] = total
        top = max(raw.values(), default=0.0)
        if top <= 0:
            return {name: 0.0 for name in raw}
        return {name: round(value / top, 6) for name, value in raw.items()}


def build_index(skills: dict[str, Any]) -> Index:
    """Индекс строится по title и description — по тому, что агент видит в выдаче."""
    return Index({name: f"{skill.title} {skill.description}"
                  for name, skill in skills.items()})
