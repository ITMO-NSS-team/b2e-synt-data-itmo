"""Каверзы канала: места, где сервис отвечает 200 OK и неверно.

Это отдельный модуль, а не разрозненные ``if`` по коду, по трём причинам.

1. Каждая каверза должна быть **включаемой и выключаемой**. С выключенными
   каверзами эмулятор ведёт себя «как в учебнике» — на этом проверяется, что
   тест ловит именно каверзу, а не общую поломку.
2. Реестр каверз — курируемый артефакт фабрики (``traps/*.yaml``). Код и реестр
   обязаны говорить об одном и том же наборе, поэтому имена здесь — те же.
3. Каверзы — главная причина, по которой эмулятор вообще нужен. Рецепт, который
   «синтаксически валиден и молча возвращает пусто», ловится только исполнением.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Молчаливая свёртка регистра. ClickHouse `lower()`/`ILIKE` не сворачивают
#: кириллицу — нужен `lowerUTF8()`. Фильтр по русскому тексту в другом регистре
#: молча возвращает пусто.
ASCII_ONLY_CASE_FOLD = "ascii_only_case_fold"

#: Строка 'false' на Bool-колонке коэрсится в true и ИНВЕРТИРУЕТ выборку.
BOOL_STRING_COERCION = "bool_string_coercion"

#: limit выше API_MAX_LIMIT ужимается молча, без ошибки; в ответе видно
#: применённое значение.
SILENT_LIMIT_CLAMP = "silent_limit_clamp"

#: Вектор-поиск без фильтра has_embedding = true падает на исполнении:
#: cosineDistance применяется к пустому вектору.
EMBEDDING_DIMENSION_MISMATCH = "embedding_dimension_mismatch"

#: NULL при сортировке идёт первым, если не задан nulls: last — нескоренные
#: всплывают наверх выдачи «сильнейших».
NULLS_FIRST_BY_DEFAULT = "nulls_first_by_default"

ALL_QUIRKS = (
    ASCII_ONLY_CASE_FOLD,
    BOOL_STRING_COERCION,
    SILENT_LIMIT_CLAMP,
    EMBEDDING_DIMENSION_MISMATCH,
    NULLS_FIRST_BY_DEFAULT,
)


@dataclass
class Quirks:
    """Набор включённых каверз. По умолчанию включены все — как в бою."""

    enabled: set[str] = field(default_factory=lambda: set(ALL_QUIRKS))

    def __contains__(self, name: str) -> bool:
        return name in self.enabled

    @classmethod
    def none(cls) -> "Quirks":
        return cls(enabled=set())

    @classmethod
    def only(cls, *names: str) -> "Quirks":
        return cls(enabled=set(names))


def fold_case(text: str, quirks: Quirks) -> str:
    """Свернуть регистр так, как это делает канал.

    С включённой каверзой сворачивается только ASCII: «Иванов» и «иванов»
    остаются разными строками, и ILIKE по кириллице молча не находит ничего.
    """
    if ASCII_ONLY_CASE_FOLD in quirks:
        return "".join(c.lower() if "A" <= c <= "Z" else c for c in text)
    return text.lower()
