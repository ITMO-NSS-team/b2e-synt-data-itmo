"""Сборка локальных JSON Schema и шаблонов сравнения перед запуском кейса."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


def resolve_schema(value, base: Path, stack: tuple = ()):
    """Встроить локальные $ref, сохраняя ограничения соседних ключей через allOf."""
    if isinstance(value, list):
        return [resolve_schema(item, base, stack) for item in value]
    if not isinstance(value, dict):
        return value
    # Значения этих keywords — данные, не вложенные схемы.
    result = {k: (v if k in {'const', 'enum', 'default', 'examples'}
                  else resolve_schema(v, base, stack))
              for k, v in value.items() if k != '$ref'}
    if '$ref' not in value:
        return result
    ref = urlsplit(value['$ref'])
    if ref.scheme or ref.netloc or ref.query:
        raise ValueError('Поддерживаются только локальные ссылки JSON Schema.')
    path = (base.parent / unquote(ref.path)).resolve() if ref.path else base.resolve()
    key = (path, ref.fragment)
    if key in stack:
        raise ValueError(f'Циклическая ссылка JSON Schema: {value["$ref"]}')
    target = json.loads(path.read_text(encoding='utf-8-sig'))
    pointer = unquote(ref.fragment)
    if pointer:
        if not pointer.startswith('/'):
            raise ValueError('Поддерживаются только фрагменты JSON Pointer.')
        for token in pointer[1:].split('/'):
            token = token.replace('~1', '/').replace('~0', '~')
            target = target[int(token)] if isinstance(target, list) else target[token]
    resolved = resolve_schema(target, path, stack + (key,))
    return {'allOf': [resolved, result]} if result else resolved


def load_case(path: str | Path) -> dict:
    path = Path(path).resolve()
    case = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(case, dict):
        raise ValueError('Кейс должен быть JSON-объектом.')
    if 'gold_contract' in case:
        case['gold_contract'] = resolve_schema(case['gold_contract'], path)
    comparison = case.get('gold_comparison')
    if isinstance(comparison, dict) and 'template' in comparison:
        if comparison['template'] != 'simple_comparison':
            raise ValueError(f'Неизвестный шаблон сравнения: {comparison["template"]}')
        template = Path(__file__).resolve().parents[2] / 'gold_dataset/templates/simple_comparison.json'
        defaults = json.loads(template.read_text(encoding='utf-8'))
        overrides = {k: v for k, v in comparison.items() if k != 'template'}
        if set(overrides) - set(defaults):
            raise ValueError('Неизвестные поля gold_comparison.')
        case['gold_comparison'] = {**defaults, **overrides}
    return case
