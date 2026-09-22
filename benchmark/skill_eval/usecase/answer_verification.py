"""Сравнение структурированного ответа с gold_answer (не валидация JSON Schema)."""
import json
import math


def comparison_options(case):
    options = {"ordered": False, "row_key": [], "allow_extra_rows": False,
               "numeric_absolute_tolerance": 0}
    supplied = case.get("gold_comparison", {})
    if not isinstance(supplied, dict) or set(supplied) - set(options):
        raise ValueError("Некорректный gold_comparison.")
    options.update(supplied)
    for key in ("ordered", "allow_extra_rows"):
        if not isinstance(options[key], bool):
            raise ValueError(f"gold_comparison.{key} должен быть boolean.")
    keys = options["row_key"]
    if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("row_key должен быть массивом уникальных имён полей.")
    tolerance = options["numeric_absolute_tolerance"]
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("numeric_absolute_tolerance должен быть конечным неотрицательным числом.")
    return options


def equal(expected, actual, tolerance=0):
    if type(expected) in (int, float) and type(actual) in (int, float):
        return math.isfinite(expected) and math.isfinite(actual) and abs(expected - actual) <= tolerance
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(equal(v, actual[k], tolerance) for k, v in expected.items())
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(equal(x, y, tolerance) for x, y in zip(expected, actual))
    return expected == actual


def row_keys(rows, keys):
    result = []
    for row in rows:
        if not isinstance(row, dict) or any(k not in row for k in keys):
            raise ValueError("В строке rows отсутствуют поля row_key.")
        # Тип входит в ключ: true не равен 1; ключи не сравниваются с допуском.
        key = tuple(json.dumps(row[k], sort_keys=True, ensure_ascii=False) for k in keys)
        if key in result:
            raise ValueError(f"Повторяющийся row_key: {key}")
        result.append(key)
    return result


def verify_answer(case, answer):
    """Вернуть список расхождений; пустой список означает совпадение."""
    options = comparison_options(case)
    expected = case["gold_answer"]
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except ValueError:
            return ["Поле answer не содержит корректный JSON."]
    if not isinstance(answer, dict):
        return ["Поле answer должно содержать JSON-объект."]
    tolerance = options["numeric_absolute_tolerance"]
    ignored = {"rows", "comment", "comments"}
    gold_fields = {k: v for k, v in expected.items() if k not in ignored}
    actual_fields = {k: v for k, v in answer.items() if k not in ignored}
    errors = []
    if not equal(gold_fields, actual_fields, tolerance):
        errors.append("Поля ответа вне rows (включая outcome) отличаются от gold_answer.")
    gold_rows, actual_rows = expected.get("rows"), answer.get("rows")
    if not isinstance(gold_rows, list) or not isinstance(actual_rows, list):
        if not equal(gold_rows, actual_rows, tolerance) or ("rows" in expected) != ("rows" in answer):
            errors.append("rows отличается от gold_answer.")
        return errors
    if answer.get("outcome") != "answer" and actual_rows:
        errors.append("При outcome != answer массив rows должен быть пустым.")
    keys = options["row_key"]
    try:
        gold_keys = row_keys(gold_rows, keys) if keys else None
        actual_keys = row_keys(actual_rows, keys) if keys else None
    except ValueError as exc:
        return errors + [str(exc)]
    if len(actual_rows) < len(gold_rows) or (not options["allow_extra_rows"] and len(actual_rows) != len(gold_rows)):
        errors.append(f"Количество rows: ожидалось {len(gold_rows)}, получено {len(actual_rows)}.")

    def matches(i, j):
        return (not keys or gold_keys[i] == actual_keys[j]) and equal(gold_rows[i], actual_rows[j], tolerance)

    if options["ordered"]:
        cursor = 0
        for i, row in enumerate(gold_rows):
            if options["allow_extra_rows"]:
                while cursor < len(actual_rows) and not matches(i, cursor):
                    cursor += 1
            if cursor >= len(actual_rows) or not matches(i, cursor):
                errors.append(f"rows[{i}]: отсутствует ожидаемая строка в нужном порядке: {json.dumps(row, ensure_ascii=False)}")
            cursor += 1
    else:
        # Взаимно однозначное сопоставление сохраняет дубликаты и корректно
        # работает с пересекающимися числовыми допусками, в отличие от greedy.
        assigned = {}
        def match(i, seen):
            for j in range(len(actual_rows)):
                if j in seen or not matches(i, j):
                    continue
                seen.add(j)
                if j not in assigned or match(assigned[j], seen):
                    assigned[j] = i
                    return True
            return False
        for i, row in enumerate(gold_rows):
            if not match(i, set()):
                errors.append(f"rows[{i}]: нет совпадения для {json.dumps(row, ensure_ascii=False)}")
    return errors
