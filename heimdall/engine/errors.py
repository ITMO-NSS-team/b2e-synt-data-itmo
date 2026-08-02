"""Таксономия ошибок канала Heimdall.

Реестр собран из трёх источников, потому что ни один по отдельности не полон:

* схема ``ErrorResponse`` и тексты OpenAPI — конверт ``{code, detail, hint}``;
* тексты приёмов ``errors``, ``rules``, ``limits``, ``compare_people``,
  ``history``, ``param_metrics`` — там нашлись коды, которых нет в CLAUDE.md
  (``composite-key-arity-mismatch``, ``composite-key-operator-unsupported``,
  ``array-operator-requires-array-column``);
* поведение, описанное прозой (молчаливое ужатие ``limit`` — НЕ ошибка).

``hint`` обязателен: агент читает именно его, и текстовый градиент фабрики
работает с настоящими формулировками, а не с придуманными.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorSpec:
    code: str
    status: int
    hint: str


_SPECS: tuple[ErrorSpec, ...] = (
    # --- доступ
    ErrorSpec("forbidden", 403,
              "Нет доступа к модели или инструменту. Проверь action_ids токена и "
              "allowed_groups модели."),
    ErrorSpec("auth-failed", 403,
              "Заголовок Authorization отсутствует или токен невалиден. "
              "Формат: Authorization: Bearer <JWT>."),
    # --- каталог
    ErrorSpec("model-not-found", 404,
              "Пары schema + logic_model нет в каталоге. Возьми имена из list_models; "
              "схема system скрыта и недоступна."),
    ErrorSpec("skill-not-found", 404,
              "Скилла с таким name нет. Найди подходящий через find_skills."),
    ErrorSpec("unknown-column", 400,
              "Имя колонки отсутствует в каталоге модели. Возьми точное имя из "
              "describe_model; угадывать нельзя."),
    ErrorSpec("unknown-metric", 400,
              "Имя метрики отсутствует в каталоге модели. Метрики и колонки — разные "
              "списки в describe_model."),
    # --- тело запроса
    ErrorSpec("request-validation-error", 400,
              "Тело не прошло валидацию. Лишние поля запрещены: additionalProperties=false. "
              "Смотри errors[] — там loc, msg и type по каждому полю."),
    ErrorSpec("query-empty", 422,
              "Не заданы ни columns, ни metrics, ни time_dimensions — нечего возвращать. "
              "Режим выводится из структуры тела."),
    ErrorSpec("filter-value-invalid", 400,
              "Значение в фильтре не соответствует типу колонки. Число передавай числом, "
              "булево — булевым, дату строкой YYYY-MM-DD."),
    # --- фильтры
    ErrorSpec("array-operator-requires-array-column", 422,
              "Операторы has/hasAny/hasAll работают только по Array-колонке. "
              "Для скаляра используй condition или condition_in."),
    ErrorSpec("composite-key-operator-unsupported", 422,
              "Для колонки Tuple(...) допустимы только =, !=, IN, NOT IN."),
    ErrorSpec("composite-key-arity-mismatch", 422,
              "Арность кортежа не совпадает с типом колонки. Для emp_key передавай "
              "[company, employee_id]."),
    # --- сортировка
    ErrorSpec("order-by-metric-not-allowed-in-rows", 422,
              "В режиме строк сортировать по метрике нельзя: метрик в выдаче нет. "
              "Добавь metrics (это включит агрегат) или сортируй по колонке."),
    ErrorSpec("order-by-kind-mismatch", 422,
              "Поле order_by.kind не совпадает с классом члена: column для колонки, "
              "metric для метрики."),
    # --- история
    ErrorSpec("column-not-a-time-dimension", 422,
              "Колонка не объявлена time-dimension. Ось истории — report_date, и только "
              "у исторических моделей."),
    ErrorSpec("granularity-not-allowed", 422,
              "Гранулярность не входит в column.granularities. Допустимы day, week_end, "
              "month_end, quarter_end, year_end."),
    ErrorSpec("range-exceeds-lookback", 400,
              "Окно шире допустимой глубины истории. Сузь диапазон [from, to)."),
    ErrorSpec("cross-history-join-forbidden", 400,
              "Соединение исторической модели с неисторической запрещено."),
    ErrorSpec("limit-by-not-supported-in-history", 422,
              "limit_by в режиме истории не поддержан. Убери его или откажись от "
              "time_dimensions."),
    ErrorSpec("array-join-not-supported-in-history", 422,
              "Модель-развёртка (ARRAY JOIN) не поддерживает режим истории."),
    # --- параметрические члены
    ErrorSpec("param-metric-not-supported", 422,
              "Модель не имеет параметрических метрик."),
    ErrorSpec("param-virtual-not-supported", 422,
              "Параметрическую колонку нельзя указывать голым именем в columns — "
              "передавай её в param_columns как {name, args}."),
    ErrorSpec("param-metric-not-supported-in-history", 422,
              "Параметрические метрики в режиме истории не поддержаны."),
    ErrorSpec("param-virtual-not-supported-in-history", 422,
              "Параметрические колонки в режиме истории не поддержаны."),
    ErrorSpec("missing-parameter", 422,
              "В args не передан объявленный параметр члена. Список параметров — "
              "в describe_model, поле parameters."),
    ErrorSpec("unknown-parameter", 422,
              "В args передан параметр, которого у члена нет."),
    ErrorSpec("parameter-type-invalid", 422,
              "Тип значения аргумента не совпадает с объявленным типом параметра."),
    ErrorSpec("parameter-arity-mismatch", 422,
              "Число значений аргумента не совпадает с ожидаемым."),
    ErrorSpec("parameter-too-many-values", 422,
              "Слишком много значений в аргументе параметрического члена."),
    ErrorSpec("parameter-not-literal", 422,
              "Аргумент параметрического члена должен быть литералом, а не выражением."),
    # --- ресурсы
    # ВНИМАНИЕ: этот код смоделирован эмулятором, а не взят из спеки. Боевой
    # сервис поверх ClickHouse на абсурдно широком запросе тоже откажет, но
    # текста его отказа у нас нет — и выдавать выдуманный за документированный
    # нельзя. Обоснование потолка — docs/query-budget.md.
    ErrorSpec("query-too-expensive", 422,
              "Запрос читает слишком много данных. Колонка поднимается целиком, "
              "поэтому стоимость задают число колонок и число строк витрины, а НЕ "
              "limit: уменьшать limit бесполезно. Сократи columns, а также поля в "
              "filters, order_by и limit_by, либо возьми витрину поуже."),
    # --- исполнение
    ErrorSpec("internal-error", 500,
              "Внутренняя ошибка сервиса. Сохрани request_id и повтори позже."),
)

ERRORS: dict[str, ErrorSpec] = {e.code: e for e in _SPECS}


class HeimdallError(Exception):
    """Доменная ошибка канала: конверт ``{code, detail, hint}`` и HTTP-статус."""

    def __init__(self, code: str, detail: str, *, errors: list[dict] | None = None,
                 request_id: str | None = None, status: int | None = None) -> None:
        spec = ERRORS.get(code)
        if spec is None:  # pragma: no cover - защита от опечатки в коде вызова
            raise KeyError(f"неизвестный код ошибки: {code}")
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        # Один и тот же код может приходить с разным HTTP-статусом на разных
        # ручках: спека документирует у моделей 400 «некорректное тело», а у
        # заливки скилла — 422. Статус по умолчанию берётся из реестра,
        # переопределяется только там, где спека говорит иначе.
        self.status = status or spec.status
        self.hint = spec.hint
        self.errors = errors
        self.request_id = request_id

    def envelope(self) -> dict:
        body: dict = {"code": self.code, "detail": self.detail, "hint": self.hint}
        if self.errors:
            body["errors"] = self.errors
        if self.request_id:
            body["request_id"] = self.request_id
        return body


def fail(code: str, detail: str, **kw) -> "HeimdallError":
    """Сахар: ``raise fail("unknown-column", "нет колонки foo")``."""
    return HeimdallError(code, detail, **kw)
