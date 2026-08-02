"""Локальный эмулятор Heimdall: три MCP-канала, REST-зеркало, dev-ручки.

Почему эмулятор, а не библиотека
--------------------------------
Топология совпадает с боевой: агент говорит с сетевым сервисом, получает
настоящие HTTP-статусы и настоящие коды ошибок в конверте ``{code, detail,
hint}``. Появляется место для ``POST /api/v2/dev/add_skill_for_debug/`` — той
самой ручки, на которой замыкается конвейер рецептов: «сгенерировал → залил →
получил 201 или 422».

Почему локальный, а не стенд
----------------------------
Герметичность прогона в этом проекте держится на ``manifest_hash`` от хэшей
смонтированных файлов. Удалённый стенд хэшировать нечем — и весь оффлайн-контур
воспроизводимости (кэш прогонов, фикстуры replay) рассыпался бы. Локальный
эмулятор поверх колоночного снимка сохраняет всё: снимок детерминирован, его
``snapshot_id`` входит в манифест, а данные при этом бесплатны.

Каналы различаются заголовком ``x-heimdall-mcp-version`` — как в бою, где его
проставляет собственный HTTP-клиент канала.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, File, Header, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from .catalog import Catalog, Model
from .engine.budget import Budget
from .engine.compile import API_MAX_LIMIT
from .engine.errors import HeimdallError, fail
from .engine.execute import execute
from .engine.quirks import Quirks
from .skills import Registry
from .store import Snapshot

#: Канал по умолчанию. Скиллы есть только здесь.
DEFAULT_CHANNEL = "v2"
CHANNELS = ("v1", "v2", "orion")


def create_app(snapshot_root: str | Path, catalog_path: str | Path = "catalog/snapshot.json",
               skills_root: str | Path = "heimdall-skills",
               quirks: Quirks | None = None,
               budget: Budget | None = None) -> FastAPI:
    """Собрать приложение эмулятора."""
    catalog = Catalog.load(catalog_path)
    # Хвост каталога не материализован: снимок отдаёт такие колонки процедурно.
    # Если пакет генератора недоступен, поведение прежнее — колонка из NULL.
    try:
        from b2e.store import ProceduralSnapshot
        snapshot = ProceduralSnapshot(snapshot_root, catalog)
    except Exception:                                    # pragma: no cover
        snapshot = Snapshot(snapshot_root)
    skills_root = Path(skills_root)
    state = _State(catalog=catalog, snapshot=snapshot, skills_root=skills_root,
                   quirks=quirks or Quirks(), budget=budget or Budget())

    app = FastAPI(title="Heimdall Sandbox", version="01.002.00")
    app.state.heimdall = state
    _install_error_handler(app)
    app.include_router(_mcp_v1_router(state))
    app.include_router(_mcp_v2_router(state))
    app.include_router(_rest_router(state))
    app.include_router(_dev_router(state))
    return app


class _State:
    """Состояние эмулятора: каталог, данные, каталог скиллов с ленивым кэшем."""

    def __init__(self, catalog: Catalog, snapshot: Snapshot, skills_root: Path,
                 quirks: Quirks, budget: Budget | None = None) -> None:
        self.catalog = catalog
        self.snapshot = snapshot
        self.skills_root = skills_root
        self.quirks = quirks
        self.budget = budget or Budget()
        self._registry: Registry | None = None

    @property
    def registry(self) -> Registry:
        if self._registry is None:
            self._registry = Registry.load(self.skills_root)
        return self._registry

    def invalidate(self) -> None:
        """Сбросить кэш каталога скиллов — как делает DELETE у dev-ручки."""
        self._registry = None

    def model(self, schema: str, logic_model: str, channel: str) -> Model:
        model = self.catalog.get(schema, logic_model)
        if model is None or channel not in model.channels:
            raise fail("model-not-found",
                       f"модели {schema}.{logic_model} нет в каталоге канала {channel}")
        return model

    def visible(self, channel: str) -> list[Model]:
        return sorted(self.catalog.visible_in(channel), key=lambda m: m.key)


# ------------------------------------------------------------------ ошибки

def _install_error_handler(app: FastAPI) -> None:
    @app.exception_handler(HeimdallError)
    async def _handle(_: Request, exc: HeimdallError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=exc.envelope())


def _require_bearer(authorization: str | None = Header(default=None)) -> str:
    """Канал авторизуется технической учётной записью.

    Без валидного токена инструменты отвечают 403 с телом {"code": "forbidden"}.
    Значение токена эмулятор не проверяет: он моделирует контур, а не выдачу.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise fail("forbidden",
                   "нет заголовка Authorization: Bearer <JWT> технической учётной записи")
    return authorization.split(" ", 1)[1]


def _channel(x_heimdall_mcp_version: str | None = Header(default=None)) -> str:
    value = (x_heimdall_mcp_version or DEFAULT_CHANNEL).lower()
    return value if value in CHANNELS else DEFAULT_CHANNEL


# ------------------------------------------------------------------- MCP v1

def _mcp_v1_router(state: _State) -> APIRouter:
    router = APIRouter(prefix="/api/v1/mcp", tags=["MCP"])

    @router.get("/models/")
    def list_models(_: str = Depends(_require_bearer),
                    channel: str = Depends(_channel)) -> list[dict]:
        return [{"schema": m.schema, "logic_model": m.logic_model,
                 "description": m.summary, "is_history": m.is_history,
                 "column_count": len(m.columns), "metric_count": len(m.metrics)}
                for m in state.visible(channel)]

    @router.get("/models/{schema}/{logic_model}/")
    def describe_model(schema: str, logic_model: str,
                       _: str = Depends(_require_bearer),
                       channel: str = Depends(_channel)) -> dict:
        return _describe(state.model(schema, logic_model, channel))

    @router.post("/query/")
    def mcp_query(body: dict, _: str = Depends(_require_bearer),
                  channel: str = Depends(_channel)) -> dict:
        schema, logic_model = body.get("schema"), body.get("logic_model")
        if not schema or not logic_model:
            raise fail("request-validation-error",
                       "поля schema и logic_model обязательны")
        model = state.model(schema, logic_model, channel)
        payload = {k: v for k, v in body.items() if k not in ("schema", "logic_model")}
        payload["schema"], payload["logic_model"] = schema, logic_model
        # raw_sql через MCP недоступен: ?query_type=raw объявлен только у REST.
        return _run(state, model, payload, query_type=None)

    @router.get("/docs/")
    def get_docs(topic: str = "", _: str = Depends(_require_bearer)) -> dict:
        registry = state.registry
        if topic and topic in registry.all_names():
            return {"topic": topic, "markdown": registry.get(topic).markdown}
        return {"topic": topic,
                "markdown": "Доступные темы: " + ", ".join(registry.active())}

    return router


def _describe(model: Model) -> dict:
    """Ответ describe_model. Только здесь видно поле parameters."""
    def member(m) -> dict:
        out: dict[str, Any] = {"name": m.name, "type": m.type,
                               "description": m.description}
        if m.parameters:
            out["parameters"] = m.parameters
        return out

    return {
        "schema": model.schema,
        "logic_model": model.logic_model,
        "description": model.summary,
        "is_history": model.is_history,
        "columns": [member(m) for m in model.columns.values()],
        "metrics": [member(m) for m in model.metrics.values()],
        "time_dimensions": [{"name": t.name, "type": "Date32",
                             "description": "Ось истории",
                             "granularities": t.granularities}
                            for t in model.time_dimensions.values()],
        "notes": {
            "filters": "Дерево узлов с полем type; ровно один корень; лишние поля запрещены.",
            "limits": f"limit по умолчанию 100, потолок канала {API_MAX_LIMIT}.",
            "modes": "Режим выводится из структуры тела, отдельного поля нет.",
        },
    }


def _run(state: _State, model: Model, body: dict, query_type: str | None) -> dict:
    from .engine.metrics import DEFAULT_REGISTRY
    reader = _reader(state, model)
    return execute(body, model, reader, state.quirks, query_type=query_type,
                   metrics_registry=DEFAULT_REGISTRY, budget=state.budget)


def _reader(state: _State, model: Model):
    """Читатель витрины; отсутствующая в снимке витрина читается как пустая."""
    if state.snapshot.has(model.key):
        return state.snapshot.table(model.key)
    return _EmptyReader()


class _EmptyReader:
    rows = 0

    def column(self, name: str) -> list:
        return []


# ------------------------------------------------------------------- MCP v2

def _mcp_v2_router(state: _State) -> APIRouter:
    router = APIRouter(prefix="/api/v2/mcp", tags=["MCP v2"])

    def _only_v2(channel: str = Depends(_channel)) -> str:
        # Скиллы существуют только в канале v2. Рецепт, до которого агент
        # добрался бы из v1, там просто не существует — и это надо показать.
        if channel != "v2":
            raise fail("skill-not-found",
                       f"каталог скиллов доступен только в канале v2, запрошен {channel}")
        return channel

    @router.get("/overview/")
    def get_overview(_: str = Depends(_require_bearer),
                     __: str = Depends(_only_v2)) -> dict:
        return state.registry.overview()

    @router.get("/skills/")
    def find_skills(query: str = Query(...), domain: str = "", kind: str = "",
                    limit: int = 10, offset: int = 0, include_deprecated: bool = False,
                    _: str = Depends(_require_bearer),
                    __: str = Depends(_only_v2)) -> dict:
        if kind and kind not in ("recipe", "reference"):
            raise fail("request-validation-error", "kind ∈ {recipe, reference}")
        return state.registry.find(query=query, domain=domain or None,
                                   kind=kind or None, limit=limit, offset=offset,
                                   include_deprecated=include_deprecated)

    @router.get("/skills/{name}/")
    def get_skill(name: str, _: str = Depends(_require_bearer),
                  __: str = Depends(_only_v2)) -> dict:
        try:
            return state.registry.get(name).as_detail()
        except KeyError:
            raise fail("skill-not-found",
                       f"скилла {name} нет; найди подходящий через find_skills")

    return router


# -------------------------------------------------------------- REST-зеркало

def _rest_router(state: _State) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["REST"])

    @router.post("/{schema}/{logic_model}/")
    def query_model(schema: str, logic_model: str, body: dict,
                    query_type: str | None = None,
                    _: str = Depends(_require_bearer),
                    channel: str = Depends(_channel)) -> dict:
        model = state.model(schema, logic_model, channel)
        # У REST limit объявлен с maximum: 1000 — тело отвергается, а не ужимается.
        # Это единственное место, где REST и MCP расходятся по поведению.
        limit = body.get("limit")
        if isinstance(limit, int) and limit > API_MAX_LIMIT:
            raise fail("request-validation-error",
                       f"limit {limit} больше потолка {API_MAX_LIMIT}",
                       errors=[{"loc": ["body", "limit"],
                                "msg": f"ensure this value is less than or equal to "
                                       f"{API_MAX_LIMIT}",
                                "type": "value_error.number.not_le"}])
        payload = dict(body)
        payload["schema"], payload["logic_model"] = schema, logic_model
        return _run(state, model, payload, query_type=query_type)

    return router


# ---------------------------------------------------------------- dev-ручки

def _dev_router(state: _State) -> APIRouter:
    """Ручки заливки скиллов. Существуют только на dev — там же, где Swagger."""
    router = APIRouter(prefix="/api/v2/dev", tags=["DEV v2"])

    @router.get("/skills")
    def list_files() -> dict:
        root = state.skills_root
        return {"files": [{"path": str(f.path.relative_to(root)), "ok": f.ok,
                           "name": f.name, "error": f.error}
                          for f in state.registry.files()]}

    @router.get("/skills/raw")
    def raw(path: str) -> dict:
        target = _safe_path(state.skills_root, path)
        if not target.exists():
            raise fail("skill-not-found", f"файла {path} нет")
        return {"path": path, "text": target.read_text(encoding="utf-8")}

    @router.post("/add_skill_for_debug/", status_code=201)
    def add_skill(skill_file: UploadFile = File(...)) -> dict:
        text = skill_file.file.read().decode("utf-8")
        suffix = Path(skill_file.filename or "skill.yaml").suffix or ".yaml"
        return _save(state, text, suffix)

    @router.post("/skills/save", status_code=201)
    def save_skill(payload: dict) -> dict:
        filename = payload.get("filename") or "skill.yaml"
        text = payload.get("text") or ""
        return _save(state, text, Path(filename).suffix or ".yaml")

    @router.delete("/skills")
    def delete_skill(path: str) -> dict:
        target = _safe_path(state.skills_root, path)
        if target.exists():
            target.unlink()
        state.invalidate()
        return {"deleted": path}

    return router


def _safe_path(root: Path, relative: str) -> Path:
    """Не дать выйти за корень каталога скиллов."""
    target = (root / relative).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise fail("request-validation-error", "путь выходит за каталог скиллов")
    return target


def _save(state: _State, text: str, suffix: str) -> dict:
    """Проверить файл целиком и записать. При отказе на диск не пишется ничего.

    Имя загруженного файла игнорируется: домен и имя берутся из полей внутри —
    ровно так же, как это делает сервис.
    """
    import tempfile

    from .skills.registry import _parse

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / f"probe{suffix}"
        probe.write_text(text, encoding="utf-8")
        try:
            skill = _parse(probe)
        except Exception as exc:  # noqa: BLE001 — любой отказ разбора
            # Спека дословно: «при ошибке возвращается код 422, на диск ничего
            # не пишется». Проверка идёт на копии во временном каталоге именно
            # ради второй половины этой фразы.
            raise fail("request-validation-error",
                       f"файл не прошёл проверку: {exc}", status=422,
                       errors=[{"loc": ["skill_file"], "msg": str(exc)[:300],
                                "type": "value_error"}])

    target = state.skills_root / skill.domain / f"{skill.name}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    state.invalidate()
    return {"path": str(target.relative_to(state.skills_root)),
            "name": skill.name, "domain": skill.domain, "kind": skill.kind}
