"""Отправить query + gold_contract в последнюю сессию employee_id из кейса."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import ssl
import socket
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPSHandler, ProxyHandler, HTTPRedirectHandler


from case_loader import load_case
from answer_verification import comparison_options, verify_answer

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def build_payload(path: Path | dict) -> dict[str, str]:
    case = path if isinstance(path, dict) else load_case(path)
    if not isinstance(case, dict):
        raise ValueError("Кейс должен быть JSON-объектом.")
    query = case.get("query")
    contract = case.get("gold_contract")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("В кейсе нужен непустой query.")
    if not isinstance(contract, dict):
        raise ValueError("gold_contract должен быть объектом JSON Schema.")
    return {
        "content": (
            query.strip()
            + "\n\nВерни итоговый ответ только в виде JSON, соответствующего "
            "следующей JSON Schema. Не добавляй Markdown или текст вне JSON. "
            + "\n\nJSON Schema:\n"
            + json.dumps(contract, ensure_ascii=False, indent=2)
        )
    }


def request_json(opener, url: str, headers: dict, timeout: float,
                 payload: dict | None = None) -> dict:
    request = Request(
        url, headers=headers,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="GET" if payload is None else "POST",
    )
    # Не повторять POST автоматически: сервер мог уже создать сессию.
    with opener.open(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("API сессий вернул не JSON-объект.")
    return result


def resolve_session_or_create(opener, sessions_url: str, headers: dict,
                    timeout: float, employee_id: str) -> str:
    result = request_json(
        opener, sessions_url + "?" + urlencode({"employee_id": employee_id, "limit": 1}),
        headers, timeout,
    )
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("В ответе GET /sessions отсутствует массив sessions.")
    if sessions:
        # API сортирует по created_at DESC, limit=1 возвращает самую новую.
        session = sessions[0]
        if not isinstance(session, dict) or str(session.get("employee_id")) != employee_id:
            raise ValueError("API вернул сессию другого employee_id.")
        session_id = session.get("id")
    else:
        session = request_json(opener, sessions_url, headers, timeout,
                               {"employee_id": employee_id, "config_ref": "agent_config"})
        if str(session.get("employee_id")) != employee_id:
            raise ValueError("Созданная сессия не соответствует employee_id кейса.")
        session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("API не вернул идентификатор сессии.")
    return session_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="Путь к JSON-кейсу относительно benchmark директории")
    parser.add_argument("--base-url", default="https://10.32.1.71:8443")
    parser.add_argument("--api-prefix", default="/agent", help="Префикс API (по умолчанию /agent)")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).parent / ".env")
    parser.add_argument("--user", default="researcher", help="Пользователь Basic Auth")
    parser.add_argument("--host", help="Значение заголовка Host, если его требует Caddy")
    parser.add_argument("--timeout", type=float, default=600, help="Тайм-аут ответа в секундах")
    parser.add_argument("--insecure", action="store_true", help="Отключить проверку TLS для локального сертификата")
    parser.add_argument("--dry-run", action="store_true", help="Показать тело запроса без отправки")
    parser.add_argument("--verify", action="store_true", help="Сверить поле answer ответа с gold_answer кейса")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        case = load_case(args.case)
        payload = build_payload(case)
        if args.verify:
            if not isinstance(case.get("gold_answer"), dict):
                raise ValueError("Для --verify в кейсе нужен gold_answer — JSON-объект.")
            comparison_options(case)
        employee_id = case.get("employee_id")
        if isinstance(employee_id, bool) or not isinstance(employee_id, (str, int)) or not str(employee_id).strip():
            raise ValueError("В кейсе нужен employee_id — непустая строка или целое число.")
        employee_id = str(employee_id).strip()
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if args.verify:
                print("VERIFY: пропущена в режиме --dry-run.", file=sys.stderr)
            return 0
        env = read_env(args.env_file)
        password = os.environ.get("RESEARCHER_PASSWORD") or env.get("RESEARCHER_PASSWORD")
        host = args.host or os.environ.get("PUBLIC_HOST") or env.get("PUBLIC_HOST")
        headers = {"Host": host} if host else {}
        headers["Content-Type"] = "application/json; charset=utf-8"
        if password:
            credentials = base64.b64encode(f"{args.user}:{password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = "Basic " + credentials
        sessions_url = (
            args.base_url.rstrip("/") + "/" + args.api_prefix.strip("/")
            + "/sessions"
        )
        # Без повторных POST: при тайм-ауте обработка сообщения могла уже начаться.
        context = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
        opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context), NoRedirect())
        session_id = resolve_session_or_create(opener, sessions_url, headers, args.timeout, employee_id)
        print(f"employee_id={employee_id}, session_id={session_id}", file=sys.stderr)
        url = sessions_url + "/" + quote(session_id, safe="") + "/messages"
        request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            response = opener.open(request, timeout=args.timeout)
        except HTTPError as exc:
            response = exc
        with response:
            status = response.code
            body = response.read().decode("utf-8", errors="replace")
        try:
            print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
        except ValueError:
            print(body)
        if status >= 300:
            print(f"HTTP {status}", file=sys.stderr)
            if status == 401:
                print("Задайте RESEARCHER_PASSWORD в окружении или .env.", file=sys.stderr)
            return 1
        
        #---------verify answer-------------
        if args.verify:
            try:
                result = json.loads(body)
            except ValueError:
                result = None
            if not isinstance(result, dict) or "answer" not in result:
                errors = ["Ответ API не является JSON-объектом с полем answer."]
            else:
                errors = verify_answer(case, result["answer"])
            print("VERIFY: FAIL" if errors else "VERIFY: PASS", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
            return 2 if errors else 0
        return 0
    
    
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        if exc.code == 401:
            print("Задайте RESEARCHER_PASSWORD в окружении или .env.", file=sys.stderr)
        return 1
    except (TimeoutError, socket.timeout):
        print("Тайм-аут. Сообщение могло быть принято: проверьте сессию перед повторной отправкой.", file=sys.stderr)
        return 1
    except (OSError, ValueError, URLError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
