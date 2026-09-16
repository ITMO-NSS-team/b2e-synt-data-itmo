"""Отправить query + gold_contract из JSON-кейса в существующую сессию агента."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import ssl
import socket
from pathlib import Path
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPSHandler, ProxyHandler, HTTPRedirectHandler

from case_loader import load_case


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


def build_payload(path: Path) -> dict[str, str]:
    case = load_case(path)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="Путь к JSON-кейсу")
    parser.add_argument("--base-url", default="https://10.32.1.71:8443")
    parser.add_argument("--session-id", default="ses_d4070af2371d4b96bef3") #session for employee 3715473 (manager, has team)
    parser.add_argument("--api-prefix", default="/agent", help="Префикс API (по умолчанию /agent)")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).parent / ".env")
    parser.add_argument("--user", default="researcher", help="Пользователь Basic Auth")
    parser.add_argument("--host", help="Значение заголовка Host, если его требует Caddy")
    parser.add_argument("--timeout", type=float, default=600, help="Тайм-аут ответа в секундах")
    parser.add_argument("--insecure", action="store_true", help="Отключить проверку TLS для локального сертификата")
    parser.add_argument("--dry-run", action="store_true", help="Показать тело запроса без отправки")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        payload = build_payload(args.case)
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        env = read_env(args.env_file)
        password = os.environ.get("RESEARCHER_PASSWORD") or env.get("RESEARCHER_PASSWORD")
        host = args.host or os.environ.get("PUBLIC_HOST") or env.get("PUBLIC_HOST")
        headers = {"Host": host} if host else {}
        headers["Content-Type"] = "application/json; charset=utf-8"
        if password:
            credentials = base64.b64encode(f"{args.user}:{password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = "Basic " + credentials
        url = (
            args.base_url.rstrip("/") + "/" + args.api_prefix.strip("/")
            + "/sessions/" + quote(args.session_id, safe="") + "/messages"
        )
        # Без повторных POST: при тайм-ауте обработка сообщения могла уже начаться.
        context = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
        opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context), NoRedirect())
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
        return 0
    except (TimeoutError, socket.timeout):
        print("Тайм-аут. Сообщение могло быть принято: проверьте сессию перед повторной отправкой.", file=sys.stderr)
        return 1
    except (OSError, ValueError, URLError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
