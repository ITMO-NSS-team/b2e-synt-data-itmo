"""Admin UI security primitives.

These are not hygiene. The threat model records a working attack chain that ends
"the payload issues the approve POST with the approver's cookies, and the human
made no real decision". The controls that break that chain are exactly:

* **autoescaping** — skill descriptions and markdown are attacker-influenced text
  (``heimdall``'s ``render.py`` emits them unescaped), so the UI must never
  render them as HTML;
* **a strict CSP** — no inline script, no external origin, so an injected tag
  cannot execute even if escaping were bypassed;
* **CSRF tokens on every mutation** — so a cross-origin or injected request
  cannot act with the approver's session;
* **showing the exact bytes** being approved, with their digest.

Authentication is HTTP Basic here *and* at the reverse proxy. Defence in depth
is deliberate: the proxy is the thing exposed to the internet, and a
misconfiguration there should not leave the approve button open.
"""
from __future__ import annotations

import hmac
import os
import secrets
from hashlib import sha256

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

CSRF_COOKIE = "b2e_csrf"
CSRF_FIELD = "csrf_token"

#: No inline script, no remote anything. The UI ships its own CSS inline via a
#: nonce-free style-src 'self' and uses no JavaScript at all, which makes this
#: policy trivially satisfiable and an injected <script> inert.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; "
    "style-src 'unsafe-inline'; "
    "form-action 'self'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)

#: The one page that needs script, and what still holds it in.
#:
#: The trace explorer is a client-side renderer: filtering, the timeline and the
#: span tree are all JavaScript, and it is embedded inline because the page is
#: also produced as a single standalone file by ``build_viewer.py``. Under the
#: policy above it renders nothing at all.
#:
#: So this policy relaxes exactly one directive and nothing else. There is still
#: no ``connect-src``, no ``img-src``, no remote origin of any kind: the page
#: cannot fetch, cannot beacon, and cannot load a script it did not ship with.
#:
#: The risk that matters is that the payload is **agent-authored text** —
#: reasoning, tool results, whatever came back from Heimdall — and under
#: ``'unsafe-inline'`` a payload that escapes its container is executable. Two
#: things prevent that, and both are tested:
#:
#: * the data rides in ``<script type="application/json">``, which is not
#:   executed and is read with ``JSON.parse``;
#: * ``sim.traceview.render`` escapes ``</`` to ``<\/`` before embedding, so the
#:   literal ``</script`` that would close the element cannot occur. It is a
#:   legal JSON escape, so nothing about the data changes.
#:
#: Everything the renderer writes into the DOM goes through its own ``esc()``.
EXPLORER_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; "
    "style-src 'unsafe-inline'; "
    "script-src 'unsafe-inline'; "
    "form-action 'self'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)

#: Paths served with the relaxed policy. Matched exactly rather than by prefix:
#: a prefix rule is how one exception becomes a general one.
EXPLORER_PATHS = frozenset({"/explorer", "/admin/explorer"})

SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


def headers_for(path: str) -> dict[str, str]:
    """Security headers for one request path."""
    headers = dict(SECURITY_HEADERS)
    if path.rstrip("/") in EXPLORER_PATHS:
        headers["Content-Security-Policy"] = EXPLORER_CONTENT_SECURITY_POLICY
    return headers

_basic = HTTPBasic(auto_error=False)


def _expected_credentials() -> tuple[str, str]:
    user = os.environ.get("ADMIN_USER", "")
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not user or not password:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_USER and ADMIN_PASSWORD are unset. The admin UI holds "
                   "the approval button; it does not run without credentials.",
        )
    return user, password


def require_admin(request: Request) -> str:
    """HTTP Basic, constant-time compared."""
    import base64

    header = request.headers.get("authorization", "")
    user, password = _expected_credentials()
    if not header.lower().startswith("basic "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": 'Basic realm="b2e-admin"'})
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        got_user, _, got_password = decoded.partition(":")
    except Exception:
        raise HTTPException(status_code=401, detail="malformed credentials",
                            headers={"WWW-Authenticate": 'Basic realm="b2e-admin"'})

    ok = (hmac.compare_digest(got_user, user)
          and hmac.compare_digest(got_password, password))
    if not ok:
        raise HTTPException(status_code=401, detail="invalid credentials",
                            headers={"WWW-Authenticate": 'Basic realm="b2e-admin"'})
    return got_user


def issue_csrf() -> str:
    return secrets.token_urlsafe(32)


def csrf_digest(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def verify_csrf(request: Request, submitted: str | None) -> None:
    """Double-submit: the cookie and the form field must match.

    Rejecting a missing token rather than treating it as legacy is the point —
    an injected form that cannot read the cookie cannot forge the field.
    """
    cookie = request.cookies.get(CSRF_COOKIE)
    if not cookie or not submitted or not hmac.compare_digest(cookie, submitted):
        raise HTTPException(
            status_code=403,
            detail="CSRF check failed. Every mutation must carry the token from "
                   "the form that rendered it.")
