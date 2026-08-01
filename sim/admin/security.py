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

SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

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
