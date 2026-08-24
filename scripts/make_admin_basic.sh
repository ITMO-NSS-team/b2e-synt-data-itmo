#!/bin/sh
# Print base64("user:password") for ADMIN_BASIC in deploy/.env.
# Usage: scripts/make_admin_basic.sh [user] [password]
#        ADMIN_USER=… ADMIN_PASSWORD=… scripts/make_admin_basic.sh
set -eu
user="${1:-${ADMIN_USER:?set ADMIN_USER or pass it as argv 1}}"
pass="${2:-${ADMIN_PASSWORD:?set ADMIN_PASSWORD or pass it as argv 2}}"
printf '%s' "$user:$pass" | openssl base64 | tr -d '\n'
echo
