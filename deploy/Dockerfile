# One image for every Python service. They share ~90% of their dependencies, and
# on a 97 GB disk with 2 vCPU a single cached layer set beats five thin images
# that each rebuild numpy.
#
# The command differs per service in docker-compose.yml.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first, so a source edit does not invalidate the wheel layer.
COPY deploy/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY b2e /app/b2e
COPY heimdall /app/heimdall
COPY sim /app/sim
COPY catalog /app/catalog
COPY scripts /app/scripts

# Every service runs unprivileged. The sandbox worker additionally runs with a
# read-only root filesystem and no network; see docker-compose.yml.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin b2e \
 && mkdir -p /app/var /spool \
 && chown -R b2e:b2e /app/var /spool
USER b2e

EXPOSE 8081 8082 8083 8084
