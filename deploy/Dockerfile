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


# ---------------------------------------------------------------- agent image
#
# Only b2e-agent needs the Claude Code CLI, and it costs a Node runtime. A
# separate target keeps the emulator, research-api, admin-ui and sandbox on the
# small base rather than paying for Node five times over.
FROM base AS agent

USER root

# Node from the distro rather than a NodeSource pin: this image is rebuilt
# rarely and the CLI is what actually matters here, not the Node minor version.
RUN apt-get update \
 && apt-get install -y --no-install-recommends nodejs npm ca-certificates \
 && npm install -g @anthropic-ai/claude-code@2.1.220 \
 && npm cache clean --force \
 && apt-get purge -y npm \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# The skill runner: the single binary the agent's Bash allowlist permits.
COPY --chmod=0755 deploy/skills-run /opt/skills/run

# Claude Code writes config and caches under HOME; uid 10001 must own one.
RUN mkdir -p /home/b2e/.claude && chown -R b2e:b2e /home/b2e

USER b2e
ENV HOME=/home/b2e

# Fail the build rather than discover at runtime that the CLI is missing.
RUN claude --version
