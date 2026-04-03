# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Solar Panel Optimization RL Environment
#
# Builds a lightweight container that runs the Gradio demo on port 7860.
# Compatible with local Docker and Hugging Face Spaces (CPU Basic, free tier).
#
# Build:  docker build -t solar-rl-env .
# Run:    docker run -p 7860:7860 solar-rl-env
# Open:   http://localhost:7860
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage: base image ────────────────────────────────────────────────────────
# python:3.10-slim is ~120MB vs ~900MB for the full image.
# "slim" has pip and the stdlib but skips compilers, docs, and test suites —
# everything this project needs, nothing it doesn't.
FROM python:3.10-slim

# ── System deps ──────────────────────────────────────────────────────────────
# libgomp1  : OpenMP runtime required by some numpy/matplotlib builds on slim.
# --no-install-recommends : skip suggested packages (man pages, etc.)
# Combine into one RUN layer and clean apt cache in the same step to keep
# the layer small — splitting into two RUN commands would save nothing because
# Docker caches each layer separately, but the apt cache would persist.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
# All subsequent COPY, RUN, and CMD instructions operate relative to /app.
# Hugging Face Spaces also expects the app to live in /app by convention.
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements.txt BEFORE the rest of the source code.
# Docker layer caching means: if requirements.txt hasn't changed, this
# expensive pip install step is skipped on rebuilds — saving ~2 minutes.
COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "jinja2==3.0.3" "markupsafe==2.1.5"

# ── Copy project source ───────────────────────────────────────────────────────
# Copied AFTER pip install so editing source files doesn't bust the
# dependency cache (the most expensive layer stays cached).
COPY . .

# Install the project as a package so `env` is importable from anywhere
# regardless of working directory or sys.path tricks.
RUN pip install --no-cache-dir -e .

# ── Environment variables ─────────────────────────────────────────────────────
# Tell Python not to write .pyc files (keeps container filesystem clean)
# and not to buffer stdout/stderr (logs appear in real-time in Docker/HF logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── Port ─────────────────────────────────────────────────────────────────────
# Gradio's default port. EXPOSE is documentation — it doesn't publish the port;
# that's done with `docker run -p 7860:7860`. Hugging Face Spaces reads this
# to know which port to proxy traffic to.
EXPOSE 7860

# ── Launch command ────────────────────────────────────────────────────────────
# Use CMD (not ENTRYPOINT) so it can be overridden at runtime, e.g.:
#   docker run solar-rl-env python baseline/baseline.py --task hard
#
# server_name="0.0.0.0" is set here as an env var so app.py can read it.
# This overrides the 127.0.0.1 used for local dev without editing app.py.
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app/app.py"]