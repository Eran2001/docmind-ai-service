FROM python:3.12-slim
# Set to true to add LibreOffice, which is only needed to read old .doc files (adds ~500 MB).
ARG WITH_LIBREOFFICE=false
RUN if [ "$WITH_LIBREOFFICE" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends libreoffice-writer \
      && rm -rf /var/lib/apt/lists/*; \
    fi
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
RUN useradd --create-home app
USER app
EXPOSE 8000
# Internal only: the compose file does not publish this port.
CMD ["uvicorn", "docmind_ai.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
