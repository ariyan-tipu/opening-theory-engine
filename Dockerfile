# Serving container.
#
# The image ships the API plus a pinned, checksum-verified Stockfish and
# defaults to the stub explainer, which is what CI smoke-tests. To serve
# the real model, install the train extra at build time or mount adapters
# and point the OTE_* variables at them (see README, Serving section).

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml LICENSE README.md NOTICE.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && python -c "from pathlib import Path; from opening_theory_engine.engine.fetch import fetch; fetch(Path('/opt/stockfish'), quiet=True)"

ENV STOCKFISH_PATH=/opt/stockfish/stockfish \
    OTE_BACKEND=stub \
    OTE_ENGINE=stockfish

RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "--factory", "opening_theory_engine.serving.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
