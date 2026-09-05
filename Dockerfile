# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/document-indexer

COPY pyproject.toml README.md LICENSE ./

# Third-party deps only. This layer is reused until pyproject.toml changes.
RUN --mount=type=cache,target=/root/.cache/pip \
    python - <<'PY'
import subprocess
import tomllib
from pathlib import Path

data = tomllib.loads(Path("pyproject.toml").read_text())
deps = list(data["project"]["dependencies"])
optional = data["project"]["optional-dependencies"]
for extra in ("docling", "runtime", "smb"):
    deps.extend(optional[extra])
deps.extend(data["build-system"]["requires"])
subprocess.check_call(["python", "-m", "pip", "install", "--upgrade", "pip"])
# CPU wheels for both x86_64 and aarch64 (DGX Spark / Grace); PyPI is the fallback
# when the CPU index has no wheel for the current platform.
subprocess.check_call(
    [
        "python",
        "-m",
        "pip",
        "install",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
        "--extra-index-url",
        "https://pypi.org/simple",
        "torch",
        "torchvision",
    ]
)
subprocess.check_call(["python", "-m", "pip", "install", *deps])
PY

COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --no-build-isolation .
