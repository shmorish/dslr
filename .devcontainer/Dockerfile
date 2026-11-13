FROM mcr.microsoft.com/devcontainers/base:bookworm
# same uv as CI (.github/workflows/ci.yml); Python comes from .python-version via uv
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/
