# syntax=docker/dockerfile:1
FROM python:3.8.11-slim as base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_PATH=/opt/poetry \
    VENV_PATH=/opt/venv \
    POETRY_VERSION=1.1.6 
ENV PATH="$POETRY_PATH/bin:$VENV_PATH/bin:$PATH"


FROM base AS build

# # non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
# ENV DEBIAN_FRONTEND=noninteractive \
#     DEBCONF_NONINTERACTIVE_SEEN=true \
#     UCF_FORCE_CONFOLD=1

RUN apt-get update && \
    apt-get install -y -q build-essential \
    curl \
    git && \ 
    curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python && \
    mv /root/.poetry $POETRY_PATH && \
    python -m venv $VENV_PATH && \
    poetry config virtualenvs.create false && \
    poetry config experimental.new-installer false && \
    rm -rf /var/lib/apt/lists/*

COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-ansi -vvv

COPY . ./
ARG GIT_HEAD_COMMIT
ARG GIT_CURRENT_BRANCH

RUN if [ -z "$GIT_HEAD_COMMIT" ] ; then \
    printf '{"name": "hydro-stat", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json ; else \
    printf '{"name": "hydro-stat", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$GIT_HEAD_COMMIT" "$GIT_CURRENT_BRANCH" "$(python --version)" >> buildinfo.json ; \
    fi

FROM base as runtime

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app

ENV DEBUG_ENV=False \
    HTTP_PORT=5000 \
    STAT_DB_NAME=hydrostat

EXPOSE ${HTTP_PORT}

HEALTHCHECK --start-period=10s CMD curl http://localhost:${HTTP_PORT}/stat/health

WORKDIR /app

COPY --from=build --chown=app:app buildinfo.json buildinfo.json
COPY --from=build $VENV_PATH $VENV_PATH
COPY --chown=app:app hydro_stat /app/hydro_stat
COPY --chown=app:app tests /app/tests

CMD python -m hydro_stat.app
