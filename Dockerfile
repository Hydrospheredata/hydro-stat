FROM python:3.8.2-slim-buster as base
ENV POETRY_PATH=/opt/poetry \
    VENV_PATH=/opt/venv \
    POETRY_VERSION=1.1.6
ENV PATH="$POETRY_PATH/bin:$VENV_PATH/bin:$PATH"

FROM base AS build

RUN apt-get update && \
    apt-get install -y -q build-essential \
    git \ 
    curl

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
RUN mv /root/.poetry $POETRY_PATH
RUN python -m venv $VENV_PATH
RUN poetry config virtualenvs.create false
RUN poetry config experimental.new-installer false

COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-ansi -vvv


COPY version version
COPY .git/ ./.git/
RUN printf '{"name": "hydro-stat", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json


FROM base as runtime

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

ENV DEBUG_ENV=False
ENV HTTP_PORT=5000
ENV STAT_DB_NAME=hydrostat

EXPOSE ${HTTP_PORT}

HEALTHCHECK --start-period=10s CMD curl http://localhost:5000/stat/health

COPY --from=build --chown=app:app buildinfo.json /buildinfo.json

COPY --from=build $VENV_PATH $VENV_PATH
COPY . ./

CMD python hydro_stat/app.py
