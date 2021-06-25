FROM python:3.8.2-slim-buster AS build

RUN apt-get update && \
    apt-get install -y -q build-essential git curl

COPY requirements.txt requirements.txt
RUN pip3 install --user -r requirements.txt

COPY . ./
RUN printf '{"name": "hydro-stat", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json


FROM python:3.8.2-slim-buster

RUN useradd --create-home --shell /bin/bash app
USER app

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

ENV PATH=/home/app/.local/bin:$PATH

ENV DEBUG_ENV=False
ENV HTTP_PORT=5000
ENV STAT_DB_NAME=hydrostat

EXPOSE ${HTTP_PORT}

HEALTHCHECK --start-period=10s CMD curl http://localhost:5000/stat/health

COPY --from=build --chown=app:app /root/.local /home/app/.local
COPY --from=build --chown=app:app buildinfo.json app/buildinfo.json
COPY --chown=app:app app /app

WORKDIR /app

CMD python app.py
