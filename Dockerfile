FROM python:3.7.4-slim-stretch

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

ENV AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
ENV AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
ENV DEBUG_ENV=False
ENV HTTP_UI_ADDRESS=http://managerui:80
ENV S3_ENDPOINT=http://minio

ENV MONGO_URL=mongodb
ENV MONGO_PORT=27017
ENV STAT_DB_NAME=hydrostat

EXPOSE 5000

HEALTHCHECK CMD curl http://localhost:5000/stat/health

RUN apt-get update && \
    apt-get install -y -q build-essential git curl \
                          libatlas-base-dev libatlas3-base


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /app
WORKDIR /app

CMD python app.py
