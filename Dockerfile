FROM python:3.7.4-slim-stretch

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential \
                       python3-pip \
                       python3-numpy \
                       python3-scipy \
                       libatlas-dev \
                       libatlas3-base \
                       git



COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/Hydrospheredata/hydro-serving-sdk.git
WORKDIR hydro-serving-sdk/
RUN git checkout 134b50bedeb4187929ede47966a4fb52fcb7483b
RUN python ./setup.py install
WORKDIR ..

COPY . /app
WORKDIR /app


#ENV AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
#ENV AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>


EXPOSE 5000
CMD python app.py
