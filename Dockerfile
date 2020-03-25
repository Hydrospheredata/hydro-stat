FROM imadeddinebek/metric_eval:0.3

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/Hydrospheredata/hydro-serving-sdk.git

RUN python hydro-serving-sdk/setup.py install

COPY . /app


EXPOSE 5000
CMD python app.py
