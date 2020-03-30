FROM imadeddinebek/metric_eval:0.4

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

#RUN git clone https://github.com/Hydrospheredata/hydro-serving-sdk.git
#WORKDIR hydro-serving-sdk/
#RUN ls ./hydrosdk/
#RUN python ./setup.py install
#WORKDIR ..

COPY . /app


EXPOSE 5000
CMD python app.py
