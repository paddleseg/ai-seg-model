FROM vikings/paddle:paddlehub
COPY setup.py /setup.py
RUN  python3 /setup.py
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN  python3 -m pip install -r requirements.txt
RUN  mkdir -p /data/input && \
    mkdir -p /data/output


ADD . /app

ENTRYPOINT [ "python3","app.py" ]