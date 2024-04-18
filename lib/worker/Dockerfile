FROM python:3.9-slim

WORKDIR /usr/src/app

COPY requirements/ requirements/

RUN pip install --no-cache-dir -r requirements/worker-requirements.txt

COPY lib/ lib/
COPY data/iam/ data/iam/

ENV PYTHONPATH=/usr/src/app

ENTRYPOINT [ "python", "lib/integrated_single_scenario.py" ]

CMD ["--stride", "0"]

