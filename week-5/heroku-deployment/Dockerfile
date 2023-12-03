FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Procfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict-customer.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

CMD gunicorn predict-customer:app
