FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

COPY ./model ./model

#Install it as part of the system not venv inside of system
RUN pipenv install --system --deploy

COPY ["main.py", "./"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
