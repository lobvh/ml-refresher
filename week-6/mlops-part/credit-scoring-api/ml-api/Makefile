all: install format check-style build

install: requirements.txt
	pip install -r requirements.txt

format: install
	black *.py

check-style: format .flake8
	flake8 *.py ./streamlit-app/src/data_preprocessing_refactored.py

build: check-style Dockerfile Dockerfile.streamlit
	docker compose up -d --build

down:
	docker compose down

kill: down
	docker compose kill
