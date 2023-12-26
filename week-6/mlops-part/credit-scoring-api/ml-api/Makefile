all: run

install: requirements.txt
	pip install -r requirements.txt

format: install
	black *.py

check-style: format .flake8
	flake8 *.py

build: check-style Dockerfile
	docker build -t credit-score-prediction:0.1 .

run: build
	docker run -it --rm -p 8000:8000 credit-score-prediction:0.1
