#Base Image we need
FROM python:3.12-slim

#Not to create a new Python environment but to install all the dependencies from Pipfile
#After all the base image is Python, so we just have to build on top of that system. 
RUN pip install pipenv

#Create and change the working directory to /app
WORKDIR /app

#Copy files Pipfile and Pipfile.lock inside current directory (that is why the "./" is after those
COPY ["Pipfile", "Pipfile.lock", "./"]

#Run pipenv but install those packages directly to our "system" since it is already isolated via Docker. No need for creating Python environment inside that
RUN pipenv install --system --deploy

#Copy neccessary files to the current directory /app
COPY ["predict-customer.py", "model_C=1.0.bin", "./"]

#In Dockerfile you can only expose the container's port, but can't do port mapping like in Docker Compose
EXPOSE 9696

#This is basically what we were doing when we 'made our Flask built-in server into gunicorn'
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict-customer:app"]
