FROM python:3.8-slim

RUN apt-get update && apt-get install -y gcc unixodbc-dev
RUN apt-get update && apt-get install -y tk

ENV FLASK_APP manage.py

RUN mkdir detector
WORKDIR /home/detector

COPY requirements requirements
RUN pip install -r requirements/docker.txt
RUN python3 -c "import nltk; nltk.download('punkt');"

COPY app app
COPY src src
COPY manage.py boot.sh ./

# run-time configuration
EXPOSE 5000
ENTRYPOINT ["./boot.sh"]
