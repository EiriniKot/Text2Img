FROM python:3.9-slim

EXPOSE 8501

ENV QDRANT_HOST = 'localhost'
ENV PORT = 6333

#COPY dataset dataset
COPY main.py main.py
COPY requirements.txt requirements.txt
COPY tools.py tools.py

RUN pip3 install -r requirements.txt

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
