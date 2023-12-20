FROM python:3.9.1-slim-buster

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

WORKDIR /app

CMD ["python", "train_model.py"]