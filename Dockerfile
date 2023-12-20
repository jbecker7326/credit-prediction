FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN python3 -m pip install --upgrade pip setuptools wheel                                                                                                                                                                                                
RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]