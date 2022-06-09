FROM python:3.8.13-slim-buster

RUN python3.8 -m pip install poetry

WORKDIR /app

COPY pyproject.toml .
COPY poetry.lock .

SHELL ["/bin/bash", "-c"]

RUN poetry install --no-dev --no-interaction

COPY . .

RUN poetry install --no-interaction && \
    poetry cache clear  pypi --all -n

ENTRYPOINT [ "sh" ]