#!/bin/bash

docker compose build opensearch_fastapi
docker compose push opensearch_fastapi
