# Local check of the Docker image

Done on 2026-10-26 (UTC), on the laptop, with Docker Desktop. Nothing was deployed to Azure. The outputs below are copied from that run.
The model in the image is `deploy/model` (version 1 of `complaint-classifier-tfidf`, exported from the local MLflow registry).

## Build

```bash
docker build -t complaint-classifier:local .
```

Result: exit code 0. The image is 424,536,828 bytes (425 MB). The container runs as user id 10001, not as root:

```bash
docker run --rm complaint-classifier:local id -u
# 10001
```

## Run on localhost only

```bash
docker run -d --name cc-local -p 127.0.0.1:8000:8000 complaint-classifier:local
```

### GET /health

```bash
curl http://127.0.0.1:8000/health
```

```
{"status":"ok","model_version":"1"}
```

### POST /classify with one example text

```bash
curl -X POST http://127.0.0.1:8000/classify -H "Content-Type: application/json" \
  -d '{"text": "The air bag warning light stayed on and the air bag did not deploy during the crash at highway speed."}'
```

```
{"label":"AIR BAGS","confidence":1.0,"needs_human_review":false,"threshold_used":0.718,"top":[{"label":"AIR BAGS","confidence":1.0},{"label":"VEHICLE SPEED CONTROL","confidence":0.0},{"label":"OTHER","confidence":0.0}],"model_version":"1"}
```

### A text of 5,001 characters (the limit is 5,000)

The status was `HTTP 422`. The answer is 5,157 characters long because it repeats the input; this is the message part:

```
"msg":"Value error, the text has 5001 characters, the limit is 5000"
```

### Container health check

After about 35 seconds Docker reported the container as healthy:

```bash
docker inspect --format '{{.State.Health.Status}}' cc-local
# healthy
```

The container log showed `POST /classify` with 200 and 422 and `GET /health` with 200.

## Stop

```bash
docker stop cc-local
docker rm cc-local
```

Afterwards no container named `cc-local` was left.

## What this does not show

This is a local run of one image. It does not show that the image runs on Azure Container Apps, how long a cold start takes there, or what it costs.
