# Fast_Api

A cleaned and modernized FastAPI service for **banknote authenticity prediction**.

## Endpoints

- `GET /` - service welcome and docs pointer
- `GET /health` - simple health check
- `POST /predict` - classify a banknote from 4 numerical features

## Local setup

```bash
pip install -r requirements.txt
uvicorn Fast_Api.app:app --reload
```

Then open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Example prediction request

```bash
curl -X POST 'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "variance": 3.6216,
    "skewness": 8.6661,
    "curtosis": -2.8073,
    "entropy": -0.44699
  }'
```
