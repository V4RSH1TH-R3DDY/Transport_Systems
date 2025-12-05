# TRAF-GNN Prediction API

Flask backend for serving traffic speed predictions from the TRAF-GNN model.

## Quick Start

```bash
# Install dependencies
pip install flask flask-cors

# Run the API
cd api
python app.py
```

API will be available at `http://localhost:5000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/sensors` | GET | List all 207 sensors |
| `/api/sensors/<id>` | GET | Get sensor details |
| `/api/predict` | POST | Single prediction |
| `/api/predict/batch` | POST | Batch predictions |
| `/api/freeways` | GET | List freeways |
| `/api/model/info` | GET | Model information |

## Example: Make a Prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sensor_id": 42, "timestamp": "2025-12-05T18:00:00"}'
```

Response:
```json
{
  "sensor": {"id": 42, "name": "I-405 Sensor 43", "freeway": "I-405"},
  "predictions": [
    {"horizon": "15min", "speed_mph": 45.2, "status": "moderate"},
    {"horizon": "30min", "speed_mph": 42.8, "status": "moderate"},
    {"horizon": "60min", "speed_mph": 51.3, "status": "light"}
  ]
}
```

## React Integration

See `frontend/react-integration.js` for example React code.

## Model Performance

- **MAE**: 3.45 mph
- **RMSE**: 7.31 mph
- **MAPE**: 7.87%
