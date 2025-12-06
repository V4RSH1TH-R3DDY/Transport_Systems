# TRAF-GNN API Documentation

## Base URL
```
http://localhost:5000/api
```

## Authentication
Currently no authentication required.

---

## Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-06T18:00:00"
}
```

---

### List Sensors
```http
GET /sensors
```

**Response:**
```json
{
  "count": 207,
  "sensors": [
    {
      "id": 0,
      "name": "I-405 Sensor 1",
      "freeway": "I-405",
      "lat": 33.95,
      "lon": -118.40
    }
  ]
}
```

---

### List Places
```http
GET /places
```

**Response:**
```json
{
  "count": 20,
  "places": [
    {
      "id": "Hollywood",
      "name": "Hollywood",
      "lat": 34.0928,
      "lon": -118.3287,
      "icon": "🎬"
    }
  ]
}
```

---

### Predict by Sensor
```http
POST /predict
Content-Type: application/json

{
  "sensor_id": 42,
  "timestamp": "2025-12-06T18:00:00"
}
```

**Response:**
```json
{
  "sensor": {"id": 42, "name": "I-10 Sensor 3", ...},
  "predictions": [
    {"horizon": "15min", "speed_mph": 45.2, "status": "light", "color": "#22c55e"},
    {"horizon": "30min", "speed_mph": 42.8, "status": "moderate", "color": "#f59e0b"},
    {"horizon": "60min", "speed_mph": 51.3, "status": "light", "color": "#22c55e"}
  ],
  "model_version": "TRAF-GNN v1.0",
  "mae": 3.45
}
```

---

### Predict by Place
```http
POST /predict/place
Content-Type: application/json

{
  "place": "Hollywood",
  "timestamp": "2025-12-06T18:00:00"
}
```

**Response:**
```json
{
  "place": {"name": "Hollywood", "lat": 34.09, "lon": -118.33, "icon": "🎬"},
  "timestamp": "2025-12-06T18:00:00",
  "nearby_sensors": [
    {
      "sensor": {"id": 45, "name": "US-101 Sensor 12", ...},
      "prediction": {"speed_mph": 42.5, "status": "moderate", "color": "#f59e0b"},
      "distance_miles": 1.2
    }
  ]
}
```

---

### Route Prediction
```http
POST /route/places
Content-Type: application/json

{
  "start_place": "Santa Monica",
  "end_place": "Downtown LA",
  "timestamp": "2025-12-06T18:00:00"
}
```

**Response:**
```json
{
  "start": {"name": "Santa Monica", "lat": 34.02, "lon": -118.49, "icon": "🏖️"},
  "end": {"name": "Downtown LA", "lat": 34.05, "lon": -118.24, "icon": "🏙️"},
  "route_geometry": [[34.02, -118.49], ...],
  "route": [
    {
      "sensor": {"id": 23, "name": "I-10 Sensor 5", ...},
      "prediction": {"speed_mph": 38.2, "status": "moderate"},
      "segment_index": 0
    }
  ],
  "summary": {
    "total_sensors": 8,
    "total_distance_miles": 15.3,
    "estimated_time_minutes": 25,
    "average_speed_mph": 42.5,
    "status": "moderate",
    "color": "#f59e0b"
  }
}
```

---

### Batch Predictions
```http
POST /predict/batch
Content-Type: application/json

{
  "sensor_ids": [0, 1, 2, 5, 10],
  "timestamp": "2025-12-06T18:00:00"
}
```

---

### Model Info
```http
GET /model/info
```

**Response:**
```json
{
  "name": "TRAF-GNN",
  "version": "1.0",
  "architecture": {
    "type": "Multi-View Graph Neural Network",
    "gnn_layers": 2,
    "hidden_dim": 128,
    "views": ["physical", "proximity", "correlation"]
  },
  "performance": {
    "MAE": 3.45,
    "RMSE": 7.31,
    "MAPE": 7.87,
    "dataset": "METR-LA"
  }
}
```

---

## Error Responses

```json
{
  "error": "Error message here"
}
```

HTTP Status Codes:
- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `500` - Server Error
