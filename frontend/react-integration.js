// TRAF-GNN React Integration Examples
// Copy these into your React components

// ============================================
// API Configuration
// ============================================
const API_BASE_URL = 'http://localhost:5000/api';

// ============================================
// API Service Functions
// ============================================

// Get all sensors
export async function getSensors() {
    const response = await fetch(`${API_BASE_URL}/sensors`);
    return response.json();
}

// Get sensors by freeway
export async function getFreeways() {
    const response = await fetch(`${API_BASE_URL}/freeways`);
    return response.json();
}

// Get single prediction
export async function predictTraffic(sensorId, timestamp) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            sensor_id: sensorId,
            timestamp: timestamp
        })
    });
    return response.json();
}

// Get batch predictions (for map view)
export async function predictBatch(sensorIds, timestamp) {
    const response = await fetch(`${API_BASE_URL}/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            sensor_ids: sensorIds,
            timestamp: timestamp
        })
    });
    return response.json();
}

// Get model info
export async function getModelInfo() {
    const response = await fetch(`${API_BASE_URL}/model/info`);
    return response.json();
}


// ============================================
// Example React Component
// ============================================

/*
import React, { useState, useEffect } from 'react';
import { getSensors, predictTraffic } from './api';

function TrafficPredictor() {
  const [sensors, setSensors] = useState([]);
  const [selectedSensor, setSelectedSensor] = useState(null);
  const [timestamp, setTimestamp] = useState(new Date().toISOString().slice(0, 16));
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getSensors().then(data => setSensors(data.sensors));
  }, []);

  const handlePredict = async () => {
    if (!selectedSensor) return;

    setLoading(true);
    const result = await predictTraffic(selectedSensor, timestamp);
    setPrediction(result);
    setLoading(false);
  };

  return (
    <div className="traffic-predictor">
      <h1>🚦 TRAF-GNN Traffic Predictor</h1>

      <div className="controls">
        <select
          value={selectedSensor || ''}
          onChange={e => setSelectedSensor(parseInt(e.target.value))}
        >
          <option value="">Select Location...</option>
          {sensors.map(s => (
            <option key={s.id} value={s.id}>
              {s.name} ({s.freeway})
            </option>
          ))}
        </select>

        <input
          type="datetime-local"
          value={timestamp}
          onChange={e => setTimestamp(e.target.value)}
        />

        <button onClick={handlePredict} disabled={loading}>
          {loading ? 'Predicting...' : '🔮 Predict Traffic'}
        </button>
      </div>

      {prediction && (
        <div className="results">
          <h2>📊 Predictions for {prediction.sensor.name}</h2>

          {prediction.predictions.map(p => (
            <div
              key={p.horizon}
              className="prediction-card"
              style={{ borderColor: p.color }}
            >
              <span className="horizon">{p.horizon}</span>
              <span className="speed">{p.speed_mph} mph</span>
              <span className={`status ${p.status}`}>{p.status}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default TrafficPredictor;
*/


// ============================================
// Example CSS (paste in your styles)
// ============================================

/*
.traffic-predictor {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.controls {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
}

.controls select,
.controls input {
  padding: 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 14px;
}

.controls button {
  padding: 12px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
}

.controls button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.results {
  background: #f8fafc;
  border-radius: 12px;
  padding: 20px;
}

.prediction-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: white;
  padding: 16px 20px;
  border-radius: 8px;
  margin-bottom: 12px;
  border-left: 4px solid;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.horizon {
  font-weight: 600;
  color: #64748b;
}

.speed {
  font-size: 24px;
  font-weight: 700;
}

.status {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
}

.status.heavy { background: #fee2e2; color: #dc2626; }
.status.moderate { background: #fef3c7; color: #d97706; }
.status.light { background: #dcfce7; color: #16a34a; }
*/
