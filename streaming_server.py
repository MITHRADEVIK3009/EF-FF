import asyncio
import websockets
import aiohttp
import json
import time
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from model_worker import HybridWorker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirQualityStreamProcessor:
    def __init__(self):
        # Initialize the hybrid worker
        self.worker = HybridWorker()
        self.worker.init({
            "feature_cols": ["pm25", "pm10", "no2", "o3", "so2", "co"],
            "target_col": None,  # Will derive from Skip_Rate equivalent
            "retain_size": 500,
            "n_trees": 20,
            "correlation_threshold": 0.3,
            "warmup_size": 200
        })
        
        # Data buffers for streaming
        self.data_buffer = []
        self.metrics_history = []
        self.last_fetch_time = None
        
        # Air quality thresholds (WHO standards)
        self.thresholds = {
            "pm25": 25,  # µg/m³
            "pm10": 50,  # µg/m³
            "no2": 200,  # µg/m³
            "o3": 100,   # µg/m³
            "so2": 500,  # µg/m³
            "co": 10     # mg/m³
        }

    async def fetch_air_quality(self, session, city="Delhi", country="IN"):
        """Fetch air quality data from OpenAQ API"""
        try:
            # Use the v2 API endpoint
            url = f"https://api.openaq.org/v2/latest"
            params = {
                "city": city,
                "country": country,
                "limit": 100
            }
            
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    logger.error(f"API request failed with status {resp.status}")
                    return []
                
                data = await resp.json()
                measurements = []
                
                for result in data.get("results", []):
                    location = result.get("location", "Unknown")
                    coordinates = result.get("coordinates", {})
                    
                    for measurement in result.get("measurements", []):
                        # Normalize parameter names
                        param = measurement.get("parameter", "").lower()
                        value = measurement.get("value", 0)
                        unit = measurement.get("unit", "")
                        timestamp = measurement.get("lastUpdated", "")
                        
                        # Convert units to standard format
                        if param == "co" and unit == "µg/m³":
                            value = value / 1000  # Convert to mg/m³
                        
                        measurements.append({
                            "location": location,
                            "parameter": param,
                            "value": float(value),
                            "unit": unit,
                            "timestamp": timestamp,
                            "latitude": coordinates.get("latitude"),
                            "longitude": coordinates.get("longitude")
                        })
                
                logger.info(f"Fetched {len(measurements)} measurements from {city}")
                return measurements
                
        except asyncio.TimeoutError:
            logger.error("API request timed out")
            return []
        except Exception as e:
            logger.error(f"Error fetching air quality data: {e}")
            return []

    def process_measurements(self, measurements):
        """Process measurements through the hybrid worker"""
        if not measurements:
            return []
        
        # Group measurements by timestamp to create feature vectors
        measurements_by_time = {}
        for m in measurements:
            timestamp = m["timestamp"]
            if timestamp not in measurements_by_time:
                measurements_by_time[timestamp] = {}
            measurements_by_time[timestamp][m["parameter"]] = m["value"]
        
        # Convert to feature vectors
        processed_data = []
        for timestamp, params in measurements_by_time.items():
            # Create feature vector with all parameters
            features = {
                "pm25": params.get("pm25", 0.0),
                "pm10": params.get("pm10", 0.0),
                "no2": params.get("no2", 0.0),
                "o3": params.get("o3", 0.0),
                "so2": params.get("so2", 0.0),
                "co": params.get("co", 0.0),
                "Skip_Rate": self._calculate_skip_rate(params)  # Anomaly score
            }
            
            processed_data.append(features)
        
        return processed_data

    def _calculate_skip_rate(self, params):
        """Calculate anomaly score based on air quality thresholds"""
        total_score = 0
        count = 0
        
        for param, value in params.items():
            if param in self.thresholds:
                threshold = self.thresholds[param]
                if value > threshold:
                    # Calculate how much it exceeds the threshold
                    excess_ratio = (value - threshold) / threshold
                    total_score += min(excess_ratio, 2.0)  # Cap at 2x threshold
                    count += 1
        
        return total_score / max(count, 1)

    def update_model(self, processed_data):
        """Update the hybrid worker with new data"""
        if not processed_data:
            return None
        
        try:
            if len(self.data_buffer) < self.worker.warmup_size:
                # Still in warmup phase
                result = self.worker.warmup(processed_data)
                logger.info(f"Warmup: {result}")
            else:
                # Update model with new data
                result = self.worker.update(processed_data)
                logger.info(f"Model updated: {result}")
            
            # Store metrics
            if result and "metrics" in result:
                self.metrics_history.append({
                    "timestamp": datetime.now().isoformat(),
                    **result["metrics"]
                })
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return None

    def get_current_status(self):
        """Get current model status and metrics"""
        try:
            status = self.worker.status()
            return status
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"ok": False, "error": str(e)}

async def stream_server(websocket, path, processor):
    """WebSocket server for streaming data"""
    try:
        logger.info(f"New client connected: {websocket.remote_address}")
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    # Fetch new data
                    measurements = await processor.fetch_air_quality(session)
                    
                    if measurements:
                        # Process through model
                        processed_data = processor.process_measurements(measurements)
                        model_result = processor.update_model(processed_data)
                        
                        # Prepare streaming response
                        stream_data = {
                            "timestamp": datetime.now().isoformat(),
                            "measurements": measurements,
                            "model_metrics": model_result.get("metrics") if model_result else None,
                            "data_count": len(measurements)
                        }
                        
                                                # Send to client
                        await websocket.send(json.dumps(stream_data))
                        logger.info(f"Sent {len(measurements)} measurements to client")
                    
                    # Wait before next fetch
                    await asyncio.sleep(10)  # Fetch every 10 seconds for more frequent updates
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Client disconnected")
                    break
                except Exception as e:
                    logger.error(f"Error in stream loop: {e}")
                    await asyncio.sleep(5)
                    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

async def main():
    """Main server function"""
    processor = AirQualityStreamProcessor()
    
    # Start WebSocket server
    server = await websockets.serve(
        lambda ws: stream_server(ws, "", processor),
        "localhost",
        8765
    )
    
    logger.info("Air Quality Streaming Server running at ws://localhost:8765")
    logger.info("Model initialized and ready for streaming data")
    
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.close()
        await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
