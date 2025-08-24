# Air Quality Streaming Application

A real-time air quality monitoring application that combines machine learning with live data streaming for environmental monitoring and analysis.

##  Quick Start

### Option 1: Quick Start (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete application
python start_application.py
```

### Option 2: Manual Start
```bash
# Terminal 1: Start streaming server
python streaming_server.py

# Terminal 2: Start dashboard
streamlit run streamlit_dashboard.py
```

### Option 3: Test Individual Components
```bash
# Test the hybrid worker
python test_model_worker.py

# Run the demo
python demo.py
```

##  Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OpenAQ API    │───▶│ Streaming Server │───▶│ Hybrid Worker   │
│                 │    │                  │    │                 │
│ Air Quality     │    │ WebSocket        │    │ Forgetful       │
│ Data            │    │ Real-time        │    │ Forest +        │
└─────────────────┘    │ Processing       │    │ EdgeFrame       │
                       └──────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Streamlit        │◀───│ Processed       │
                       │ Dashboard        │    │ Metrics &       │
                       │                  │    │ Data            │
                       │ Real-time Viz    │    │                 │
                       └──────────────────┘    └─────────────────┘
```

##  Features

### Forgetful Forest Implementation
- **Incremental Learning**: Model updates with new streaming data
- **Memory Management**: Fixed-size buffer (configurable: 200-500 samples)
- **Adaptive Depth**: Tree depth based on data size
- **Prequential Evaluation**: Real-time performance assessment

### EdgeFrame Analysis
- **Correlation Matrix**: Feature relationship analysis
- **Sparsity Control**: Configurable threshold (default: 0.3)
- **Memory Efficiency**: Sparse representation storage
- **Real-time Updates**: Continuous structure evolution

### Interactive Dashboard
- **Real-time Visualization**: Live air quality parameter charts
- **Model Performance Monitoring**: Accuracy, memory, sparsity tracking
- **EdgeFrame Visualization**: Network structure and correlation analysis
- **Data Analytics**: Comprehensive data summary and statistics
- **Responsive Design**: Modern UI with real-time updates

##  Monitored Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| PM2.5 | Fine particulate matter | µg/m³ |
| PM10 | Coarse particulate matter | µg/m³ |
| NO2 | Nitrogen dioxide | µg/m³ |
| O3 | Ozone | µg/m³ |
| SO2 | Sulfur dioxide | µg/m³ |
| CO | Carbon monoxide | mg/m³ |

## Configuration

### Model Parameters
```json
{
    "feature_cols": ["pm25", "pm10", "no2", "o3", "so2", "co"],
    "target_col": null,
    "retain_size": 500,
    "n_trees": 20,
    "correlation_threshold": 0.3,
    "warmup_size": 200
}
```

### API Settings
```json
{
    "city": "Delhi",
    "country": "IN",
    "limit": 100
}
```

## Performance Metrics

The application tracks the following real-time metrics:

- **Model Accuracy**: Real-time prediction performance
- **Memory Usage**: Efficient data storage utilization (MB)
- **Sparsity Ratio**: EdgeFrame structure efficiency (0-1)
- **Data Retention**: Active sample count
- **Processing Speed**: Real-time update latency
- **Forest Size**: Number of active trees

### Benchmark Results

| Metric | Performance |
|--------|-------------|
| Warmup Time | ~0.05s for 60 samples |
| Update Time | ~0.064s average per batch |
| Memory Usage | ~0.01 MB for 200 samples |
| Accuracy | 30-80% (varies with data quality) |
| Sparsity | 0.067-0.333 (configurable) |

## 🛠Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies
```bash
pip install numpy pandas scikit-learn websockets aiohttp streamlit plotly asyncio
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
EF-FF/
├── model_worker.py          # Core ML implementation
├── streaming_server.py      # Real-time data server
├── streamlit_dashboard.py   # Interactive dashboard
├── data_analysis.py         # Offline analysis utilities
├── test_model_worker.py     # Testing suite
├── demo.py                  # Demonstration script
├── start_application.py     # Startup script
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## 🔧 Troubleshooting

### Common Issues

#### Connection Failed
- Check if streaming server is running
- Verify WebSocket port 8765 is available
- Check firewall settings

#### No Data Displayed
- Verify OpenAQ API connectivity
- Check city/country parameters
- Monitor server logs for errors

#### Model Not Updating
- Ensure sufficient data for warmup (200 samples)
- Check feature column configuration
- Verify data format compatibility

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

##  Testing

### Test Coverage
- Worker Initialization
-  Data Processing
-  Warmup Phase
-  Model Updates
-  EdgeFrame Analysis
-  Memory Management
-  Performance Metrics

Run tests:
```bash
python test_model_worker.py
```

##  Access Points

- **Dashboard**: http://localhost:8501
- **WebSocket**: ws://localhost:8765
- **API**: OpenAQ v2 endpoint

##  Future Enhancements

### Immediate Improvements
- [ ] Data validation and input quality checks
- [ ] Automatic reconnection and error recovery
- [ ] Performance tuning and correlation threshold optimization
- [ ] Anomaly detection alert system

### Future Features
- [ ] Multi-city support for processing multiple locations
- [ ] Advanced ML ensemble methods
- [ ] Database persistence for historical data
- [ ] API rate limiting for OpenAQ integration
- [ ] Mobile-responsive dashboard design

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
