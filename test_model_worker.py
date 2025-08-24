#!/usr/bin/env python3
"""
Test script for the HybridWorker model implementation.
This script validates the Forgetful Forest + EdgeFrame hybrid approach.
"""

import json
import sys
import numpy as np
from model_worker import HybridWorker

def test_worker_initialization():
    """Test worker initialization and configuration"""
    print(" Testing Worker Initialization...")
    
    worker = HybridWorker()
    
    # Test default configuration
    assert worker.retain_size == 500
    assert worker.n_trees == 20
    assert worker.corr_threshold == 0.3
    assert worker.warmup_size == 200
    
    # Test custom configuration
    config = {
        "feature_cols": ["pm25", "pm10", "no2", "o3", "so2", "co"],
        "target_col": None,
        "retain_size": 300,
        "n_trees": 15,
        "correlation_threshold": 0.4,
        "warmup_size": 150
    }
    
    result = worker.init(config)
    assert result["ok"] == True
    assert worker.retain_size == 300
    assert worker.n_trees == 15
    assert worker.corr_threshold == 0.4
    assert worker.warmup_size == 150
    
    print("Worker initialization tests passed!")
    return worker

def test_data_processing():
    """Test data processing and feature extraction"""
    print(" Testing Data Processing...")
    
    worker = HybridWorker()
    worker.init({
        "feature_cols": ["pm25", "pm10", "no2", "o3", "so2", "co"],
        "target_col": None,
        "retain_size": 100,
        "n_trees": 10,
        "correlation_threshold": 0.3,
        "warmup_size": 50
    })
    
    # Create synthetic air quality data
    test_data = []
    for i in range(10):
        test_data.append({
            "pm25": np.random.uniform(10, 100),
            "pm10": np.random.uniform(20, 200),
            "no2": np.random.uniform(5, 50),
            "o3": np.random.uniform(20, 80),
            "so2": np.random.uniform(2, 20),
            "co": np.random.uniform(0.5, 5.0),
            "Skip_Rate": np.random.uniform(0, 2.0)
        })
    
    # Test data conversion
    X, y, skip_vals = worker._rows_to_Xy(test_data)
    
    assert X.shape == (10, 6)  # 10 samples, 6 features
    assert y.shape == (10,)     # 10 targets
    assert len(skip_vals) == 10
    
    print("Data processing tests passed!")
    return test_data

def test_warmup_phase():
    """Test the warmup phase of the worker"""
    print("Testing Warmup Phase...")
    
    worker = HybridWorker()
    worker.init({
        "feature_cols": ["pm25", "pm10", "no2", "o3", "so2", "co"],
        "target_col": None,
        "retain_size": 100,
        "n_trees": 10,
        "correlation_threshold": 0.3,
        "warmup_size": 50
    })
    
    # Create warmup data
    warmup_data = []
    for i in range(60):  # More than warmup_size
        warmup_data.append({
            "pm25": np.random.uniform(10, 100),
            "pm10": np.random.uniform(20, 200),
            "no2": np.random.uniform(5, 50),
            "o3": np.random.uniform(20, 80),
            "so2": np.random.uniform(2, 20),
            "co": np.random.uniform(0.5, 5.0),
            "Skip_Rate": np.random.uniform(0, 2.0)
        })
    
    # Test warmup
    result = worker.warmup(warmup_data)
    assert result["ok"] == True
    assert "buffer" in result
    
    # Check that median_skip was computed
    assert worker.median_skip is not None
    assert worker.median_skip >= 0
    
    # Check that forest was trained
    assert worker.forest is not None
    
    print("Warmup phase tests passed!")
    return warmup_data

def test_model_updates():
    """Test incremental model updates"""
    print(" Testing Model Updates...")
    
    worker = HybridWorker()
    worker.init({
        "feature_cols": ["pm25", "pm10", "no2", "o3", "so2", "co"],
        "target_col": None,
        "retain_size": 100,
        "n_trees": 10,
        "correlation_threshold": 0.3,
        "warmup_size": 50
    })
    
    # First, warmup the model
    warmup_data = []
    for i in range(60):
        warmup_data.append({
            "pm25": np.random.uniform(10, 100),
            "pm10": np.random.uniform(20, 200),
            "no2": np.random.uniform(5, 50),
            "o3": np.random.uniform(20, 80),
            "so2": np.random.uniform(2, 20),
            "co": np.random.uniform(0.5, 5.0),
            "Skip_Rate": np.random.uniform(0, 2.0)
        })
    
    worker.warmup(warmup_data)
    
    # Now test incremental updates
    update_data = []
    for i in range(10):
        update_data.append({
            "pm25": np.random.uniform(10, 100),
            "pm10": np.random.uniform(20, 200),
            "no2": np.random.uniform(5, 50),
            "o3": np.random.uniform(20, 80),
            "so2": np.random.uniform(2, 20),
            "co": np.random.uniform(0.5, 5.0),
            "Skip_Rate": np.random.uniform(0, 2.0)
        })
    
    result = worker.update(update_data)
    assert result["ok"] == True
    assert "metrics" in result
    
    metrics = result["metrics"]
    assert "accuracy" in metrics
    assert "memory_usage_mb" in metrics
    assert "sparsity_ratio" in metrics
    assert "edgeframe_nodes" in metrics
    assert "edgeframe_edges" in metrics
    
    print("Model update tests passed!")
    return result

def test_edgeframe_analysis():
    """Test EdgeFrame correlation analysis"""
    print("Testing EdgeFrame Analysis...")
    
    worker = HybridWorker()
    worker.init({
        "feature_cols": ["pm25", "pm10", "no2", "o3", "so2", "co"],
        "target_col": None,
        "retain_size": 100,
        "n_trees": 10,
        "correlation_threshold": 0.3,
        "warmup_size": 50
    })
    
    # Create correlated data
    correlated_data = []
    for i in range(100):
        base_value = np.random.uniform(0, 100)
        correlated_data.append({
            "pm25": base_value + np.random.normal(0, 5),
            "pm10": base_value * 2 + np.random.normal(0, 10),
            "no2": base_value * 0.5 + np.random.normal(0, 2),
            "o3": 100 - base_value + np.random.normal(0, 5),
            "so2": base_value * 0.1 + np.random.normal(0, 1),
            "co": base_value * 0.05 + np.random.normal(0, 0.5),
            "Skip_Rate": np.random.uniform(0, 2.0)
        })
    
    # Process data
    X, y, _ = worker._rows_to_Xy(correlated_data)
    
    # Test EdgeFrame analysis
    ef_result = worker._edgeframe(X)
    
    assert "nodes" in ef_result
    assert "edges" in ef_result
    assert "sparsity_ratio" in ef_result
    assert ef_result["nodes"] == 6  # 6 features
    assert ef_result["sparsity_ratio"] >= 0 and ef_result["sparsity_ratio"] <= 1
    
    print("EdgeFrame analysis tests passed!")
    return ef_result

def test_memory_management():
    """Test memory management and buffer limits"""
    print("Testing Memory Management...")
    
    worker = HybridWorker()
    worker.init({
        "feature_cols": ["pm25", "pm10", "no2", "o3", "so2", "co"],
        "target_col": None,
        "retain_size": 50,  # Small buffer for testing
        "n_trees": 10,
        "correlation_threshold": 0.3,
        "warmup_size": 30
    })
    
    # Add more data than buffer size
    large_data = []
    for i in range(100):
        large_data.append({
            "pm25": np.random.uniform(10, 100),
            "pm10": np.random.uniform(20, 200),
            "no2": np.random.uniform(5, 50),
            "o3": np.random.uniform(20, 80),
            "so2": np.random.uniform(2, 20),
            "co": np.random.uniform(0.5, 5.0),
            "Skip_Rate": np.random.uniform(0, 2.0)
        })
    
    # Warmup with large data
    worker.warmup(large_data)
    
    # Check buffer size is maintained
    assert len(worker.buffer_X) <= worker.retain_size
    assert len(worker.buffer_y) <= worker.retain_size
    
    print(" Memory management tests passed!")

def test_performance_metrics():
    """Test performance metrics calculation"""
    print(" Testing Performance Metrics...")
    
    worker = HybridWorker()
    worker.init({
        "feature_cols": ["pm25", "pm10", "no2", "o3", "so2", "co"],
        "target_col": None,
        "retain_size": 100,
        "n_trees": 10,
        "correlation_threshold": 0.3,
        "warmup_size": 50
    })
    
    # Create test data
    test_data = []
    for i in range(60):
        test_data.append({
            "pm25": np.random.uniform(10, 100),
            "pm10": np.random.uniform(20, 200),
            "no2": np.random.uniform(5, 50),
            "o3": np.random.uniform(20, 80),
            "so2": np.random.uniform(2, 20),
            "co": np.random.uniform(0.5, 5.0),
            "Skip_Rate": np.random.uniform(0, 2.0)
        })
    
    # Warmup and update
    worker.warmup(test_data)
    
    # Test status
    status = worker.status()
    assert status["ok"] == True
    assert "metrics" in status
    
    metrics = status["metrics"]
    if metrics is not None:  # Check if metrics exist
        required_fields = [
            "seen", "accuracy", "retained_size", "memory_usage_mb",
            "forest_size", "edgeframe_nodes", "edgeframe_edges", "sparsity_ratio"
        ]
        
        for field in required_fields:
            assert field in metrics, f"Missing field: {field}"
    else:
        print("  No metrics available yet (this is normal during warmup)")
    
    print("Performance metrics tests passed!")

def run_all_tests():
    """Run all test functions"""
    print(" Starting HybridWorker Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        test_worker_initialization()
        test_data_processing()
        test_warmup_phase()
        test_model_updates()
        test_edgeframe_analysis()
        test_memory_management()
        test_performance_metrics()
        
        print("\n" + "=" * 50)
        print(" All tests passed successfully!")
        print(" HybridWorker is working correctly")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
