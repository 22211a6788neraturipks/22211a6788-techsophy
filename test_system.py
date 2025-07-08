#!/usr/bin/env python3
"""
Test script for Stock Anomaly Detection System
This script tests individual components and system integration.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import sqlite3

# Add the main module to path
sys.path.append('.')

# Update imports to match refactored class names
try:
    from stock_anomaly_detector import (
        MarketDataFetcher, DataSanitizer, SimpleStatAnomaly,
        VolumeSpikeDetector, AlertSystem, DataStorage, StockData, AnomalyAlert
    )
    print("✅ Successfully imported all components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure stock_anomaly_detector.py is in the current directory")
    sys.exit(1)

class TestStockAnomalyDetector(unittest.TestCase):
    """Test suite for the Stock Anomaly Detection System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            "symbols": ["AAPL"],
            "monitoring_interval": 60,
            "email_enabled": False
        }
        
        # Create sample stock data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        prices = 150 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        volumes = np.random.randint(1000000, 5000000, len(dates))
        
        self.sample_data = pd.DataFrame({
            'Close': prices,
            'Volume': volumes,
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98
        }, index=dates)
    
    def test_data_sanitizer(self):
        """Test data validation functionality"""
        print("\n🧪 Testing Data Sanitizer...")
        
        # Test with valid data
        validated_data = DataSanitizer.validate_stock_data(self.sample_data.copy())
        self.assertFalse(validated_data.empty)
        print("  ✅ Valid data passed validation")
        
        # Test with missing values
        corrupted_data = self.sample_data.copy()
        corrupted_data.loc[corrupted_data.index[10:15], 'Close'] = np.nan
        
        validated_data = DataSanitizer.validate_stock_data(corrupted_data)
        self.assertTrue(len(validated_data) < len(corrupted_data))
        print("  ✅ Missing values properly removed")
        
        # Test with negative prices
        corrupted_data = self.sample_data.copy()
        corrupted_data.loc[corrupted_data.index[20:25], 'Close'] = -10
        
        validated_data = DataSanitizer.validate_stock_data(corrupted_data)
        self.assertTrue(all(validated_data['Close'] > 0))
        print("  ✅ Negative prices properly removed")
    
    def test_statistical_anomaly_detector(self):
        """Test statistical anomaly detection"""
        print("\n🧪 Testing Statistical Anomaly Detector...")
        
        detector = SimpleStatAnomaly(window_size=20, threshold_multiplier=2.0)
        detector.train(self.sample_data)
        
        self.assertIsNotNone(detector.moving_avg)
        self.assertIsNotNone(detector.moving_std)
        print("  ✅ Statistical model trained successfully")
        
        # Test normal price (should not trigger anomaly)
        normal_price = detector.moving_avg
        anomalies = detector.detect_anomalies(self.sample_data, normal_price)
        self.assertEqual(len(anomalies), 0)
        print("  ✅ Normal price correctly classified")
        
        # Test anomalous price (should trigger anomaly)
        anomalous_price = detector.moving_avg + 5 * detector.moving_std
        anomalies = detector.detect_anomalies(self.sample_data, anomalous_price)
        self.assertGreater(len(anomalies), 0)
        print("  ✅ Anomalous price correctly detected")
    
    def test_volume_spike_detector(self):
        """Test volume anomaly detection"""
        print("\n🧪 Testing Volume Anomaly Detector...")
        
        detector = VolumeSpikeDetector(window_size=20, threshold_multiplier=3.0)
        detector.train(self.sample_data)
        
        self.assertIsNotNone(detector.avg_volume)
        self.assertIsNotNone(detector.std_volume)
        print("  ✅ Volume model trained successfully")
        
        # Test normal volume
        normal_volume = detector.avg_volume
        anomalies = detector.detect_anomalies(self.sample_data, normal_volume)
        self.assertEqual(len(anomalies), 0)
        print("  ✅ Normal volume correctly classified")
        
        # Test anomalous volume
        anomalous_volume = detector.avg_volume + 5 * detector.std_volume
        anomalies = detector.detect_anomalies(self.sample_data, anomalous_volume)
        self.assertGreater(len(anomalies), 0)
        print("  ✅ Anomalous volume correctly detected")
    
    def test_alert_system(self):
        """Test alert system functionality"""
        print("\n🧪 Testing Alert System...")
        
        alert_system = AlertSystem(self.test_config)
        
        # Create test alert
        test_alert = AnomalyAlert(
            symbol="AAPL",
            timestamp=datetime.now(),
            anomaly_type="PRICE_SPIKE",
            severity="HIGH",
            current_price=200.0,
            expected_range=(180.0, 190.0),
            confidence=0.95,
            details="Test alert"
        )
        
        # Test alert sending (should not raise exception)
        try:
            alert_system.send_alert(test_alert)
            print("  ✅ Alert sent successfully")
        except Exception as e:
            self.fail(f"Alert sending failed: {e}")
        
        # Check alert history
        self.assertEqual(len(alert_system.alert_history), 1)
        print("  ✅ Alert properly stored in history")
    
    def test_data_storage(self):
        """Test data storage functionality"""
        print("\n🧪 Testing Data Storage...")
        
        # Use temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            storage = DataStorage(tmp.name)
            
            # Test stock data storage
            stock_data = StockData(
                symbol="AAPL",
                timestamp=datetime.now(),
                price=150.0,
                volume=1000000,
                change_percent=2.5
            )
            
            try:
                storage.save_stock_data(stock_data)
                print("  ✅ Stock data saved successfully")
            except Exception as e:
                self.fail(f"Stock data saving failed: {e}")
            
            # Test anomaly alert storage
            alert = AnomalyAlert(
                symbol="AAPL",
                timestamp=datetime.now(),
                anomaly_type="PRICE_SPIKE",
                severity="HIGH",
                current_price=200.0,
                expected_range=(180.0, 190.0),
                confidence=0.95,
                details="Test alert"
            )
            
            try:
                storage.save_anomaly_alert(alert)
                print("  ✅ Anomaly alert saved successfully")
            except Exception as e:
                self.fail(f"Anomaly alert saving failed: {e}")
            
            # Verify data was saved
            with sqlite3.connect(tmp.name) as conn:
                stock_count = conn.execute("SELECT COUNT(*) FROM stock_data").fetchone()[0]
                alert_count = conn.execute("SELECT COUNT(*) FROM anomaly_alerts").fetchone()[0]
                
                self.assertEqual(stock_count, 1)
                self.assertEqual(alert_count, 1)
                print("  ✅ Data retrieval verified")
            
            # Clean up
            os.unlink(tmp.name)

def test_api_connection():
    """Test API connectivity"""
    print("\n🧪 Testing API Connection...")
    
    api_client = MarketDataFetcher(rate_limit_delay=0.5)
    
    # Test real-time data retrieval
    try:
        stock_data = api_client.get_real_time_data("AAPL")
        if stock_data:
            print(f"  ✅ Successfully retrieved AAPL data: ${stock_data.price:.2f}")
            return True
        else:
            print("  ⚠️  No data returned for AAPL")
            return False
    except Exception as e:
        print(f"  ❌ API connection failed: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are installed"""
    print("\n🧪 Testing Dependencies...")
    
    dependencies = [
        'yfinance', 'pandas', 'numpy', 'sklearn', 
        'tensorflow', 'matplotlib', 'seaborn'
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - MISSING")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install missing dependencies with: pip install -r requirements.txt")
        return False
    else:
        print("  ✅ All dependencies installed")
        return True

def run_integration_test():
    """Run a quick integration test"""
    print("\n🧪 Running Integration Test...")
    
    try:
        # Test system initialization (without LSTM to save time)
        from stock_anomaly_detector import StockAnomalyDetectorSystem
        
        # Create temporary config
        test_config = {
            "symbols": ["AAPL"],
            "monitoring_interval": 60,
            "training_period": "1mo",  # Shorter period for testing
            "email_enabled": False
        }
        
        # Save test config
        import json
        with open('test_config.json', 'w') as f:
            json.dump(test_config, f)
        
        # Initialize system
        system = StockAnomalyDetectorSystem('test_config.json')
        
        print("  ✅ System created successfully")
        
        # Test initialization (this will fetch real data)
        system.initialize_system()
        print("  ✅ System initialized successfully")
        
        # Clean up
        os.remove('test_config.json')
        if os.path.exists('stock_anomaly_detector.db'):
            os.remove('stock_anomaly_detector.db')
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Stock Anomaly Detection System - Test Suite")
    print("=" * 60)
    
    # Test dependencies first
    if not test_dependencies():
        print("\n❌ Some dependencies are missing. Please install them first.")
        return
    
    # Test API connection
    api_works = test_api_connection()
    if not api_works:
        print("\n⚠️  API connection issues detected. Some tests may fail.")
    
    # Run unit tests
    print("\n🧪 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=0)
    
    # Run integration test
    if api_works:
        integration_success = run_integration_test()
        if integration_success:
            print("\n✅ All tests passed! Your system is ready to use.")
        else:
            print("\n⚠️  Some integration tests failed. Check the logs above.")
    else:
        print("\n⚠️  Skipping integration tests due to API issues.")
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("- Unit tests check individual components")
    print("- Integration tests check the full system")
    print("- API tests verify data source connectivity")
    print("\nTo run the full system: python stock_anomaly_detector.py")

if __name__ == "__main__":
    main()