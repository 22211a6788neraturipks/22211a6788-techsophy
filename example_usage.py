#!/usr/bin/env python3
"""
Example Usage Script for Stock Anomaly Detection System
This script demonstrates how to use individual components and features.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Update imports to match refactored class names
try:
    from stock_anomaly_detector import (
        MarketDataFetcher, SimpleStatAnomaly, VolumeSpikeDetector,
        SequencePatternDetector, AlertSystem, DataStorage, DataSanitizer
    )
    print("✅ Successfully imported components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def example_1_basic_data_fetching():
    """Example 1: Basic data fetching and validation"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Data Fetching and Validation")
    print("="*60)
    
    # Initialize API client
    api_client = MarketDataFetcher(delay=1.0)
    
    # Fetch historical data
    print("Fetching historical data for AAPL...")
    historical_data = api_client.fetch_history("AAPL", span="3mo")
    
    if not historical_data.empty:
        print(f"✅ Retrieved {len(historical_data)} historical records")
        print(f"Date range: {historical_data.index[0]} to {historical_data.index[-1]}")
        print(f"Price range: ${historical_data['Close'].min():.2f} - ${historical_data['Close'].max():.2f}")
        
        # Validate data
        validated_data = DataSanitizer.clean(historical_data)
        print(f"✅ Validated data: {len(validated_data)} clean records")
        
        # Show basic statistics
        print("\nBasic Statistics:")
        print(f"Average price: ${validated_data['Close'].mean():.2f}")
        print(f"Price volatility (std): ${validated_data['Close'].std():.2f}")
        print(f"Average volume: {validated_data['Volume'].mean():,.0f}")
        
        return validated_data
    else:
        print("❌ Failed to fetch historical data")
        return None

def example_2_statistical_anomaly_detection():
    """Example 2: Statistical anomaly detection"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Statistical Anomaly Detection")
    print("="*60)
    
    # Get data from example 1
    data = example_1_basic_data_fetching()
    if data is None:
        return
    
    # Initialize statistical detector
    detector = SimpleStatAnomaly(window=20, multiplier=2.0)
    
    # Train the detector
    print("\nTraining statistical anomaly detector...")
    detector.train(data)
    
    if detector.mean is not None:
        print(f"✅ Model trained successfully")
        print(f"Moving Average: ${detector.mean:.2f}")
        print(f"Standard Deviation: ${detector.stdev:.2f}")
        
        # Test with current price
        current_price = data['Close'].iloc[-1]
        print(f"\nTesting with current price: ${current_price:.2f}")
        
        anomalies = detector.detect_anomalies(data, current_price)
        
        if anomalies:
            print(f"🚨 {len(anomalies)} anomalies detected!")
            for anomaly in anomalies:
                print(f"  - Type: {anomaly.anomaly_type}")
                print(f"  - Severity: {anomaly.severity}")
                print(f"  - Confidence: {anomaly.confidence:.2%}")
                print(f"  - Details: {anomaly.details}")
        else:
            print("✅ No anomalies detected - price is within normal range")
        
        # Test with artificial anomaly
        print(f"\nTesting with artificial anomaly...")
        artificial_price = detector.mean + 4 * detector.stdev
        print(f"Artificial price: ${artificial_price:.2f}")
        
        anomalies = detector.detect_anomalies(data, artificial_price)
        if anomalies:
            print(f"🚨 Artificial anomaly successfully detected!")
            print(f"  - Severity: {anomalies[0].severity}")
            print(f"  - Confidence: {anomalies[0].confidence:.2%}")
        
        return detector
    else:
        print("❌ Failed to train statistical detector")
        return None

def example_3_volume_analysis():
    """Example 3: Volume anomaly detection"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Volume Anomaly Detection")
    print("="*60)
    
    # Get data
    api_client = MarketDataFetcher()
    data = api_client.fetch_history("AAPL", span="3mo")
    
    if data.empty:
        print("❌ Failed to fetch data")
        return
    
    # Initialize volume detector
    volume_detector = VolumeSpikeDetector(window=20, multiplier=3.0)
    
    # Train detector
    print("Training volume anomaly detector...")
    volume_detector.train(data)
    
    if volume_detector.avg is not None:
        print(f"✅ Volume model trained successfully")
        print(f"Average Volume: {volume_detector.avg:,.0f}")
        print(f"Volume Std Dev: {volume_detector.std:,.0f}")
        
        # Test with recent volume
        recent_volume = data['Volume'].iloc[-1]
        print(f"\nTesting with recent volume: {recent_volume:,}")
        
        anomalies = volume_detector.detect_anomalies(data, recent_volume)
        
        if anomalies:
            print(f"🚨 Volume anomaly detected!")
            for anomaly in anomalies:
                print(f"  - Details: {anomaly.details}")
        else:
            print("✅ Volume within normal range")
        
        # Test with artificial high volume
        artificial_volume = volume_detector.avg + 5 * volume_detector.std
        print(f"\nTesting with artificial high volume: {artificial_volume:,.0f}")
        
        anomalies = volume_detector.detect_anomalies(data, artificial_volume)
        if anomalies:
            print(f"🚨 Artificial volume anomaly detected!")
            print(f"  - Confidence: {anomalies[0].confidence:.2%}")
        
        return volume_detector
    else:
        print("❌ Failed to train volume detector")
        return None

def example_4_lstm_detection():
    """Example 4: LSTM anomaly detection"""
    print("\n" + "="*60)
    print("EXAMPLE 4: LSTM Anomaly Detection")
    print("="*60)
    
    # Get data
    api_client = MarketDataFetcher()
    data = api_client.fetch_history("AAPL", span="6mo")  # More data for LSTM
    
    if data.empty or len(data) < 100:
        print("❌ Insufficient data for LSTM training")
        return
    
    print(f"Training LSTM with {len(data)} data points...")
    print("⚠️  This may take a few minutes...")
    
    # Initialize LSTM detector
    lstm_detector = SequencePatternDetector(seq_len=30, threshold_pct=95)
    
    # Train detector
    lstm_detector.train(data)
    
    if lstm_detector.model is not None:
        print("✅ LSTM model trained successfully")
        print(f"Threshold: {lstm_detector.threshold:.6f}")
        
        # Test with current price
        current_price = data['Close'].iloc[-1]
        print(f"\nTesting LSTM with current price: ${current_price:.2f}")
        
        anomalies = lstm_detector.detect_anomalies(data, current_price)
        
        if anomalies:
            print(f"🚨 LSTM anomaly detected!")
            for anomaly in anomalies:
                print(f"  - Details: {anomaly.details}")
                print(f"  - Confidence: {anomaly.confidence:.2%}")
        else:
            print("✅ LSTM: No pattern anomalies detected")
        
        return lstm_detector
    else:
        print("❌ LSTM training failed")
        return None

def example_5_alert_system():
    """Example 5: Alert system demonstration"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Alert System Demonstration")
    print("="*60)
    
    # Configure alert system
    config = {
        "email_enabled": False,  # Disable email for demo
        "monitoring_interval": 60
    }
    
    alert_system = AlertSystem(config)
    
    # Create sample alerts
    from stock_anomaly_detector import AnomalyAlert
    
    alerts = [
        AnomalyAlert(
            symbol="AAPL",
            timestamp=datetime.now(),
            anomaly_type="PRICE_SPIKE",
            severity="HIGH",
            current_price=185.50,
            expected_range=(175.20, 182.80),
            confidence=0.854,
            details="Z-score: 3.2, Moving average: $179.00"
        ),
        AnomalyAlert(
            symbol="GOOGL",
            timestamp=datetime.now(),
            anomaly_type="VOLUME_SPIKE",
            severity="MEDIUM",
            current_price=2650.30,
            expected_range=(0, 0),  # Not applicable for volume
            confidence=0.723,
            details="Volume Z-score: 4.1, Current: 2,500,000, Avg: 1,200,000"
        )
    ]
    
    # Send alerts
    print("Sending sample alerts...")
    for alert in alerts:
        alert_system.send_alert(alert)
    
    print(f"\n✅ {len(alerts)} alerts sent and stored in history")
    print(f"Alert history contains {len(alert_system.alert_history)} total alerts")

def example_6_data_persistence():
    """Example 6: Data storage and retrieval"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Data Storage and Retrieval")
    print("="*60)
    
    # Initialize storage
    storage = DataStorage("example_database.db")
    
    # Create sample data
    from stock_anomaly_detector import StockData, AnomalyAlert
    
    # Sample stock data
    stock_data = StockData(
        symbol="AAPL",
        timestamp=datetime.now(),
        price=180.50,
        volume=1500000,
        change_percent=2.3
    )
    
    # Sample anomaly
    anomaly = AnomalyAlert(
        symbol="AAPL",
        timestamp=datetime.now(),
        anomaly_type="PRICE_SPIKE",
        severity="HIGH",
        current_price=180.50,
        expected_range=(175.0, 185.0),
        confidence=0.85,
        details="Example anomaly for demonstration"
    )
    
    # Save data
    print("Saving sample data to database...")
    storage.save_stock_data(stock_data)
    storage.save_anomaly_alert(anomaly)
    
    # Retrieve and display data
    import sqlite3
    with sqlite3.connect("example_database.db") as conn:
        # Get stock data count
        stock_count = conn.execute("SELECT COUNT(*) FROM stock_data").fetchone()[0]
        alert_count = conn.execute("SELECT COUNT(*) FROM anomaly_alerts").fetchone()[0]
        
        print(f"✅ Database contains:")
        print(f"  - Stock data records: {stock_count}")
        print(f"  - Anomaly alerts: {alert_count}")
        
        # Show recent data
        recent_stocks = pd.read_sql_query('''
            SELECT symbol, timestamp, price, volume, change_percent 
            FROM stock_data 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''', conn)
        
        if not recent_stocks.empty:
            print(f"\nRecent stock data:")
            print(recent_stocks.to_string(index=False))
        
        recent_alerts = pd.read_sql_query('''
            SELECT symbol, timestamp, anomaly_type, severity, confidence 
            FROM anomaly_alerts 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''', conn)
        
        if not recent_alerts.empty:
            print(f"\nRecent alerts:")
            print(recent_alerts.to_string(index=False))

def example_7_visualization():
    """Example 7: Basic data visualization"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Data Visualization")
    print("="*60)
    
    # Get data
    api_client = MarketDataFetcher()
    data = api_client.fetch_history("AAPL", span="3mo")
    
    if data.empty:
        print("❌ No data available for visualization")
        return
    
    try:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price chart with moving average
        ax1.plot(data.index, data['Close'], label='Close Price', alpha=0.7)
        
        # Calculate and plot moving average
        ma_20 = data['Close'].rolling(window=20).mean()
        ax1.plot(data.index, ma_20, label='20-day MA', color='red')
        
        # Add standard deviation bands
        ma_std = data['Close'].rolling(window=20).std()
        upper_band = ma_20 + 2 * ma_std
        lower_band = ma_20 - 2 * ma_std
        
        ax1.fill_between(data.index, upper_band, lower_band, alpha=0.2, color='gray', label='±2σ bands')
        
        ax1.set_title('AAPL Stock Price with Anomaly Detection Bands')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2.bar(data.index, data['Volume'], alpha=0.7, color='blue')
        
        # Volume moving average
        vol_ma = data['Volume'].rolling(window=20).mean()
        ax2.plot(data.index, vol_ma, color='red', label='20-day Volume MA')
        
        ax2.set_title('AAPL Trading Volume')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stock_analysis_example.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization created and saved as 'stock_analysis_example.png'")
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        print("Note: GUI display might not work in some environments")

def main():
    """Run all examples"""
    print("🚀 Stock Anomaly Detection System - Usage Examples")
    print("=" * 60)
    print("This script demonstrates various components and features.")
    print("Each example can be run independently.\n")
    
    examples = [
        ("Basic Data Fetching", example_1_basic_data_fetching),
        ("Statistical Anomaly Detection", example_2_statistical_anomaly_detection),
        ("Volume Analysis", example_3_volume_analysis),
        ("LSTM Detection (Advanced)", example_4_lstm_detection),
        ("Alert System", example_5_alert_system),
        ("Data Persistence", example_6_data_persistence),
        ("Visualization", example_7_visualization)
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\nOptions:")
    print("- Press Enter to run all examples")
    print("- Enter a number (1-7) to run a specific example")
    print("- Enter 'q' to quit")
    
    choice = input("\nYour choice: ").strip()
    
    if choice.lower() == 'q':
        return
    elif choice == '':
        # Run all examples
        for name, func in examples:
            try:
                func()
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        # Run specific example
        name, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
    else:
        print("❌ Invalid choice")
    
    print(f"\n{'='*60}")
    print("Examples completed!")
    print("To run the full system: python stock_anomaly_detector.py")

if __name__ == "__main__":
    main()