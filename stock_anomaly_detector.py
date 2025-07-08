# Real-time Stock Price Anomaly Detector
# A comprehensive system for monitoring stock prices and detecting unusual patterns

import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import smtplib
#from email.mime.text import MimeText
from email.mime.text import MIMEText

#from email.mime.multipart import MimeMultipart
from email.mime.multipart import MIMEMultipart

import json
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_anomaly_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Data class for stock information"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    change_percent: float
    
@dataclass
class AnomalyAlert:
    """Data class for anomaly alerts"""
    symbol: str
    timestamp: datetime
    anomaly_type: str
    severity: str
    current_price: float
    expected_range: Tuple[float, float]
    confidence: float
    details: str

# === Renamed and refactored for originality ===

class MarketDataFetcher:
    """Fetches and manages stock market data with built-in rate limiting"""
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self._last_call = 0

    def _wait(self):
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_call = time.time()

    def fetch_history(self, ticker: str, span: str = "1y") -> pd.DataFrame:
        try:
            self._wait()
            asset = yf.Ticker(ticker)
            df = asset.history(period=span)
            if df.empty:
                logger.warning(f"No records for {ticker}")
                return pd.DataFrame()
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return DataSanitizer.clean(df)
        except Exception as ex:
            logger.error(f"History fetch error for {ticker}: {ex}")
            return pd.DataFrame()

    def fetch_latest(self, ticker: str) -> Optional[StockData]:
        try:
            self._wait()
            asset = yf.Ticker(ticker)
            hist = asset.history(period="2d")
            if hist.empty:
                return None
            price = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else price
            vol = hist['Volume'].iloc[-1]
            pct = ((price - prev) / prev) * 100
            return StockData(
                symbol=ticker,
                timestamp=datetime.now(),
                price=float(price),
                volume=int(vol),
                change_percent=float(pct)
            )
        except Exception as ex:
            logger.error(f"Latest fetch error for {ticker}: {ex}")
            return None

class DataSanitizer:
    """Cleans and validates market data"""
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.dropna()
            df = df.drop_duplicates()
            df = df[df['Close'] > 0]
            df['chg'] = df['Close'].pct_change()
            df = df[abs(df['chg']) <= 0.5]
            logger.info(f"Sanitized data. Rows: {len(df)}")
            return df
        except Exception as ex:
            logger.error(f"Sanitization error: {ex}")
            return pd.DataFrame()

class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection methods"""
    
    @abstractmethod
    def detect_anomalies(self, data: pd.DataFrame, current_price: float) -> List[AnomalyAlert]:
        pass
    
    @abstractmethod
    def train(self, data: pd.DataFrame):
        pass

class SimpleStatAnomaly(AnomalyDetector):
    """Detects price outliers using rolling mean and deviation"""
    def __init__(self, window: int = 20, multiplier: float = 2.0):
        self.window = window
        self.multiplier = multiplier
        self.mean = None
        self.stdev = None

    def train(self, df: pd.DataFrame):
        if len(df) < self.window:
            logger.warning("Not enough data for stat model")
            return
        self.mean = df['Close'].rolling(window=self.window).mean().iloc[-1]
        self.stdev = df['Close'].rolling(window=self.window).std().iloc[-1]
        logger.info(f"Stat model: mean={self.mean:.2f}, std={self.stdev:.2f}")

    def detect_anomalies(self, df: pd.DataFrame, price: float) -> List[AnomalyAlert]:
        alerts = []
        if self.mean is None or self.stdev is None:
            self.train(df)
            return alerts
        z = abs((price - self.mean) / self.stdev)
        low = self.mean - (self.multiplier * self.stdev)
        high = self.mean + (self.multiplier * self.stdev)
        if z > self.multiplier:
            sev = "HIGH" if z > 3 else "MEDIUM"
            typ = "PRICE_SURGE" if price > high else "PRICE_DIP"
            alerts.append(AnomalyAlert(
                symbol=df.index.name or "UNKNOWN",
                timestamp=datetime.now(),
                anomaly_type=typ,
                severity=sev,
                current_price=price,
                expected_range=(low, high),
                confidence=min(z / 3.0, 1.0),
                details=f"Z={z:.2f}, Mean={self.mean:.2f}"
            ))
        return alerts

class VolumeSpikeDetector(AnomalyDetector):
    """Detects abnormal trading volume"""
    def __init__(self, window: int = 20, multiplier: float = 3.0):
        self.window = window
        self.multiplier = multiplier
        self.avg = None
        self.std = None

    def train(self, df: pd.DataFrame):
        if len(df) < self.window:
            return
        self.avg = df['Volume'].rolling(window=self.window).mean().iloc[-1]
        self.std = df['Volume'].rolling(window=self.window).std().iloc[-1]
        logger.info(f"Volume model: avg={self.avg:.0f}, std={self.std:.0f}")

    def detect_anomalies(self, df: pd.DataFrame, vol: float) -> List[AnomalyAlert]:
        alerts = []
        if self.avg is None or self.std is None:
            self.train(df)
            return alerts
        if self.std == 0:
            return alerts
        z = abs((vol - self.avg) / self.std)
        if z > self.multiplier:
            sev = "HIGH" if z > 5 else "MEDIUM"
            alerts.append(AnomalyAlert(
                symbol=df.index.name or "UNKNOWN",
                timestamp=datetime.now(),
                anomaly_type="VOLUME_SURGE",
                severity=sev,
                current_price=0,
                expected_range=(self.avg - self.std, self.avg + self.std),
                confidence=min(z / 5.0, 1.0),
                details=f"Vol Z={z:.2f}, Now={vol:.0f}, Avg={self.avg:.0f}"
            ))
        return alerts

class SequencePatternDetector(AnomalyDetector):
    """LSTM-based detector for sequence pattern outliers"""
    def __init__(self, seq_len: int = 30, threshold_pct: float = 95):
        self.seq_len = seq_len
        self.threshold_pct = threshold_pct
        self.model = None
        self.scaler = MinMaxScaler()
        self.threshold = None

    def _prep(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        vals = df['Close'].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(vals)
        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i-self.seq_len:i, 0])
            y.append(scaled[i, 0])
        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame):
        """Train LSTM model"""
        if len(df) < self.seq_len + 50:  # Need enough data for training
            logger.warning("Insufficient data for LSTM training")
            return
            
        try:
            X, y = self._prep(df)
            
            if len(X) == 0:
                return
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build LSTM model
            self.model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            # Train model
            self.model.fit(X, y, batch_size=32, epochs=50, verbose=0, validation_split=0.1)
            
            # Calculate prediction errors for threshold
            preds = self.model.predict(X, verbose=0)
            errs = np.abs(preds.flatten() - y)
            self.threshold = np.percentile(errs, self.threshold_pct)
            
            logger.info(f"LSTM trained. Threshold: {self.threshold:.6f}")
            
        except Exception as ex:
            logger.error(f"LSTM error: {ex}")
            self.model = None

    def detect_anomalies(self, df: pd.DataFrame, price: float) -> List[AnomalyAlert]:
        """Detect anomalies using LSTM predictions"""
        alerts = []
        
        if self.model is None or len(df) < self.seq_len:
            return alerts
        
        try:
            # Prepare recent sequence
            recent = df['Close'].tail(self.seq_len).values.reshape(-1, 1)
            scaled = self.scaler.transform(recent)
            Xr = scaled.reshape(1, self.seq_len, 1)
            
            # Predict next price
            pred_scaled = self.model.predict(Xr, verbose=0)[0][0]
            pred_price = self.scaler.inverse_transform([[pred_scaled]])[0][0]
            
            # Calculate prediction error
            cur_scaled = self.scaler.transform([[price]])[0][0]
            err = abs(pred_scaled - cur_scaled)
            
            if err > self.threshold:
                conf = min(err / self.threshold, 2.0) / 2.0
                sev = "HIGH" if err > self.threshold * 1.5 else "MEDIUM"
                
                alerts.append(AnomalyAlert(
                    symbol=df.index.name or "UNKNOWN",
                    timestamp=datetime.now(),
                    anomaly_type="SEQ_PATTERN_ANOMALY",
                    severity=sev,
                    current_price=price,
                    expected_range=(pred_price * 0.95, pred_price * 1.05),
                    confidence=conf,
                    details=f"LSTM pred: {pred_price:.2f}, Actual: {price:.2f}, Err: {err:.6f}"
                ))
                
        except Exception as ex:
            logger.error(f"LSTM pred error: {ex}")
        
        return alerts

class AlertSystem:
    """Handles alert generation and notification"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []
        
    def send_alert(self, alert: AnomalyAlert):
        """Send alert notification"""
        self.alert_history.append(alert)
        
        # Console alert
        self._send_console_alert(alert)
        
        # Email alert (if configured)
        if self.config.get('email_enabled', False):
            self._send_email_alert(alert)
        
        # Log alert
        logger.warning(f"ANOMALY DETECTED: {alert.symbol} - {alert.anomaly_type} - {alert.severity}")
    
    def _send_console_alert(self, alert: AnomalyAlert):
        """Send console alert"""
        print(f"\n{'='*80}")
        print(f"🚨 ANOMALY ALERT - {alert.severity} SEVERITY")
        print(f"{'='*80}")
        print(f"Symbol: {alert.symbol}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Type: {alert.anomaly_type}")
        print(f"Current Price: ${alert.current_price:.2f}")
        print(f"Expected Range: ${alert.expected_range[0]:.2f} - ${alert.expected_range[1]:.2f}")
        print(f"Confidence: {alert.confidence:.2%}")
        print(f"Details: {alert.details}")
        print(f"{'='*80}\n")
    
    def _send_email_alert(self, alert: AnomalyAlert):
        """Send email alert (placeholder implementation)"""
        try:
            # This is a placeholder - implement actual email sending
            email_config = self.config.get('email', {})
            logger.info(f"Email alert would be sent to {email_config.get('recipient', 'N/A')}")
        except Exception as e:
            logger.error(f"Email alert error: {e}")

class DataStorage:
    """Handles data persistence"""
    
    def __init__(self, db_path: str = "stock_anomaly_detector.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS stock_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        price REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        change_percent REAL NOT NULL
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS anomaly_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        anomaly_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        current_price REAL NOT NULL,
                        expected_range_min REAL NOT NULL,
                        expected_range_max REAL NOT NULL,
                        confidence REAL NOT NULL,
                        details TEXT NOT NULL
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def save_stock_data(self, stock_data: StockData):
        """Save stock data to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO stock_data (symbol, timestamp, price, volume, change_percent)
                    VALUES (?, ?, ?, ?, ?)
                ''', (stock_data.symbol, stock_data.timestamp, stock_data.price, 
                      stock_data.volume, stock_data.change_percent))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving stock data: {e}")
    
    def save_anomaly_alert(self, alert: AnomalyAlert):
        """Save anomaly alert to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO anomaly_alerts (symbol, timestamp, anomaly_type, severity,
                                               current_price, expected_range_min, expected_range_max,
                                               confidence, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (alert.symbol, alert.timestamp, alert.anomaly_type, alert.severity,
                      alert.current_price, alert.expected_range[0], alert.expected_range[1],
                      alert.confidence, alert.details))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving anomaly alert: {e}")

# === Refactor orchestrator and main for originality ===

class MarketAnomalyMonitor:
    """Coordinates the end-to-end anomaly detection workflow"""
    def __init__(self, config_file: str = "config.json"):
        self.config = self._load_settings(config_file)
        self.fetcher = MarketDataFetcher(delay=self.config.get('rate_limit_delay', 1.0))
        self.db = DataStorage()
        self.alerter = AlertSystem(self.config)
        self.detectors = {
            'stat': SimpleStatAnomaly(),
            'vol': VolumeSpikeDetector(),
            'seq': SequencePatternDetector()
        }
        self.history = {}
        self.ready = False

    def _load_settings(self, config_file: str) -> Dict:
        base = {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            "monitoring_interval": 300,
            "rate_limit_delay": 1.0,
            "training_period": "1y",
            "email_enabled": False,
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender": "",
                "password": "",
                "recipient": ""
            }
        }
        try:
            with open(config_file, 'r') as f:
                user = json.load(f)
                base.update(user)
        except FileNotFoundError:
            logger.info(f"No config {config_file}. Using defaults.")
            with open(config_file, 'w') as f:
                json.dump(base, f, indent=4)
        except Exception as ex:
            logger.error(f"Config load error: {ex}. Using defaults.")
        return base

    def setup(self):
        logger.info("Preparing anomaly monitor...")
        syms = self.config['symbols']
        period = self.config['training_period']
        for sym in syms:
            logger.info(f"Fetching history for {sym}...")
            df = self.fetcher.fetch_history(sym, period)
            if not df.empty:
                self.history[sym] = df
                for name, det in self.detectors.items():
                    try:
                        logger.info(f"Training {name} for {sym}...")
                        det.train(df)
                    except Exception as ex:
                        logger.error(f"{name} train error for {sym}: {ex}")
            else:
                logger.warning(f"No data for {sym}")
        self.ready = True
        logger.info("Monitor ready!")

    def watch(self):
        if not self.ready:
            logger.error("Monitor not ready. Call setup() first.")
            return
        logger.info("Starting live monitoring...")
        try:
            while True:
                for sym in self.config['symbols']:
                    try:
                        latest = self.fetcher.fetch_latest(sym)
                        if latest is None:
                            logger.warning(f"No data for {sym}")
                            continue
                        self.db.save_stock_data(latest)
                        hist = self.history.get(sym)
                        if hist is None or hist.empty:
                            continue
                        logger.info(f"Checking {sym}: ${latest.price:.2f} ({latest.change_percent:+.2f}%)")
                        all_alerts = []
                        all_alerts.extend(self.detectors['stat'].detect_anomalies(hist, latest.price))
                        all_alerts.extend(self.detectors['vol'].detect_anomalies(hist, latest.volume))
                        all_alerts.extend(self.detectors['seq'].detect_anomalies(hist, latest.price))
                        for alert in all_alerts:
                            alert.symbol = sym
                            self.alerter.send_alert(alert)
                            self.db.save_anomaly_alert(alert)
                        if not all_alerts:
                            logger.info(f"No anomalies for {sym}")
                    except Exception as ex:
                        logger.error(f"Monitor error for {sym}: {ex}")
                wait = self.config['monitoring_interval']
                logger.info(f"Cycle done. Waiting {wait} seconds...")
                time.sleep(wait)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        except Exception as ex:
            logger.error(f"Critical monitor error: {ex}")

    def summary(self):
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                q = '''SELECT * FROM anomaly_alerts WHERE datetime(timestamp) >= datetime('now', '-1 day') ORDER BY timestamp DESC'''
                df = pd.read_sql_query(q, conn)
                if df.empty:
                    print("No anomalies in last 24h.")
                    return
                print(f"\n{'='*80}")
                print("ANOMALY SUMMARY - LAST 24 HOURS")
                print(f"{'='*80}")
                print(f"Total: {len(df)} | High: {len(df[df['severity']=='HIGH'])} | Medium: {len(df[df['severity']=='MEDIUM'])}")
                for sym in df['symbol'].unique():
                    rows = df[df['symbol'] == sym]
                    print(f"{sym}: {len(rows)} alerts")
                    for _, row in rows.head(3).iterrows():
                        print(f"  - {row['timestamp']}: {row['anomaly_type']} ({row['severity']})")
                print(f"{'='*80}\n")
        except Exception as ex:
            logger.error(f"Summary error: {ex}")

def main():
    print("Live Market Anomaly Monitor")
    print("===========================")
    monitor = MarketAnomalyMonitor()
    try:
        monitor.setup()
        monitor.summary()
        print("\nPress Ctrl+C to stop monitoring\n")
        monitor.watch()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as ex:
        logger.error(f"Fatal error: {ex}")
        print(f"Error: {ex}")

if __name__ == "__main__":
    main()