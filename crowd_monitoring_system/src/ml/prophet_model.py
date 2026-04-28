from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled=True

class ForecastModel:
    def __init__(self):
        self.model = None

    def train(self, df):
        if df.empty or len(df) < 3:
            return False
        self.model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True)
        # Add extra regressors for hour and day
        if 'hour' in df.columns:
            self.model.add_regressor('hour')
        if 'day' in df.columns:
            self.model.add_regressor('day')
        self.model.fit(df)
        return True
        
    def save(self, filepath="models/prophet_model.json"):
        import json
        if self.model is None:
            return False
        with open(filepath, 'w') as f:
            json.dump(model_to_json(self.model), f)
        return True
        
    def load(self, filepath="models/prophet_model.json"):
        import json
        import os
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'r') as f:
                self.model = model_from_json(json.load(f))
            return True
        except Exception:
            # Fallback if text-based
            with open(filepath, 'r') as f:
                self.model = model_from_json(f.read())
            return True

    def predict(self, periods=30, freq='5min'):
        if self.model is None:
            return None
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        # Add regressors to future dataframe
        future['hour'] = future['ds'].dt.hour
        future['day'] = future['ds'].dt.dayofweek
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat']].tail(periods)
