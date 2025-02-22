"""Use LSTM to predict the next number in a lottery game."""

from dataclasses import dataclass, field
from typing import List, Tuple, Any


import logging
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # pylint: disable=E0401, E0611
from tensorflow.keras.layers import LSTM, Dense, Dropout  # pylint: disable=E0401, E0611
from statsmodels.tsa.arima.model import ARIMA

from .simple_ml_predictors import MLPredictor

logger = logging.getLogger(__name__)


@dataclass
class LSTMPredictor(MLPredictor):
    """Implement LSTM predictor for the lottery data"""

    sequence_length: int = field(default=10)
    model: Any = field(init=False)
    scaler: Any = field(init=False)

    def _data_normalizer(self, x_data: Any, y_data: Any) -> Tuple[Any, Any]:
        """Normalize the data using MinMaxScaler"""
        self.scaler = MinMaxScaler()
        x_data = self.scaler.fit_transform(x_data.reshape(-1, 7)).reshape(x_data.shape)
        y_data = self.scaler.transform(y_data)
        return x_data, y_data

    def prepare_data(self) -> Tuple[Any, Any]:
        """Prepare the data for the LSTM model"""
        temp = self.data_selection("all")
        x_all, y_all = [], []
        data = temp[self.headers[3:]].values[::-1]
        for i in range(len(data) - self.sequence_length):
            x_all.append(data[i : i + self.sequence_length])
            y_all.append(data[i + self.sequence_length])
        return self._data_normalizer(np.array(x_all), np.array(y_all))

    def train_model(self) -> None:
        """Train the LSTM model"""
        x_train, y_train = self.prepare_data()
        self.model = Sequential(
            [
                LSTM(
                    100,
                    return_sequences=True,
                    activation="relu",
                    input_shape=(self.sequence_length, 7),
                ),
                LSTM(100, return_sequences=True, activation="relu"),
                Dropout(0.2),
                LSTM(75, return_sequences=True, activation="elu"),
                LSTM(75, return_sequences=True, activation="elu"),
                Dropout(0.2),
                LSTM(50, return_sequences=True, activation="relu"),
                LSTM(50, return_sequences=False, activation="relu"),
                Dense(7, activation="linear"),  # Predict 7 digits
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")
        logger.info("Training the LSTM model ...")
        self.model.fit(
            x_train, y_train, epochs=500, batch_size=8, validation_split=0.2, verbose=0
        )

    def predict(self) -> Any:
        """Predict the next number using the LSTM model"""
        self.train_model()
        temp = self.data_selection("all")
        data = temp[self.headers[3:]].values[::-1]
        last_numbers = data[-self.sequence_length :]
        last_numbers = self.scaler.transform(last_numbers.reshape(-1, 7)).reshape(
            1, self.sequence_length, 7
        )
        predicted_number = self.model.predict(last_numbers)
        predicted_number = self.scaler.inverse_transform(predicted_number)
        logger.info(
            "Predicted numbers using LSTM model: %s",
            np.round(predicted_number).astype(int).tolist()[0],
        )
        return np.round(predicted_number).astype(int).tolist()[0]


@dataclass
class ARIMAPredictor(MLPredictor):
    """Implement ARIMA predictor for the lottery data"""

    def prepare_data(self, digit: str = "d1") -> Tuple[Any, Any]:
        """Prepare the data for the ARIMA model"""
        temp = self.data_selection("all")
        data = temp[digit].values[::-1]
        return data, []

    def train_model(self, digit: str = "d1") -> Any:
        """Train the ARIMA model"""
        data = self.prepare_data(digit)[0]
        model = ARIMA(data, order=(5, 3, 1))
        model_fit = model.fit()
        return model_fit

    def predict(self) -> Tuple[List[int], List[float]]:
        """Predict the next number using the ARIMA model"""
        result = []
        for col in self.headers[3:]:
            model_fit = self.train_model(col)
            result.append(int(model_fit.forecast(steps=1)[0]))
        logger.info("Predicted numbers using ARIMA model: %s", result)
        return result, []
