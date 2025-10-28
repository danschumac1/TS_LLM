'''
2025-10-09
Author: Dan Schumacher
How to run:
   python ./src/method/forecasting/arima.py
'''
import sys; sys.path.append("./src")
import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from utils.file_io import load_jsonl

INFILE = "./data/datasets/_raw_data/MONSTER/AudioMNIST/test.jsonl"

def main():
    data = load_jsonl(INFILE)
    print(f"Loaded {len(data)} records from {INFILE}")
    exit()

    # Example data: a simple sine wave with noise
    np.random.seed(0)
    n = 100
    time = np.arange(n)
    y = np.sin(0.2 * time) + 0.1 * np.random.randn(n)

    # Fit ARIMA model: order=(p,d,q)
    model = ARIMA(y, order=(2,1,2))
    model_fit = model.fit()

    # Summary
    print(model_fit.summary())

    # Forecast next 10 steps
    forecast = model_fit.forecast(steps=10)
    print("Forecast:", forecast)

    plt.plot(time, y, label="observed")
    plt.plot(np.arange(n, n+10), forecast, label="forecast", marker="o")
    plt.legend()
    os.makedirs("./images", exist_ok=True)
    plt.savefig("./images/arima_forecast.png")
    plt.show()


if __name__ == "__main__":
    main()