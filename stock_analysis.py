import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from prophet import Prophet
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    return (
        Prophet,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        plt,
        yf,
    )


@app.cell
def _(yf):
    # Gets the last 5 years of data for the ticker specified
    ticker = "PYPL"
    stock_data = yf.download(ticker, period = "5y", auto_adjust = True, multi_level_index = False)
    stock_data
    return stock_data, ticker


@app.cell
def _(stock_data):
    # Rename the Date and Closing Price columns (this is the format that's needed for the Prophet library)
    closing_price = stock_data[["Close"]].reset_index()
    closing_price.rename(columns = {"Date": "ds", "Close": "y"}, inplace = True)
    closing_price
    return (closing_price,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Splitting the data into test and training data""")
    return


@app.cell
def _(closing_price):
    # The last 90 days for the test data
    num_days = 90
    test = closing_price[-num_days:]
    train = closing_price[:-num_days]
    return num_days, test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Training the Prophet model on the data""")
    return


@app.cell
def _(Prophet, train):
    model = Prophet(daily_seasonality=True)
    model.fit(train)
    return (model,)


@app.cell
def _(model, num_days):
    future = model.make_future_dataframe(periods = num_days) # Extrapolating future values up to 90 days but this can be tweaked
    prediction = model.predict(future)
    prediction
    return future, prediction


@app.cell
def _(prediction, test):
    forecast = prediction[["ds", "yhat"]].set_index("ds").join(test.set_index("ds"))
    forecast.dropna(subset = "y", inplace = True)
    forecast
    return (forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot the forecasted values alongside the actual values""")
    return


@app.cell
def _(forecast, plt, ticker):
    plt.figure(figsize = (10, 5))
    plt.plot(forecast.index, forecast["y"], label = "Actual")
    plt.plot(forecast.index, forecast["yhat"], label = "Predicted")
    plt.legend()
    plt.title(f"{ticker} Forecasted vs. Actual")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(forecast, mean_absolute_error, mean_squared_error, np):
    mae = mean_absolute_error(forecast['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(forecast['y'], forecast['yhat']))

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    return mae, rmse


if __name__ == "__main__":
    app.run()
