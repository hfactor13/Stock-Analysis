import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from prophet import Prophet
    import yfinance as yf
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo
    return Prophet, mean_absolute_error, mean_squared_error, mo, pd, plt, yf


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
    forecast = prediction[["ds", "yhat"]].join(test.set_index("ds"), on = "ds")
    forecast
    return (forecast,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
