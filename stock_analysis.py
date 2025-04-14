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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
