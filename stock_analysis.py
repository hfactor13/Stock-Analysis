

import marimo

__generated_with = "0.13.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/stock_analysis.grid.json",
)

with app.setup:
    # Initialization code that runs before all other cells
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    from datetime import timedelta, datetime
    from pathlib import Path
    from yahooquery import Ticker
    import pandas as pd
    import marimo as mo
    import subprocess
    import warnings
    warnings.filterwarnings("ignore")


@app.function
def get_stock_data(symbol, cache_dir='cache', period = "5y", max_age=timedelta(days=1)):
    """
    Fetches stock data for the given symbol with caching to CSV.
    Deletes the cache file if it's older than max_age.

    Args:
        symbol (str): Stock ticker symbol.
        cache_dir (str): Directory to store cache files.
        max_age (timedelta): Maximum cache age allowed (default = 1 day).

    Returns:
        pd.DataFrame: Stock data.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    file_path = cache_path / f"{symbol}.csv"

    # Check if cache exists
    if file_path.exists():
        modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        age = datetime.now() - modified_time
        if age < max_age:
            print(f"Loading {symbol} data from cache...")
            return pd.read_csv(file_path)
        else:
            print(f"Deleting expired cache for {symbol}...")
            file_path.unlink()  # Delete expired file

    print(f"Fetching {symbol} data from Yahoo Finance...")
    ticker = Ticker(symbol)
    data = ticker.history(period = period)

    if data.empty:
        raise ValueError(f"No data returned for {symbol}")

    data.to_csv(file_path)
    return data


@app.cell(hide_code=True)
def _():
    mo.md(r"""Ticker Box""")
    return


@app.cell
def _():
    text_box = mo.ui.text(value = "PYPL", label = "Ticker: ")
    text_box
    return (text_box,)


@app.cell
def _(text_box):
    # Gets the last 5 years of data for the ticker specified
    ticker = text_box.value
    stock_data = get_stock_data(ticker, period = "5y").iloc[:,1:] # omits symbol column
    stock_data.reset_index(inplace = True)
    stock_data["date"] = pd.to_datetime(stock_data["date"].apply(lambda x: x[:10]))
    stock_data
    return (stock_data,)


@app.cell
def _(stock_data):
    stock_field = mo.ui.dropdown(options = stock_data.columns[1:], value = stock_data.columns[4], label = "Stock Field: ")
    stock_field
    return (stock_field,)


@app.cell
def _(stock_data, stock_field):
    # Rename the date and closing price columns (this is the format that's needed for the Prophet library)
    data_for_analysis = stock_data.loc[:,["date", stock_field.value]]
    data_for_analysis.rename(columns = {"date": "ds", stock_field.value: "y"}, inplace = True)
    data_for_analysis
    return (data_for_analysis,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""Splitting the data into test and training data""")
    return


@app.cell
def _(data_for_analysis):
    # The last 90 days for the test data
    num_days = 90
    test = data_for_analysis[-num_days:]
    train = data_for_analysis[:-num_days]
    return num_days, train


@app.cell(hide_code=True)
def _():
    mo.md(r"""Training the Prophet model on the data""")
    return


@app.cell
def _(train):
    model = Prophet(daily_seasonality=True)
    model.fit(train)
    return (model,)


@app.cell
def _(model, num_days):
    future = model.make_future_dataframe(periods = num_days) # Extrapolating future values up to 90 days but this can be tweaked
    forecast = model.predict(future)
    forecast
    return (forecast,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""Plot the forecasted values alongside the actual values""")
    return


@app.cell
def _(forecast, model, stock_field):
    plot_plotly(model, forecast, xlabel = "Date", ylabel = f"{stock_field.value}")
    return


@app.cell
def _(forecast, model):
    plot_components_plotly(model, forecast)
    return


if __name__ == "__main__":
    app.run()
