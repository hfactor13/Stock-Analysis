

import marimo

__generated_with = "0.13.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/stock_analysis.grid.json",
)


@app.cell
def _():
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import marimo as mo
    import warnings
    warnings.filterwarnings("ignore")
    return (
        Prophet,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        plot_components_plotly,
        plot_plotly,
        yf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Ticker Box""")
    return


@app.cell
def _(mo):
    text_box = mo.ui.text(value = "PYPL", label = "Ticker: ")
    text_box
    return (text_box,)


@app.cell
def _(text_box, yf):
    # Gets the last 5 years of data for the ticker specified
    ticker = text_box.value
    stock_data = yf.download(ticker, period = "5y", auto_adjust = True, multi_level_index = False)
    stock_data
    return (stock_data,)


@app.cell
def _(mo, stock_data):
    stock_field = mo.ui.dropdown(options = stock_data.columns, value = stock_data.columns[0], label = "Stock Field: ")
    stock_field
    return (stock_field,)


@app.cell
def _(stock_data, stock_field):
    # Rename the Date and Closing Price columns (this is the format that's needed for the Prophet library)
    data_for_analysis = stock_data[[stock_field.value]].reset_index()
    data_for_analysis.rename(columns = {"Date": "ds", stock_field.value: "y"}, inplace = True)
    data_for_analysis
    return (data_for_analysis,)


@app.cell(hide_code=True)
def _(mo):
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
    forecast = model.predict(future)
    forecast
    return (forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot the forecasted values alongside the actual values""")
    return


@app.cell
def _(forecast, model, plot_plotly, stock_field):
    plot_plotly(model, forecast, xlabel = "Date", ylabel = f"{stock_field.value}")
    return


@app.cell
def _(forecast, model, plot_components_plotly):
    plot_components_plotly(model, forecast)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculate the Error""")
    return


@app.cell
def _(
    data_for_analysis,
    forecast,
    mean_absolute_error,
    mean_squared_error,
    mo,
    np,
):
    mae = mo.ui.text(value = mean_absolute_error(data_for_analysis['y'], forecast['yhat']))
    rmse = mo.ui.text(value = np.sqrt(mean_squared_error(data_for_analysis['y'], forecast['yhat'])))
    return mae, rmse


@app.cell
def _(mae):
    f"Mean Absolute Error: {mae.value:.2f}"
    return


@app.cell
def _(rmse):
    f"Root Mean Squared Error: {rmse.value:.2f}"
    return


if __name__ == "__main__":
    app.run()
