import streamlit as st
import pandas as pd
import plotly.express as px
from fuzzywuzzy import process
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from polygon import WebSocketClient, RESTClient

# Set up your Polygon API key
API_KEY = "YOUR_POLYGON_API_KEY"
client = RESTClient(API_KEY)

st.title("Stock Price Analyzer")

st.write("This tool is developed to analyze stock data, generate plots using technical indicators, and predict stock prices")

# Load the Excel sheet with company data
company_data = pd.read_excel("tickers.xlsx")
company_names = company_data["Name"].tolist()

# Default company and ticker
default_company = "Tesla"
default_ticker = "TSLA"

st.sidebar.header("Enter a Company Name")
company_input = st.sidebar.text_input("Type to search for a company", value=default_company)

# Find the best matches for the company name input dynamically
if company_input:
    best_matches = process.extractBests(company_input, company_names, score_cutoff=70, limit=5)
    suggested_companies = [match[0] for match in best_matches]

    if suggested_companies:
        selected_company = suggested_companies[0]
        selected_ticker = company_data.loc[company_data["Name"] == selected_company, "Ticker"].values[0]
    else:
        selected_company = default_company
        selected_ticker = default_ticker
else:
    selected_company = default_company
    selected_ticker = default_ticker

# Sidebar selection box for company name
selected_company = st.sidebar.selectbox("Select a Company Name", suggested_companies, index=0 if company_input else -1)

# Years of historical data slider
years = st.sidebar.slider("Select Number of years of Historical Data", min_value=1, max_value=10, value=5)

# Sidebar options for 52 Week High graph
st.sidebar.subheader(f"52 Week High Graph for {selected_company}")
show_moving_average = st.sidebar.checkbox("50 Moving Average", value=True)

years_prediction = st.sidebar.slider("Select Number of years to predict", min_value=2, max_value=10, value=5)

# Comparison checkbox
enable_comparison = st.sidebar.checkbox("Compare with Another Company")

# Function to fetch stock data using Polygon.io
def get_stock_data(ticker_symbol, year_list):
    try:
        end = pd.to_datetime('today').strftime("%Y-%m-%d")
        data_frames = []

        for year in year_list:
            start = (pd.to_datetime('today') - pd.DateOffset(years=year)).strftime("%Y-%m-%d")

            # Fetch the historical data from Polygon.io
            try:
                aggs = client.get_aggs(ticker_symbol, 1, "day", from_=start, to=end)
                data = pd.DataFrame(aggs)
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)
                data_frames.append(data)
            except Exception as e:
                st.error(f"Error downloading data for {ticker_symbol} for the year range starting from {start} to {end}: {e}")
                return pd.DataFrame()

        # Combine the data for the years
        yearly_data = pd.concat(data_frames)
        yearly_data.index = pd.to_datetime(yearly_data.index)
        yearly_data = yearly_data.resample('Y').agg({"high": "max", "low": "min", "open": "first", "close": "last"})
        yearly_data.index = yearly_data.index.year.astype(str)

        # Add P/E ratio and Market Cap (You can fetch P/E and market cap from the company's fundamental data)
        pe_ratios = []
        market_caps = []

        for year in yearly_data.index:
            pe_ratio, market_cap = calculate_pe_ratio_and_market_cap(ticker_symbol, int(year))
            pe_ratios.append(pe_ratio)
            market_caps.append(market_cap)

        yearly_data["P/E Ratio"] = pe_ratios
        yearly_data["Market Capacity"] = market_caps
        yearly_data.index.names = ["Year"]
        yearly_data.rename(columns={"high": "52 Week High", "low": "52 Week Low", "open": "Year Open", "close": "Year Close"}, inplace=True)

        return yearly_data

    except KeyError as e:
        st.error(f"Error: {e}. The symbol '{ticker_symbol}' was not found. Please check the symbol and try again.")

# Function to prepare data for predictions
def prepare_data_for_prediction(data, years_prediction):
    # Use the closing price for prediction
    data = data[['close']].copy()
    data['Date'] = pd.to_datetime(data.index)
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data.set_index('Date', inplace=True)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close']])

    # Train the model (SVR)
    X = np.array(range(len(scaled_data))).reshape(-1, 1)
    y = scaled_data

    svr = SVR(kernel='rbf')
    svr.fit(X, y)

    # Make future predictions
    future_days = np.array(range(len(scaled_data), len(scaled_data) + years_prediction * 252)).reshape(-1, 1)
    predicted_stock_price = svr.predict(future_days)

    # Inverse transform to get back to the original scale
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))

    # Create a DataFrame with the predicted data
    predicted_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=years_prediction * 252, freq='B')
    prediction_df = pd.DataFrame(data=predicted_stock_price, index=predicted_dates, columns=['Predicted Close'])
    return prediction_df

# Rest of your functions for plotting

# Fetch and plot stock data
with st.spinner("Fetching stock data..."):
    stock_data = get_stock_data(selected_ticker, [years])

    if stock_data is not None and not stock_data.empty:
        st.write(f"{selected_company} Stock Data:")
        st.write(stock_data)

        # Handle comparison if enabled
        if enable_comparison:
            compare_stock_data = get_stock_data(compare_ticker, [years])

            if compare_stock_data is not None and not compare_stock_data.empty:
                st.write(f"{compare_company} Stock Data:")
                st.write(compare_stock_data)
                plot_stock_data(stock_data, compare_stock_data, selected_company, compare_company, f"{selected_company} vs {compare_company} 52 Week High Comparison", show_moving_average, enable_comparison)
            else:
                st.warning(f"No data available for {compare_company}.")
        else:
            plot_stock_data(stock_data, stock_data, selected_company, selected_company, f"{selected_company} 52 Week High", show_moving_average)

        # Predict stock prices for the selected company
        predicted_data = prepare_data_for_prediction(stock_data, years_prediction)

        if enable_comparison:
            if compare_stock_data is not None and not compare_stock_data.empty:
                compare_predicted_data = prepare_data_for_prediction(compare_stock_data, years_prediction)
                plot_predicted_stock_prices(stock_data, compare_stock_data, predicted_data, compare_predicted_data, selected_company, compare_company, years_prediction, enable_comparison)
            else:
                st.warning(f"No prediction data available for {compare_company}.")
        else:
            plot_predicted_stock_prices(stock_data, stock_data, predicted_data, predicted_data, selected_company, selected_company, years_prediction)
    else:
        st.warning(f"No data available for {selected_company}.")
