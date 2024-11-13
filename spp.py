import streamlit as st
import pandas as pd
import plotly.express as px
from fuzzywuzzy import process
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import intrinio_sdk
from intrinio_sdk.rest import ApiException

# Set up Intrinio API key
intrinio_sdk.ApiClient().configuration.api_key['api_key'] = "OjA2MDhjYTFmNGZhMTViYzFiMjkxM2Q0ZWU5MjVkYjY0"
securities_api = intrinio_sdk.SecurityApi()

st.title("Stock Price Analyzer")
st.write("This tool is developed to analyze stock data, generate plots using technical indicators, and predict stock prices")

# Load Excel sheet with company data
company_data = pd.read_excel("tickers.xlsx")
company_names = company_data["Name"].tolist()

# Default company and ticker
default_company = "Tesla"
default_ticker = "TSLA"

st.sidebar.header("Enter a Company Name")
company_input = st.sidebar.text_input("Type to search for a company", value=default_company)

# Find the best matches for the company name input
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

selected_company = st.sidebar.selectbox("Select a Company Name", suggested_companies, index=0 if company_input else -1)

# Historical data years slider
years = st.sidebar.slider("Select Number of years of Historical Data", min_value=1, max_value=10, value=5)

# Sidebar options for 52 Week High graph
st.sidebar.subheader(f"52 Week High Graph for {selected_company}")
show_moving_average = st.sidebar.checkbox("50 Moving Average", value=True)

years_prediction = st.sidebar.slider("Select Number of years to predict", min_value=2, max_value=10, value=5)

# Comparison checkbox
enable_comparison = st.sidebar.checkbox("Compare with Another Company")

# Function to fetch stock data using Intrinio API
def get_stock_data(ticker_symbol, years):
    try:
        end_date = pd.to_datetime("today").strftime("%Y-%m-%d")
        start_date = (pd.to_datetime("today") - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
        
        # Fetch historical stock prices
        stock_data = []
        try:
            response = securities_api.get_security_stock_prices(ticker_symbol, start_date=start_date, end_date=end_date, frequency='daily')
            for stock_price in response.stock_prices:
                stock_data.append({
                    "date": stock_price.date,
                    "open": stock_price.open,
                    "high": stock_price.high,
                    "low": stock_price.low,
                    "close": stock_price.close,
                    "volume": stock_price.volume
                })
            data = pd.DataFrame(stock_data)
            data.set_index("date", inplace=True)
            return data
        except ApiException as e:
            st.error(f"Error fetching data for {ticker_symbol}: {e}")
            return pd.DataFrame()
    except KeyError as e:
        st.error(f"Error: {e}. The symbol '{ticker_symbol}' was not found. Please check the symbol and try again.")

# Function to prepare data for predictions
def prepare_data_for_prediction(data, years_prediction):
    data = data[['close']].copy()
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close']])

    X = np.array(range(len(scaled_data))).reshape(-1, 1)
    y = scaled_data

    svr = SVR(kernel='rbf')
    svr.fit(X, y)

    future_days = np.array(range(len(scaled_data), len(scaled_data) + years_prediction * 252)).reshape(-1, 1)
    predicted_stock_price = svr.predict(future_days)

    predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))

    predicted_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=years_prediction * 252, freq='B')
    prediction_df = pd.DataFrame(data=predicted_stock_price, index=predicted_dates, columns=['Predicted Close'])
    return prediction_df

# Function to plot stock data and predictions
def plot_stock_data(data, title, show_moving_average=False):
    fig = px.line(data, x=data.index, y='close', title=title)
    if show_moving_average:
        data['MA50'] = data['close'].rolling(window=50).mean()
        fig.add_scatter(x=data.index, y=data['MA50'], mode='lines', name='50-day MA')
    st.plotly_chart(fig)

def plot_predicted_stock_prices(data, predicted_data, company_name, years_prediction):
    fig = px.line(data, x=data.index, y='close', title=f"{company_name} Stock Price Prediction")
    fig.add_scatter(x=predicted_data.index, y=predicted_data['Predicted Close'], mode='lines', name=f"{years_prediction} Year Prediction")
    st.plotly_chart(fig)

# Fetch and plot stock data
with st.spinner("Fetching stock data..."):
    stock_data = get_stock_data(selected_ticker, years)
    if stock_data is not None and not stock_data.empty:
        st.write(f"{selected_company} Stock Data:")
        st.write(stock_data)

        # Plot the stock data with options
        plot_stock_data(stock_data, f"{selected_company} 52 Week High", show_moving_average)

        # Predict stock prices
        predicted_data = prepare_data_for_prediction(stock_data, years_prediction)
        plot_predicted_stock_prices(stock_data, predicted_data, selected_company, years_prediction)
    else:
        st.warning(f"No data available for {selected_company}.")
