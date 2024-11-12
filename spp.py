import streamlit as st

import pandas as pd

import yfinance as yf

import plotly.express as px

from fuzzywuzzy import process

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVR


st.title("Stock Price Analyzer")

st.write("This tool is developed to analyze stock data, generate plots using technical indicators, and predict stock prices")


# Load the Excel sheet

company_data = pd.read_excel("C:/Users/KARTHIKA/Downloads/tickers.xlsx")

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


# Second company for comparison if checkbox is enabled

if enable_comparison:

    st.sidebar.header("Compare with Another Company")

    compare_company_input = st.sidebar.text_input("Type to search for another company", value="Microsoft")

    compare_best_matches = process.extractBests(compare_company_input, company_names, score_cutoff=70, limit=5)

    compare_suggested_companies = [match[0] for match in compare_best_matches]

    if compare_suggested_companies:

        compare_company = compare_suggested_companies[0]

        compare_ticker = company_data.loc[company_data["Name"] == compare_company, "Ticker"].values[0]

    else:

        compare_company = "Microsoft"

        compare_ticker = "MSFT"


# Years to predict slider



def get_stock_data(ticker_symbol, year_list):

    try:

        end = pd.to_datetime('today').strftime("%Y-%m-%d")

        data_frames = []

        for year in year_list:

            start = (pd.to_datetime('today') - pd.DateOffset(years=year)).strftime("%Y-%m-%d")

            try:

                df = yf.download(ticker_symbol, start=start, end=end, progress=False)

                data_frames.append(df)

            except Exception as e:

                st.error(f"Error downloading data for {ticker_symbol} for the year range starting from {start} to {end}: {e}")

                return pd.DataFrame()


        yearly_data = pd.concat(data_frames)


        yearly_data.index = pd.to_datetime(yearly_data.index)

        yearly_data = yearly_data.resample('Y').agg({"High": "max", "Low": "min", "Open": "first", "Close": "last"})

        yearly_data.index = yearly_data.index.year.astype(str)


        pe_ratios = []

        market_caps = []

        for year in yearly_data.index:

            pe_ratio, market_cap = calculate_pe_ratio_and_market_cap(ticker_symbol, int(year))

            pe_ratios.append(pe_ratio)

            market_caps.append(market_cap)


        yearly_data["P/E Ratio"] = pe_ratios

        yearly_data["Market Capacity"] = market_caps


        yearly_data.index.names = ["Year"]

        yearly_data.rename(columns={"High": "52 Week High", "Low": "52 Week Low", "Open": "Year Open", "Close": "Year Close"}, inplace=True)


        return yearly_data


    except KeyError as e:

        st.error(f"Error: {e}. The symbol '{ticker_symbol}' was not found. Please check the symbol and try again.")


def calculate_pe_ratio_and_market_cap(ticker_symbol, year):

    try:

        start_date = pd.to_datetime(f"{year}-01-01")

        end_date = pd.to_datetime(f"{year}-12-31")


        stock_info = yf.Ticker(ticker_symbol)

        info = stock_info.history(start=start_date, end=end_date)


        if not info.empty:

            close_price = info['Close'].mean()

            eps = stock_info.info.get('trailingEps', 'N/A')

            market_cap = close_price * stock_info.info.get('sharesOutstanding', 'N/A')

            if eps != 'N/A' and close_price > 0:

                pe_ratio = close_price / eps

            else:

                pe_ratio = 'N/A'

        else:

            pe_ratio = 'N/A'

            market_cap = 'N/A'


        return pe_ratio, market_cap


    except KeyError as e:

        st.error(f"Error: {e}. There was an issue with retrieving data for the specified year.")


def plot_stock_data(data, compare_data, company_name, compare_company_name, title, show_moving_average=True, enable_comparison=False):

    # Create the figure and add the main company's 52 Week High

    fig = px.line(data, x=data.index, y='52 Week High', labels={'52 Week High': 'Stock Price'},

                  title=title if not enable_comparison else f"{company_name} vs {compare_company_name} - {title}")


    # Plot main company's 52 Week High

    fig.add_scatter(x=data.index, y=data['52 Week High'], mode='lines', name=f'{company_name} 52 Week High')


    # Add comparison company's 52 Week High if comparison is enabled

    if enable_comparison:

        fig.add_scatter(x=compare_data.index, y=compare_data['52 Week High'], mode='lines', name=f'{compare_company_name} 52 Week High')


    # Plot 50-day moving average for main company

    if show_moving_average:

        window_50 = 50

        sma_50 = data['Year Close'].rolling(window=window_50, min_periods=1).mean()

        fig.add_scatter(x=data.index, y=sma_50, mode='lines', name=f'{company_name} 50-Day Moving Avg', line=dict(dash='dash'))


        # Add 50-day moving average for comparison company if comparison is enabled

        if enable_comparison:

            compare_sma_50 = compare_data['Year Close'].rolling(window=window_50, min_periods=1).mean()

            fig.add_scatter(x=compare_data.index, y=compare_sma_50, mode='lines', name=f'{compare_company_name} 50-Day Moving Avg', line=dict(dash='dash'))


    # Update layout

    fig.update_layout(

        xaxis_title="Year",

        yaxis_title="Stock Price",

        legend_title="Indicators"

    )


    st.plotly_chart(fig)




def predict_stock_prices(data, company_name, years_prediction):

    if not isinstance(data.index, pd.DatetimeIndex):

        data.index = pd.to_datetime(data.index)


    closing_prices = data['Year Close'].values.reshape(-1, 1)


    # Scale the data

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(closing_prices)


    # Prepare the data for SVR

    time_step = min(60, len(scaled_data) // 2)

    X_train, y_train = [], []

    

    for i in range(time_step, len(scaled_data)):

        X_train.append(scaled_data[i - time_step:i, 0])

        y_train.append(scaled_data[i, 0])


    X_train, y_train = np.array(X_train), np.array(y_train)


    # Reshape data to fit SVR requirements

    X_train = X_train.reshape(X_train.shape[0], time_step)


    # Initialize and train the SVR model

    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

    model.fit(X_train, y_train)


    # Predict future prices

    predictions = []

    last_sequence = X_train[-1]  # Start with the last known sequence


    for _ in range(years_prediction):

        prediction = model.predict(last_sequence.reshape(1, -1))

        predictions.append(prediction[0])

        last_sequence = np.append(last_sequence[1:], prediction, axis=0)


    # Transform the predictions back to original scale

    if len(predictions) > 0:

        future_data = pd.DataFrame(index=pd.date_range(start=f"{pd.to_datetime('today').year + 1}-01-01", periods=years_prediction, freq='Y'), columns=['Predicted Year Close'])

        future_data['Predicted Year Close'] = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        return future_data

    else:

        return pd.DataFrame() 

    

def plot_predicted_stock_prices(stock_data, compare_stock_data, predicted_data, compare_predicted_data, company_name, compare_company_name, years_prediction, enable_comparison=False):

    # Check if predicted_data is None or empty

    if predicted_data is None or predicted_data.empty:

        st.error(f"No predicted data available for {company_name}.")

        return


    # Create the figure and add the main company's predicted stock price

    fig = px.line(predicted_data, x=predicted_data.index, y='Predicted Year Close', labels={'Predicted Year Close': 'Predicted Stock Price'},

                  title=f"{company_name} Predicted Stock Price" if not enable_comparison else f"{company_name} vs {compare_company_name} Predicted Stock Price Comparison")


    fig.add_scatter(x=predicted_data.index, y=predicted_data['Predicted Year Close'], mode='lines', name=f'{company_name} Predicted Price')


    # Only add comparison company's predicted stock price if comparison is enabled

    if enable_comparison:

        if compare_predicted_data is None or compare_predicted_data.empty:

            st.error(f"No predicted data available for {compare_company_name}.")

            return

        fig.add_scatter(x=compare_predicted_data.index, y=compare_predicted_data['Predicted Year Close'], mode='lines', name=f'{compare_company_name} Predicted Price')


    # Update layout

    fig.update_layout(

        xaxis_title="Year",

        yaxis_title="Stock Price",

        legend_title="Company"

    )


    st.plotly_chart(fig)



with st.spinner("Fetching stock data..."):

    stock_data = get_stock_data(selected_ticker, [years])

    if stock_data is not None and not stock_data.empty:

        st.write(f"{selected_company} Stock Data:")

        st.write(stock_data)

        

        if enable_comparison:

            compare_stock_data = get_stock_data(compare_ticker, [years])

            if compare_stock_data is not None and not compare_stock_data.empty:

                st.write(f"{compare_company} Stock Data:")

                st.write(compare_stock_data)

                plot_stock_data(stock_data, compare_stock_data, selected_company, compare_company,

                            f"{selected_company} vs {compare_company} 52 Week High Comparison", show_moving_average, enable_comparison)

            else:

                st.warning(f"No data available for {compare_company}.")

        else:

            plot_stock_data(stock_data, stock_data, selected_company, selected_company, f"{selected_company} 52 Week High", show_moving_average)

            

        predicted_data = predict_stock_prices(stock_data, selected_company, years_prediction)

        if enable_comparison:

            if compare_stock_data is not None and not compare_stock_data.empty:

                compare_predicted_data = predict_stock_prices(compare_stock_data, compare_company, years_prediction)

                if compare_predicted_data is not None and not compare_predicted_data.empty:

                    plot_predicted_stock_prices(stock_data, compare_stock_data, predicted_data, compare_predicted_data, selected_company, compare_company, years_prediction, enable_comparison)

                else:

                    st.warning(f"No prediction data available for {compare_company}.")

            else:

                st.warning(f"No data available for {compare_company} to make predictions.")

        else:

            plot_predicted_stock_prices(stock_data, stock_data, predicted_data, predicted_data, selected_company, selected_company, years_prediction)

            

    else:

        st.warning(f"No data available for {selected_company}.")

