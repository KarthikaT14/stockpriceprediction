import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process

# Set the title of the app
st.title("Stock Price Analyzer")
st.write("This tool is developed to analyze stock data, generate plots using technical indicators, and predict stock prices.")

# Load the Excel sheet
company_data = pd.read_excel("tickers.xlsx")  # Ensure this file exists in your working directory
company_names = company_data["Name"].tolist()

# Default company and ticker
default_company = "Tesla"
default_ticker = "TSLA"

# Sidebar input for company name with default value
st.sidebar.header("Enter a Company Name")
company_input = st.sidebar.text_input("Type to search for a company", value=default_company)

# Find the best matches for the company name input dynamically
if company_input:
    best_matches = process.extractBests(company_input, company_names, score_cutoff=70, limit=5)
    suggested_companies = [match[0] for match in best_matches]
else:
    suggested_companies = []

# Sidebar selection box for company name
selected_company = st.sidebar.selectbox("Select a Company Name", suggested_companies, index=0)

if selected_company:
    selected_ticker = company_data.loc[company_data["Name"] == selected_company, "Ticker"].values[0]

    # Years of historical data slider
    years = st.sidebar.slider("Select Number of years of Historical Data", min_value=1, max_value=10, value=5)

    # Sidebar options for 52 Week High graph
    st.sidebar.subheader(f"52 Week High Graph for {selected_company}")
    show_moving_average = st.sidebar.checkbox("50 Moving Average", value=True)

    # Years to predict slider
    years_prediction = st.sidebar.slider("Select Number of years to predict", min_value=2, max_value=10, value=5)

    def get_stock_data(ticker_symbol, year_list):
        try:
            end = pd.to_datetime('today').strftime("%Y-%m-%d")
            data_frames = []
            for year in year_list:
                start = (pd.to_datetime('today') - pd.DateOffset(years=year)).strftime("%Y-%m-%d")
                try:
                    df = yf.download(ticker_symbol, start=start, end=end, progress=False)
                    st.write(f"Data fetched for {ticker_symbol} from {start} to {end}:")
                    st.write(df)  # Display the DataFrame in Streamlit

                    if df.empty or not all(col in df.columns for col in ['Close', 'High', 'Low', 'Open']):
                        st.warning(f"Data for {ticker_symbol} is incomplete or unavailable. Please check the symbol and date range.")
                        return pd.DataFrame()  # Return an empty DataFrame

                    data_frames.append(df)
                except Exception as e:
                    st.error(f"Error downloading data for {ticker_symbol} for the year range starting from {start} to {end}: {e}")
                    return pd.DataFrame()

            if data_frames:
                yearly_data = pd.concat(data_frames)
                yearly_data.index = pd.to_datetime(yearly_data.index)
                yearly_data = yearly_data.resample('Y').agg({"High": "max", "Low": "min", "Open": "first", "Close": "last"})
                yearly_data.index = yearly_data.index.year.astype(str)

                return yearly_data
            else:
                return pd.DataFrame()  # If no data was appended
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

    def plot_stock_data(data, company_name, title, show_moving_average=True):
        fig = px.line(data, x=data.index, y='High', labels={'High': 'Stock Price'})

        if show_moving_average:
            window_50 = 50
            sma_50 = data['Close'].rolling(window=window_50, min_periods=1).mean()
            fig.add_scatter(x=data.index, y=sma_50, mode='lines', name=f'{window_50}-Day Moving Average',
                            line=dict(dash='dash'))

        fig.update_layout(title=f"{title} for {company_name} Over Time",
                          xaxis_title="Year",
                          yaxis_title="Stock Price",
                          legend_title="Indicators",
                          )

        st.plotly_chart(fig)

    def predict_stock_prices(data, company_name, years_prediction):
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        closing_prices = data['Close'].values

        model = ARIMA(closing_prices, order=(5, 1, 2))
        results = model.fit()

        current_year = pd.to_datetime('today').year
        future_years = pd.date_range(start=f"{current_year + 1}-01-01", periods=years_prediction, freq='Y')
        forecast = results.get_forecast(steps=len(future_years))

        future_data = pd.DataFrame(index=future_years, columns=['Predicted Year Close'])
        future_data['Predicted Year Close'] = forecast.predicted_mean

        return future_data

    def main():
        st.subheader(f"Yearly Stock Data for {selected_company} ({selected_ticker})")

        stock_data = get_stock_data(selected_ticker, [years])
        if not stock_data.empty:
            st.write(stock_data)

            st.subheader(f"52 Week High {'with Moving Average' if show_moving_average else ''} for {selected_company}")
            plot_stock_data(stock_data, selected_company, "Stock Data", show_moving_average)

            # Prepare data for regression
            mlr_data = stock_data[['High', 'Close']].copy()
            mlr_data.dropna(inplace=True)
            X = mlr_data[['High']]
            y = mlr_data['Close']
            mlr_model = LinearRegression()
            mlr_model.fit(X, y)

            # Predict future stock prices
            future_stock_prices = predict_stock_prices(stock_data, selected_company, years_prediction)

            st.subheader(f"Predicted Year Close for the Next {years_prediction} Years")
            fig_pred = px.line(future_stock_prices, x=future_stock_prices.index, y='Predicted Year Close',
                               labels={'Predicted Year Close': 'Predicted Stock Price'})

            fig_pred.update_layout(
                title=f"Predicted Year Close Over Time for {selected_company} (ARIMA)",
                xaxis_title="Year",
                yaxis_title="Predicted Stock Price",
                legend_title="Indicators",
            )

            st.plotly_chart(fig_pred)

    if __name__ == "__main__":
        main()
else:
    st.subheader("Please enter and select a company to display its stock data and predictions.")
