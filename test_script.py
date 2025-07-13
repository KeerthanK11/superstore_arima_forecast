import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

print("Starting ARIMA model building process for multiple categories, countries, and specific totals...")

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('superstore_forecasting_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'superstore_forecasting_dataset.csv' not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# Convert 'Order Date' to datetime objects
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Ensure 'Sales', 'Category', and 'Country' columns exist
if 'Sales' not in df.columns:
    print("Error: 'Sales' column not found in the dataset. Please check the column name.")
    exit()
if 'Category' not in df.columns:
    print("Error: 'Category' column not found in the dataset. Please check the column name.")
    exit()
if 'Country' not in df.columns:
    print("Error: 'Country' column not found in the dataset. Please check the column name.")
    exit()

def run_arima_forecast(category_label, country_label, main_df, forecast_type="category_country"):
    """
    Performs SARIMA forecasting based on the specified forecast_type.

    Args:
        category_label (str, optional): The name of the category to forecast. Required for "category_country" and "total_category".
        country_label (str, optional): The name of the country to forecast. Required for "category_country".
        main_df (pd.DataFrame): The main DataFrame containing all data.
        forecast_type (str): Specifies the type of forecast:
                             - "category_country": Forecasts for a specific category within a specific country.
                             - "total_category": Forecasts for a specific category across all countries.
                             - "total_overall": Forecasts for total sales across all categories and countries.
    """
    ts = pd.Series() # Initialize an empty series
    title_suffix = ""
    search_label = ""

    if forecast_type == "total_overall":
        print(f"\n📈 Total Sales Forecasting (All Categories & Countries)")
        ts = main_df.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum()
        title_suffix = "Total Monthly Sales"
        search_label = "Total Sales"
    elif forecast_type == "total_category":
        if category_label is None:
            print("Error: category_label must be provided for 'total_category' forecast_type.")
            return
        print(f"\n📈 Total {category_label} Sales Forecasting (All Countries)")
        df_filtered = main_df[main_df['Category'] == category_label]
        ts = df_filtered.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum()
        title_suffix = f"Total {category_label} Monthly Sales"
        search_label = f"Total {category_label} Sales"
    elif forecast_type == "category_country":
        if category_label is None or country_label is None:
            print("Error: category_label and country_label must be provided for 'category_country' forecast_type.")
            return
        print(f"\n📦 Category: {category_label}, Country: {country_label}")
        df_filtered = main_df[(main_df['Category'] == category_label) & (main_df['Country'] == country_label)]
        ts = df_filtered.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum()
        title_suffix = f"{category_label} in {country_label} Monthly Sales"
        search_label = f"{category_label} in {country_label}"
    else:
        print(f"Error: Invalid forecast_type '{forecast_type}'.")
        return

    ts = ts.dropna() # Drop any months with no sales after aggregation
    # Filter data to start from a specific date if needed
    ts = ts[ts.index >= '2014-01-01']

    if ts.empty:
        print(f"Skipping {search_label}: No sales data available for the specified period.")
        return
    if len(ts) < 24: # A heuristic: need at least 2 years of data for monthly seasonality
        print(f"Skipping {search_label}: Not enough data points ({len(ts)}) for effective seasonal forecasting. At least 24 recommended.")
        return

    # Plot historical data
    plt.figure(figsize=(12, 4))
    plt.plot(ts, label='Historical Monthly Sales', color='blue')
    plt.title(title_suffix)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Grid search for optimal SARIMA parameters ---
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

    best_aic = np.inf
    best_model = None
    best_param = None
    best_seasonal = None

    print(f"Searching for optimal SARIMA parameters for {search_label} (this may take a moment)...")
    for param in pdq:
        for seasonal in seasonal_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(ts,
                                                  order=param,
                                                  seasonal_order=seasonal,
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=100)
                
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_model = result
                    best_param = param
                    best_seasonal = seasonal
            except Exception as e:
                # print(f"Error fitting ARIMA{param}x{seasonal} for {search_label}: {e}")
                continue

    if best_model is None:
        print(f"❌ No valid SARIMA model found for {search_label} after grid search. Skipping forecast.")
        return

    print(f"✅ Best SARIMA Model for {search_label}: ARIMA{best_param}x{best_seasonal} - AIC: {best_aic:.2f}")
    # print(best_model.summary())

    # --- In-sample forecasting (for past values) ---
    # Get predictions for the training data
    in_sample_pred = best_model.predict(start=ts.index[0], end=ts.index[-1])

    plt.figure(figsize=(14, 5))
    plt.plot(ts, label='Historical Sales (Actual)', color='blue')
    plt.plot(in_sample_pred, label='In-sample Forecast (Fitted)', color='purple', linestyle=':')
    plt.title(f"{search_label} - In-sample Forecast (Model Fit)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"In-sample forecast for {search_label} (how well the model fits past data):")
    print(in_sample_pred)


    # --- Forecast next 12 months (out-of-sample) ---
    n_forecast = 12
    forecast_results = best_model.get_forecast(steps=n_forecast)
    forecast_mean = forecast_results.predicted_mean
    forecast_ci = forecast_results.conf_int()

    forecast_index = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=n_forecast, freq='MS')
    forecast_series = pd.Series(forecast_mean, index=forecast_index)
    lower_bound = pd.Series(forecast_ci.iloc[:, 0], index=forecast_index)
    upper_bound = pd.Series(forecast_ci.iloc[:, 1], index=forecast_index)

    print(f"Forecast for the next {n_forecast} months for {search_label} (Out-of-sample):")
    print(forecast_series)

    # --- Plot forecast (out-of-sample) ---
    plt.figure(figsize=(14, 5))
    plt.plot(ts, label='Historical Sales', color='blue')
    plt.plot(forecast_series, label='Forecasted Sales', color='red', linestyle='--')
    plt.fill_between(forecast_index,
                     lower_bound,
                     upper_bound,
                     color='pink', alpha=0.3, label='95% Confidence Interval')
    plt.title(f"{search_label} - 12 Month Sales Forecast (Out-of-sample)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Forecast accuracy (backtesting on last 12 months) ---
    test_size = 12
    if len(ts) > test_size * 2:
        train = ts[:-test_size]
        test = ts[-test_size:]

        try:
            model_train = sm.tsa.statespace.SARIMAX(train,
                                                    order=best_param,
                                                    seasonal_order=best_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
            results_train = model_train.fit(disp=False, maxiter=100)
            pred = results_train.forecast(steps=test_size)

            mae = mean_absolute_error(test, pred)
            rmse = np.sqrt(mean_squared_error(test, pred))

            print(f"\n📊 Accuracy Metrics for {search_label} (last {test_size} months backtest):")
            print(f"🔹 MAE  : {mae:.2f}")
            print(f"🔹 RMSE : {rmse:.2f}")
        except Exception as e:
            print(f"⚠️ Error during backtesting for {search_label}: {e}")
    else:
        print(f"⚠️ Not enough data for backtesting accuracy for {search_label}. Need at least {test_size * 2} data points.")

# --- Main execution loop for each category-country combination ---
print("\nIdentifying unique category-country combinations...")
unique_combinations = df[['Category', 'Country']].drop_duplicates().values

print(f"Found {len(unique_combinations)} unique category-country combinations.")

# for category, country in unique_combinations:
#     run_arima_forecast(category, country, df, forecast_type="category_country")

# --- Total Sales Forecasting for specific categories ---
all_categories = df['Category'].unique()

if 'Furniture' in all_categories:
    run_arima_forecast(category_label='Furniture', country_label=None, main_df=df, forecast_type="total_category")
else:
    print("\nSkipping 'Total Furniture' forecast: 'Furniture' category not found in the dataset.")

if 'Office Supplies' in all_categories:
    run_arima_forecast(category_label='Office Supplies', country_label=None, main_df=df, forecast_type="total_category")
else:
    print("\nSkipping 'Total Office Supplies' forecast: 'Office Supplies' category not found in the dataset.")

# --- Overall Total Sales Forecasting ---
run_arima_forecast(category_label=None, country_label=None, main_df=df, forecast_type="total_overall")

print("\nARIMA forecasting process for all category-country combinations, specific category totals, and overall total sales completed.")
