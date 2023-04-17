import os
import pickle
import re

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Get the parent directory path
parent_dir_path = os.path.dirname(os.path.dirname(current_script_path))

# Dataset path
data_dir = os.path.join(parent_dir_path, 'pre-processed-data',
                        'modified_pump_data_with_fault_status.xlsx')

# load data from a Excel file
df = pd.read_excel(data_dir)

df_copy = df.copy()  # create a copy of the original DataFrame
# Convert the DateTime column to the pandas DatetimeIndex
df['DateTime'] = pd.to_datetime(df['DateTime'])
# Set the DateTime column as the DataFrame index
df.set_index('DateTime', inplace=True)
df = df.sort_index()

# Set the frequency to 10 minutes
df = df.asfreq('10T')

# replace null with np.nan
df.replace('Null', np.nan, inplace=True)

# drop all rows with null values
df = df.dropna()


# check for stationarity using Augmented Dickey-Fuller test
def test_stationarity(timeseries, pump_feature, pump):
    # The rolling mean is also called the moving average and is calculated by taking the mean of a rolling
    # window of data points. For example, if we have a time series with 100 data points and we want to
    # calculate the rolling mean over a window of 10 data points, we would calculate the mean of the first
    # 10 data points, then move the window one data point to the right and recalculate the mean of the next
    # 10 data points, and so on. The resulting rolling mean time series will have the same length as the original
    # time series, but the data points will be smoother and less noisy.
    # The rolling standard deviation is similar to the rolling mean, but it measures the variability of the data points
    # in the rolling window. It is calculated by taking the standard deviation of a rolling window of data points in the same way as the rolling mean.

    # calculate rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # plot rolling statistics
    plt.plot(timeseries, color='teal', label='Original')
    plt.plot(rolmean, color='deeppink', label='Rolling Mean')
    plt.plot(rolstd, color='deepskyblue', label='Rolling Std')
    plt.legend(loc='best')
    plt.title(
        f'Rolling Mean & Standard Deviation for {pump_feature} (Pump {pump})')
    plt.show()

    # perform Augmented Dickey-Fuller test
    print(
        f'Results of Augmented Dickey-Fuller Test for {pump_feature} (Pump {pump}):')
    adf_test = adfuller(timeseries, autolag='AIC')
    adf_output = pd.Series(adf_test[0:4], index=[
                           'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in adf_test[4].items():
        adf_output['Critical Value (%s)' % key] = value
    print(adf_output)


# decompose time series into trend, seasonal, and residual components
def decompose(timeseries, pump_feature, pump):
    decomposition = seasonal_decompose(timeseries, period=144)
    # Trend component: This component shows the overall direction of the time series over a long period of time.
    # It represents the long-term changes in the series and can help identify any underlying patterns or trends that may be present.
    trend = decomposition.trend
    # Seasonal component: This component shows the repeated patterns that occur in the time series over a fixed period, such as daily,
    # weekly, or monthly cycles. It can help identify any seasonal patterns or cycles that may exist in the series.
    seasonal = decomposition.seasonal
    # Residual component: This component represents the variation in the time series that is not accounted for by the trend or seasonal
    # components. It is the difference between the observed values and the values predicted by the trend and seasonal components.
    # The residual component can provide insights into the randomness or noise in the series, and can be useful in identifying any unusual
    # or anomalous behavior.
    residual = decomposition.resid

    # plot decomposed components
    plt.subplot(411)
    plt.plot(timeseries, label=f'Original ({pump_feature} , Pump {pump})')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label=f'Seasonality ({pump_feature} , Pump {pump}')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label=f'Residuals ({pump_feature} , Pump {pump}')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# plot autocorrelation and partial autocorrelation functions
def plot_acf_pacf(timeseries, pump_feature, pump):
    # Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) are two important tools used in
    # time series analysis to identify the possible order of the Autoregressive (AR) and Moving Average (MA) models.
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 6))
    plot_acf(timeseries, lags=50, ax=ax1,
             title=f'Autocorrelation Plot ({pump_feature} , Pump {pump})')
    plot_pacf(timeseries, lags=50, ax=ax2,
              title=f'Partial Autocorrelation Plot ({pump_feature} , Pump {pump})')
    plt.show()


# Run time series analysis on each pump
def time_series_analysis(df):
    for pump in range(1, 4):
        pump_current = f'KTF{pump}_Current'
        pump_power = f'KTF{pump}_Power'
        pump_temp = f'KTF{pump}_Temp'
        pump_voltage = f'KTF{pump}_Voltage'
        pump_vsd_speed = f'KTF{pump}_VSDSpeed(%)'
        pump_fault_status = f'KTF{pump}_Fault_Status'

        # check for stationarity using Augmented Dickey-Fuller test
        print(f'Stationarity test for {pump_current}')
        test_stationarity(df[pump_current], pump_current, pump)
        print(f'Stationarity test for {pump_power}')
        test_stationarity(df[pump_power], pump_power, pump)
        print(f'Stationarity test for {pump_temp}')
        test_stationarity(df[pump_temp], pump_temp, pump)

        # decompose time series into trend, seasonal, and residual components
        print(f'Decomposition of {pump_current}')
        decompose(df[pump_current], pump_current, pump)
        print(f'Decomposition of {pump_power}')
        decompose(df[pump_power], pump_power, pump)
        print(f'Decomposition of {pump_temp}')
        decompose(df[pump_temp], pump_power, pump)

        # plot autocorrelation and partial autocorrelation functions
        print(f'ACF and PACF for {pump_current}')
        plot_acf_pacf(df[pump_current], pump_current, pump)
        print(f'ACF and PACF for {pump_power}')
        plot_acf_pacf(df[pump_power], pump_power, pump)
        print(f'ACF and PACF for {pump_temp}')
        plot_acf_pacf(df[pump_temp], pump_temp, pump)


# Fill NaN values in target columns with their corresponding values from original shifted targets
def replace_target_nan_values(df, _days):
    for _pump in range(1, 4):
        for _lag in range(1, _days + 1):
            df[f"KTF{_pump}_Fault_Status_next_" + str(_lag) + "_day(s)_sum"].fillna(
                df[f"KTF{_pump}_Fault_Status_next_" + str(_lag) + "_day(s)"], inplace=True)
    return df


# Create lagged features
def create_lag_features(df, aggregate_data, lags, _days):
    # lagged_columns will be used later for showing the imortance of each of these features
    lagged_columns = []
    # Since the data is aggregated and converted from 10 min interval to 1H, they don't need to be multiplied by 6
    df_new_list = []
    for _lag in range(1, lags+1):
        # create lagged columns for each feature
        for feature in list(df.columns):
            if aggregate_data:
                df_new_list.append(df[feature].shift(
                    24*_lag).rename(f'{feature}_lag_{_lag}_day(s)'))
            else:
                df_new_list.append(df[feature].shift(
                    6*24*_lag).rename(f'{feature}_lag_{_lag}_day(s)'))
            lagged_columns.append(f'{feature}_lag_{_lag}_day(s)')

    for _pump in range(1, 4):
        for _lag in range(1, _days + 1):
            if aggregate_data:
                df_new_list.append(df[f"KTF{_pump}_Fault_Status_sum"].rolling(
                    24*_lag).sum().shift(-24*_lag).rename(f"KTF{_pump}_Fault_Status_next_" + str(_lag) + "_day(s)_sum"))
                df_new_list[-1].fillna(
                    df[f"KTF{_pump}_Fault_Status_sum"], inplace=True)
            else:
                df_new_list.append(df[f"KTF{_pump}_Fault_Status"].rolling(
                    6*24*_lag).sum().shift(-6*24*_lag).rename(f"KTF{_pump}_Fault_Status_next_" + str(_lag) + "_day(s)"))
                df_new_list[-1].fillna(
                    df[f"KTF{_pump}_Fault_Status"], inplace=True)

    # Concatenate all new DataFrame columns
    new_df = pd.concat(df_new_list, axis=1)
    new_df = pd.concat([df, new_df], axis=1)
    new_df = new_df.dropna()

    return new_df, lagged_columns


def save_lagged_features(new_df, lagged_columns, _df_name, _df_columns):
    # Save lagged_cols to a file using pickle
    with open(os.path.join(parent_dir_path, 'pre-processed-data', f'{_df_columns}.pkl'), 'wb') as f:
        pickle.dump(lagged_columns, f)
    # Save DataFrame as Excel file, Exclude the first lags[-1] rows to avoid empty cells created because of the lags
    new_df = new_df.fillna(-1)
    new_df.to_csv(os.path.join(parent_dir_path,
                  'pre-processed-data', f'{_df_name}.csv'))
    # new_df.to_excel(
    #     f'{os.getcwd()}/{_df_name}.xlsx', engine='xlsxwriter')
    print(f"{_df_name} created !!!!!!!! ")

# Create time-based features from the original ones for future prediction


def aggregate_features(df):

    # Create a new Dataframe to stir aggregated data
    df_merge = pd.DataFrame()

    interval = '1H'  # 1D
    # reset the index of df to create a new index with consecutive integers
    # df = df.reset_index(drop=True)
    # resample the data to 1 hour intervals and calculate the min, max, and mean values for each column
    for _pump in range(1, 4):
        df_agg = df.resample(interval).agg({
            # f'KTF{_pump}_Current': ['sum'],
            f'KTF{_pump}_Voltage': ['sum'],
            f'KTF{_pump}_Power': ['sum'],
            f'KTF{_pump}_Temp': ['min', 'max', 'mean', 'sum', 'std'],
            f'KTF{_pump}_VSDSpeedHz': ['sum'],
            # f'KTF{_pump}_VSDSpeed(%)': ['sum'],
            f'KTF{_pump}_Run_Status': ['sum'],
            f'KTF{_pump}_Fault_Status': ['min', 'max', 'mean', 'sum', 'std'],
        })

        # flatten the column names
        df_agg.columns = ['_'.join(col).strip()
                          for col in df_agg.columns.values]
        # Merge the new DataFrame with the original DataFrame
        df_merge = pd.concat([df_merge, df_agg], axis=1)
    # reset the index to bring the datetime index back into the DataFrame as a regular column
    # df_merge = df_merge.reset_index()

    return df_merge


# Create time-based features from the original ones for future prediction
def create_time_based_features(df, df_new, aggregate_data):
    # Create time-based features
    # Hour of day
    df_new['hour_of_day'] = df.index.hour
    # Day of week
    df_new['day_of_week'] = df.index.dayofweek
    # Days of a week
    # _days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    # for _day in range(len(_days)):
    #     df_new[f'day_of_week_{_days[_day]}'] = (
    #         df_new['day_of_week'] == _day).astype(int)

    # Day of month
    # df_new['day_of_month'] = df.index.day
    # Week of year
    # df_new['week_of_year'] = df.index.isocalendar().week
    # Month of year
    # df_new['month_of_year'] = df.index.month
    # Quarter of year
    # df_new['quarter'] = df.index.quarter
    # Year
    # df_new['year'] = df.index.year
    # Holidays
    # Create a holiday object for the US
    au_holidays = holidays.AU()
    # Create a binary feature for holidays
    # df_new['is_holiday'] = [
    #     1 if date in au_holidays else 0 for date in df.index.date]

    # Merge the new DataFrame with the original DataFrame
    df = pd.concat([df, df_new], axis=1)

    # Create columns for pump start and fault times
    for _pump in range(1, 4):
        if aggregate_data:
            pump_running_col = f'KTF{_pump}_Run_Status_mean'
            pump_fault_col = f'KTF{_pump}_Fault_Status_mean'
        else:
            pump_running_col = f'KTF{_pump}_Run_Status'
            pump_fault_col = f'KTF{_pump}_Fault_Status'

        for period, (start_hour, end_hour) in {'late_at_night': (0, 6), 'in_early_morning': (6, 9), 'in_morning': (9, 12), 'in_afternoon': (12, 18), 'in_evening': (18, 21), 'at_night': (21, 24)}.items():
            start_col = f'KTF{_pump}_start_{period}'
            fault_col = f'KTF{_pump}_fault_{period}'
            df[start_col] = ((df.index.hour >= start_hour) & (
                df.index.hour < end_hour) & (df[pump_running_col] == 1)).astype(int)
            df[fault_col] = ((df.index.hour >= start_hour) & (
                df.index.hour < end_hour) & (df[pump_fault_col] == 1)).astype(int)

    # Assuming the fault event is recorded in a column named 'fault_event'
    # for pump in range(1, 4):
    #     df[f'time_since_last_fault_{pump}'] = df_copy.groupby((df_copy[f'KTF{pump}_Fault_Status'] != df_copy[f'KTF{pump}_Fault_Status'].shift()).cumsum())['DateTime'].apply(
    #             lambda x: x.diff().dt.total_seconds().fillna(0).cumsum() / 86400).astype(int)

    return df


# create a new column for the time since last fault event for each pump
def time_since_last_fault(df):
    for pump in range(1, 4):
        fault_col = f'KTF{pump}_Fault_Status'
        df[fault_col] = df[fault_col].astype(bool)
        df.reset_index(inplace=True)
        time_since_fault_col = f'time_since_last_fault_{pump}'
        # calculate time since last fault event
        # df[time_since_fault_col] = df[fault_col].diff().ne(0).cumsum()
        # calculate the time since the last fault event using the boolean column
        df[time_since_fault_col] = df[fault_col].groupby(
            (df[fault_col] != df[fault_col].shift()).cumsum()).cumsum()

        # replace 0s with NaNs
        df[time_since_fault_col].replace(0, pd.NaT, inplace=True)
        # Convert the time_since_fault_col column to datetime format
        df[time_since_fault_col] = pd.to_datetime(df[time_since_fault_col])
        # calculate time difference since last fault event in hours
        df[time_since_fault_col] = df.groupby(time_since_fault_col).apply(
            lambda x: x.index.to_series().diff().shift(-1).dt.total_seconds() / 3600
            if x[time_since_fault_col].dtype == 'datetime64[ns]' else None
        )
        # drop the temporary column
        df.drop(columns=[time_since_fault_col], inplace=True)
        # rename the calculated column
        df.rename(columns={
                  f'{fault_col}_time_since_last_fault': time_since_fault_col}, inplace=True)


# %%%%%%%%%%%%%%%%%%%%%%%%% These data is not provided yet %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Assuming the maintenance event is recorded in a column named 'maintenance_event'
# df['time_since_last_maintenance'] = df.groupby('Pump')['DateTime'].diff().dt.days.where(df['maintenance_event'] == 1, 0)
# # Assuming the maintenance event is recorded in a column named 'maintenance_event'
# df['time_to_next_maintenance'] = df.groupby('Pump')['DateTime'].diff(periods=-1).dt.days.where(df['maintenance_event'] == 1, 0)
# # Assuming the inspection event is recorded in a column named 'inspection_event'
# df['time_to_next_inspection'] = df.groupby('Pump')['DateTime'].diff(periods=-1).dt.days.where(df['inspection_event'] == 1, 0)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Create heatmap for all pumps
# the colors in the heatmap show the level of correlation between each parameter and fault status.
# The scale ranges from blue (negative correlation) to red (positive correlation), with white
# representing no correlation (a correlation coefficient of 0). The intensity of the color reflects
# the strength of the correlation. Darker colors indicate a stronger correlation, while lighter colors
# indicate a weaker correlation.
def plot_heatmap(df):
    # params = list(df.columns)[:-24]
    feature_cols = df.columns.tolist()
    for _pump in range(1, 4):
        for _week in range(1, 5):
            feature_cols.remove(
                f'KTF{_pump}_Fault_Status_next_{_week}_week(s)_sum')
            feature_cols.remove(
                f'KTF{_pump}_Fault_Status_next_{_week}_week(s)')
    faults = ['Normal', 'Warning', 'Fault']
    corr_data = pd.DataFrame(index=feature_cols, columns=[
                             'Pump 1', 'Pump 2', 'Pump 3'])
    for param in feature_cols:
        for _pump in range(1, 4):
            if re.search(f"KTF{_pump}", param):
                fault_col = f'KTF{_pump}_Fault_Status_next_1_week(s)'
                df_n = df[[param, fault_col]].dropna()
                corr = df_n[param].corr(df_n[fault_col])
                corr_data.loc[param, f'Pump {_pump}'] = corr
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_data.astype(float), cmap='coolwarm',
                annot=True, fmt='.2f', ax=ax)
    ax.set_title('Correlation between Parameters and Fault Status')
    plt.show()


def plot_feature_importance(df):
    pumps = ['KTF1', 'KTF2', 'KTF3']

    for pump in pumps:
        # Select columns for the current pump
        cols = [col for col in df.columns if col.startswith(pump)]
        X = df[cols].drop(f'{pump}_Fault_Status_next_1_day(s)', axis=1)
        y = df[f'{pump}_Fault_Status_next_1_day(s)']
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # Fit random forest model
        _model = RandomForestClassifier(n_estimators=100, random_state=42)
        _model.fit(X_train, y_train)

        # Plot feature importances
        importances = _model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in _model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title(f"Feature importances for {pump}")
        plt.bar(range(X_train.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]),
                   X_train.columns[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()


aggregate_data = False
merge_agg_and_org = True
lags = 14  # Number of days to lag
_days = 7 # Number of days to predict

if merge_agg_and_org:
    # Aggregate Features
    aggregated_df = aggregate_features(df)

    df_agg_new = pd.DataFrame(index=aggregated_df.index)
    df_new = pd.DataFrame(index=df.index)

    # Create time_based_features from aggregated
    updated_df_agg = create_time_based_features(
        aggregated_df, df_agg_new, merge_agg_and_org)

    # Create time_based_features from original
    updated_df = create_time_based_features(df, df_new, aggregate_data)
    # Create lag features from aggregated
    lagged_features_agg, lagged_columns_agg = create_lag_features(
        aggregated_df, merge_agg_and_org, lags, _days)
    # Create lag features from original
    lagged_features, lagged_columns = create_lag_features(
        updated_df, aggregate_data, lags, _days)

    # perform a left join on the 'key' column
    # merged_df = pd.merge(
    #     lagged_features, lagged_features_agg, left_index=True, right_index=True, how='left')

    # Merge the DataFrames with a suffix added to columns from each DataFrame
    merged_df = pd.merge(lagged_features, lagged_features_agg,
                         left_index=True, right_index=True, how='left')

    # Remove the suffix from the remaining column names
    merged_df.columns = merged_df.columns.str.replace('_y', '')
    merged_df.columns = merged_df.columns.str.replace('_x', '')

    # Drop the columns with the same name from the right DataFrame
    duplicated_cols = merged_df.columns[merged_df.columns.duplicated()]
    merged_df = merged_df.drop(columns=duplicated_cols)

    # Replace NaN values from aggregated target after merge
    merged_df = replace_target_nan_values(merged_df, _days)

    # time_series_analysis(merged_df)
    # plot_heatmap(merged_df)
    # plot_feature_importance(merged_df)

    merged_lagged_columns = lagged_columns + lagged_columns_agg
    _df_name = f'merged_lagged_features_{lags}_days'
    _lagged_columns_name = f'merged_columns_{lags}_days'
    save_lagged_features(merged_df, merged_lagged_columns,
                         _df_name, _lagged_columns_name)

elif aggregate_data:
    # Aggregate Features
    aggregated_df = aggregate_features(df)
    df_new = pd.DataFrame(index=aggregated_df.index)
    # Create time_based_features from aggregated
    updated_df_agg = create_time_based_features(
        aggregated_df, df_new, aggregate_data)
    # Create lag features from aggregated
    lagged_features_agg, lagged_columns_agg = create_lag_features(
        updated_df_agg, aggregate_data, lags, _days)
    _df_name = f'aggregated_lagged_features_{lags}_days'
    _lagged_columns_name = f'aggregated_columns_{lags}_days'
    save_lagged_features(lagged_features_agg,
                         lagged_columns_agg, _df_name, _lagged_columns_name)

else:
    df_new = pd.DataFrame(index=df.index)
    # Create time_based_features from original
    updated_df = create_time_based_features(df, df_new, aggregate_data=False)
    # Create lag features from original
    lagged_features, lagged_columns = create_lag_features(
        updated_df, aggregate_data, lags, _days)
    _df_name = f'original_lagged_features_{lags}_days'
    _lagged_columns_name = f'original_columns_{lags}_days'
    save_lagged_features(lagged_features, lagged_columns,
                         _df_name, _lagged_columns_name)

    # Merge the new DataFrame with the original DataFrame using the DateTime index
    # df_merged = pd.merge(df, df_merge, left_index=True, right_index=True, how="left")

# plot_heatmap(lagged_features)
# plot_feature_importance(df)
