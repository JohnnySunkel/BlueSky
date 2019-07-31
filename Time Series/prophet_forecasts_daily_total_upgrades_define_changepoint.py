# Prophet forecasts of daily total upgrades
import pandas as pd
from fbprophet import Prophet

# Load the dataset
df = pd.read_csv('daily_total_upgrades_fbp.csv')
df.head()
df.tail()

# Define holidays
easter = pd.DataFrame({
        'holiday': 'easter',
        'ds': pd.to_datetime(['2017-04-16', 
                              '2018-04-01',
                              '2019-04-21',
                              '2020-04-12']),
        'lower_window': 0,
        'upper_window': 0,
})

samsung_preorder = pd.DataFrame({
        'holiday': 'samsung_preorder',
        'ds': pd.to_datetime(['2017-03-30',
                              '2018-03-02',
                              '2019-02-21']),
        'lower_window': 0,
        'upper_window': 0,
})

samsung_launch = pd.DataFrame({
        'holiday': 'samsung_launch',
        'ds': pd.to_datetime(['2017-04-21',
                              '2018-03-16',
                              '2019-03-08',
                              '2019-08-07',
                              '2020-03-16']),
        'lower_window': 0,
        'upper_window': 1,
})

apple_preorder = pd.DataFrame({
        'holiday': 'apple_preorder',
        'ds': pd.to_datetime(['2017-09-15',
                              '2017-10-27',
                              '2018-09-14',
                              '2018-10-19']),
        'lower_window': 0,
        'upper_window': 0,
})

apple_launch = pd.DataFrame({
        'holiday': 'apple_launch',
        'ds': pd.to_datetime(['2017-09-22',
                              '2017-11-03',
                              '2018-09-21',
                              '2018-10-26',
                              '2019-09-21',
                              '2019-10-26',
                              '2020-09-21',
                              '2020-10-26']),
        'lower_window': 0,
        'upper_window': 1,
})

black_friday = pd.DataFrame({
        'holiday': 'black_friday',
        'ds': pd.to_datetime(['2017-11-24',
                              '2018-11-23',
                              '2019-11-29',
                              '2020-11-27']),
        'lower_window': 0,
        'upper_window': 1,
})

cyber_monday = pd.DataFrame({
        'holiday': 'cyber_monday',
        'ds': pd.to_datetime(['2017-11-27',
                              '2018-11-26',
                              '2019-12-02',
                              '2020-11-30']),
        'lower_window': -1,
        'upper_window': 0,
})

thanksgiving = pd.DataFrame({
        'holiday': 'thanksgiving',
        'ds': pd.to_datetime(['2017-11-23',
                              '2018-11-22',
                              '2019-11-28',
                              '2020-11-26']),
        'lower_window': 0,
        'upper_window': 0,
})

christmas = pd.DataFrame({
        'holiday': 'christmas',
        'ds': pd.to_datetime(['2017-12-25',
                              '2018-12-25',
                              '2019-12-25',
                              '2020-12-25']),
        'lower_window': 0,
        'upper_window': 0,
})

independence_day = pd.DataFrame({
        'holiday': 'independence_day',
        'ds': pd.to_datetime(['2017-07-04',
                              '2018-07-04',
                              '2019-07-04',
                              '2020-07-04']),
         'lower_window': 0,
         'upper_window': 1,
})

holidays = pd.concat((easter,
                      samsung_preorder,
                      samsung_launch,
                      apple_preorder,
                      apple_launch,
                      black_friday,
                      cyber_monday,
                      thanksgiving,
                      christmas,
                      independence_day))

# Define and fit the model
m = Prophet(holidays = holidays,
            daily_seasonality = True,
            yearly_seasonality = 6,
            changepoints = ['2018-01-01'])
# m.add_country_holidays(country_name = 'US')
m.train_holiday_names
m.fit(df)

# Create a dataframe to hold predictions
future = m.make_future_dataframe(periods = 31)
future.tail()

# Make predictions
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(31)
# forecast.to_csv('jan_jul_forecast.csv')

# Plot forecasts
m.plot(forecast)

# Plot forecast components
m.plot_components(forecast)