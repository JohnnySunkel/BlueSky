import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Create dataframes for each data source
attendance_df = pd.read_csv('nba_2017_attendance.csv')
attendance_df.head()

endorsement_df = pd.read_csv('nba_2017_endorsements.csv')
endorsement_df.head()

valuations_df = pd.read_csv('nba_2017_team_valuations.csv')
valuations_df.head()

salary_df = pd.read_csv('nba_2017_salary.csv')
salary_df.head()

pie_df = pd.read_csv('nba_2017_pie.csv')
pie_df.head()

plus_minus_df = pd.read_csv('nba_2017_real_plus_minus.csv')
plus_minus_df.head()

br_stats_df = pd.read_csv('nba_2017_br.csv')
br_stats_df.head()

elo_df = pd.read_csv('nba_2017_elo.csv')
elo_df.head()

# Merge attendance and valuation data
attendance_valuation_df = attendance_df.merge(valuations_df,
                                              how = "inner",
                                              on = "TEAM")
attendance_valuation_df.head()

# Pairplot 
sns.pairplot(attendance_valuation_df, hue = "TEAM")

# Correlation heatmap for variables
corr = attendance_valuation_df.corr()
sns.heatmap(corr,
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values)

# Correlation heatmap for teams
valuations = attendance_valuation_df.pivot("TEAM",
                                           "TOTAL_MILLIONS",
                                           "VALUE_MILLIONS")
plt.subplots(figsize = (20, 15))
ax = plt.axes()
ax.set_title("NBA Team AVG Attendance vs. \
             Valuation in Millions: 2016-2017 Season")
sns.heatmap(valuations, 
            linewidths = 0.5,
            annot = True,
            fmt = 'g')

# Linear regression with StatsModels
results = smf.ols('VALUE_MILLIONS ~TOTAL_MILLIONS',
                  data = attendance_valuation_df).fit()
print(results.summary())

# Plot residuals
sns.residplot(y = "VALUE_MILLIONS",
              x = "TOTAL_MILLIONS",
              data = attendance_valuation_df)

# Measure RMSE 
attendance_valuation_predictions_df = attendance_valuation_df.copy()
attendance_valuation_predictions_df["predicted"] = results.predict()

rmse = statsmodels.tools.eval_measures. \
    rmse(attendance_valuation_predictions_df["predicted"],
         attendance_valuation_predictions_df["VALUE_MILLIONS"])
rmse

# Plot predictions vs actual values
sns.lmplot(x = "predicted",
           y = "VALUE_MILLIONS",
           data = attendance_valuation_predictions_df)

# Create a new dataframe 
val_housing_win_df = pd.read_csv('nba_2017_att_val_elo_win_housing.csv')
val_housing_win_df.columns

# Scale the data
numerical_df = val_housing_win_df.loc[:, ["TOTAL_ATTENDANCE_MILLIONS",
                                          "ELO",
                                          "VALUE_MILLIONS",
                                          "MEDIAN_HOME_PRICE_COUNTY_MILLIONS"]]
scaler = MinMaxScaler()
print(scaler.fit(numerical_df))
print(scaler.transform(numerical_df))

# K-Means clustering
k_means = KMeans(n_clusters = 3)
kmeans = k_means.fit(scaler.transform(numerical_df))

# Attach cluster results to the dataframe
val_housing_win_df['cluster'] = kmeans.labels_
val_housing_win_df.head()

# Write the new dataframe to a csv file
val_housing_win_df.to_csv('nba_2017_att_val_elo_win_housing_cluster.csv')
