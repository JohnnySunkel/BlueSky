import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

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
