import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import datetime as date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio

df = pd.read_csv('C:/Users/btran/Downloads/Python/project/Project/Project/mass_shootings.csv')
# The dataset contains a list of all the mass shootings from 1924 to 2022

# Date: Date of the shooting
# City: City where the shooting occured
# State: State where the shooting occured
# Dead: Number of casualties
# Injured: Number of people injured
# Total: Dead + Injured
# Description: Description of the shooting

df.head(8)
# Taking care of Null values!

# Plot to see null values in our data
sns.heatmap(df.isnull(), cbar=False)

# Null Value Percentage Per column
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
missing_value_df

df.dropna(axis=0, inplace=True)
# Date column should be of type date and time so we will convert it intro datetime type. Also we are going to create three now columns of year, month, day of week and day from date column.

df.dtypes

df['Date'] = pd.to_datetime(df['Date'])

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.day_name()

# Combine Year to form intervals
df['Interval'] = (10 * (df['year'] // 10)).astype(str) + 's'

df.dtypes

df.sample(5)

for elem in df['State'].unique():
    print(elem)
for elem in df['Interval'].unique():
    print(elem)

df.sample(5)
# Create State Code Colomn from State Names for our Map

state_code = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

df['state_code'] = df.State.replace(state_code)
df.head(5)

data = df
city_df = data[['City', 'Total']].groupby(['City']).sum().reset_index().sort_values('Total', ascending=False)
city_dfa = city_df.head(25)

city_dfa

# Years Trend
ata = df
interval_df = data[['Interval', 'Total']].groupby(['Interval']).sum().reset_index().sort_values('Total',
                                                                                                ascending=True)
interval_df = interval_df.head(12)

plt.figure(figsize=(20, 12))
ax = sns.barplot(x='Interval', y='Total', data=interval_df, palette='Paired')
plt.xticks(rotation=90, fontsize=13, color='midnightblue')
plt.yticks(fontsize=13, color='midnightblue')
plt.xlabel('Interval', size=18, color='midnightblue')
plt.ylabel('Total', size=17, color='midnightblue')
plt.title('Mass Shootings Trend with Year Passings', size=20, color='midnightblue')
plt.show()

# Top Cities
data = df
city_df = data[['City', 'Total']].groupby(['City']).sum().reset_index().sort_values('Total', ascending=False)
city_dfa = city_df.head(25)

plt.figure(figsize=(20, 16))
ax = sns.barplot(x='Total', y='City', data=city_dfa, palette='Paired')
plt.xticks(fontsize=13, color='midnightblue')
plt.yticks(fontsize=13, color='midnightblue')
plt.xlabel('Total', size=18, color='midnightblue')
plt.ylabel('City', size=17, color='midnightblue')
plt.title('Top 25 Cities of USA Mass Shootings Between 1920 - 2022', size=25, color='midnightblue')
plt.show()

# map
fig = px.choropleth(df,
                    locations='state_code',
                    locationmode="USA-states",
                    scope="usa",
                    color='Total',
                    range_color=[1, 40],
                    template='plotly'
                    )

fig.update_layout(
    title_text='Mass Shootings Map of USA states 1920 - 2022',
)

fig.show()

# scatter plots
fig = px.scatter(df, x="Injured", y="Dead", color="Total",
                 size='Dead', hover_data=['City', 'State', 'year'], template='plotly',
                 title='Scatter Plot of cross Injured, Dead And Total number of incidences')
fig.show()

data.State.value_counts().plot(kind='bar', figsize=(20, 5))


