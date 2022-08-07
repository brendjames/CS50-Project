import os
import csv
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

df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/2328589/4015443/History_of_Mass_Shootings_in_the_USA.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220807%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220807T115535Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3a363971b64ff7c66cdb8d9488a090910612a2b82f034e3006eef972494b1e880c20ac422e01108f6e45ac3b873ec556b2a375cad158e5c25d7dcbf732d5a11c1bb28a35e24f01e412d701c53018f1bf79fb0be9ae085ce98b0e9e256a76a9801bb0c51c6f6cab4ea404690c85b44b7d2d692da431274062e1bad7f6d5869265361bea13745be4f964f4fe2c764f05f7b4d5c59eeca9ad6d55b9e0e0a16a34ec953c853d92577782870507de9daa353c92d5ddefced6bb28c8ad52ff1527a21b65e50ef48067fda2aa579d247bb922eda053d05fa4ebf0c2fd9e864b63629f8a8b2568ce6da605c3bcaf5e0f711f8e0f01f38401e66d2da4685ad0643d50c95e')
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


