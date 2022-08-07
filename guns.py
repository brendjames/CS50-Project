from datetime import time
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
# To store the data
import pandas as pd

# To do linear algebra
import numpy as np

# To create plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# To create nicer plots
import seaborn as sns

# To create interactive maps
from plotly.offline import init_notebook_mode, iplot
import plotly.offline as offline
from plotly import tools


# To create interactive plots
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool, ResetTool, PanTool
from bokeh.plotting import figure, show
from bokeh.io import output_notebook


# To sort dictionaries
import operator

# To generate word clouds
from pygments.lexers import go
from wordcloud import WordCloud

# Load the data
df = pd.read_csv('https://storage.googleapis.com/kaggle-data-sets/21619/27807/compressed/gun-violence-data_01-2013_03-2018.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220807%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220807T115913Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=6883ef767f4e53529482c3105dc894cf77fa717a7ed4a158087266196aa5612fd103b28c5fa0669f0b206be294c370caf76539d665f5acf56e5bc2a24bda0740cda48a6fb0fc7863be8c62142f2ab0a97631e4e10a26aece31a5d4d0b95d554c8a1498b5b15c41d858fc330c2b37076c8a2436efaccc993b783fe154f8ee91e6361d8bac42b94a4b26c0267566ff604ac8f6fb9bb8a01b559a99398d992b6eada062a1dece51118963c0b4a217d7fe8a9421a2f14d315f397c37fbaa10b4f6b43c1018bbaf6151017bddc6f68f7aebc3ae325c1826edd4a14095709f808957625722f0b38add68e9acbde974c0f33f2892df1a2b22245054d94b4e7d5e23268b')
# Parse the date and set the index
df.date = pd.to_datetime(df.date)
df.set_index('date', inplace=True)

# Display
print('The Dataset has {} entries with {} features.'.format(df.shape[0], df.shape[1]))
print('Here is an example entry:')
df.head(1)

# Group by day, count incidens, plot graph
incident_df = df.groupby(pd.Grouper(freq='d')).agg({'incident_id':'count'}).rename(columns={'incident_id':'incidents'})
incident_df.plot(figsize=(16,5), title='Daily incidents in the US', color='#ff4500')

text = 'The data seems\nunreliable\nbefore 2014'
plt.annotate(text, (incident_df.index[364], incident_df.incidents[364]), xytext=(-120, 50), textcoords='offset points', arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel('Date')
plt.ylabel('Incidents')
plt.show()

# Exclude the unreliable data
df = df.loc['2014':]
print('The remaining DataFrame as {} entries.'.format(df.shape[0]))

# Check for missing values, compute the percentage, reverse the series, plot the graph
df.isna().mean().mul(100).iloc[::-1].plot(kind='barh', figsize=(14,5), grid=True, title='Percentage of missing values for each feature', color='#ff4500')
plt.xlabel('Percentage')
plt.ylabel('Feature')
plt.show()

# Create a DataFrame grouped by state and compute the count, sum and mean
state_df = df.groupby('state').agg({'incident_id': 'count', 'n_killed': ['mean', sum]})
state_df.columns = ['_'.join([first, second]) for first, second in
                    zip(state_df.columns.get_level_values(0), state_df.columns.get_level_values(1))]

# Round the data and reset the index for the plot
state_df = state_df.apply(lambda x: round(x, 3)).reset_index()

# Rename the states for a correct mapping
state_to_code = {'District of Columbia': 'dc', 'Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE',
                 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN',
                 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN',
                 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA',
                 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA',
                 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
                 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL',
                 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH',
                 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ',
                 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV',
                 'Maine': 'ME'}
state_df['state_code'] = state_df['state'].apply(lambda x: state_to_code[x])

# Store the data of the subplots
data = []
# Layout for the whole plot
layout = dict(title='Deaths grouped by state',
              width=1024,
              height=768,
              hovermode=False)

# Count-Plot (data and layout)
data.append(dict(type='choropleth',
                 colorscale=[[0.0, '#baff00'], [1.0, '#000000']],
                 autocolorscale=False,
                 locations=state_df['state_code'],
                 geo='geo',
                 z=state_df['incident_id_count'],
                 text=state_df['state'],
                 locationmode='USA-states',
                 marker=dict(line=dict(color='rgb(255,255,255)',
                                       width=2)),
                 colorbar=dict(title="Incidents",
                               x=0.29,
                               thickness=10)))

layout['geo'] = dict(scope='usa',
                     showland=True,
                     projection=dict(type='albers usa'),
                     showlakes=True,
                     lakecolor='rgb(255, 255, 255)',
                     landcolor='rgb(229, 229, 229)',
                     subunitcolor="rgb(255, 255, 255)",
                     domain=dict(x=[0 / 3, 1 / 3], y=[0, 1]))

# Sum-Plot (data and layout)
data.append(dict(type='choropleth',
                 colorscale=[[0.0, '#00baff'], [1.0, '#000000']],
                 autocolorscale=False,
                 locations=state_df['state_code'],
                 geo='geo2',
                 z=state_df['n_killed_sum'],
                 text=state_df['state'],
                 locationmode='USA-states',
                 marker=dict(line=dict(color='rgb(255,255,255)',
                                       width=2)),
                 colorbar=dict(title="Deaths",
                               x=0.6225,
                               thickness=10)))
layout['geo2'] = dict(scope='usa',
                      showland=True,
                      projection=dict(type='albers usa'),
                      showlakes=True,
                      lakecolor='rgb(255, 255, 255)',
                      landcolor='rgb(229, 229, 229)',
                      subunitcolor="rgb(255, 255, 255)",
                      domain=dict(x=[1 / 3, 2 / 3], y=[0, 1]))

# Mean-Plot (data and layout)
data.append(dict(type='choropleth',
                 colorscale=[[0.0, '#4400ff'], [1.0, '#000000']],
                 autocolorscale=False,
                 locations=state_df['state_code'],
                 geo='geo3',
                 z=state_df['n_killed_mean'],
                 text=state_df['state'],
                 locationmode='USA-states',
                 marker=dict(line=dict(color='rgb(255,255,255)',
                                       width=2)),
                 colorbar=dict(title="Deaths/Incident",
                               x=0.96,
                               thickness=10)))

layout['geo3'] = dict(scope='usa',
                      showland=True,
                      projection=dict(type='albers usa'),
                      showlakes=True,
                      lakecolor='rgb(255, 255, 255)',
                      landcolor='rgb(229, 229, 229)',
                      subunitcolor="rgb(255, 255, 255)",
                      domain=dict(x=[2 / 3, 3 / 3], y=[0, 1]))

# Create the subplots
fig = {'data': data, 'layout': layout}
iplot(fig)


# Plot the states ranked by their deadliness
state_df.set_index('state').sort_values('n_killed_mean', ascending=False)['n_killed_mean'].plot(kind='bar', figsize=(18, 4), title='Where is an incident most deadly?', grid=True, color='#4400ff')
plt.xlabel('State')
plt.ylabel('Mean deaths in an incident')
plt.xticks(rotation=75)
plt.show()


# Number of cities
n = 100

city_df = df.groupby('city_or_county').agg({'n_killed':['mean', sum, 'count'], 'longitude':'mean', 'latitude':'mean'})
city_df.columns = ['_'.join([first, second]) for first, second in zip(city_df.columns.get_level_values(0), city_df.columns.get_level_values(1))]
#tmp[tmp.n_killed_count>50].sort_values('n_killed_mean', ascending=False).head(10)
city_df = city_df[city_df.n_killed_count>50].sort_values('n_killed_sum').tail(n).reset_index()

# Death colorscale
scl = [[0.0, '#00baff'],[1.0, '#000000']]

# Data for the map
data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = city_df['longitude_mean'],
        lat = city_df['latitude_mean'],
        text = city_df['city_or_county'] +': '+ city_df['n_killed_sum'].astype(str),
        mode = 'markers',
        marker = dict(
            size = 8,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = city_df['n_killed_sum'],
            cmax = city_df['n_killed_sum'].max(),
            colorbar=dict(
                title="Deaths"
            )
        ))]

# Layout for the map
layout = dict(
        title = '{} Cities with the most deaths in 4.25 years'.format(n),
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(0, 0, 0)",
            countrycolor = "rgb(0, 0, 0)",
            countrywidth = 0.5,
            subunitwidth = 0.5))

# Create the map
fig = dict(data=data, layout=layout)
iplot(fig)

# Store the data
data = []

# Iterate over each row
for row in df.values:
    # Get the data
    length = []
    state = row[1]
    n_killed = row[4]
    n_injured = row[5]
    participant_age = row[18]

    # Split the entries in the columns
    if type(participant_age) == str:
        participant_age = participant_age.replace('||', '|').split('|')
        length.append(len(participant_age))
    participant_gender = row[20]
    if type(participant_gender) == str:
        participant_gender = participant_gender.replace('||', '|').split('|')
        length.append(len(participant_gender))
    participant_status = row[23]
    if type(participant_status) == str:
        participant_status = participant_status.replace('||', '|').split('|')
        length.append(len(participant_status))
    participant_type = row[24]
    if type(participant_type) == str:
        participant_type = participant_type.replace('||', '|').split('|')
        length.append(len(participant_type))

    # Combine the splitted entries
    if length:
        for i in range(max(length)):
            try:
                p_a = participant_age[i].replace('::', ':').split(':')[-1]
            except:
                p_a = np.nan
            try:
                p_g = participant_gender[i].replace('::', ':').split(':')[-1]
            except:
                p_g = np.nan
            try:
                p_s = participant_status[i].replace('::', ':').split(':')[-1]
            except:
                p_s = np.nan
            try:
                p_t = participant_type[i].replace('::', ':').split(':')[-1]
            except:
                p_t = np.nan

            # Store the data
            data.append([state, n_killed, n_injured, p_a, p_g, p_s, p_t])

# Create the DataFrame
people_df = pd.DataFrame(data, columns=['state', 'n_killed', 'n_injured', 'age', 'gender', 'status', 'type'])
people_df['age'] = people_df['age'].astype(float)

# Groupby categories, compute mean
# participant_df = people_df.groupby(['gender', 'status', 'type']).agg({'n_killed':['mean', sum], 'n_injured':['mean', sum], 'age':'mean', 'state':'count'})
# participant_df.columns = ['_'.join([first, second]) for first, second in zip(participant_df.columns.get_level_values(0), participant_df.columns.get_level_values(1))]


# Color palette
palette = ["#ff4500", "#00baff"]

people_df[(people_df.type == 'Victim') & (people_df.gender.isin(['Male', 'Female']))].groupby(
    'gender').state.count().plot(kind='bar', color=palette, title='Involved victims grouped by gender')
plt.ylabel('Victims')
plt.show()

# Color palette
palette ={"Female":"#ff4500","Male":"#00baff"}

# Create plots
f, axarr = plt.subplots(1, 2, sharey=True, figsize=(14,6))
sns.barplot(data=people_df[(people_df.gender.isin(['Male', 'Female'])) & (people_df.type=='Subject-Suspect')], x='gender', y='n_injured', ax=axarr[0], palette=palette)
sns.barplot(data=people_df[(people_df.gender.isin(['Male', 'Female'])) & (people_df.type=='Subject-Suspect')], x='gender', y='n_killed', ax=axarr[1], palette=palette)
axarr[0].set_title('Mean injured grouped by attacker gender')
axarr[1].set_title('Mean killed grouped by attacker gender')
axarr[0].set_ylabel('mean injured')
axarr[1].set_ylabel('mean killed')
plt.show()


# Create plots
sns.barplot(data=people_df[(people_df.gender.isin(['Male', 'Female'])) & (people_df.type=='Subject-Suspect')], x='gender', y='age', palette=palette)
plt.title('Attacker age grouped by gender')
plt.show()

age_df = people_df[people_df.type == 'Subject-Suspect'].groupby('state').agg({'age': 'mean'}).apply(
    lambda x: round(x, 2)).reset_index()
age_df['state_code'] = age_df['state'].apply(lambda x: state_to_code[x])

scl = [[0.0, '#ff4500'], [1.0, '#000000']]

# Data for the map
data = [dict(
    type='choropleth',
    colorscale=scl,
    autocolorscale=False,
    locations=age_df['state_code'],
    z=age_df['age'].astype(float),
    locationmode='USA-states',
    text=age_df['state'],
    marker=dict(line=dict(color='rgb(255,255,255)',
                          width=2)),
    colorbar=dict(title="Age"))]

# Layout for the map
layout = dict(title='Attacker age grouped by state',
              geo=dict(
                  scope='usa',
                  projection=dict(type='albers usa'),
                  showlakes=True,
                  lakecolor='rgb(255, 255, 255)'))

fig = dict(data=data, layout=layout)
iplot(fig)






