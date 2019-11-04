import dash
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_core_components as dcc
#import dash_auth
import plotly.graph_objs as go
import pandas as pd

#USERNAME_PASSWORD_PAIRS = [
#    ['kali', 'rabbit']
#]

token = "pk.eyJ1Ijoia2FsaTYiLCJhIjoiY2swejc4ZXQzMDZzdTNrbzV1eXpwejhzeiJ9.Xw7BdlQY-wbll9qirrcnuw"
df_01 = pd.read_csv("data/final_admin_mapped_data.csv", dtype={'Actor1Religion2Code':str,
 'Actor1Type3Code':str,
 'Actor2KnownGroupCode':str,
 'Actor2Religion2Code':str,
 'Actor2Type3Code':str,
 'Actor2Geo_ADM2Code':str})

df_01 = df_01[['Year', 'Actor1Type1CodeName', 'ActionGeo_Lat', 'ActionGeo_Long', 'admin1_fipsnames', 'Actor1Name']]

app = dash.Dash()
server = app.server
app.title = 'GDELT EVENTS APP by Kali'
#auth = dash_auth.BasicAuth(app,USERNAME_PASSWORD_PAIRS)
# options for year drop down
year_options = []
for y in df_01['Year'].unique():
    year_options.append({'label':str(y), 'value':y})

# option for admin1 Dropdown
admin_options = []
for a in df_01['admin1_fipsnames'].unique():
    admin_options.append({'label':a, 'value':a})

# app layout
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='year_drop_id',
                options=year_options,
                searchable=False,#value=df_01['Year'].min(),
                placeholder="Select a Year",
                multi=True,
                value = [2008]
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='admin1_drop_id',
                options=admin_options,
                searchable=False, #value=admin_options[0],
                placeholder="Select admin Level",
                multi=True,
                value=df_01['admin1_fipsnames'].unique()[0]
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    html.Div([dcc.Graph(id='map_id', style={'width': '90%', 'height': '100%'})]),
], style={'padding':10})


@app.callback(Output('map_id', 'figure'),
              [Input('year_drop_id', 'value'),
              Input('admin1_drop_id', 'value')])

def update_map(input_year, district_name):
    input_year_list = input_year
    district_name_list = district_name
    if type(input_year) != list:
        input_year_list = [input_year]
    if type(district_name) != list:
        district_name_list = [district_name]

    filtered_df = df_01[df_01['Year'].isin(input_year_list) & df_01['admin1_fipsnames'].isin(district_name_list)]
    #if type(input_year) == int and type(district_name) == str:
    #    filtered_df = df_01[df_01['Year'].isin([input_year]) & df_01['admin1_fipsnames'].isin([district_name])]
    #else:
    #    filtered_df = df_01[df_01['Year'].isin(input_year) & df_01['admin1_fipsnames'].isin(district_name)]
    traces = []
    for religion in filtered_df['Actor1Type1CodeName'].unique():
        df_religion_level = filtered_df[filtered_df['Actor1Type1CodeName'].isin([religion])]

        traces.append(go.Scattermapbox(
        lat=df_religion_level['ActionGeo_Lat'],
        lon=df_religion_level['ActionGeo_Long'],
        mode='markers',
        name=religion,
        text=df_religion_level['Actor1Name']))

    return {'data': traces,
            'layout': {'title':'GDELT EVENT DATA BY YEAR and Admin Levels',
            'mapbox': {'accesstoken': (token)},
            'autosize':True
            }}


if __name__ == '__main__':
    app.run_server()
