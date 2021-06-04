import pandas as pd
import numpy as np
import dash
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import plotly.express as px

import dash_table
import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import scipy
from scipy import stats

import math

import dash_bootstrap_components as dbc

#app = dash.Dash(__name__ , external_stylesheets= [dbc.themes.MINTY])
#app = dash.Dash(__name__ )

#external_stylesheets = ['dbc.themes.BOOTSTRAP']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def read_dataframe():
    """ Reading the igra2 file as a dataframe.
        Data cleaning:
            - removing wrong latitude and longitude values
        Return:
            - pandas dataframe with 8 columns
             ['code', 'latitude', 'longitude', 'elevation', 'dummy', 'station',
                          'start', 'end', 'records'] """


    '''
    # data is saved irregularly spaced, need to use fixed-width formatted lines read function
    df = pd.read_fwf('data/igra2.csv', widths=(11, 9, 10, 7, 4, 30, 5, 5, 7),
                          names=( 'code', 'latitude', 'longitude', 'elevation', 'dummy',
                                  'station', 'start', 'end', 'records') )
    '''

    file_url = "https://raw.githubusercontent.com/fambrogi/Visualization2/main/data/igra2.csv"
    df = pd.read_fwf(file_url, widths=(11, 9, 10, 7, 4, 30, 5, 5, 7),
                          names=( 'code', 'latitude', 'longitude', 'elevation', 'dummy',
                                  'station', 'start', 'end', 'records') )


    df = df.loc[ (df['latitude'] <= 90) & (df['latitude'] >= -90) ]
    df = df.loc[ (df['longitude'] <= 180) & (df['latitude'] >= -180) ]
    df = df.sort_values(['latitude','longitude'])
    df = df.drop(columns = ["dummy"])
    return df[0:2500]

    #return df

def distance(point_lat, point_lon, dest_lat, dest_lon):
    """
    Calculate the Haversine distance.
    """
    radius = 6371  # km

    dlat = math.radians(dest_lat - point_lat)
    dlon = math.radians(dest_lon - point_lon)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(point_lat)) * math.cos(math.radians(dest_lat)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d



MSIZE = 5

dataframe = read_dataframe()

def combine_points(df, zoom_level):
    """ Core implementation of the Algorithm described in Janicke et al. 2012.
        Calculates if the bubble marker needs to be merged,
        calculates the position of the new bubble,
        calculates the radius of the new bubble.
        Initial size is fixed to XXX. """

    point_size = MSIZE
    max_radius = 15
    def weighted_average(v1, w1, v2, w2):
        """ Return the weighted average """
        r = (v1 * w1 + v2 * w2) / (w1 + w2)
        #print( v1, w1, v2, w2 , r )
        return r

    # columns : 'code', 'latitude', 'longitude', 'elevation', 'dummy',
    #                                   'station', 'start', 'end', 'records'

    LAT, LON = list(df['latitude'][:]), list(df['longitude'][:])
    STAT = list(df['station'][:])
    RADIUS = len(LAT) * [point_size] # initialize an empty array for radius = 1
    RECORDS = list(df['records'][:])

    # For each zoom level, I empirically extracted the distance in km for which two points with default size
    # do not overlap.
    scale_distance = {1: 25,  # nemayer, sanae south pole , size_marker = 5
                      2: 75,
                      3: 50,
                      5: 30,
                      10: 15}

    # 1:700 is the minimum distance between two points to be visually separated with marker size=5
    # means: 350 per each point minimum
    limit = scale_distance[zoom_level]

    # Starting the loop for points merging
    merge = True
    i = 0

    tot_points = len(LAT)
    while merge:

        #if i == 0:
        #    tot_points = len(LAT)

        while i < tot_points:
            #print("i: " , i , "tot_point: " , tot_points, "len LAT: ", len(LAT) )
            lat1, lon1 = LAT[i], LON[i]
            r1 = RADIUS[i]
            s1 = STAT[i]
            rec1 = RECORDS[i]

            for j in range(i+1, tot_points):
                #if i==j:
                #    continue
                #print(i, LAT, LON, j)
                #print(len(LAT), j )
                lat2, lon2 = LAT[j], LON[j]
                s2 = STAT[j]
                r2 = RADIUS[j]
                rec2 = RECORDS[j]

                # should be irrelevant
                #if lat1==lat2 and lon1==lon2:
                #    continue

                max_points = ( distance(lat1, lon1, lat2, lon2) ) / limit # max number of distinct points with r=1
                tot_points = r1 + r2

                if max_points < tot_points :

                    #print('*** Merging two points of ', len(LAT) , dist_to_limit , limit )

                    # Updating the new merged data
                    LAT.append( weighted_average(lat1, r1, lat2, r2 ))
                    LON.append( weighted_average(lon1, r1, lon2, r2 ))
                    tot_stat = len( s1.split(',')) + len(s2.split(','))
                    RADIUS.append( min(point_size + 0.3*(tot_stat) , max_radius) )
                    STAT.append(s1+','+s2)
                    RECORDS.append(rec1 + rec2 )
                    # Remove old points from lists

                    #print('LAT', len(LAT), i, j )
                    #print('LON', len(LAT), i, j )

                    for index in [j,i]:
                        LAT.pop(index)
                        LON.pop(index)
                        STAT.pop(index)
                        RECORDS.pop(index)
                        RADIUS.pop(index)

                    merge = True
                    i = 0
                    tot_points = len(LAT)
                    break

                else:  # if not distance below limit, pass to following point
                    continue
            i = i+1
            tot_points = len(LAT)
            merge = True

        merge = False

    #RADIUS = point_size * np.array(RADIUS)

    dic = {'latitude': LAT , 'longitude': LON,
               'radius': RADIUS,
               'station' : STAT,
               'records' : RECORDS }

    df_resized = pd.DataFrame.from_dict(dic)

    return df_resized





#a = combine_points( dataframe, 3)
#print(0)

""" Setting the APP LAYOUT 
What goes inside the app layout is your dash components,
with the graphs, layouts, checkboxes
anything that is listed here: https://dash.plotly.com/dash-core-components 
"""

dates = [1900 + 10*i for i in range(0,21)]

app.layout = html.Div([

    html.Br(),  # Br is a break i.e. a space in between

    dbc.Row (dbc.Col (html.H1("Visualization 2 - SS2021",
                      style={'color': 'blue', 'fontSize': 50}),
                      width={'size': 6}
                      )
             ),

    dbc.Row(dbc.Col(html.H1("Federico Ambrogi , e1449911@student.tuwien.ac.at" ,
                            style={'color': 'black', 'fontSize': 25}),
                    width={'size': 6}
                    )
            ),

    dbc.Row(dbc.Col(html.H2("Tool for the visualization of the IGRA2 upper-air/radiosonde stations using the size adapting technique from Janicke et al. 2012",
                            style={'color': 'black', 'fontSize': 25}),
                    width={'size': 6}
                    )
            ),

    dbc.Row(dbc.Col(html.A(
        "Article: COMPARATIVE VISUALIZATION OF GEOSPATIAL-TEMPORAL DATA",
        href="https://www.informatik.uni-leipzig.de/~stjaenicke/Comparative_Visualization_Of_Geospatial-Temporal_Data.pdf",
        target="_blank",
        style={'color': 'black', 'fontSize': 15}),
                    width={'size': 6}
                    )
            ),

    dbc.Row(dbc.Col(html.A(
        "IGRA2 Data Source",
        href="https://www.ncdc.noaa.gov/data-access/weather-balloon/integrated-global-radiosonde-archive",
        target="_blank",
        style={'color': 'black', 'fontSize': 15}),
        width={'size': 6}
    )
    ),


    html.Br(),  # Br is a break i.e. a space in between
    html.Br(),  # Br is a break i.e. a space in between

    dbc.Row([html.H2("Zoom Level",
                             style={'color': 'red', 'fontSize': 22}),
             html.H2("Select the desired zoom level to visualize the station distribution on the map. "
                     "Note that the automatic mouse-wheel zooming will not work",
                     style={'color': 'gray', 'fontSize': 17}),


            dcc.Dropdown(id="scale",
                     options=[
                     {"label": "1", "value": 1},
                     {"label": "2", "value": 2},
                     {"label": "3", "value": 3},
                     {"label": "5", "value": 5},
                     {"label": "10", "value": 10},
                     ],
                     multi=False,
                     value=1, # this is the initial value displayed in the dropwdow
                     style={'width': "40%"}
                     ),

            html.H2("Projection Type ",
                    style={'color': 'red', 'fontSize': 22}),

            html.H2("Select the desired type of map projection. "
                    "The visualization is optimized for the 'Mercator' projection",
                     style={'color': 'gray', 'fontSize': 17}),

            dcc.Dropdown(id="projection",
            options=[
             {"label": "Aitoff", "value": "aitoff"},
             {"label": "Albers(USA)", "value": "albers usa"},
             {"label": "Azimuthal equal area", "value": "azimuthal equal area"},
             {"label": "Azimuthal equidistant", "value": "azimuthal equidistant"},
             {"label": "Conic conformal", "value": "conic conformal"},
             {"label": "Conic equal area", "value": "conic equal area"},
             {"label": "Conic equidistant", "value": "conic equidistant"},
             {"label": "Eckert4", "value": "eckert4"},
             {"label": "Equirectangular", "value": "equirectangular"},
             {"label": "Gnomonic", "value": "gnomonic"},
             {"label": "Hammer", "value": "hammer"},
             {"label": "Kavrayskiy7", "value": "kavrayskiy7"},
             {"label": "Mercator", "value": "mercator"},
             {"label": "Mollweide", "value": "mollweide"},
             {"label": "Natural earth", "value": "natural earth"},
             {"label": "Orthographic", "value": "orthographic"},
             {"label": "Robinson", "value": "robinson"},
             {"label": "Sinusoidal", "value": "sinusoidal"},
             {"label": "Stereographic", "value": "stereographic"},
             {"label": "Transverse mercator", "value": "transverse mercator"},
             {"label": "Winkel tripel", "value": "winkel tripel"},

             ],
            multi=False,
            value="mercator",
            style={'width': "40%"}
            ),
         ]),

    html.Br(),  # Br is a break i.e. a space in between
    #html.Br(),  # Br is a break i.e. a space in between

    html.P([
        html.Label("Start and End date of Observations "),

        html.H2("Select the desired time range of the observations ",
                style={'color': 'gray', 'fontSize': 17}),

        dcc.RangeSlider(id='year_slider',
                        min=1900,
                        max=2020,
                        step=10,
                        marks = { i: str(i) for i in dates },
                        value=[1900, 1950],

                        ) ],
        style={'width': '97%',
                'fontSize': '22px',
               'color' : "red",
                #'padding-left': '50px',
                'display': 'inline-block'} ),

    html.Br(),  # Br is a break i.e. a space in between
    html.Br(),  # Br is a break i.e. a space in between
    html.Br(),  # Br is a break i.e. a space in between

    # Placeholder for the maps (original and following Janicke et al. 2012 )
    html.Div([ html.Div([ dcc.Graph(id='map', figure={},)],
                        style={'width': '48%',
                               'display': 'inline-block',
                               'padding-left': '50px' }),

              html.Div([dcc.Graph(id='map_2', figure={} ,
                        clickData={'station': 'CONCORDIA'}
                        ),
                       ],
                       style={'width': '48%',
                              'display': 'inline-block',
                              'padding-left': '50px' }),
              ]),


    html.Br(),  # Br is a break i.e. a space in between
    html.Br(),  # Br is a break i.e. a space in between

    html.Div([ html.Div([

                 html.H2("IGRA2 Data ",
                         style={'color': 'red', 'fontSize': 22}),
                 html.H2("Select a data point from the Janicke map to display a sumamrz of the data of all stations included in the point",
                         style={'color': 'gray', 'fontSize': 17}),

                 html.Div([ dash_table.DataTable(
                            id='table',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": False} for i in dataframe.columns ],
                            data=dataframe[:10].to_dict('records'),
                            editable=False,
                            filter_action="native",
                            #sort_action="native",
                            #sort_mode="multi",
                            column_selectable=False,
                            row_selectable="single",
                            row_deletable=False,
                            page_action="native",
                            #page_size= 1,)
                        )],
                         style={
                               'display': 'inline-block',
                               'padding-left': '5px'}),
            ]),


            html.Div([

                html.H2("Temperature Trend ",
                        style={'color': 'red', 'fontSize': 22}),
                html.H2(
                    "Select a row from the table to display the Temperature time series [P=100hPa]",
                    style={'color': 'gray', 'fontSize': 17}),

                 html.Div([ dcc.Graph(id='time_series',
                                     )],
                         style={'width': '10%',
                                'display': 'inline-block',
                                'padding-left': '10px'}),


             ]),
        ])




])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})





# time series
@app.callback(
        #Output(component_id='time_series', component_property='figure'),
        Output(component_id='map_2', component_property='figure'),
        Output(component_id='map', component_property='figure'),

#        Output(component_id='time_series', component_property = 'figure'),

        Input('year_slider', 'value'),
        Input(component_id='scale', component_property='value'),
        Input(component_id='projection', component_property='value') )



def update_plots(year_range, scale, projection):
    """ map plot """
    font = 15
    marker_size = MSIZE
    # select the dataset

    df = dataframe

    start_date = year_range[0]
    end_date = year_range[1]

    df = df.loc[ (df['start'] >= start_date) & (df['end'] <= end_date )]

    def standard_map(df):
        """ Main function to create a geo-scatter plot using standard plotly functionality """

        map = go.Figure(data=go.Scattergeo(lat=df.latitude, lon=df.longitude,
                                           text=df["station"],
                                           mode='markers',
                                           marker=dict(size=marker_size, opacity=0.8, reversescale=True,
                                                       autocolorscale=False,
                                                       line=dict(
                                                           width=1,
                                                           # color='rgba(102, 102, 102)'
                                                       ),
                                                       colorscale='Plotly3', cmin=0,
                                                       color=df['records'],
                                                       cmax=df['records'].max(),
                                                       colorbar_title="Number of Records"
                                                       )
                                           )
                        )

        map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(projection_type=projection,
                     showland=True)
        )

        map.update_geos(
            resolution=50,
            showcoastlines=True, coastlinecolor="RebeccaPurple",
            showland=True, landcolor="White",
            showocean=True, oceancolor="LightBlue",
            showlakes=True, lakecolor="Blue",
            showrivers=True, rivercolor="Blue"
        )

        #map.update_layout(height=500, margin={"r": 20, "t": 20, "l": 20, "b": 20})
        map.update_layout(height=700, width = 1100,
               margin={"r": 20, "t": 50, "l": 20, "b": 20} )

        vienna_lat, vienna_lon = 48.2, 16.3

        map.update_layout(
            paper_bgcolor="LightSteelBlue",
            title='Default Map',
            geo=dict(
                projection_scale=scale,  # this is kind of like zoom
                center=dict(lat=vienna_lat, lon=vienna_lon),  # this will center on the point
            ))

        return map

    def paper_map(df):

        """ Main function to create a geo-scatter plot.
        Will scale points size according to the algorithm adapted from Janicke et al. 2012 """
        df = combine_points(df, scale)

        map = go.Figure(data=go.Scattergeo(lat=df.latitude, lon=df.longitude,
                                           text=df["station"],
                                           mode='markers',
                                           marker=dict(size=df['radius'], opacity=0.8, reversescale=True,
                                                       autocolorscale=False,
                                                       line=dict(
                                                           width=1,
                                                           #color='red',
                                                       ),
                                                       colorscale='Plotly3', cmin=0,
                                                       color=df['records'],
                                                       #cmax=df['records'].max(),
                                                       colorbar_title="Number of Records"
                                                       )
                                           )
                        )


        map.update_layout(
            geo=dict(projection_type=projection,
                     showland=True)
        )

        map.update_geos(
            resolution=50,
            showcoastlines=True, coastlinecolor="RebeccaPurple",
            showland=True, landcolor="LightGreen",
            showocean=True, oceancolor="LightBlue",
            showlakes=True, lakecolor="Blue",
            showrivers=True, rivercolor="Blue"
        )

        map.update_layout(height=700, width = 1100,
               margin={"r": 20, "t": 50, "l": 20, "b": 20}

                          )

        vienna_lat, vienna_lon = 48.2, 16.3

        map.update_layout(

            paper_bgcolor="LightSteelBlue",
            title='Janicke et Al. , 2012',
            geo=dict(
                projection_scale=scale,  # this is kind of like zoom
                center=dict(lat=vienna_lat, lon=vienna_lon),  # this will center on the point
            ))

        return map

    '''
    def time_series():
        """ Plots a temperature time series retirevend the station name from the selected row in the table """

        dir = "https://raw.githubusercontent.com/fambrogi/Visualization2/main/data/igra2_temp/"
        station = "ACM00078861"

        file = dir + "/" + station + ".csv"
        data = pd.read_csv(file,
                           sep="\t",
                           names=["index", "date", "temp"],
                           header=1 )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.date, y=data.temp,
                                 mode='lines',
                                 name='lines',
                                 marker_color='pink') )

        fig.update_layout( title = {
                                    "text": "Time series for the station " + station ,
                                    "y":1.,
                                    "x":0.5,
                                    "xanchor": "center",
                                    "yanchor": "top", } ,

                            height = 400, width = 1100,
                            margin = {"r": 20, "t": 50, "l": 20, "b": 20},
                            yaxis_title="Temperature [K]",
                            xaxis_title="Year",

                        )

        return fig

    '''

    standard_map = standard_map(df)
    paper_map = paper_map(df)
    #time_series = time_series()
    #return [paper_map, standard_map, time_series]  # NB must always return a list even if you have one output only, due to @app

    return [paper_map, standard_map]  # NB must always return a list even if you have one output only, due to @app



@app.callback(
    Output('table', 'data'),
    Input('map_2', 'clickData') )

def update_table(clickData):
    print("click data is: " , clickData)


    df_stat = np.array(dataframe['station'])

    if 'points' in clickData.keys():
        """
        indices = []
        for s in stations:
            indices.append(np.where(df_stat == s)[0][0])
        """
        try:
            stations = clickData['points'][0]['text'].split(',')
            indices = []
            for s in stations:
                 indices.append( np.where( df_stat == s)[0][0] )
            df = dataframe.iloc[indices]
            return df.to_dict('records')

        except:
            print("fail")
            df = dataframe[:10]
            return df.to_dict('records')


    else:
        stations = "WIEN/HOHE WARTE"
        indices = np.where( df_stat == stations)[0][0]
        df = dataframe[indices:indices+9]
        #df = dataframe.iloc[indices]
        return  df.to_dict('records')







@app.callback(
    Output(component_id='time_series', component_property='figure'),

    Input('table', 'active_cell'),
    State('table', 'data')
)

def upate_time_series(active_cell, data):
    """ Plots a temperature time series retrieving the station name from the selected row in the table """


    # Github directory containing the temperature data for each station
    dir = "https://raw.githubusercontent.com/fambrogi/Visualization2/main/data/igra2_temp/"

    print("active cell" , active_cell)
    print("data", data )

    try:
        cell = active_cell["row"]
        station = data[cell]["code"]

        file = dir + "/" + station + ".csv"
        data = pd.read_csv(file,
                       sep="\t",
                       names=["index", "date", "temp"],
                       header=1)

    except:
        print("I have a problem with the station selected :-( ")
        station = "ACM00078861"
        file = dir + "/" + station + ".csv"
        data = pd.read_csv(file,
                       sep="\t",
                       names=["index", "date", "temp"],
                       header=1)



    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.date, y=data.temp,
                             mode='lines',
                             name='lines',
                             marker_color='pink'))

    fig.update_layout(title={
        "text": "Time series for the station " + station,
        "y": 1.,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top", },

        height=400, width=500,
        margin={"r": 20, "t": 50, "l": 20, "b": 20},
        yaxis_title="Temperature [K]",
        xaxis_title="Year",

    )

    return fig

    print('no cell selected')

    return 0



if __name__ == '__main__':

    app.run_server(debug=True)
