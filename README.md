# Visualization2
### TU Wien, SS2021

Contributors:

* F. Ambrogi , federico.ambrogi88@gmail.com

Contents:
* [files](#Files)
* [code](#Code)
* [data](#Data)
* [requirements](#Requirements)
* [run](#Run)
* [dashboard](#Dashboard)

# Files

The present repository contains one main python file containing the executable, i.e. *station_visualizer.py*,
and one *data* directory.


# Code

There is only one main python script, called *station_visualizer.py*.
When running the file, e.g. with *python station_visualizer.py*,
a new webpage on the the user's default browser will appear, where the dashboard is run.
See the dashboard section for details.

# Data

The data directory contains one file called *igra2.csv*, which contains the main information regarding the **IGRA2** inventory stations distributions.
The *igra2_temp* directory contains a list of 2789 station files, each structured as tsv files.

The first columns represent the index of the observation, then the second and third columns correspond to the time-stamps and observed temperature respectively.
Note that the original file in the **IGRA2** contained much more data than the data here collected and processed. In particular,
only the time-stamps and the observed temperature at a pressure of 1000 hPa are stored here.
The total size of the dataset amounts to 471Mb. However, it is not necessary to download the data, since the time-series plot will automatically
read the files from this GitHub repository.

The original dataset as	well as	extensive information can be consulted at https://catalog.data.gov/dataset/integrated-global-radiosonde-archive-igra-version-2 .

# Requirements

The following libraries are needed to run the script:
- python3.8 
- pandas (v. 1.1.3)
- numpy (v. 1.19.1)
- scipy (v. 1.5.2)
- plotly (v. 4.14.1)
- dash (v. 1.18.1)

# Run

To run the script, simply call it as *python station_visualizer.py*.
On the shell, a link to an html page will appear. By clicking on it,
a local server will be launched, and the dashboard can be then accessed and used by the user.

# Dashboard

The dashboard is structured in the following way.

1. A dropdown menu lets the user select the type of projection.
Note that the code is tested and optimized only with the *Mercator* projection.
Nevertheless, other projections are available

2. A dropdown menu lets the user select the level of zooming between [1,2,3,5,10].
Nothe that using the interactive zooming with the mouse wheel will not produce correct results.
However, it is possible to drag the map (which is, by default, centred around the coordinates of Vienna).

3. The date range can be selected with a date slider.
The stations represented in the plots have valid data in between the selected years.

4. Two geo-scatter maps willbe displayed.
On the right, a version created using the basic plotly functionalities.
On the left, the version using the technique adapted from Janicke et Al. 20120
(https://www.informatik.uni-leipzig.de/~stjaenicke/Comparative_Visualization_Of_Geospatial-Temporal_Data.pdf)
The color code represents the total number of records (i.e. amount of observations) inside each bubble, which consists
of the agglomeration of (potentially several) stations.

5. The Janicke plot will show agglomerated stations into a bigger bubble. By clicking on a bubble,
the table below will shoe the summary of *all* the stations included in the bubble.
This is useful to obtain information regarding each single radiosonde station.

6. The time series plot will show the values of the temperature measured across all the available years,
for the pressure level of 1000 hPa. The user can select the station he likes to investigate by clicking on any cell
of the table (of the row of the desired station). Note that the temperature data might not be available, and in this case
the plot will not be updated, but it will show the last valid or the default data.

