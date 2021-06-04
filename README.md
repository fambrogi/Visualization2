# Visualization2
## TU Wien, SS2021

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

The present repository contains one main pytohn file containing the executable, i.e. "station_visualizer.py",
and one "data" directory.


# Code

There is only one main python script, called "station_visualizer.py".
When running the file, e.g. with python station_visualizer.py,
a new webpage on the 

# Data

The data directory contains one file called igra2.csv, which contains the main information regarding the IGRA2 inventory stations distributions.
The "igra2_temp" directory contains a list of 2789 station files, each structured as tsv files.
The first columns represent the index of the observation, then the second and third columns correpsond to the time-stamps and observed temperature respectively.

# Requirements

The following libraries are needed to run the script:
- python3.8 (tested)
- pandas
- numpy
- scipy
- plotly
- dash

# Run

To run the script, simply call it as "python station_visualizer.py"


# Dashboard

The dashboard is structured in the following way.



