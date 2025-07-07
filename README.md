# Time Series Analysis

1. The script in TimeSeries.py imports essential libraries and suppresses warnings. It reads data from a CSV file, filters out invalid geolocations, and converts timestamps to datetime format. The data is then aggregated to obtain weekly counts of service requests, and lagged features are created. An ARIMA model is applied to the data for each unique hex ID to forecast the next four weeks, with the results saved to a CSV file and visualized. To run the code successfully, download and extract sr_hex.csv from https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com/sr.csv.gz and save it in the same directory as the TimeSeries.py file.

2. The SolutionSummary.docx file provides an overview of the initial solution and suggests ways to improve it.
