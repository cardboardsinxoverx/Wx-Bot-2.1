import discord
from discord.ext import commands
import matplotlib.pyplot as plt
#from metpy.plots import Meteogram
import requests
from bs4 import BeautifulSoup
import pandas as pd
import metpy.calc as mpcalc
from metpy.units import units
from matplotlib.dates import DayLocator, HourLocator, DateFormatter
import warnings  # For suppressing warnings
from datetime import datetime, timedelta
import os
from metpy.plots import add_metpy_logo
from metpy.calc import dewpoint_from_relative_humidity
from matplotlib.ticker import MultipleLocator
import metar
import json

# --- METAR fetching and parsing ---

def get_metar_meteogram(icao, hoursback=None):
    """
    Download METARs for the specified airport code from ADDS.

    Parameters
    ----------
    icao : str
        ICAO identifier used when reporting METARs
    hoursback : str or int
        Number of hours before present to query

    Returns
    ----------
    raw_metars : list
        List of raw METAR observations
    """

    try:
        if hoursback:
            metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format=json&hours={hoursback}'
        else:
            metar_url = f'https://aviationweather.gov/api/data/metar?ids={icao}&format=json'  # Default to latest METAR

        src = requests.get(metar_url).content
        json_data = json.loads(src)

        # Check if any METAR data was found at all
        if not json_data:
            raise ValueError(f"No METAR data found for {icao}.")

        # Extract raw METAR observations
        raw_metars = [entry['rawOb'] for entry in json_data]

        return raw_metars

    except requests.exceptions.RequestException as e:
        # Handle network errors during fetching
        raise Exception(f"Error fetching METAR data for {icao}: {e}")
    except (KeyError, ValueError) as e:
        # Handle potential parsing errors
        raise Exception(f"Error parsing METAR data for {icao}: {e}")

def parse_metar_to_dataframe(metar_data):
    """Parses METAR data from the API response into a Pandas DataFrame."""

    # Create a list to store parsed METAR data
    parsed_data = []

    # Extract METAR strings from the API response (assuming raw_metars is a list)
    metar_lines = metar_data

    for line in metar_lines:
        try:
            # Parse each METAR observation using the correct class
            obs = metar.Metar(line)  # Use metar.Metar directly

            # Extract relevant data and append to the list
            data = {
                'station_id': obs.station_id,
                'date_time': obs.time.replace(tzinfo=None),  # Convert to naive datetime
                'air_temperature': obs.temp.value('C') if obs.temp else np.nan,
                'dew_point_temperature': obs.dewpt.value('C') if obs.dewpt else np.nan,
                'wind_speed': obs.wind_speed.value('KT') if obs.wind_speed else np.nan,
                'wind_direction': obs.wind_dir.value() if obs.wind_dir else np.nan,
                'wind_gust': obs.wind_gust.value('KT') if obs.wind_gust else np.nan,
                'visibility': obs.vis.value('M') if obs.vis else np.nan,
                'altimeter': obs.press.value('IN') if obs.press else np.nan,
                'low_cloud_level': obs.sky[0][1].value('FT') if obs.sky and len(obs.sky) > 0 and obs.sky[0][1] else np.nan,
                'medium_cloud_level': obs.sky[1][1].value('FT') if obs.sky and len(obs.sky) > 1 and obs.sky[1][1] else np.nan,
                'high_cloud_level': obs.sky[2][1].value('FT') if obs.sky and len(obs.sky) > 2 and obs.sky[2][1] else np.nan,
                'air_pressure_at_sea_level': obs.press_sea_level.value('hPa') if obs.press_sea_level else np.nan,
                'present_weather': obs.present_weather() if obs.present_weather() else np.nan
            }
            parsed_data.append(data)

        except Exception as e:
            print(f"Error parsing METAR: {line}. Skipping this observation. Error: {e}")

    # Create a DataFrame from the parsed data
    df = pd.DataFrame(parsed_data)

    # Convert 'date_time' column to datetime format if needed
    df['date_time'] = pd.to_datetime(df['date_time'])

    return df

class Meteogram:
    """ Plot a time series of meteorological data from a particular station as a
    meteogram with standard variables to visualize, including thermodynamic,
    kinematic, and pressure. The functions below control the plotting of each
    variable.
    TO DO: Make the subplot creation dynamic so the number of rows is not
    static as it is currently. """

    def __init__(self, fig, dates, probeid, time=None, axis=0):
        """
        Required input:
            fig: figure object
            dates: array of dates corresponding to the data
            probeid: ID of the station
        Optional Input:
            time: Time the data is to be plotted
            axis: number that controls the new axis to be plotted (FOR FUTURE)
        """
        if not time:
            time = dt.datetime.now(dt.timezone.utc)
        self.start = dates[0]
        self.fig = fig
        self.end = dates[-1]
        self.axis_num = 0
        self.dates = mpl.dates.date2num(dates)
        self.time = time.strftime('%Y-%m-%d %H:%M UTC')
        self.title = f'Latest Ob Time: {self.time}\nProbe ID: {probeid}'
        self.probeid = probeid

    def plot_winds(self, ws, wd, wsmax, plot_range=None):
        """
        Required input:
            ws: Wind speeds (knots)
            wd: Wind direction (degrees)
            wsmax: Wind gust (knots)
        Optional Input:
            plot_range: Data range for making figure (list of (min,max,step))
        """
        # PLOT WIND SPEED AND WIND DIRECTION
        self.ax1 = fig.add_subplot(4, 1, 1)
        ln1 = self.ax1.plot(self.dates, ws, label='Wind Speed')
        self.ax1.fill_between(self.dates, ws, 0)
        self.ax1.set_xlim(self.start, self.end)
        ymin, ymax, ystep = plot_range if plot_range else (0, 20, 2)
        self.ax1.set_ylabel('Wind Speed (knots)', multialignment='center')
        self.ax1.set_ylim(ymin, ymax)
        self.ax1.yaxis.set_major_locator(MultipleLocator(ystep))
        self.ax1.grid(which='major', axis='y', color='k', linestyle='--', linewidth=0.5)
        ln2 = self.ax1.plot(self.dates, wsmax, '.r', label='3-sec Wind Speed Max')

        ax7 = self.ax1.twinx()
        ln3 = ax7.plot(self.dates, wd, '.k', linewidth=0.5, label='Wind Direction')
        ax7.set_ylabel('Wind\nDirection\n(degrees)', multialignment='center')
        ax7.set_ylim(0, 360)
        ax7.set_yticks(np.arange(45, 405, 90))
        ax7.set_yticklabels(['NE', 'SE', 'SW', 'NW'])
        lines = ln1 + ln2 + ln3
        labs = [line.get_label() for line in lines]
        ax7.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%H UTC'))
        ax7.legend(lines, labs, loc='upper center',
                   bbox_to_anchor=(0.5, 1.2), ncol=3, prop={'size': 12})

    def plot_thermo(self, t, td, plot_range=None):
        """
        Required input:
            T: Temperature (deg C)
            TD: Dewpoint (deg C)
        Optional Input:
            plot_range: Data range for making figure (list of (min,max,step))
        """
        # PLOT TEMPERATURE AND DEWPOINT

        ymin, ymax, ystep = plot_range if plot_range else (-10, 40, 5)  # Adjust range for Celsius
        self.ax2 = fig.add_subplot(6, 1, 2, sharex=self.ax1)  # 6 rows now
        ln4 = self.ax2.plot(self.dates, t, 'r-', label='Temperature')
        self.ax2.fill_between(self.dates, t, td, color='r')

        self.ax2.set_ylabel('Temperature\n(°C)',
        multialignment='center')
        self.ax2.grid(which='major', axis='y', color='k', linestyle='--', linewidth=0.5)
        self.ax2.set_ylim(ymin, ymax)
        self.ax2.yaxis.set_major_locator(MultipleLocator(ystep))

        ln5 = self.ax2.plot(self.dates, td, 'g-', label='Dewpoint')
        self.ax2.fill_between(self.dates, td, self.ax2.get_ylim()[0], color='g')

        ax_twin = self.ax2.twinx()
        ax_twin.set_ylim(ymin, ymax)
        ax_twin.yaxis.set_major_locator(MultipleLocator(ystep))
        lines = ln4 + ln5
        labs = [line.get_label() for line in lines]
        ax_twin.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%H UTC'))

        self.ax2.legend(lines, labs, loc='upper center',
                        bbox_to_anchor=(0.5, 1.2), ncol=2, prop={'size': 12})

    def plot_heat_index(self, hi, plot_range=None):
        """
        Required input:
            HI: Heat Index (deg C)
        Optional Input:
            plot_range: Data range for making figure (list of (min,max,step))
        """
        # PLOT HEAT INDEX
        ymin, ymax, ystep = plot_range if plot_range else (-10, 50, 5)  # Adjust as needed
        self.ax3 = fig.add_subplot(6, 1, 3, sharex=self.ax1)
        self.ax3.plot(self.dates, hi, 'orange', label='Heat Index')
        self.ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), prop={'size': 12})
        self.ax3.grid(which='major', axis='y', color='k', linestyle='--', linewidth=0.5)
        self.ax3.set_ylim(ymin, ymax)
        self.ax3.yaxis.set_major_locator(MultipleLocator(ystep))
        self.ax3.set_ylabel('Heat Index\n(°C)', multialignment='center')
        self.ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%H UTC'))
        axtwin = self.ax3.twinx()
        axtwin.set_ylim(ymin, ymax)
        axtwin.yaxis.set_major_locator(MultipleLocator(ystep))

    def plot_pressure(self, p, plot_range=None):
        """
        Required input:
            P: Mean Sea Level Pressure (hPa)
        Optional Input:
            plot_range: Data range for making figure (list of (min,max,step))
        """
        # PLOT PRESSURE
        ymin, ymax, ystep = plot_range if plot_range else (970, 1030, 4)
        self.ax4 = fig.add_subplot(4, 1, 4, sharex=self.ax1)
        self.ax4.plot(self.dates, p, 'm', label='Mean Sea Level Pressure')
        self.ax4.set_ylabel('Mean Sea\nLevel Pressure\n(mb)', multialignment='center')
        self.ax4.set_ylim(ymin, ymax)
        self.ax4.yaxis.set_major_locator(MultipleLocator(ystep))

        axtwin = self.ax4.twinx()
        axtwin.set_ylim(ymin, ymax)
        axtwin.yaxis.set_major_locator(MultipleLocator(ystep))
        axtwin.fill_between(self.dates, p, axtwin.get_ylim()[0], color='m')
        axtwin.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%H UTC'))

        self.ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12})
        self.ax4.grid(which='major', axis='y', color='k', linestyle='--', linewidth=0.5)

    def plot_cloud_cover(self, cc, plot_range=None):
        """
        Required input:
            CC: Cloud Cover (%, 0-100)
        Optional Input:
            plot_range: Data range for making figure (list of (min,max,step))
        """
        # PLOT CLOUD COVER (using shading)
        ymin, ymax, ystep = plot_range if plot_range else (0, 100, 10)
        self.ax4 = fig.add_subplot(6, 1, 4, sharex=self.ax1)
        self.ax4.fill_between(self.dates, 0, cc, color='gray', alpha=0.5, label='Cloud Cover')
        self.ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), prop={'size': 12})
        self.ax4.grid(which='major', axis='y', color='k', linestyle='--', linewidth=0.5)
        self.ax4.set_ylim(ymin, ymax)
        self.ax4.yaxis.set_major_locator(MultipleLocator(ystep))
        self.ax4.set_ylabel('Cloud Cover\n(%)', multialignment='center')
        self.ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%H UTC'))
        axtwin = self.ax4.twinx()
        axtwin.set_ylim(ymin, ymax)
        axtwin.yaxis.set_major_locator(MultipleLocator(ystep))

    def plot_meteogram(data, station_id):
        with plt.rc_context({'figure.figsize': (20, 20)}):  # Increased height
            # Add MetPy logo (if applicable)
            add_metpy_logo(plt, 250, 180)  # Uncomment if you have MetPy installed

            meteogram = Meteogram(data['date_time'], station_id)
            meteogram.plot_winds(data['wind_speed'], data['wind_direction'], data['wind_gust'])
            meteogram.plot_thermo(data['air_temperature'], data['dew_point_temperature'])
            meteogram.plot_heat_index(data['heat_index'])  # Assuming you have heat index data
            meteogram.plot_cloud_cover(data['cloud_cover'])  # Assuming you have cloud cover data
            meteogram.plot_pressure(data['air_pressure_at_sea_level'])

            plt.subplots_adjust(hspace=0.5)

        # Save the plot
        temp_image_path = f"/home/evanl/Documents/meteogram_{station_id}.png"
        try:
            plt.savefig(temp_image_path, bbox_inches='tight')  # Save the image
        except Exception as e:
            print(f"Error saving the image: {e}")
        finally:
            plt.close()  # Close the figure

        return temp_image_path
