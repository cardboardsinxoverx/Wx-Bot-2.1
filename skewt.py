import datetime
import os
import pytz
import requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units
from matplotlib.patches import Rectangle
from metpy.plots.wx_symbols import sky_cover
from metpy.plots import add_metpy_logo
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, LinearSegmentedColormap
from siphon.simplewebservice.wyoming import WyomingUpperAir
import discord
from discord.ext import commands
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import asyncio
import logging
import pint
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define intents
intents = discord.Intents.default()
intents.message_content = True

# Custom Help Command
class CustomHelpCommand(commands.DefaultHelpCommand):
    async def send_bot_help(self, mapping):
        # General help message when no specific command is provided
        ctx = self.context
        help_text = (
            "**Bot Commands**\n"
            "Use `$help <command>` for details on a specific command.\n\n"
            "**Available Commands:**\n"
            "- **skewt**: Generate a Skew-T diagram from observed or forecast sounding data.\n"
            "Type `$help skewt` for more information."
        )
        await ctx.send(help_text)

    async def send_command_help(self, command):
        # Detailed help for a specific command using its docstring
        ctx = self.context
        if command.name == 'skewt':
            help_text = command.help
            await ctx.send(help_text)
        else:
            # Fallback to default help for other commands (if any)
            await super().send_command_help(command)

# Initialize bot with custom help command
bot = commands.Bot(command_prefix='$', intents=intents, help_command=CustomHelpCommand())

# Station coordinates (latitude, longitude)
STATIONS = {
    'FFC': (33.36, -84.57),  # Atlanta, GA (KATL)
    'OUN': (35.18, -97.44),  # Norman, OK (KOUN)
    'HOU': (29.65, -95.28),  # Houston, TX (KHOU)
    'OKC': (35.39, -97.60),  # Oklahoma City, OK (KOKC)
    'LAX': (33.94, -118.41), # Los Angeles, CA (KLAX)
    'BOS': (42.36, -71.01),  # Boston, MA (KBOS)
    'ORD': (41.98, -87.90),  # Chicago, IL (KORD)
    'DEN': (39.86, -104.67), # Denver, CO (KDEN)
    'DFW': (32.90, -97.04),  # Dallas-Fort Worth, TX (KDFW)
    'IND': (39.72, -86.29),  # Indianapolis, IN (KIND)
    'LAS': (36.08, -115.15), # Las Vegas, NV (KLAS)
    'MIA': (25.79, -80.29),  # Miami, FL (KMIA)
    'PHX': (33.43, -112.01), # Phoenix, AZ (KPHX)
    'SEA': (47.45, -122.31), # Seattle, WA (KSEA)
    'MSP': (44.88, -93.22),  # Minneapolis, MN (KMSP)
    'ASH': (42.78, -71.51),  # Nashua, NH (KASH)
    'JFK': (40.64, -73.78),  # New York, NY (KJFK)
    'PDX': (45.59, -122.60), # Portland, OR (KPDX)
    'PHL': (39.87, -75.24),  # Philadelphia, PA (KPHL)
    'PIT': (40.49, -80.23),  # Pittsburgh, PA (KPIT)
    'SAN': (32.73, -117.19), # San Diego, CA (KSAN)
    'SFO': (37.62, -122.37), # San Francisco, CA (KSFO)
    'STL': (38.75, -90.37),  # St. Louis, MO (KSTL)
    'TPA': (27.98, -82.53),  # Tampa, FL (KTPA)
    'TVC': (44.74, -85.58),  # Traverse City, MI (KTVC)
    'IAD': (38.94, -77.46),  # Washington, DC (KIAD)
}

# Mapping to NWS station IDs with working BUFKIT links
NWS_STATIONS = {
    'FFC': 'KATL',  # Atlanta, GA
    'OUN': 'KOUN',  # Norman, OK
    'HOU': 'KHOU',  # Houston, TX
    'OKC': 'KOKC',  # Oklahoma City, OK
    'LAX': 'KLAX',  # Los Angeles, CA
    'BOS': 'KBOS',  # Boston, MA
    'ORD': 'KORD',  # Chicago, IL
    'DEN': 'KDEN',  # Denver, CO
    'DFW': 'KDFW',  # Dallas-Fort Worth, TX
    'IND': 'KIND',  # Indianapolis, IN
    'LAS': 'KLAS',  # Las Vegas, NV
    'MIA': 'KMIA',  # Miami, FL
    'PHX': 'KPHX',  # Phoenix, AZ
    'SEA': 'KSEA',  # Seattle, WA
    'MSP': 'KMSP',  # Minneapolis, MN
    'ASH': 'KASH',  # Nashua, NH
    'JFK': 'KJFK',  # New York, NY
    'PDX': 'KPDX',  # Portland, OR
    'PHL': 'KPHL',  # Philadelphia, PA
    'PIT': 'KPIT',  # Pittsburgh, PA
    'SAN': 'KSAN',  # San Diego, CA
    'SFO': 'KSFO',  # San Francisco, CA
    'STL': 'KSTL',  # St. Louis, MO
    'TPA': 'KTPA',  # Tampa, FL
    'TVC': 'KTVC',  # Traverse City, MI
    'IAD': 'KIAD',  # Washington, DC
}

def parse_bufkit_text(bufkit_text, forecast_hour):
    lines = bufkit_text.splitlines()
    model_run = None
    for line in lines:
        if 'TIME =' in line:
            match = re.search(r'TIME =\s*(\d{6}/\d{4})', line)
            if match:
                time_str = match.group(1)
                try:
                    model_run = datetime.datetime.strptime(time_str, "%y%m%d/%H%M").replace(tzinfo=pytz.UTC)
                    break
                except ValueError:
                    continue
    if not model_run:
        raise ValueError("Could not find a valid model run time in BUFKIT file.")
    print(f"Model run time: {model_run}")
    profile_starts = []
    for i, line in enumerate(lines):
        if 'STIM =' in line:
            try:
                stim = int(line.split('=')[1].strip())
                profile_starts.append((stim, i))
            except (IndexError, ValueError):
                continue
    if not profile_starts:
        raise ValueError("No valid STIM entries found in BUFKIT file.")
    closest_stim, start_idx = min(profile_starts, key=lambda x: abs(x[0] - forecast_hour))
    valid_time = model_run + datetime.timedelta(hours=closest_stim)
    end_idx = len(lines)
    for stim, idx in profile_starts:
        if idx > start_idx:
            end_idx = idx
            break
    profile_lines = lines[start_idx:end_idx]
    data_start = None
    for i, line in enumerate(profile_lines):
        if line.strip().startswith('PRES'):
            data_start = i
            break
    if data_start is None:
        raise ValueError(f"No sounding data found for forecast hour {closest_stim}.")
    header = profile_lines[data_start].split()
    data_lines = [line.split() for line in profile_lines[data_start + 1:] if line.strip() and not line.startswith('%')]
    if not data_lines:
        raise ValueError(f"No data rows found for forecast hour {closest_stim}.")
    header_len = len(header)
    valid_data_lines = [line for line in data_lines if len(line) == header_len]
    if not valid_data_lines:
        raise ValueError(f"No valid data lines found for forecast hour {closest_stim} with {header_len} columns.")
    df = pd.DataFrame(valid_data_lines, columns=header)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df, valid_time

async def fetch_bufkit_data_async(ctx, station_code, model, forecast_hour, max_retries=10, retry_delay=5):
    logger.info("Starting BUFKIT data fetch process")
    nws_station = NWS_STATIONS.get(station_code)
    if not nws_station:
        error_msg = f"No NWS station mapping found for {station_code}. Supported stations: {', '.join(STATIONS.keys())}"
        logger.error(error_msg)
        await ctx.send(error_msg)
        return None, None
    url = f"http://www.meteor.iastate.edu/~ckarsten/bufkit/data/gfs/gfs3_{nws_station.lower()}.buf"
    logger.info(f"Fetching BUFKIT file from {url}")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 404:
                logger.error(f"BUFKIT file not found at {url}")
                await ctx.send(f"No BUFKIT file found for {station_code} ({nws_station}) with model {model}.")
                return None, None
            elif response.status_code != 200:
                raise ValueError(f"Failed to fetch BUFKIT file: {response.status_code} - {response.reason}")
            bufkit_text = response.text
            df, valid_time = parse_bufkit_text(bufkit_text, forecast_hour)

            df.replace(-9999.00, np.nan, inplace=True)
            df = df[df['PRES'] >= 100].copy()  # Filter out pressures below 100 hPa

            required_cols = ['PRES', 'TMPC', 'DWPC', 'DRCT', 'SKNT']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"BUFKIT data missing columns: {missing_cols}")
            df = df[required_cols].copy()
            df = df.dropna(subset=required_cols)  # Drop NaN values in required columns
            df = df.drop_duplicates(subset=['PRES'], keep='first')  # Remove duplicate pressures
            df.rename(columns={'PRES': 'pressure', 'TMPC': 'temperature', 'DWPC': 'dewpoint'}, inplace=True)
            wind_speed = df['SKNT'].values * units.knots
            wind_dir = df['DRCT'].values * units.degrees
            u, v = mpcalc.wind_components(wind_speed, wind_dir)
            df['u'] = u.to('m/s').magnitude
            df['v'] = v.to('m/s').magnitude
            pressure_array = df['pressure'].to_numpy() * units.hPa
            height_array = mpcalc.pressure_to_height_std(pressure_array)
            df['height'] = height_array.to('meters').magnitude
            logger.info(f"Successfully fetched and parsed BUFKIT data for {station_code} at forecast hour {forecast_hour}")
            return df, valid_time
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            if attempt < max_retries - 1:
                await ctx.send(f"Request timed out. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                await ctx.send(f"Failed to fetch BUFKIT data for {station_code} after {max_retries} attempts due to timeout.")
                return None, None
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await ctx.send(f"Error fetching BUFKIT data: {str(e)}. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                await ctx.send(f"Failed to fetch BUFKIT data for {station_code} after {max_retries} attempts: {str(e)}")
                return None, None
    return None, None

async def fetch_sounding_data(ctx, time, station_code, data_type, max_retries=10, retry_delay=5):
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Fetching data for {station_code} at {time}")
            df = await asyncio.to_thread(WyomingUpperAir.request_data, time, station_code)
            if df is None or df.empty:
                raise ValueError("Received empty or None DataFrame from WyomingUpperAir")

            if data_type == "observed":
                df.rename(columns={'u_wind': 'u', 'v_wind': 'v'}, inplace=True)

            # Filter out pressures below 100 hPa
            df = df[df['pressure'] >= 100]

            # Clean essential columns and drop NaNs
            essential_cols = ['pressure', 'temperature', 'dewpoint', 'u', 'v']
            df = df.dropna(subset=essential_cols)

            # Sort by pressure (descending) and remove duplicates
            df = df.sort_values(by='pressure', ascending=False)
            df = df.drop_duplicates(subset=['pressure'], keep='first')

            logger.info(f"Successfully fetched and cleaned data for {station_code}: {len(df)} rows")
            return df
        except ValueError as e:
            error_str = str(e)
            if "503" in error_str or "Server Error" in error_str:
                if attempt < max_retries - 1:
                    await ctx.send(f"Server busy. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                    logger.warning(f"503 error on attempt {attempt + 1} for {station_code}: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Max retries ({max_retries}) reached for {station_code}: {e}")
                    await ctx.send(f"Failed to fetch data for {station_code} after {max_retries} attempts due to server overload.")
                    return None
            else:
                logger.error(f"Value error fetching data for {station_code}: {e}")
                await ctx.send(f"Error fetching data: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {station_code}: {e}")
            if attempt < max_retries - 1:
                await ctx.send(f"Error fetching data. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                await ctx.send(f"Failed to fetch data for {station_code} after {max_retries} attempts.")
                return None
    return None

def calculate_temperature_advection(p, u, v, lat):
    try:
        f = mpcalc.coriolis_parameter(lat * units.degrees)
        R_d = 287 * units('J/(kg K)')
        p_pa = p.to('Pa')
        delta_p = np.diff(p_pa)
        delta_u = np.diff(u)
        delta_v = np.diff(v)
        p_avg = (p_pa[:-1] + p_pa[1:]) / 2
        dTdx_layers = - (f * p_avg / R_d) * (delta_v / delta_p)
        dTdy_layers = (f * p_avg / R_d) * (delta_u / delta_p)
        n = len(p)
        dTdx = np.zeros(n) * dTdx_layers.units
        dTdy = np.zeros(n) * dTdy_layers.units
        dTdx[0], dTdy[0] = dTdx_layers[0], dTdy_layers[0]
        dTdx[n-1], dTdy[n-1] = dTdx_layers[-1], dTdy_layers[-1]
        for k in range(1, n-1):
            dTdx[k] = (dTdx_layers[k-1] + dTdx_layers[k]) / 2
            dTdy[k] = (dTdy_layers[k-1] + dTdy_layers[k]) / 2
        advection = - (u * dTdx + v * dTdy)
        return (advection * 3600 * units('s/hour')).to('degC/hour')
    except Exception as e:
        logger.error(f"Error in temperature advection calculation: {e}")
        return np.full(len(p), np.nan) * units('degC/hour')

def calculate_total_totals(p, T, Td):
    try:
        idx_850 = np.argmin(np.abs(p - 850 * units.hPa))
        idx_500 = np.argmin(np.abs(p - 500 * units.hPa))
        T_850 = T[idx_850].to('degC')
        Td_850 = Td[idx_850].to('degC')
        T_500 = T[idx_500].to('degC')
        TT = (T_850.magnitude + Td_850.magnitude) - 2 * T_500.magnitude
        return TT * units.dimensionless
    except Exception as e:
        logger.error(f"Error in Total Totals calculation: {e}")
        return np.nan * units.dimensionless

def label_ccl(skew, p, T, Td):
    try:
        p_surface = p[0]
        Td_surface = Td[0]
        e_surface = mpcalc.saturation_vapor_pressure(Td_surface)
        w_surface = mpcalc.mixing_ratio(e_surface, p_surface)
        e_p = (w_surface * p) / (0.622 + w_surface)
        Td_p = mpcalc.dewpoint(e_p)
        diff = (T - Td_p).magnitude
        idx = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(idx) > 0:
            i = idx[0]
            p1, p2 = p[i], p[i+1]
            diff1, diff2 = diff[i], diff[i+1]
            p_CCL = p1 - (p1 - p2) * (diff1 / (diff1 - diff2))
            T_CCL = np.interp(p_CCL.magnitude, p.magnitude[::-1], T.magnitude[::-1]) * T.units
            skew.ax.scatter(T_CCL, p_CCL, color='purple', marker='o', s=50)
            skew.ax.text(T_CCL.magnitude + 2, p_CCL.magnitude, 'CCL', fontsize=10, color='purple')
        else:
            logger.warning("No CCL found in the sounding.")
    except Exception as e:
        logger.error(f"Error labeling CCL: {e}")

def label_mixing_ratios(skew):
    mixing_ratios = [1, 2, 4, 8, 16]  # g/kg
    label_pressure = 850 * units.hPa
    for mr in mixing_ratios:
        try:
            w = (mr / 1000.0) * units('kg/kg')
            e = (w * label_pressure) / (0.622 + w)
            Td = mpcalc.dewpoint(e)
            x_pos = Td.to('degC').magnitude + 5
            y_pos = label_pressure.magnitude
            skew.ax.text(
                x_pos, y_pos, f'{mr} g/kg',
                fontsize=13, color='#02a312',
                verticalalignment='center',
                horizontalalignment='left'
            )
        except Exception as e:
            logger.warning(f"Could not label mixing ratio {mr} g/kg: {e}")

@bot.command()
async def skewt(ctx, *args):
    """
    Generate an enhanced Skew-T diagram from observed or forecast sounding data.

    **Usage:**
    - **Observed Sounding:** `$skewt <station> <time>`
      - `<station>`: Three-letter station code (see list below)
      - `<time>`: '00Z' or '12Z' (UTC times for observed soundings)
      - Example: `$skewt FFC 12Z`

    - **Forecast Sounding:** `$skewt <station> <model> <forecast_hour>`
      - `<station>`: Three-letter station code (see list below)
      - `<model>`: 'gfs' (currently the only supported model)
      - `<forecast_hour>`: Integer (0-384) representing hours ahead from the latest GFS run
      - Example: `$skewt HOU gfs 6`

    **Available Station Codes:**
    - **ASH**: Nashua, New Hampshire
    - **BOS**: Boston, Massachusetts
    - **DEN**: Denver, Colorado
    - **DFW**: Dallas-Fort Worth, Texas
    - **FFC**: Atlanta, Georgia
    - **HOU**: Houston, Texas
    - **IAD**: Washington, District of Columbia
    - **IND**: Indianapolis, Indiana
    - **JFK**: New York, New York
    - **LAS**: Las Vegas, Nevada
    - **LAX**: Los Angeles, California
    - **MIA**: Miami, Florida
    - **MSP**: Minneapolis, Minnesota
    - **OKC**: Oklahoma City, Oklahoma
    - **ORD**: Chicago, Illinois
    - **OUN**: Norman, Oklahoma
    - **PDX**: Portland, Oregon
    - **PHL**: Philadelphia, Pennsylvania
    - **PHX**: Phoenix, Arizona
    - **PIT**: Pittsburgh, Pennsylvania
    - **SAN**: San Diego, California
    - **SEA**: Seattle, Washington
    - **SFO**: San Francisco, California
    - **STL**: St. Louis, Missouri
    - **TPA**: Tampa, Florida
    - **TVC**: Traverse City, Michigan

    **Notes:**
    - Station codes are three letters without the 'K' prefix.
    - Observed soundings use University of Wyoming data; forecast soundings use NOAA GFS data from Iowa State University.
    - For forecast hours, 0 is the analysis time, and values increase in 3-hour increments (e.g., 3, 6, 9, up to 384).
    """
    logger.info(f"Received skewt command with args: {args}")

    utc_time = datetime.datetime.now(pytz.UTC)

    try:
        if len(args) == 2:
            station_code, sounding_time = args
            station_code = station_code.upper()
            sounding_time = sounding_time.upper()
            if sounding_time not in ['00Z', '12Z']:
                raise ValueError("Invalid time. Use '00Z' or '12Z'.")
            year, month, day = utc_time.year, utc_time.month, utc_time.day
            hour = 12 if sounding_time == "12Z" else 0
            if sounding_time == "00Z" and utc_time.hour >= 12:
                day += 1
            now = datetime.datetime(year, month, day, hour, 0, 0, tzinfo=pytz.UTC)
            data_type = "observed"  # Set data_type for observed soundings
            df = await fetch_sounding_data(ctx, now, station_code, data_type)  # Pass data_type
            if df is None:
                return
            station_lat = df['latitude'][0]
            station_lon = df['longitude'][0]
            title = f"{station_code} - {now.strftime('%Y-%m-%d %HZ')}"
        elif len(args) == 3:
            station_code, model, forecast_hour = args
            station_code = station_code.upper()
            model = model.lower()
            forecast_hour = int(forecast_hour)
            if model != 'gfs':
                raise ValueError("Only 'gfs' model is supported currently.")
            if station_code not in NWS_STATIONS:
                raise ValueError(f"Station {station_code} not found in NWS station mapping.")
            df, valid_time = await fetch_bufkit_data_async(ctx, station_code, model, forecast_hour)
            if df is None:
                return
            station_lat, station_lon = STATIONS[station_code]
            title = f"{station_code} {model.upper()} Forecast {forecast_hour}-hr ({valid_time.strftime('%Y-%m-%d %HZ')})"
            data_type = "forecast"
        else:
            raise ValueError("Usage: `$skewt <station> <time>` or `$skewt <station> <model> <forecast_hour>`")

        df.drop_duplicates(subset=['pressure'], keep='first', inplace=True)
        essential_cols = ['pressure', 'height', 'temperature', 'dewpoint', 'u', 'v']
        df = df.dropna(subset=essential_cols)
        if df.empty:
            raise ValueError(f"No valid data for {station_code} after cleaning.")

        z = df['height'].values * units.m
        p = df['pressure'].values * units.hPa
        T = df['temperature'].values * units.degC
        Td = df['dewpoint'].values * units.degC
        u = df['u'].values * units('m/s')
        v = df['v'].values * units('m/s')
        sort_indices = np.argsort(-p.magnitude)
        z, p, T, Td, u, v = [arr[sort_indices] for arr in [z, p, T, Td, u, v]]

        Td = np.minimum(Td, T)

        if np.any(p.magnitude < 0):
            logger.warning("Negative pressure values detected")
        if np.any(p.magnitude < 100):
            logger.warning(f"Very low pressure values detected: min p = {np.min(p)}")

        T_kelvin = T.to('kelvin')
        Td_kelvin = Td.to('kelvin')

        for name, arr in [('pressure', p), ('temperature', T), ('dewpoint', Td), ('u', u), ('v', v)]:
            if np.any(np.isnan(arr.magnitude)):
                logger.warning(f"NaN values detected in {name} array")

        try:
            theta_e = mpcalc.equivalent_potential_temperature(p, T, Td)
        except Exception as e:
            logger.error(f"Error calculating theta_e: {e}")
            theta_e = np.full_like(T, np.nan) * units.kelvin

        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])

        p_truncated = p[p >= 100 * units.hPa]
        T_truncated = T[p >= 100 * units.hPa]
        Td_truncated = Td[p >= 100 * units.hPa]

        if data_type == "forecast":
            try:
                prof = mpcalc.parcel_profile(p_truncated, T[0], Td[0]).to('degC')
            except Exception as e:
                logger.error(f"Failed to calculate SB parcel profile for forecast: {e}")
                prof = None
            sbcape, sbcin = mpcalc.surface_based_cape_cin(p, T, Td)
            ml_depth = 300 * units.hPa
            ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=ml_depth)
            try:
                ml_prof = mpcalc.parcel_profile(p_truncated, ml_t, ml_td).to('degC')
            except Exception as e:
                logger.error(f"Failed to calculate ML parcel profile for forecast: {e}")
                ml_prof = None
            mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, prof if prof is not None else T, depth=ml_depth)
            mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, T, Td, depth=300 * units.hPa)
            mpl_height = np.interp(mu_p.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if mu_p is not None and not np.isnan(mu_p.magnitude) else np.nan * units.m
            logger.info(f"MPL pressure: {mu_p}, height: {mpl_height}")
            try:
                mu_prof = mpcalc.parcel_profile(p_truncated, mu_t, mu_td).to('degC')
            except Exception as e:
                logger.error(f"Failed to calculate MU parcel profile for forecast: {e}")
                mu_prof = None
            mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=100 * units.hPa)
            valid_idx = np.where((p > 500 * units.hPa) & (~np.isnan(theta_e.magnitude)))[0]
            if len(valid_idx) > 0:
                min_theta_e_idx = valid_idx[np.argmin(theta_e[valid_idx].magnitude)]
                try:
                    down_prof = mpcalc.parcel_profile(p_truncated, T[min_theta_e_idx], Td[min_theta_e_idx]).to('degC')
                except Exception as e:
                    logger.error(f"Failed to calculate downdraft parcel profile for forecast: {e}")
                    down_prof = None
            else:
                logger.warning("No valid theta_e values above 500 hPa for downdraft parcel")
                down_prof = None
        else:
            prof = mpcalc.parcel_profile(p_truncated, T[0], Td[0]).to('degC')
            sbcape, sbcin = mpcalc.surface_based_cape_cin(p, T, Td)
            ml_depth = 300 * units.hPa
            ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=ml_depth)
            ml_prof = mpcalc.parcel_profile(p_truncated, ml_t, ml_td).to('degC')
            mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, prof, depth=ml_depth)
            mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, T, Td, depth=300 * units.hPa)
            mpl_height = np.interp(mu_p.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if mu_p is not None and not np.isnan(mu_p.magnitude) else np.nan * units.m
            mu_prof = mpcalc.parcel_profile(p_truncated, mu_t, mu_td).to('degC')
            mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=100 * units.hPa)
            min_theta_e_idx = np.argmin(theta_e[p > 500 * units.hPa])
            down_prof = mpcalc.parcel_profile(p_truncated, T[min_theta_e_idx], Td[min_theta_e_idx]).to('degC')

        wet_bulb = mpcalc.wet_bulb_temperature(p, T_kelvin, Td_kelvin)
        kindex = mpcalc.k_index(p, T_kelvin, Td_kelvin)
        total_totals = calculate_total_totals(p, T, Td)
        temperature_advection = calculate_temperature_advection(p, u, v, station_lat)
        lfc_pressure, lfc_temperature = mpcalc.lfc(p, T, Td)
        el_pressure, el_temperature = mpcalc.el(p, T, Td)

    except ValueError as e:
        logger.error(f"Value error in data prep: {e}")
        await ctx.send(str(e))
        return
    except Exception as e:
        logger.error(f"Unexpected error in data prep: {e}", exc_info=True)
        await ctx.send("An unexpected error occurred while preparing the data. Please try again later.")
        return

    try:
        idx = np.where(T.magnitude < 0)[0]
        p_freeze = None
        if len(idx) > 0 and idx[0] > 0:
            idx = idx[0]
            T1, T2 = T[idx-1], T[idx]
            p1, p2 = p[idx-1], p[idx]
            if T2 != T1 and not np.any(np.isnan([T1.magnitude, T2.magnitude, p1.magnitude, p2.magnitude])):
                p_freeze = p1 + (0 * units.degC - T1) * (p2 - p1) / (T2 - T1)
                if p_freeze < 100 * units.hPa or p_freeze > 1000 * units.hPa:
                    logger.warning(f"Freezing level pressure {p_freeze} out of bounds")
                    p_freeze = None

        p_CCL, T_CCL, ccl_height, Tc = None, None, None, None
        if len(p) > 1:
            try:
                e_surface = mpcalc.saturation_vapor_pressure(Td[0])
                w_surface = mpcalc.mixing_ratio(e_surface, p[0])
                e_p = (w_surface * p) / (0.622 + w_surface)
                Td_p = mpcalc.dewpoint(e_p)
                diff = (T - Td_p).magnitude
                idx = np.where(np.diff(np.sign(diff)) != 0)[0]
                if len(idx) > 0:
                    i = idx[0]
                    p1, p2 = p[i], p[i+1]
                    diff1, diff2 = diff[i], diff[i+1]
                    p_CCL = p1 - (p1 - p2) * (diff1 / (diff1 - diff2))
                    T_CCL = np.interp(p_CCL.magnitude, p.magnitude[::-1], T.magnitude[::-1]) * T.units
                    ccl_height = np.interp(p_CCL.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m
                    if p_CCL is not None and T_CCL is not None:
                        T_CCL_K = T_CCL.to('K')
                        p_surface = p[0]
                        Tc_K = T_CCL_K * (p_surface / p_CCL) ** 0.286
                        Tc = Tc_K.to('degC')
                    else:
                        Tc = np.nan * units.degC
                else:
                    p_CCL, T_CCL, ccl_height = None, None, np.nan * units.m
                    Tc = np.nan * units.degC
            except Exception as e:
                logger.error(f"Error calculating CCL/Tc: {e}")
                p_CCL, T_CCL, ccl_height, Tc = None, None, None, np.nan * units.degC
        else:
            p_CCL, T_CCL, ccl_height, Tc = None, None, np.nan * units.m, np.nan * units.degC

        max_T = np.max(T) if len(T) > 0 else np.nan * units.degC

        RM, LM, MW = mpcalc.bunkers_storm_motion(p, u, v, z)
        storm_u, storm_v = RM
        total_helicity1, _, _ = mpcalc.storm_relative_helicity(z, u, v, depth=1 * units.km, storm_u=storm_u, storm_v=storm_v)
        total_helicity3, _, _ = mpcalc.storm_relative_helicity(z, u, v, depth=3 * units.km, storm_u=storm_u, storm_v=storm_v)
        total_helicity6, _, _ = mpcalc.storm_relative_helicity(z, u, v, depth=6 * units.km, storm_u=storm_u, storm_v=storm_v)
        bshear1 = mpcalc.bulk_shear(p, u, v, depth=1 * units.km)
        bshear3 = mpcalc.bulk_shear(p, u, v, depth=3 * units.km)
        bshear6 = mpcalc.bulk_shear(p, u, v, depth=6 * units.km)
        bshear1_mag = np.sqrt(bshear1[0]**2 + bshear1[1]**2) if bshear1 is not None else np.nan * units('m/s')
        bshear3_mag = np.sqrt(bshear3[0]**2 + bshear3[1]**2) if bshear3 is not None else np.nan * units('m/s')
        bshear6_mag = np.sqrt(bshear6[0]**2 + bshear6[1]**2) if bshear6 is not None else np.nan * units('m/s')

        lcl_height = np.interp(lcl_pressure.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if lcl_pressure is not None else np.nan * units.m
        lfc_height = np.interp(lfc_pressure.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if lfc_pressure is not None and not np.isnan(lfc_pressure.magnitude) else np.nan * units.m
        el_height = np.interp(el_pressure.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if el_pressure is not None and not np.isnan(el_pressure.magnitude) else np.nan * units.m
        fl_height = np.interp(p_freeze.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if p_freeze is not None and not np.isnan(p_freeze.magnitude) else np.nan * units.m
        mpl_height = np.interp(mu_p.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m if mu_p is not None and not np.isnan(mu_p.magnitude) else np.nan * units.m

        try:
            sig_tor = mpcalc.significant_tornado(sbcape, lcl_height, total_helicity1, bshear6_mag).to_base_units()
        except Exception as e:
            logger.error(f"Error calculating sig_tor: {e}")
            sig_tor = np.nan * units.dimensionless

        try:
            super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3_mag).to_base_units()
        except Exception as e:
            logger.error(f"Error calculating super_comp: {e}")
            super_comp = np.nan * units.dimensionless

        RH = mpcalc.relative_humidity_from_dewpoint(T, Td) * 100
        e = mpcalc.saturation_vapor_pressure(Td)
        ω = mpcalc.mixing_ratio(e, p).to('g/kg')
        z_km = z / 1000

        layers = [
            (0, 0.5, '0 - 0.5 km'), (0, 1, '0 - 1 km'), (1, 3, '1 - 3 km'),
            (3, 6, '3 - 6 km'), (6, 9, '6 - 9 km')
        ]
        layer_data = []
        for lower, upper, label in layers:
            if lower == 0:
                idx = (z_km <= upper * units.km)
            else:
                idx = (z_km > lower * units.km) & (z_km <= upper * units.km)
            if np.any(idx):
                mean_RH = np.mean(RH[idx].magnitude)
                mean_ω = np.mean(ω[idx].magnitude)
                layer_data.append((label, mean_RH, mean_ω))
            else:
                layer_data.append((label, np.nan, np.nan))

        PWAT = mpcalc.precipitable_water(p, Td).to('inch').magnitude
        WB_surface = wet_bulb[0].to('degC').magnitude
        idx_wb0 = np.where(wet_bulb.magnitude < 0)[0]
        if len(idx_wb0) > 0 and idx_wb0[0] > 0:
            i = idx_wb0[0]
            wb1, wb2 = wet_bulb[i-1], wet_bulb[i]
            p1, p2 = p[i-1], p[i]
            z1, z2 = z[i-1], z[i]
            p_wb0 = p1 + (0 - wb1.magnitude) * (p2 - p1) / (wb2.magnitude - wb1.magnitude)
            wb0_height = np.interp(p_wb0.magnitude, p.magnitude[::-1], z.magnitude[::-1]) * units.m
        else:
            wb0_height = np.nan * units.m

        surface_RH = mpcalc.relative_humidity_from_dewpoint(T[0], Td[0]) * 100
        surface_wet_bulb = wet_bulb[0].to('degC')

        fig = plt.figure(figsize=(30, 15))
        fig.set_facecolor('lightsteelblue')
        skew = SkewT(fig, rotation=45, rect=[0.03, 0.15, 0.55, 0.8])

        x1 = np.linspace(-100, 40, 8)
        x2 = np.linspace(-90, 50, 8)
        y = [1050, 100]
        for i in range(0, 8):
            skew.shade_area(y=y, x1=x1[i], x2=x2[i], color='#ebbe2a', alpha=0.25, zorder=1)

        skew.plot(p, T, 'r', linewidth=2, label='Temperature')
        skew.plot(p, Td, 'g', linewidth=2, label='Dewpoint')
        skew.plot(p, wet_bulb.to('degC'), 'b', linestyle='--', linewidth=2, label='Wet Bulb')
        if prof is not None:
            skew.plot(p, prof, 'k', linewidth=2.5, label='SB Parcel')
        if ml_prof is not None:
            skew.plot(p, ml_prof, 'm--', linewidth=2, label='ML Parcel')
        if mu_prof is not None:
            skew.plot(p, mu_prof, 'y--', linewidth=2, label='MU Parcel')
        if down_prof is not None:
            skew.plot(p, down_prof, color='#a87308', linestyle='-.', linewidth=2.5, label='Downdraft')
        skew.plot_barbs(p[::2], u[::2], v[::2], color='#000000')
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)
        if prof is not None:
            skew.shade_cin(p, T, prof, Td)
            skew.shade_cape(p, T, prof)
        skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
        skew.ax.set_xlabel('Temperature (°C)', weight='bold')
        skew.ax.set_ylabel(f'Pressure ({p.units:~P})', weight='bold')
        skew.plot_dry_adiabats(linewidth=1.5, color='brown', label='Dry Adiabat')
        skew.plot_moist_adiabats(linewidth=1.5, color='purple', label='Moist Adiabat')
        skew.plot_mixing_lines(linewidth=1.5, color='#02a312', label='Mixing Ratio (g/kg)')
        label_mixing_ratios(skew)
        label_ccl(skew, p, T, Td)

        def plot_point(skew, pressure, temperature, marker, color, label, log_message):
            if (pressure is not None and temperature is not None and
                not np.isnan(pressure.magnitude) and not np.isnan(temperature.magnitude) and
                100 <= pressure.magnitude <= 1000 and -40 <= temperature.magnitude <= 60):
                try:
                    logger.debug(f"Plotting {label}: pressure={pressure}, temperature={temperature}")
                    skew.ax.scatter(temperature, pressure, marker='o', s=50, color=color)
                    skew.ax.text(temperature.magnitude + 2, pressure.magnitude, label,
                                 fontsize=10, color='k' if label in ['LFC', 'EL'] else color)
                    logger.info(log_message)
                except Exception as e:
                    logger.error(f"Error plotting {label}: {e}")
            else:
                logger.info(f"Skipping {label}: Invalid pressure={pressure}, temperature={temperature}")

        plot_point(skew, lcl_pressure, lcl_temperature, 'o', 'black', 'LCL', f"LCL plotted at {lcl_pressure}, {lcl_temperature}")
        plot_point(skew, lfc_pressure, lfc_temperature, 'o', 'black', 'LFC', f"LFC plotted at {lfc_pressure}, {lcl_temperature}")
        plot_point(skew, el_pressure, el_temperature, 'o', 'black', 'EL', f"EL plotted at {el_pressure}, {el_temperature}")
        plot_point(skew, p_freeze, 0 * units.degC if p_freeze is not None else None, 'o', 'blue', 'FL', f"Freezing Level plotted at {p_freeze}, 0°C")
        plot_point(skew, p_CCL, T_CCL, 'o', 'purple', 'CCL', f"CCL plotted at {p_CCL}, {T_CCL}")

        adv_ax = plt.axes([0.58, 0.05, 0.08, 0.9])
        temp_advection = temperature_advection.magnitude
        max_adv = 5
        norm = Normalize(vmin=-max_adv, vmax=max_adv)
        colors = plt.cm.coolwarm(norm(temp_advection))
        adv_ax.barh(p.magnitude, temp_advection, color=colors, height=8, align='center')
        adv_ax.set_ylim(1000, 100)
        adv_ax.set_xlim(-max_adv, max_adv)
        adv_ax.set_xlabel('Temp. Adv. (°C/hr)', weight='bold')
        adv_ax.set_ylabel('Pressure (hPa)', weight='bold')
        adv_ax.axvline(0, color='black', linewidth=0.5)
        adv_ax.set_facecolor('#fafad2')

        ax_hodo = plt.axes((0.61, 0.32, 0.4, 0.4))
        ax_hodo.set_facecolor('#fafad2')
        h = Hodograph(ax_hodo, component_range=60.)
        h.add_grid(increment=20, linestyle='--', linewidth=1)
        cmap = LinearSegmentedColormap.from_list('my_cmap', ['purple', 'blue', 'green', 'yellow', 'orange', 'red'])
        norm = Normalize(vmin=z.min().magnitude, vmax=z.max().magnitude)
        colored_line = h.plot_colormapped(u, v, c=z.magnitude, linewidth=6, cmap=cmap, label='0-12km WIND')
        cbar = plt.colorbar(colored_line, ax=h.ax, orientation='vertical', pad=0.01)
        cbar.set_label('Height (m)')
        ax_hodo.set_xlabel('U component (m/s)', weight='bold')
        ax_hodo.set_ylabel('V component (m/s)', weight='bold')
        ax_hodo.set_title(f'Hodograph {title}')

        idx_6km = np.argmin(np.abs(z.magnitude - 6000))
        h.ax.scatter(u[0], v[0], color='blue', marker='o', s=100, edgecolor='black', label='Surface Wind', zorder=3)
        h.ax.scatter(u[idx_6km], v[idx_6km], color='cyan', marker='x', s=100, label='6 km Wind', zorder=3)
        h.ax.text(u[idx_6km].magnitude + 2, v[idx_6km].magnitude + 2, '6 km', color='cyan', fontsize=10)
        h.ax.scatter(u[-1], v[-1], color='red', marker='o', s=100, edgecolor='black', label='12km Wind', zorder=3)

        for motion, label, offset in [(RM, 'RM', (0.5, -0.5)), (LM, 'LM', (0.5, -0.5)), (MW, 'MW', (0.5, -0.5))]:
            h.ax.text(motion[0].magnitude + offset[0], motion[1].magnitude + offset[1], label,
                      weight='bold', ha='left', fontsize=13, alpha=0.6)
        h.ax.arrow(0, 0, RM[0].magnitude - 0.3, RM[1].magnitude - 0.3, linewidth=2, color='red',
                   alpha=0.5, label='Bunkers RM Vector', length_includes_head=True, head_width=2)

        h.ax.fill([u[0].magnitude, u[idx_6km].magnitude, RM[0].magnitude],
                  [v[0].magnitude, v[idx_6km].magnitude, RM[1].magnitude],
                  color='blue', alpha=0.3, label='Inflow Shading')

        w_ax = 0.08
        h_ax = 2 * w_ax
        theta_e_ax = fig.add_axes([0.46, -0.06, w_ax, h_ax])
        theta_e_ax.plot(theta_e, p, color='purple', linewidth=1.5)
        theta_e_ax.set_xlabel('Theta-E (K)', fontsize=8)
        theta_e_ax.set_ylabel('Pressure (hPa)', fontsize=8)
        theta_e_ax.set_title('Theta-e(K)/Pressure(hPa)', fontsize=10, weight='bold')
        theta_e_ax.invert_yaxis()
        theta_e_ax.set_xlim(280, 360)
        theta_e_ax.set_ylim(1000, 100)
        theta_e_ax.tick_params(axis='both', labelsize=6)
        theta_e_ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        theta_e_ax.set_facecolor('#fafad2')

        storm_slinky_ax = fig.add_axes([0.34, -0.06, w_ax, h_ax])
        u_sr = u - storm_u
        v_sr = v - storm_v
        z_km = z.to('km').magnitude
        mask_3km = z_km <= 3
        u_sr_3km = u_sr[mask_3km]
        v_sr_3km = v_sr[mask_3km]
        z_3km = z[mask_3km]
        storm_slinky_ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        storm_slinky_ax.axvline(0, color='black', linewidth=2, zorder=0)
        storm_slinky_ax.axhline(0, color='black', linewidth=2, zorder=0)
        storm_slinky_ax.set_facecolor('#fafad2')
        norm = Normalize(vmin=0, vmax=3)
        scatter = storm_slinky_ax.scatter(u_sr_3km.magnitude, v_sr_3km.magnitude, c=z_3km.magnitude, cmap='brg', s=30, label='Storm Slinky (0–3 km)')
        storm_slinky_ax.set_xlabel('SR U (m/s)', fontsize=8)
        storm_slinky_ax.set_ylabel('SR V (m/s)', fontsize=8)
        storm_slinky_ax.set_title('Storm Slinky', fontsize=10, weight='bold')
        storm_slinky_ax.set_xlim(-50, 50)
        storm_slinky_ax.set_ylim(-50, 50)
        storm_slinky_ax.tick_params(axis='both', labelsize=6)
        cbar = plt.colorbar(scatter, ax=storm_slinky_ax, orientation='vertical', pad=0.03, fraction=0.046)
        cbar.set_label('Height (km)', fontsize=8)

        bbox_props = dict(facecolor='#fafad2', alpha=0.7, edgecolor='none', pad=3)
        indices = [
            ('SBCAPE', sbcape, 'red', 0.15, None), ('SBCIN', sbcin, 'purple', 0.13, None),
            ('MLCAPE', mlcape, 'red', 0.11, None), ('MLCIN', mlcin, 'purple', 0.09, None),
            ('MUCAPE', mucape, 'red', 0.07, None), ('MUCIN', mucin, 'purple', 0.05, None),
            ('TT-INDEX', total_totals, 'red', 0.03, None), ('K-INDEX', kindex, 'red', 0.01, None),
            ('SIG TORNADO', sig_tor, 'red', -0.01, None),
            ('0-1km SRH', total_helicity1, 'navy', 0.15, 0.88), ('0-1km SHEAR', bshear1_mag, 'blue', 0.13, 0.88),
            ('0-3km SRH', total_helicity3, 'navy', 0.11, 0.88), ('0-3km SHEAR', bshear3_mag, 'blue', 0.09, 0.88),
            ('0-6km SRH', total_helicity6, 'navy', 0.07, 0.88), ('0-6km SHEAR', bshear6_mag, 'blue', 0.05, 0.88),
            ('SUPERCELL COMP', super_comp, 'red', -0.01, 0.88),
            ('CCL', ccl_height, 'purple', -0.03, 0.88),
            ('LCL', lcl_height, 'black', -0.03, None), ('LFC', lfc_height, 'black', -0.05, None),
            ('EL', el_height, 'black', -0.07, None), ('MPL', mpl_height, 'black', -0.09, None),
            ('FL', fl_height, 'blue', -0.11, None),
            ('Surface RH', surface_RH, 'green', -0.13, 0.88),
            ('Surface Wet Bulb', surface_wet_bulb, 'blue', -0.15, 0.88),
            ('PWAT', PWAT * units.inch, 'green', -0.17, 0.88),
            ('Convective Temp', Tc, 'purple', -0.19, 0.88),
            ('Max Temp', max_T, 'red', -0.21, 0.88),
        ]

        left_indices = [idx for idx in indices if idx[4] is None]
        right_indices = [idx for idx in indices if idx[4] == 0.88]
        max_rows = max(len(left_indices), len(right_indices))
        y_start = 0.15
        y_step = 0.02
        y_positions = [y_start - i * y_step for i in range(max_rows)]

        def format_value(label, value):
            if label == 'Surface RH':
                if isinstance(value, pint.Quantity):
                    magnitude = value.magnitude
                else:
                    magnitude = value
                if np.isscalar(magnitude):
                    return f'{magnitude:.0f}%' if not np.isnan(magnitude) else 'N/A'
                return 'N/A'
            elif isinstance(value, pint.Quantity):
                magnitude = value.magnitude
                if not np.isscalar(magnitude):
                    magnitude = magnitude[0] if len(magnitude) == 1 else np.nan
                if label in ['Surface Wet Bulb', 'Convective Temp', 'Max Temp']:
                    return f'{magnitude:.1f} °C' if not np.isnan(magnitude) else 'N/A'
                elif label == 'PWAT':
                    return f'{magnitude:.2f} {value.units}' if not np.isnan(magnitude) else 'N/A'
                elif value.dimensionality == {}:
                    return f'{magnitude:.0f}' if not np.isnan(magnitude) else 'N/A'
                else:
                    return f'{value:.0f~P}' if not np.isnan(magnitude) else 'N/A'
            else:
                return 'N/A' if np.isnan(value) else f'{value:.0f}'

        for i, (label, value, color, _, _) in enumerate(left_indices):
            y = y_positions[i]
            x_left = 0.71
            x_right = 0.80
            value_str = format_value(label, value)
            plt.figtext(x_left, y, f'{label}: ', weight='bold', fontsize=12, color='black', ha='left', bbox=bbox_props)
            plt.figtext(x_right, y, value_str, weight='bold', fontsize=12, color=color, ha='right', bbox=bbox_props)

        for i, (label, value, color, _, _) in enumerate(right_indices):
            y = y_positions[i]
            x_left = 0.88
            x_right = 0.96
            value_str = format_value(label, value)
            plt.figtext(x_left, y, f'{label}: ', weight='bold', fontsize=12, color='black', ha='left', bbox=bbox_props)
            plt.figtext(x_right, y, value_str, weight='bold', fontsize=12, color=color, ha='right', bbox=bbox_props)

        plt.figtext(0.71, 0.23, f"Plot Created With MetPy (C) Evan J Lane 2024\nData Source: {'University of Wyoming' if data_type == 'observed' else 'NOAA (GFS)'}\nImage Created: " +
                    utc_time.strftime('%H:%M UTC'), fontsize=16, fontweight='bold', bbox=bbox_props)
        add_metpy_logo(fig, 85, 85, size='small')
        logo_paths = {
            'boxlogo': '/home/evanl/Documents/bot/boxlogo2.png',
            'bulldogs': '/home/evanl/Documents/bot/Georgia_Bulldogs_logo.png'
        }
        for name, path, pos, zoom in [('boxlogo', logo_paths['boxlogo'], (1.10, 1.20), 0.2),
                                      ('bulldogs', logo_paths['bulldogs'], (0.45, 1.20), 0.97)]:
            try:
                logo_img = plt.imread(path)
                imagebox = OffsetImage(logo_img, zoom=zoom)
                ab = AnnotationBbox(imagebox, pos, xycoords='axes fraction', frameon=False, box_alignment=(1, 0))
                ax_hodo.add_artist(ab)
            except FileNotFoundError:
                logger.error(f"Logo file not found: {path}")
                await ctx.send(f"Warning: Logo '{name}' not found at {path}")

        skew.ax.legend(loc='upper left', fontsize=14, frameon=True, title='Skew-T Legend', title_fontsize=10)
        h.ax.legend(loc='upper left', fontsize=14, frameon=True, title='Hodograph Legend', title_fontsize=10)
        skew.ax.set_facecolor('#d9deda')
        plt.suptitle('UPPER AIR SOUNDING', fontsize=24, fontweight='bold', y=1.00)
        plt.figtext(0.7, 0.98, title, fontsize=20, fontweight='bold', ha='center')
        skew.ax.set_title(f'Skew-T Log-P Diagram - Latitude: {station_lat:.2f}, Longitude: {station_lon:.2f}')

        temp_image_path = f"skewt_{station_code}_{data_type}.png"
        plt.savefig(temp_image_path, format='png', bbox_inches='tight')
        plt.close(fig)
        await ctx.send(file=discord.File(temp_image_path))
        os.remove(temp_image_path)
        logger.info(f"Skew-T diagram generated and sent for {station_code}")

    except Exception as e:
        logger.error(f"Unexpected error in plotting: {e}", exc_info=True)
        await ctx.send("An unexpected error occurred while generating the plot. Please try again later.")

# Uncomment and add your bot token to run
# bot.run('YOUR_DISCORD_BOT_TOKEN')
