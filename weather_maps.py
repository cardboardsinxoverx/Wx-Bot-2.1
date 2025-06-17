import discord
from discord.ext import commands
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.units import units
import metpy.calc as mpcalc
from scipy import ndimage
import asyncio
import io
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
import logging
from PIL import Image
import urllib.error
import matplotlib.colors as mcolors
from pydap.exceptions import ServerError, ClientError
import requests

# Set up logging for debugging and error tracking
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Define paths to logo images (adjust these paths to match your system)
LOGO_PATHS = [
    "/media/evanl/EXTRA/bot/boxlogo2.png",
    "/media/evanl/EXTRA/bot/uga.png"
]

# Helper function to validate PNG files
def is_valid_png(file_path):
    """Check if a file is a valid PNG."""
    try:
        with Image.open(file_path) as img:
            return img.format == 'PNG'
    except Exception as e:
        logging.warning(f"Invalid PNG file {file_path}: {e}")
        return False

def get_previous_run_time(now):
    """Calculate the most recent GFS run time (00Z, 06Z, 12Z, or 18Z) at least 6 hours before now."""
    logging.debug(f"Current UTC time (now): {now}")
    T_offset = now - timedelta(hours=6)
    logging.debug(f"T_offset (now - 6 hours): {T_offset}")
    run_hours = [18, 12, 6, 0]
    for rh in run_hours:
        if T_offset.hour >= rh:
            run_time = rh
            break
    else:
        run_time = 18
        T_offset -= timedelta(days=1)

    run_date = T_offset.replace(hour=run_time, minute=0, second=0, microsecond=0)
    if run_time > T_offset.hour:
        run_date -= timedelta(days=1)
    logging.debug(f"Calculated GFS run date: {run_date}")
    return run_date

def get_gfs_data_for_level(level):
    logging.debug(f"Fetching GFS data for level {level} Pa from NOMADS")
    now = datetime.utcnow()
    run_date = get_previous_run_time(now)
    nomads_base_url = "https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs"

    attempts = 0
    max_attempts = 4
    min_run_date = now - timedelta(days=14)

    lon_slice = slice(360 - 125, 360 - 65)  # -125 to -65 -> 235 to 295
    lat_slice = slice(50, 25)

    while attempts < max_attempts and run_date >= min_run_date:
        date_str = run_date.strftime("%Y%m%d")
        hour_str = run_date.strftime("%H")
        nomads_url = f"{nomads_base_url}{date_str}/gfs_0p25_{hour_str}z"
        logging.info(f"Trying NOMADS for run at {run_date}: {nomads_url}")

        try:
            response = requests.get(nomads_url, timeout=10)
            logging.debug(f"HTTP status code: {response.status_code}")
        except requests.RequestException as e:
            logging.warning(f"Failed to fetch raw response from {nomads_url}: {e}")

        try:
            ds_full = xr.open_dataset(nomads_url, engine='pydap')
            ds = ds_full.sel(lon=lon_slice, lat=lat_slice)

            if level is None:
                required_vars = ['tmp2m', 'msletmsl', 'ugrd10m', 'vgrd10m']
                ds = ds[required_vars].load()
            else:
                pressure_dim = 'lev'
                level_hpa = level / 100
                required_vars = ['hgtprs', 'ugrdprs', 'vgrdprs', 'tmpprs', 'rhprs']
                ds = ds[required_vars].sel({pressure_dim: level_hpa}, method='nearest').load()

            ds = ds.metpy.parse_cf()
            logging.debug(f"Dataset variables after parsing: {ds.data_vars}")
            return ds, run_date

        except (ServerError, ClientError) as e:
            logging.warning(f"NOMADS server or client error for {nomads_url}: {e}, trying previous run.")
            run_date -= timedelta(hours=6)
            attempts += 1
        except urllib.error.HTTPError as e:
            if e.code in [403, 404]:
                logging.warning(f"HTTP error {e.code} for {nomads_url}, trying previous run.")
            else:
                logging.error(f"HTTP error {e.code} fetching GFS data: {e}", exc_info=True)
                raise
            run_date -= timedelta(hours=6)
            attempts += 1
        except Exception as e:
            logging.error(f"Unexpected error fetching GFS data: {e}", exc_info=True)
            raise

    raise Exception("No available GFS data found in the last 14 days")

async def fetch_gfs_data(level):
    """Asynchronously fetch GFS data with a 300-second timeout."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(get_gfs_data_for_level, level),
            timeout=300
        )
    except asyncio.TimeoutError:
        raise Exception("Data fetching timed out. It took longer than 5 minutes. Please try again later.")
    except Exception as e:
        raise Exception(f"Error fetching data: {e}")

def plot_background(ax):
    """Add geographical features and gridlines to the plot."""
    logging.debug("Setting up plot background")
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor='#3f664a')
    ax.add_feature(cfeature.OCEAN, facecolor='#a5a2fa')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

def select_data(ds, level, variable):
    """Select and return data for a specific variable at a given level or surface."""
    try:
        logging.debug(f"Selecting data for variable {variable} at level {level}")
        if level is None:
            var_map = {
                'Temperature_surface': 'tmp2m',
                'MSLET': 'msletmsl',
                'u10': 'ugrd10m',
                'v10': 'vgrd10m',
            }
        else:
            var_map = {
                'Geopotential_height_isobaric': 'hgtprs',
                'u-component_of_wind_isobaric': 'ugrdprs',
                'v-component_of_wind_isobaric': 'vgrdprs',
                'Temperature_isobaric': 'tmpprs',
                'Relative_humidity_isobaric': 'rhprs',
            }

        actual_var = var_map.get(variable)
        if actual_var is None:
            raise ValueError(f"Variable {variable} not mapped for level {level}")

        da = ds[actual_var]

        # Select the first time step if 'time' dimension exists
        if 'time' in da.dims:
            da = da.isel(time=0)
            logging.debug(f"Selected first time step for {variable}, shape: {da.shape}")

        # Explicitly assign units for wind components if needed
        if variable in ['u-component_of_wind_isobaric', 'v-component_of_wind_isobaric', 'u10', 'v10']:
            has_metpy_units = hasattr(da, 'metpy') and hasattr(da.metpy, 'units')
            is_speed_unit = False
            if has_metpy_units:
                try:
                    is_speed_unit = da.metpy.units.dimensionality == units.speed.dimensionality
                except Exception as e:
                    logging.warning(f"Error checking unit dimensionality for {variable}: {e}")
            if not is_speed_unit:
                logging.warning(f"MetPy did not find valid speed units for {variable}, quantifying as m/s.")
                if has_metpy_units:
                    da = da.metpy.dequantify() * units('m/s')
                else:
                    da = da * units('m/s')
            else:
                return da
        else:
            da = da.metpy.quantify()

        # Explicitly assign units for temperature if needed
        if variable == 'Temperature_surface' and (not hasattr(da, 'metpy') or not hasattr(da.metpy, 'units') or da.metpy.units.dimensionality == units.dimensionless.dimensionality):
            logging.warning(f"MetPy did not find valid temperature units for {variable}, quantifying as Kelvin.")
            if hasattr(da, 'metpy') and hasattr(da.metpy, 'units'):
                return da.metpy.dequantify() * units.kelvin
            else:
                return da * units.kelvin
        return da

    except Exception as e:
        logging.error(f"Error in select_data for {variable}: {e}", exc_info=True)
        raise

def get_height_contour_levels(height_data_array):
    """Determine appropriate contour levels for geopotential heights."""
    h_min = height_data_array.min().item()
    h_max = height_data_array.max().item()
    interval = 60
    start_level = int(np.floor(h_min / interval)) * interval
    end_level = int(np.ceil(h_max / interval)) * interval
    levels = np.arange(start_level, end_level + interval, interval)
    logging.debug(f"Calculated height contour levels: {levels}")
    return levels

def generate_map(ds, run_date: datetime, level: int, parameter: str, cmap=None, title: str = None, cbar_label: str = None, levels=None, plot_contour_lines=False):
    """Generate a weather map for a specified parameter at a given pressure level."""
    try:
        logging.debug(f"Starting generate_map for {parameter} at {level} Pa")
        lon_plot = np.where(ds['lon'].values > 180, ds['lon'].values - 360, ds['lon'].values)
        lat_plot = ds['lat'].values
        lon_2d, lat_2d = np.meshgrid(lon_plot, lat_plot)

        heights = select_data(ds, level, 'Geopotential_height_isobaric')
        u_wind = select_data(ds, level, 'u-component_of_wind_isobaric')
        v_wind = select_data(ds, level, 'v-component_of_wind_isobaric')
        heights_smooth = ndimage.gaussian_filter(heights.values, sigma=3, order=0)

        dx_full, dy_full = mpcalc.lat_lon_grid_deltas(lon_plot, lat_plot)
        lat_coords_with_units = xr.DataArray(lat_plot, dims=('lat',), coords={'lat': lat_plot}).metpy.quantify()

        data = None
        if parameter == 'vorticity':
            vorticity = mpcalc.absolute_vorticity(u_wind, v_wind, dx=dx_full, dy=dy_full, latitude=lat_coords_with_units)
            data = vorticity.values * 1e5
            if levels is None:
                levels = np.linspace(-20, 20, 41)
            if cmap is None:
                cmap = plt.get_cmap('RdBu_r')
            if title is None:
                title = f'{level/100:.0f}-hPa Absolute Vorticity ($10^{{-5}}$ s$^{{-1}}$) and Heights'
            if cbar_label is None:
                cbar_label = r'Absolute Vorticity ($10^{-5}$ s$^{-1}$)'
            extend = 'neither'

        elif parameter == 'pva':
            absolute_vorticity = mpcalc.absolute_vorticity(u_wind, v_wind, dx=dx_full, dy=dy_full, latitude=lat_coords_with_units)
            if not isinstance(absolute_vorticity, xr.DataArray):
                absolute_vorticity = xr.DataArray(
                    absolute_vorticity.magnitude,
                    dims=('lat', 'lon'),
                    coords={'lat': lat_plot, 'lon': lon_plot}
                ).metpy.quantify(absolute_vorticity.units)
            pva = mpcalc.advection(absolute_vorticity, u_wind, v_wind, dx=dx_full, dy=dy_full)
            data = pva.metpy.convert_units('1/s**2').values * 1e10
            if levels is None:
                levels = np.linspace(-20, 20, 41)
            if cmap is None:
                cmap = plt.get_cmap('RdBu_r')
            if title is None:
                title = f'{level/100:.0f}hPa Abs. Vorticity Advection ($10^{{-10}}$ s$^{{-2}}$) and Heights'
            if cbar_label is None:
                cbar_label = r'Absolute Vorticity Advection ($10^{-10}$ s$^{-2}$)'
            extend = 'neither'

        elif parameter == 'wind_speed':
            wind_speed = mpcalc.wind_speed(u_wind, v_wind)
            data = wind_speed.metpy.convert_units('knots').values
            if levels is None:
                levels = np.linspace(0, np.max(data), 41)
            if cmap is None:
                cmap = plt.get_cmap('viridis')
            if title is None:
                title = f'{level/100:.0f}hPa Wind Speed (knots) and Heights'
            if cbar_label is None:
                cbar_label = 'Wind Speed (knots)'
            extend = 'both'

        elif parameter == 'relative_humidity':
            rh = select_data(ds, level, 'Relative_humidity_isobaric')
            data = rh.values
            if levels is None:
                levels = np.linspace(0, 100, 41)
            if cmap is None:
                cmap = plt.get_cmap('Greens')
            if title is None:
                title = f'{level/100:.0f}hPa Relative Humidity (%) and Heights'
            if cbar_label is None:
                cbar_label = 'Relative Humidity (%)'
            extend = 'both'

        else:
            raise ValueError(f"Unsupported parameter: {parameter}")

        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        fig.patch.set_facecolor('lightsteelblue')
        ax.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.STATES, linewidth=1.0, edgecolor='darkgray')
        ax.set_extent([lon_plot.min(), lon_plot.max(), lat_plot.min(), lat_plot.max()], crs=ccrs.PlateCarree())

        cf = ax.contourf(lon_2d, lat_2d, data, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), extend=extend)

        if plot_contour_lines:
            c_data = ax.contour(lon_2d, lat_2d, data, levels=levels, colors='black', linestyles='-.', linewidths=1.0, transform=ccrs.PlateCarree())
            ax.clabel(c_data, inline=True, fontsize=8, fmt='%.0f', colors='black')

        height_levels = get_height_contour_levels(heights)
        c = ax.contour(lon_2d, lat_2d, heights_smooth, levels=height_levels, colors='black', linewidths=2.0, transform=ccrs.PlateCarree())
        ax.clabel(c, inline=True, fontsize=10, fmt='%i')

        ax.barbs(
            lon_2d[::5, ::5], lat_2d[::5, ::5],
            u_wind.values[::5, ::5],
            v_wind.values[::5, ::5],
            transform=ccrs.PlateCarree(), length=6
        )

        ax.set_title(title, fontsize=16)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label(cbar_label, size='large')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=24, y=1.02)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        logging.debug(f"{parameter} map generated successfully")
        return buf

    except Exception as e:
        logging.error(f"An error occurred in generate_map: {e}", exc_info=True)
        return None

def generate_temp_advection_map(ds, run_date):
    """Generate an 850 hPa temperature advection map."""
    try:
        logging.debug("Starting generate_temp_advection_map")
        lon_plot = np.where(ds['lon'].values > 180, ds['lon'].values - 360, ds['lon'].values)
        lat_plot = ds['lat'].values
        lon_2d, lat_2d = np.meshgrid(lon_plot, lat_plot)

        heights = select_data(ds, 85000, 'Geopotential_height_isobaric')
        u_wind = select_data(ds, 85000, 'u-component_of_wind_isobaric')
        v_wind = select_data(ds, 85000, 'v-component_of_wind_isobaric')
        temp_850 = select_data(ds, 85000, 'Temperature_isobaric')

        if not (hasattr(temp_850, 'metpy') and hasattr(temp_850.metpy, 'units') and
                temp_850.metpy.units.dimensionality == units.temperature.dimensionality):
            logging.warning("temp_850 lacks temperature units, quantifying as Kelvin.")
            if hasattr(temp_850, 'metpy') and hasattr(temp_850.metpy, 'units'):
                temp_850 = temp_850.metpy.dequantify() * units.kelvin
            else:
                temp_850 = temp_850 * units.kelvin

        dx_full, dy_full = mpcalc.lat_lon_grid_deltas(lon_plot, lat_plot)
        dT_dy, dT_dx = mpcalc.gradient(temp_850, deltas=(dy_full, dx_full), axes=(0, 1))
        advection = - (u_wind * dT_dx + v_wind * dT_dy)
        temp_advection = advection.metpy.convert_units('degC / hour')

        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        fig.patch.set_facecolor('lightsteelblue')
        ax.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.STATES, linewidth=1.0, edgecolor='darkgray')
        ax.set_extent([lon_plot.min(), lon_plot.max(), lat_plot.min(), lat_plot.max()], crs=ccrs.PlateCarree())

        levels = np.linspace(-10, 10, 21)
        cf = ax.contourf(lon_2d, lat_2d, temp_advection.values, cmap='coolwarm', levels=levels, extend='both')
        c_temp_adv = ax.contour(lon_2d, lat_2d, temp_advection.values, levels=levels,
                                colors='black', linestyles='-.', linewidths=1.0, transform=ccrs.PlateCarree())
        ax.clabel(c_temp_adv, inline=True, fontsize=8, fmt='%.1f', colors='black')

        height_levels = get_height_contour_levels(heights)
        c = ax.contour(lon_2d, lat_2d, heights.values, levels=height_levels, colors='black', linewidths=2.0)
        ax.clabel(c, inline=True, fontsize=10, fmt='%i')

        ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind.values[::5, ::5], v_wind.values[::5, ::5], length=6)

        ax.set_title('850-hPa Temperature Advection and Heights', fontsize=16)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label('Temperature Advection (°C/hour)', size='large')

        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=24, y=1.02)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        logging.debug("Temperature advection map generated successfully")
        return buf

    except Exception as e:
        logging.error(f"An error occurred in generate_temp_advection_map: {e}", exc_info=True)
        return None

def generate_moisture_advection_map(ds, run_date):
    """Generate an 850 hPa moisture advection map."""
    try:
        logging.debug("Starting generate_moisture_advection_map")
        lon_plot = np.where(ds['lon'].values > 180, ds['lon'].values - 360, ds['lon'].values)
        lat_plot = ds['lat'].values
        lon_2d, lat_2d = np.meshgrid(lon_plot, lat_plot)

        heights = select_data(ds, 85000, 'Geopotential_height_isobaric')
        u_wind = select_data(ds, 85000, 'u-component_of_wind_isobaric')
        v_wind = select_data(ds, 85000, 'v-component_of_wind_isobaric')
        rh_850 = select_data(ds, 85000, 'Relative_humidity_isobaric')

        dx_full, dy_full = mpcalc.lat_lon_grid_deltas(lon_plot, lat_plot)
        dRH_dy, dRH_dx = mpcalc.gradient(rh_850, deltas=(dy_full, dx_full), axes=(0, 1))
        advection = - (u_wind * dRH_dx + v_wind * dRH_dy)
        moisture_advection_scaled = (advection * 1e4)
        moisture_advection = xr.DataArray(
            moisture_advection_scaled.values,
            dims=('lat', 'lon'),
            coords={'lat': lat_plot, 'lon': lon_plot}
        )

        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        fig.patch.set_facecolor('lightsteelblue')
        ax.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.STATES, linewidth=1.0, edgecolor='darkgray')
        ax.set_extent([lon_plot.min(), lon_plot.max(), lat_plot.min(), lat_plot.max()], crs=ccrs.PlateCarree())

        levels = np.linspace(-10, 10, 41)
        cf = ax.contourf(lon_2d, lat_2d, moisture_advection, cmap='PRGn', levels=levels, extend='both')

        height_levels = get_height_contour_levels(heights)
        c = ax.contour(lon_2d, lat_2d, heights.values, levels=height_levels, colors='black', linewidths=2.0)
        ax.clabel(c, inline=True, fontsize=10, fmt='%i')

        ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind.values[::5, ::5], v_wind.values[::5, ::5], length=6)

        ax.set_title('850-hPa Moisture Advection and Heights', fontsize=16)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label(r'Moisture Advection ($10^4$ s⁻¹)', size='large')

        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=24, y=1.02)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        logging.debug("Moisture advection map generated successfully")
        return buf

    except Exception as e:
        logging.error(f"An error occurred in generate_moisture_advection_map: {e}", exc_info=True)
        return None

def generate_dewpoint_map(ds, run_date):
    """Generate an 850 hPa dewpoint map with temperature contours."""
    try:
        logging.debug("Starting generate_dewpoint_map")
        lon_plot = np.where(ds['lon'].values > 180, ds['lon'].values - 360, ds['lon'].values)
        lat_plot = ds['lat'].values
        lon_2d, lat_2d = np.meshgrid(lon_plot, lat_plot)

        temp_850 = select_data(ds, 85000, 'Temperature_isobaric')
        rh_850 = select_data(ds, 85000, 'Relative_humidity_isobaric')
        heights = select_data(ds, 85000, 'Geopotential_height_isobaric')
        u_wind = select_data(ds, 85000, 'u-component_of_wind_isobaric')
        v_wind = select_data(ds, 85000, 'v-component_of_wind_isobaric')

        dewpoint_850 = mpcalc.dewpoint_from_relative_humidity(temp_850, rh_850)
        dewpoint_850_c = dewpoint_850.metpy.convert_units('degC').values
        temp_850_c = temp_850.metpy.convert_units('degC').values
        heights_values = heights.values

        dewpoint_da = xr.DataArray(
            dewpoint_850_c,
            dims=('lat', 'lon'),
            coords={'lat': lat_plot, 'lon': lon_plot}
        )

        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        fig.patch.set_facecolor('lightsteelblue')
        ax.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.STATES, linewidth=1.0, edgecolor='darkgray')
        ax.set_extent([lon_plot.min(), lon_plot.max(), lat_plot.min(), lat_plot.max()], crs=ccrs.PlateCarree())

        dewpoint_levels = [0, 6, 8, 10, 12, 14, 16, 20]
        cf = ax.contourf(lon_2d, lat_2d, dewpoint_da, levels=dewpoint_levels, cmap='BuGn', extend='both')

        temp_levels = np.arange(-20, 22, 2)
        c_temp = ax.contour(lon_2d, lat_2d, temp_850_c, levels=temp_levels, colors='red', linestyles='dashed')
        ax.clabel(c_temp, inline=True, fontsize=10, fmt='%i')

        height_levels = get_height_contour_levels(heights)
        c_heights = ax.contour(lon_2d, lat_2d, heights_values, levels=height_levels, colors='black', linewidths=2.0)
        ax.clabel(c_heights, inline=True, fontsize=10, fmt='%i')

        ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind.values[::5, ::5], v_wind.values[::5, ::5], length=6)

        ax.set_title('850-hPa Dewpoint, Temperature, and Heights', fontsize=16)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label('Dewpoint (°C)', size='large')

        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=24, y=1.02)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        logging.debug("Dewpoint map generated successfully")
        return buf

    except Exception as e:
        logging.error(f"An error occurred in generate_dewpoint_map: {e}", exc_info=True)
        return None

def generate_surface_temp_map():
    """Generate a surface temperature map with MSLP isobars and wind barbs."""
    try:
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        fig.patch.set_facecolor('lightsteelblue')

        ds, run_date = get_gfs_data_for_level(None)
        lon_plot = np.where(ds['lon'].values > 180, ds['lon'].values - 360, ds['lon'].values)
        lat_plot = ds['lat'].values
        lon_2d, lat_2d = np.meshgrid(lon_plot, lat_plot)

        temp_surface = select_data(ds, None, 'Temperature_surface').metpy.convert_units('degF').values
        try:
            pressure_mslp = (select_data(ds, None, 'MSLET') / 100).values
            pressure_label = 'MSLP'
        except KeyError:
            logging.warning("MSLP variable 'MSLET' not found.")
            pressure_mslp = None
            pressure_label = 'Surface Temperature'
        u_wind_surface = select_data(ds, None, 'u10').metpy.convert_units('knots').values
        v_wind_surface = select_data(ds, None, 'v10').metpy.convert_units('knots').values

        plot_background(ax)
        logo_size = 1.00
        logo_pad = 0.25
        fig_width, fig_height = fig.get_size_inches()
        logo_size_axes = logo_size / fig_width
        logo_pad_axes = logo_pad / fig_width

        for i, logo_path in enumerate(LOGO_PATHS):
            if os.path.exists(logo_path) and is_valid_png(logo_path):
                logo_img = plt.imread(logo_path)
                imagebox = OffsetImage(logo_img, zoom=logo_size_axes)
                ab = AnnotationBbox(imagebox,
                                    (1 - logo_pad_axes if i == 0 else logo_pad_axes, 1 - logo_pad / fig_height),
                                    xycoords='figure fraction',
                                    box_alignment=(1 if i == 0 else 0, 1),
                                    frameon=False)
                ax.add_artist(ab)
            else:
                logging.warning(f"Logo file not found or invalid: {logo_path}")

        cf = ax.contourf(lon_2d, lat_2d, temp_surface, cmap='coolwarm', transform=ccrs.PlateCarree(), levels=np.linspace(20, 120, 41))

        temp_contour_levels = np.arange(np.floor(temp_surface.min()), np.ceil(temp_surface.max()) + 1, 5)
        if len(temp_contour_levels) < 2 and temp_surface.max() - temp_surface.min() > 0:
            temp_contour_levels = np.linspace(temp_surface.min(), temp_surface.max(), 5)
        elif len(temp_contour_levels) < 2:
            temp_contour_levels = [temp_surface.mean()]

        c = ax.contour(lon_2d, lat_2d, temp_surface, levels=temp_contour_levels, colors='#2e2300', linewidths=2, linestyles='dashed', transform=ccrs.PlateCarree())
        ax.clabel(c, fontsize=12, inline=1, fmt='%.2f°F')
        ax.set_title(f'Surface Temperatures (°F), {pressure_label}', fontsize=16)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.05, extend='both')
        cb.set_ticks(np.arange(20, 121, 20))
        cb.set_label('Temperature (°F)', size='large')

        if pressure_mslp is not None:
            levels = np.arange(960, 1060, 2)
            isobars = ax.contour(lon_2d, lat_2d, pressure_mslp, levels=levels, colors='black', linestyles='-', linewidths=2.5, transform=ccrs.PlateCarree())
            ax.clabel(isobars, fontsize=12, inline=1, fmt='%.1f hPa')

        skip = (slice(None, None, 5), slice(None, None, 5))
        ax.barbs(lon_2d[skip], lat_2d[skip], u_wind_surface[skip], v_wind_surface[skip],
                 length=6, transform=ccrs.PlateCarree(), barbcolor='#3f0345')

        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=24, y=1.02)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        logging.debug("Surface temperature map generated successfully")
        return buf

    except Exception as e:
        logging.error(f"Error in generate_surface_temp_map: {e}", exc_info=True)
        return None

# Discord bot commands
@commands.command()
async def wind300(ctx):
    """Generate a 300 hPa wind speed map."""
    await ctx.send('Generating 300 hPa wind map, please wait...')
    image_bytes = None
    try:
        ds, run_date = await fetch_gfs_data(30000)
        image_bytes = await asyncio.to_thread(
            generate_map, ds, run_date, 30000, 'wind_speed', 'YlGnBu',
            '300-hPa Wind Speeds and Heights', 'Wind Speed (knots)'
        )
        if image_bytes:
            await ctx.send(file=discord.File(fp=image_bytes, filename='wind300.png'))
        else:
            await ctx.send('Failed to generate 300 hPa wind map.')
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')
    finally:
        if image_bytes is not None:
            image_bytes.close()

@commands.command()
async def wind500(ctx):
    """Generate a 500 hPa wind speed map."""
    await ctx.send('Generating 500 hPa wind map, please wait...')
    image_bytes = None
    try:
        ds, run_date = await fetch_gfs_data(50000)
        cmap = mcolors.LinearSegmentedColormap.from_list("white_to_orange", ["white", "orange"])
        image_bytes = await asyncio.to_thread(
            generate_map, ds, run_date, 50000, 'wind_speed', cmap,
            '500-hPa Wind Speeds and Heights', 'Wind Speed (knots)'
        )
        if image_bytes:
            await ctx.send(file=discord.File(fp=image_bytes, filename='wind500.png'))
        else:
            await ctx.send('Failed to generate 500 hPa wind map.')
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')
    finally:
        if image_bytes is not None:
            image_bytes.close()

@commands.command()
async def vort500(ctx):
    """Generate a 500 hPa absolute vorticity map."""
    await ctx.send('Generating 500 hPa vorticity map, please wait...')
    image_bytes = None
    try:
        ds, run_date = await fetch_gfs_data(50000)
        image_bytes = await asyncio.to_thread(
            generate_map, ds, run_date, 50000, 'vorticity', 'seismic',
            '500-hPa Absolute Vorticity and Heights', r'Vorticity ($10^{-5}$ s$^{-1}$)',
            levels=np.linspace(-20, 20, 41)
        )
        if image_bytes:
            await ctx.send(file=discord.File(fp=image_bytes, filename='vort500.png'))
        else:
            await ctx.send('Failed to generate 500 hPa vorticity map.')
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')
    finally:
        if image_bytes is not None:
            image_bytes.close()

@commands.command()
async def rh700(ctx):
    """Generate a 700 hPa relative humidity map."""
    await ctx.send('Generating 700 hPa relative humidity map, please wait...')
    image_bytes = None
    try:
        ds, run_date = await fetch_gfs_data(70000)
        image_bytes = await asyncio.to_thread(
            generate_map, ds, run_date, 70000, 'relative_humidity', 'BuGn',
            '700-hPa Relative Humidity and Heights', 'Relative Humidity (%)'
        )
        if image_bytes:
            await ctx.send(file=discord.File(fp=image_bytes, filename='rh700.png'))
        else:
            await ctx.send('Failed to generate 700 hPa relative humidity map.')
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')
    finally:
        if image_bytes is not None:
            image_bytes.close()

@commands.command()
async def wind850(ctx):
    """Generate an 850 hPa wind speed map."""
    await ctx.send('Generating 850 hPa wind map, please wait...')
    image_bytes = None
    try:
        ds, run_date = await fetch_gfs_data(85000)
        image_bytes = await asyncio.to_thread(
            generate_map, ds, run_date, 85000, 'wind_speed', 'PuRd',
            '850-hPa Wind Speeds and Heights', 'Wind Speed (knots)'
        )
        if image_bytes:
            await ctx.send(file=discord.File(fp=image_bytes, filename='wind850.png'))
        else:
            await ctx.send('Failed to generate 850 hPa wind map.')
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')
    finally:
        if image_bytes is not None:
            image_bytes.close()

@commands.command()
async def tAdv850(ctx):
    """Generate an 850 hPa temperature advection map."""
    await ctx.send('Generating 850 hPa Temperature Advection map, please wait...')
    image_bytes = None
    try:
        ds, run_date = await fetch_gfs_data(85000)
        image_bytes = await asyncio.to_thread(generate_temp_advection_map, ds, run_date)
        if image_bytes:
            await ctx.send(file=discord.File(fp=image_bytes, filename='tempAdvection850.png'))
        else:
            await ctx.send('Failed to generate 850 hPa temperature advection map.')
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')
    finally:
        if image_bytes is not None:
            image_bytes.close()

@commands.command()
async def mAdv850(ctx):
    """Generate an 850 hPa moisture advection map."""
    await ctx.send('Generating 850 hPa Moisture Advection map, please wait...')
    image_bytes = None
    try:
        ds, run_date = await fetch_gfs_data(85000)
        image_bytes = await asyncio.to_thread(generate_moisture_advection_map, ds, run_date)
        if image_bytes:
            await ctx.send(file=discord.File(fp=image_bytes, filename='moistureAdvection850.png'))
        else:
            await ctx.send('Failed to generate 850 hPa moisture advection map.')
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')
    finally:
        if image_bytes is not None:
            image_bytes.close()

@commands.command()
async def dew850(ctx):
    """Generate an 850 hPa dewpoint and temperature map."""
    await ctx.send('Generating 850 hPa Dewpoint and Temperature map, please wait...')
    image_bytes = None
    try:
        ds, run_date = await fetch_gfs_data(85000)
        image_bytes = await asyncio.to_thread(generate_dewpoint_map, ds, run_date)
        if image_bytes:
            await ctx.send(file=discord.File(fp=image_bytes, filename='dewpoint850.png'))
        else:
            await ctx.send('Failed to generate 850 hPa dewpoint map.')
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')
    finally:
        if image_bytes is not None:
            image_bytes.close()

@commands.command()
async def surfaceTemp(ctx):
    """Generate a surface temperature map with MSLP and wind barbs."""
    await ctx.send('Generating surface temperature map, please wait...')
    image_bytes = None
    try:
        ds, run_date = await fetch_gfs_data(None)
        image_bytes = await asyncio.to_thread(generate_surface_temp_map)
        if image_bytes:
            await ctx.send(file=discord.File(fp=image_bytes, filename='surfaceTemp.png'))
        else:
            await ctx.send('Failed to generate surface temperature map.')
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')
    finally:
        if image_bytes is not None:
            image_bytes.close()
