import discord
from discord.ext import commands
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from siphon.catalog import TDSCatalog
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import ndimage
import asyncio
import io
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents)

# Constants
EARTH_RADIUS = 6371000  # meters
OMEGA = 7.292e-5  # Earth's angular velocity (rad/s)

# Helper Functions
def compute_wind_speed(u, v):
    """Compute wind speed from u and v components (m/s to knots)."""
    wind_speed_ms = np.sqrt(u**2 + v**2)
    return wind_speed_ms * 1.94384

def compute_vorticity(u, v, lat, lon):
    """Compute absolute vorticity (scaled by 10^5 s^-1)."""
    lat_rad = np.deg2rad(lat)
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)

    dx = EARTH_RADIUS * np.cos(lat_rad) * dlon_rad
    dy = EARTH_RADIUS * np.abs(dlat_rad)

    v_x = np.full_like(v, np.nan)
    v_x[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx[:, None])
    u_y = np.full_like(u, np.nan)
    u_y[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * dy)

    zeta_r = v_x - u_y
    f = 2 * OMEGA * np.sin(lat_rad)[:, None]
    zeta_a = zeta_r + f
    return zeta_a * 1e5

def compute_advection(phi, u, v, lat, lon):
    """Compute advection of a scalar field."""
    lat_rad = np.deg2rad(lat)
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)

    dx = EARTH_RADIUS * np.cos(lat_rad) * dlon_rad
    dy = EARTH_RADIUS * np.abs(dlat_rad)

    phi_x = np.full_like(phi, np.nan)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx[:, None])
    phi_y = np.full_like(phi, np.nan)
    phi_y[1:-1, :] = (phi[:-2, :] - phi[2:, :]) / (2 * dy)

    return u * phi_x + v * phi_y

def compute_dewpoint(T, rh):
    """Compute dewpoint temperature (°C) from temperature (K) and relative humidity (%)."""
    T_C = T - 273.15
    ln_rh = np.log(rh / 100)
    numerator = 243.5 * ln_rh + (17.67 * T_C) / (243.5 + T_C)
    denominator = 17.67 - ln_rh - (17.67 * T_C) / (243.5 + T_C)
    return numerator / denominator

# Helper function to fetch GFS data for a specific isobaric level
def get_gfs_data_for_level(level):
    """
    Fetches GFS data for a specific isobaric level.

    Parameters:
        level (int): The pressure level in Pa (e.g., 50000 for 500 hPa).

    Returns:
        tuple: (dataset, run_date)
            - dataset: Xarray dataset with data for the specified level.
            - run_date: Datetime object representing the model run time.
    """
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 6:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    logging.debug(f"Selected GFS run time for level {level} Pa: {run_date}")

    catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'
    cat = TDSCatalog(catalog_url)
    latest_dataset = list(cat.datasets.values())[0]
    ncss = latest_dataset.subset()

    query = ncss.query()
    query.accept('netcdf4')
    query.time(run_date)
    query.variables('Geopotential_height_isobaric',
                    'u-component_of_wind_isobaric',
                    'v-component_of_wind_isobaric',
                    'Relative_humidity_isobaric',
                    'Temperature_isobaric')
    query.vertical_level([level])
    query.lonlat_box(north=75, south=25, east=70, west=-30)

    try:
        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
        ds = ds.metpy.parse_cf()
        logging.debug(f"Available variables for level {level}: {list(ds.variables)}")
        return ds, run_date
    except Exception as e:
        logging.error(f"Error fetching data for level {level}: {e}")
        return None, None

# Helper function to fetch GFS surface data
def get_gfs_surface_data():
    """
    Fetches GFS surface data including temperature, pressure, and 10m wind components.

    Returns:
        tuple: (dataset, run_date)
            - dataset: Xarray dataset with surface variables.
            - run_date: Datetime object representing the model run time.
    """
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    hours_since_midnight = now.hour + now.minute / 60
    for run_hour in sorted(run_hours, reverse=True):
        if hours_since_midnight >= run_hour + 6:
            run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            break
    else:
        run_date = (now - timedelta(days=1)).replace(hour=run_hours[-1], minute=0, second=0, microsecond=0)

    logging.debug(f"Selected GFS run time for surface data: {run_date}")

    catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml'
    cat = TDSCatalog(catalog_url)
    latest_dataset = list(cat.datasets.values())[0]
    ncss = latest_dataset.subset()

    query = ncss.query()
    query.accept('netcdf4')
    query.time(run_date)
    query.variables('Temperature_surface', 'Pressure_surface',
                    'u-component_of_wind_height_above_ground', 'v-component_of_wind_height_above_ground')
    query.lonlat_box(north=75, south=25, east=70, west=-30)

    try:
        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
        ds = ds.metpy.parse_cf()
        logging.debug(f"Available surface variables: {list(ds.variables)}")
        logging.debug(f"Dataset dimensions: {ds.dims}")
        return ds, run_date
    except Exception as e:
        logging.error(f"Error fetching surface data: {e}")
        return None, None

# Helper function to add map features
def plot_background(ax):
    """
    Adds background features to the map axes.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to which features are added.
    """
    ax.set_extent([-30, 70, 25, 75], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidths=2.5, edgecolor='#750b7a')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', linewidths=2, edgecolor='#750b7a')
    ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.5)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

EARTH_RADIUS = 6371000  # meters

def compute_divergence(u, v, lat, lon):
    # Convert latitude to radians
    lat_rad = np.deg2rad(lat)
    dlon = lon[1] - lon[0]  # Longitude step
    dlat = lat[1] - lat[0]  # Latitude step
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)

    # Calculate distances (dx varies with latitude, dy is constant)
    dx = EARTH_RADIUS * np.cos(lat_rad)[:, None] * dlon_rad  # Shape: (201, 1)
    dy = EARTH_RADIUS * dlat_rad  # Scalar

    # Initialize derivative arrays with NaN
    u_x = np.full_like(u, np.nan)
    v_y = np.full_like(v, np.nan)

    # Compute derivatives (dx broadcasts to match u_x shape)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)  # Fixed: no slicing on dx
    v_y[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)  # dy is scalar

    # Calculate divergence
    divergence = u_x + v_y
    return divergence

# Helper function to generate maps for isobaric levels
def generate_map(ds, run_date, level, variable, cmap, title, cb_label, levels=None):
    """
    Generates a map for a specified isobaric level and variable.

    Parameters:
        ds (xarray.Dataset): The dataset containing the data.
        run_date (datetime): The model run date.
        level (int): The pressure level in Pa.
        variable (str): The variable to plot ('wind_speed', 'vorticity', 'relative_humidity', 'temp_advection', 'moisture_advection', 'dewpoint').
        cmap (str): The colormap to use.
        title (str): The title of the map.
        cb_label (str): The colorbar label.
        levels (array-like, optional): The contour levels for the variable.

    Returns:
        io.BytesIO: A BytesIO object containing the map image, or None if an error occurs.
    """
    try:
        lon = ds.get('longitude')
        lat = ds.get('latitude')
        if lon is None or lat is None:
            raise ValueError("Longitude or latitude data is missing from the dataset.")

        lon = lon.values
        lat = lat.values
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        heights = ds['Geopotential_height_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
        heights_smooth = ndimage.gaussian_filter(heights, sigma=3, order=0)

        u_wind = ds['u-component_of_wind_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
        v_wind = ds['v-component_of_wind_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()

        if variable == 'wind_speed':
            data = compute_wind_speed(u_wind, v_wind)
        elif variable == 'vorticity':
            data = compute_vorticity(u_wind, v_wind, lat, lon)
        elif variable == 'relative_humidity':
            data = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
        elif variable == 'temp_advection':
            temp = ds['Temperature_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            data = compute_advection(temp, u_wind, v_wind, lat, lon) * 3600  # Convert to K/hour
        elif variable == 'moisture_advection':
            rh = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            data = compute_advection(rh, u_wind, v_wind, lat, lon) * 1e4  # Convert to %/hour
        elif variable == 'dewpoint':
            temp = ds['Temperature_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            rh = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            data = compute_dewpoint(temp, rh)
        elif variable == 'divergence':
            data = compute_divergence(u_wind, v_wind, lat, lon)
        else:
            raise ValueError(f"Unsupported variable type: {variable}")

        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': crs})
        fig.patch.set_facecolor('lightsteelblue')

        if levels is None:
            cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs)
        else:
            cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs, levels=levels)

        c = ax.contour(lon_2d, lat_2d, heights_smooth, colors='black', linewidths=2, transform=crs)
        ax.clabel(c, fontsize=8, inline=1, fmt='%i')

        ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind[::5, ::5], v_wind[::5, ::5], transform=crs, length=6)

        ax.set_title(f"{title} {run_date.strftime('%d %B %Y %H:%MZ')}", fontsize=16)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label(cb_label, size='large')

        plot_background(ax)

        logo_paths = ["/home/evanl/Documents/uga_logo.png", "/home/evanl/Documents/boxlogo2.png"]
        add_logos_to_figure(fig, logo_paths, logo_size=1.0, logo_pad=0.2)

        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except ValueError as e:
        logging.error(f"Data extraction error in generate_map: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred in generate_map: {e}")
        return None

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

def generate_mslp_temp_map():
    """
    Generate a Mean Sea Level Pressure (MSLP) chart with a temperature gradient overlay for Europe.

    Returns:
        tuple: (io.BytesIO, run_date) - A BytesIO object containing the map image and the model run date,
               or (None, None) if an error occurs.
    """
    try:
        # Assume get_gfs_surface_data() retrieves the dataset
        ds, run_date = get_gfs_surface_data()
        if ds is None:
            raise ValueError("Failed to retrieve surface data.")

        # Extract longitude and latitude
        lon = ds.get('longitude')
        lat = ds.get('latitude')
        if lon is None or lat is None:
            raise ValueError("Longitude or latitude data is missing.")

        lon = lon.values
        lat = lat.values
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        # Extract temperature and convert to Celsius
        temp_surface = ds['Temperature_surface'].isel(time=0).squeeze().metpy.convert_units('degC').metpy.dequantify()
        if temp_surface.ndim != 2:
            raise ValueError(f"Temperature data is not 2D, has shape {temp_surface.shape}")

        # Extract MSLP and convert to hPa
        mslp = ds['Pressure_surface'].isel(time=0).squeeze().metpy.dequantify() / 100  # Pa to hPa
        if mslp.ndim != 2:
            raise ValueError(f"MSLP data is not 2D, has shape {mslp.shape}")

        # Filter unrealistic pressure values
        mslp = np.where((mslp >= 900) & (mslp <= 1100), mslp, np.nan)

        # Set up the map
        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': crs})
        ax.set_extent([-30, 70, 25, 75], crs=ccrs.PlateCarree())
        fig.patch.set_facecolor('lightsteelblue')

        # Add geographical features
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.5)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidths=2.5, edgecolor='#750b7a')
        ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', linewidths=2, edgecolor='#750b7a')
        ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.5)
        ax.add_feature(cfeature.OCEAN, alpha=0.5)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

        # Plot temperature gradient in Celsius
        cf = ax.contourf(
            lon_2d, lat_2d, temp_surface,
            levels=np.linspace(np.nanmin(temp_surface), np.nanmax(temp_surface), 41),
            cmap='jet', transform=crs
        )

        # Plot MSLP isobars
        mslp_min = np.floor(np.nanmin(mslp) / 4) * 4
        mslp_max = np.ceil(np.nanmax(mslp) / 4) * 4
        isobar_levels = np.arange(mslp_min, mslp_max + 4, 4)
        c = ax.contour(lon_2d, lat_2d, mslp, levels=isobar_levels, colors='black', linewidths=.125, transform=crs)
        ax.clabel(c, fmt='%d hPa', inline=True, fontsize=5)

        # Set title and colorbar
        ax.set_title('Mean Sea Level Pressure (MSLP) with Temperature Gradient (°C)', fontsize=20)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label('Temperature (°C)', size='large')
        fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=16, y=1.02)

        # Add logos (assuming add_logos_to_figure is defined)
        logo_paths = ["/home/evanl/Documents/uga_logo.png", "/home/evanl/Documents/boxlogo2.png"]
        add_logos_to_figure(fig, logo_paths, logo_size=1.0, logo_pad=0.2)

        # Adjust layout and save to buffer
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf, run_date

    except Exception as e:
        logging.error(f"Error in generate_mslp_temp_map: {e}")
        return None, None

def add_logos_to_figure(fig, logo_paths, logo_size=1.0, logo_pad=0.2):
    """
    Adds logos to the figure at the top-left and top-right positions.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to which logos are added.
        logo_paths (list): List of file paths to the logo images.
        logo_size (float): Size of the logos in inches (default: 1.0).
        logo_pad (float): Padding from the edges in inches (default: 0.2).
    """
    # Get figure dimensions
    fig_width, fig_height = fig.get_size_inches()
    logo_width = logo_size / fig_width
    logo_height = logo_size / fig_height
    pad_width = logo_pad / fig_width
    pad_height = logo_pad / fig_height

    # Define positions for logos (top-left and top-right)
    positions = [
        {'left': pad_width, 'bottom': 1 - pad_height - logo_height, 'path': logo_paths[0]},  # Top-left
        {'left': 1 - pad_width - logo_width, 'bottom': 1 - pad_height - logo_height, 'path': logo_paths[1]}  # Top-right
    ]

    # Add each logo to the figure
    for pos in positions:
        try:
            ax_logo = fig.add_axes([pos['left'], pos['bottom'], logo_width, logo_height])
            ax_logo.axis('off')  # Hide axes for the logo
            logo_img = plt.imread(pos['path'])  # Read the logo image
            ax_logo.imshow(logo_img)  # Display the logo
        except FileNotFoundError:
            logging.warning(f"Logo file not found: {pos['path']}")
        except Exception as e:
            logging.error(f"Error adding logo {pos['path']}: {e}")

# Discord bot commands
@bot.command()
async def eu_wind300(ctx):
    await ctx.send('Generating 300 hPa wind map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for 300 hPa wind map")
        ds, run_date = await loop.run_in_executor(None, lambda: get_gfs_data_for_level(30000))
        if ds is None:
            await ctx.send('Failed to retrieve data for the 300 hPa wind map.')
            return
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(
            ds, run_date, 30000, 'wind_speed', 'cool', '300-hPa Wind Speeds and Heights', 'Wind Speed (knots)'
        ))
        if image_bytes is None:
            await ctx.send('Failed to generate the 300 hPa wind map due to missing or invalid data.')
            return
        filename = f'eu_wind300_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 300 hPa wind map")
    except Exception as e:
        logging.error(f"Error generating 300 hPa wind map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 300 hPa wind map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_wind500(ctx):
    """Command to generate and send the 500 hPa wind map."""
    await ctx.send('Generating 500 hPa wind map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for 500 hPa wind map")
        ds, run_date = await loop.run_in_executor(None, lambda: get_gfs_data_for_level(50000))
        if ds is None:
            await ctx.send('Failed to retrieve data for the 500 hPa wind map.')
            return
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(
            ds, run_date, 50000, 'wind_speed', 'YlOrBr', '500-hPa Wind Speeds and Heights', 'Wind Speed (knots)'
        ))
        if image_bytes is None:
            await ctx.send('Failed to generate the 500 hPa wind map due to missing or invalid data.')
            return
        filename = f'eu_wind500_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 500 hPa wind map")
    except Exception as e:
        logging.error(f"Error generating 500 hPa wind map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 500 hPa wind map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_vort500(ctx):
    await ctx.send('Generating 500 hPa vorticity map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for 500 hPa vorticity map")
        ds, run_date = await loop.run_in_executor(None, lambda: get_gfs_data_for_level(50000))
        if ds is None:
            await ctx.send('Failed to retrieve data for the 500 hPa vorticity map.')
            return
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(
            ds, run_date, 50000, 'vorticity', 'seismic', '500-hPa Absolute Vorticity and Heights',
            r'Vorticity ($10^{-5}$ s$^{-1}$)', levels=np.linspace(-20, 20, 41)
        ))
        if image_bytes is None:
            await ctx.send('Failed to generate the 500 hPa vorticity map due to missing or invalid data.')
            return
        filename = f'eu_vort500_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 500 hPa vorticity map")
    except Exception as e:
        logging.error(f"Error generating 500 hPa vorticity map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 500 hPa vorticity map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_rh700(ctx):
    await ctx.send('Generating 700 hPa relative humidity map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for 700 hPa relative humidity map")
        ds, run_date = await loop.run_in_executor(None, lambda: get_gfs_data_for_level(70000))
        if ds is None:
            await ctx.send('Failed to retrieve data for the 700 hPa relative humidity map.')
            return
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(
            ds, run_date, 70000, 'relative_humidity', 'BuGn', '700-hPa Relative Humidity and Heights', 'Relative Humidity (%)'
        ))
        if image_bytes is None:
            await ctx.send('Failed to generate the 700 hPa relative humidity map due to missing or invalid data.')
            return
        filename = f'eu_rh700_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 700 hPa relative humidity map")
    except Exception as e:
        logging.error(f"Error generating 700 hPa relative humidity map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 700 hPa relative humidity map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_wind850(ctx):
    await ctx.send('Generating 850 hPa wind map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for 850 hPa wind map")
        ds, run_date = await loop.run_in_executor(None, lambda: get_gfs_data_for_level(85000))
        if ds is None:
            await ctx.send('Failed to retrieve data for the 850 hPa wind map.')
            return
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(
            ds, run_date, 85000, 'wind_speed', 'YlOrBr', '850-hPa Wind Speeds and Heights', 'Wind Speed (knots)'
        ))
        if image_bytes is None:
            await ctx.send('Failed to generate the 850 hPa wind map due to missing or invalid data.')
            return
        filename = f'eu_wind850_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 850 hPa wind map")
    except Exception as e:
        logging.error(f"Error generating 850 hPa wind map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 850 hPa wind map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_dew850(ctx):
    """Command to generate and send the 850 hPa dewpoint map."""
    await ctx.send('Generating 850 hPa dewpoint map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for 850 hPa dewpoint map")
        ds, run_date = await loop.run_in_executor(None, lambda: get_gfs_data_for_level(85000))
        if ds is None:
            await ctx.send('Failed to retrieve data for the 850 hPa dewpoint map.')
            return
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(
            ds, run_date, 85000, 'dewpoint', 'BuGn', '850-hPa Dewpoint (°C)', 'Dewpoint (°C)'
        ))
        if image_bytes is None:
            await ctx.send('Failed to generate the 850 hPa dewpoint map due to missing or invalid data.')
            return
        filename = f'eu_dew850_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 850 hPa dewpoint map")
    except Exception as e:
        logging.error(f"Error generating 850 hPa dewpoint map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 850 hPa dewpoint map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_mAdv850(ctx):
    """Command to generate and send the 850 hPa moisture advection map."""
    await ctx.send('Generating 850 hPa moisture advection map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for 850 hPa moisture advection map")
        ds, run_date = await loop.run_in_executor(None, lambda: get_gfs_data_for_level(85000))
        if ds is None:
            await ctx.send('Failed to retrieve data for the 850 hPa moisture advection map.')
            return
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(
            ds, run_date, 85000, 'moisture_advection', 'PRGn', '850-hPa Moisture Advection', 'Moisture Advection (%/hour)'
        ))
        if image_bytes is None:
            await ctx.send('Failed to generate the 850 hPa moisture advection map due to missing or invalid data.')
            return
        filename = f'eu_mAdv850_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 850 hPa moisture advection map")
    except Exception as e:
        logging.error(f"Error generating 850 hPa moisture advection map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 850 hPa moisture advection map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_tAdv850(ctx):
    """Command to generate and send the 850 hPa temperature advection map."""
    await ctx.send('Generating 850 hPa temperature advection map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for 850 hPa temperature advection map")
        ds, run_date = await loop.run_in_executor(None, lambda: get_gfs_data_for_level(85000))
        if ds is None:
            await ctx.send('Failed to retrieve data for the 850 hPa temperature advection map.')
            return
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(
            ds, run_date, 85000, 'temp_advection', 'coolwarm', '850-hPa Temperature Advection', 'Temperature Advection (K/hour)',
            levels=np.linspace(-20, 20, 41)
        ))
        if image_bytes is None:
            await ctx.send('Failed to generate the 850 hPa temperature advection map due to missing or invalid data.')
            return
        filename = f'eu_tAdv850_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 850 hPa temperature advection map")
    except Exception as e:
        logging.error(f"Error generating 850 hPa temperature advection map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 850 hPa temperature advection map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_mslp_temp(ctx):
    """Command to generate and send the MSLP with temperature gradient map."""
    await ctx.send('Generating MSLP with temperature gradient map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for MSLP with temperature gradient map")
        image_bytes, run_date = await loop.run_in_executor(None, generate_mslp_temp_map)
        if image_bytes is None or run_date is None:
            await ctx.send('Failed to generate the MSLP with temperature gradient map due to missing or invalid data.')
            return
        filename = f'eu_mslp_temp_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent MSLP with temperature gradient map")
    except Exception as e:
        logging.error(f"Error generating MSLP with temperature gradient map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the MSLP with temperature gradient map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_divcon300(ctx):
    """Generate and send a 300 hPa divergence/convergence map."""
    await ctx.send('Generating 300 hPa divergence map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None  # Initialize image_bytes as None
    try:
        logging.info("Fetching data for 300 hPa divergence/convergence map")
        ds, run_date = await loop.run_in_executor(None, lambda: get_gfs_data_for_level(30000))
        if ds is None:
            await ctx.send('Failed to retrieve data for the 300 hPa divergence/convergence map.')
            return
        logging.info("Data fetched successfully")

        image_bytes = await loop.run_in_executor(None, lambda: generate_map(
            ds, run_date, 30000, 'divergence', 'RdBu_r',
            '300-hPa Divergence/Convergence', 'Divergence (10^5 s^-1)'
        ))
        if image_bytes is None:
            await ctx.send('Failed to generate the 300 hPa divergence/convergence map due to missing or invalid data.')
            return
        logging.info("Map generated successfully")

        filename = f'eu_divcon300_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 300 hPa divergence/convergence map")

    except Exception as e:
        logging.error(f"Error generating 300 hPa divergence/convergence map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 300 hPa divergence/convergence map: {e}')

    finally:
        if image_bytes is not None:
            image_bytes.close()

# Uncomment and add your bot token to run
# bot.run('YOUR_DISCORD_BOT_TOKEN')
