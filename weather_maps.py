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
from metpy.units import units
import metpy.calc as mpcalc

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents)

# Constants
EARTH_RADIUS = 6371000  # meters
OMEGA = 7.292e-5  # Earth's angular velocity (rad/s)
REGION = "CONUS"  # Define the region

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

def compute_divergence(u, v, lat, lon):
    """Compute horizontal divergence (s^-1) from u and v wind components."""
    lat_rad = np.deg2rad(lat)
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)

    dx = EARTH_RADIUS * np.cos(lat_rad)[:, None] * dlon_rad
    dy = EARTH_RADIUS * dlat_rad

    u_x = np.full_like(u, np.nan)
    v_y = np.full_like(v, np.nan)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    v_y[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)

    divergence = u_x + v_y
    return divergence

def get_gfs_data_for_level(level):
    """Fetches GFS data for a specific isobaric level."""
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
    query.lonlat_box(north=50, south=25, east=-65, west=-125)  # CONUS

    try:
        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
        ds = ds.metpy.parse_cf()
        logging.debug(f"Available variables for level {level}: {list(ds.variables)}")
        return ds, run_date
    except Exception as e:
        logging.error(f"Error fetching data for level {level}: {e}")
        return None, None

def get_gfs_surface_data():
    """Fetches GFS surface data including temperature, surface pressure, elevation, and 10m wind components."""
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
    query.variables(
        'Temperature_surface',
        'Pressure_surface',  # Surface pressure for MSLP calculation
        'Geopotential_height_surface',  # Elevation for MSLP calculation
        'u-component_of_wind_height_above_ground',
        'v-component_of_wind_height_above_ground'
    )
    query.lonlat_box(north=50, south=25, east=-65, west=-125)  # North America

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

def plot_background(ax):
    """Adds background features to the map axes."""
    ax.set_extent([-125, -65, 25, 50], crs=ccrs.PlateCarree())  # North America
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

def generate_map(ds, run_date, level, variable, cmap, title, cb_label, levels=None):
    """Generates a map for a specified isobaric level and variable."""
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
            data = compute_advection(temp, u_wind, v_wind, lat, lon) * 3600
        elif variable == 'moisture_advection':
            rh = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
            data = compute_advection(rh, u_wind, v_wind, lat, lon) * 1e4
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

        # Convert wind components to knots for barbs (m/s to knots: * 1.94384)
        u_wind_knots = u_wind * 1.94384
        v_wind_knots = v_wind * 1.94384
        ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind_knots[::5, ::5], v_wind_knots[::5, ::5], transform=crs, length=6)

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

def generate_mslp_temp_map():
    """Generate a Mean Sea Level Pressure (MSLP) chart with temperature gradient and meteorological features using MSLET."""
    try:
        ds, run_date = get_gfs_surface_data()
        if ds is None:
            raise ValueError("Failed to retrieve surface data.")

        lon = ds.get('longitude')
        lat = ds.get('latitude')
        if lon is None or lat is None:
            raise ValueError("Longitude or latitude data is missing.")

        lon = lon.values
        lat = lat.values
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        # Debug: Log available height levels
        logging.debug(f"Available height levels: {ds['height_above_ground2'].values}")

        # Select temperature data using reftime and validtime2
        temp_surface = ds['Temperature_surface'].isel(reftime=0, validtime2=0).metpy.convert_units('degC').metpy.dequantify()
        if temp_surface.ndim != 2:
            raise ValueError(f"Temperature data is not 2D, has shape {temp_surface.shape}")

        # Select surface pressure data using reftime and validtime2 (in Pa)
        surface_pressure = ds['Pressure_surface'].isel(reftime=0, validtime2=0).metpy.dequantify()
        if surface_pressure.ndim != 2:
            raise ValueError(f"Surface pressure data is not 2D, has shape {surface_pressure.shape}")

        # Get elevation
        try:
            elevation = ds['Geopotential_height_surface'].isel(reftime=0, validtime2=0).metpy.dequantify()
            if elevation.ndim != 2:
                raise ValueError(f"Elevation data is not 2D, has shape {elevation.shape}")
        except KeyError as e:
            logging.error(f"Elevation data not found: {e}")
            raise ValueError("Elevation data is required for MSLP calculation.")

        # Calculate MSLP manually using the hypsometric equation
        try:
            # Constants
            g = 9.80665  # m/s^2
            Rd = 287.05  # J/(kg·K)

            # Convert temperature to Kelvin
            temp_kelvin = temp_surface + 273.15  # Assuming surface temp as MSLET proxy

            # Hypsometric equation: P_mslp = P_surface * exp(g * h / (Rd * T))
            mslp = surface_pressure * np.exp((g * elevation) / (Rd * temp_kelvin))

            # Convert to hPa
            mslp = mslp / 100
        except Exception as e:
            logging.error(f"Error calculating MSLP: {e}")
            raise ValueError("Failed to calculate MSLP.")

        # Validate and smooth MSLP
        mslp = np.where((mslp >= 750) & (mslp <= 1100), mslp, np.nan)
        mslp_smooth = ndimage.gaussian_filter(mslp, sigma=3, order=0)

        # Select wind components at 10 meters height
        u_wind_da = ds['u-component_of_wind_height_above_ground'].sel(height_above_ground2=10, method='nearest').isel(reftime=0, validtime2=0).squeeze()
        v_wind_da = ds['v-component_of_wind_height_above_ground'].sel(height_above_ground2=10, method='nearest').isel(reftime=0, validtime2=0).squeeze()

        # Debug: Log selected height and shapes
        selected_height = ds['height_above_ground2'].sel(height_above_ground2=10, method='nearest').values
        logging.debug(f"Selected height for wind: {selected_height}")
        logging.debug(f"u_wind shape: {u_wind_da.shape}")
        logging.debug(f"v_wind shape: {v_wind_da.shape}")

        u_wind = u_wind_da.values
        v_wind = v_wind_da.values

        # Compute temperature gradients for front detection
        temp_grad_x, temp_grad_y = np.gradient(temp_surface, lon[1] - lon[0], lat[1] - lat[0])
        temp_grad_mag = np.sqrt(temp_grad_x**2 + temp_grad_y**2)

        # Frontogenesis (simplified as temperature gradient convergence)
        frontogenesis = compute_advection(temp_grad_mag, u_wind, v_wind, lat, lon)

        # Identify pressure centers
        mslp_grad_x, mslp_grad_y = np.gradient(mslp_smooth, lon[1] - lon[0], lat[1] - lat[0])
        mslp_lap = np.gradient(mslp_grad_x, axis=1) + np.gradient(mslp_grad_y, axis=0)

        # Compute curvature for troughs and ridges
        curvature = mslp_grad_x * np.gradient(mslp_grad_y, axis=0) - mslp_grad_y * np.gradient(mslp_grad_x, axis=1)

        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': crs})
        fig.patch.set_facecolor('lightsteelblue')

        plot_background(ax)

        # Temperature gradient fill
        cf = ax.contourf(
            lon_2d, lat_2d, temp_surface,
            levels=np.linspace(np.nanmin(temp_surface), np.nanmax(temp_surface), 41),
            cmap='jet', transform=crs
        )

        # MSLP contours with 2 hPa intervals
        mslp_min = np.floor(np.nanmin(mslp) / 2) * 2
        mslp_max = np.ceil(np.nanmax(mslp) / 2) * 2
        isobar_levels = np.arange(mslp_min, mslp_max + 2, 2)
        c = ax.contour(lon_2d, lat_2d, mslp_smooth, levels=isobar_levels, colors='black', linewidths=2, transform=crs)
        ax.clabel(c, fmt='%d hPa', inline=True, fontsize=5)

        # Convert wind components to knots for barbs (m/s to knots: * 1.94384)
        u_wind_knots = u_wind * 1.94384
        v_wind_knots = v_wind * 1.94384
        ax.barbs(
            lon_2d[::5, ::5], lat_2d[::5, ::5],
            u_wind_knots[::5, ::5], v_wind_knots[::5, ::5],
            transform=crs, length=6, color='black'
        )

        # Dry lines (simplified using temperature gradient as proxy)
        dry_line_mask = (temp_grad_mag > np.percentile(temp_grad_mag, 90)) & (frontogenesis > -0.005) & (frontogenesis < 0.005)
        ax.contour(lon_2d, lat_2d, dry_line_mask, levels=[0.5], colors='brown', linestyles='-.', linewidths=1, transform=crs)

        # Titles and labels
        main_title = f"European: MSLP with Temperature Gradient (°C) and Features"
        ax.set_title(main_title, fontsize=16)
        fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=12, y=1.02)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label('Temperature (°C)', size='large')

        logo_paths = ["/media/evanl/BACKUP/bot/metoc.png", "/media/evanl/BACKUP/bot/boxlogo2.png"]
        add_logos_to_figure(fig, logo_paths, logo_size=1.0, logo_pad=0.2)

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
    """Adds logos to the figure at the top-left and top-right positions."""
    fig_width, fig_height = fig.get_size_inches()
    logo_width = logo_size / fig_width
    logo_height = logo_size / fig_height
    pad_width = logo_pad / fig_width
    pad_height = logo_pad / fig_height

    positions = [
        {'left': pad_width, 'bottom': 1 - pad_height - logo_height, 'path': logo_paths[0]},
        {'left': 1 - pad_width - logo_width, 'bottom': 1 - pad_height - logo_height, 'path': logo_paths[1]}
    ]

    for pos in positions:
        try:
            ax_logo = fig.add_axes([pos['left'], pos['bottom'], logo_width, logo_height])
            ax_logo.axis('off')
            logo_img = plt.imread(pos['path'])
            ax_logo.imshow(logo_img)
        except FileNotFoundError:
            logging.warning(f"Logo file not found: {pos['path']}")
        except Exception as e:
            logging.error(f"Error adding logo {pos['path']}: {e}")

# Discord Commands
@bot.command()
async def wind300(ctx):
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
        filename = f'wind300_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 300 hPa wind map")
    except Exception as e:
        logging.error(f"Error generating 300 hPa wind map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 300 hPa wind map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def wind500(ctx):
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
        filename = f'wind500_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 500 hPa wind map")
    except Exception as e:
        logging.error(f"Error generating 500 hPa wind map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 500 hPa wind map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def vort500(ctx):
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
        filename = f'vort500_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 500 hPa vorticity map")
    except Exception as e:
        logging.error(f"Error generating 500 hPa vorticity map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 500 hPa vorticity map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def rh700(ctx):
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
        filename = f'rh700_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 700 hPa relative humidity map")
    except Exception as e:
        logging.error(f"Error generating 700 hPa relative humidity map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 700 hPa relative humidity map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def wind850(ctx):
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
        filename = f'wind850_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 850 hPa wind map")
    except Exception as e:
        logging.error(f"Error generating 850 hPa wind map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 850 hPa wind map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def dew850(ctx):
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
        filename = f'dew850_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 850 hPa dewpoint map")
    except Exception as e:
        logging.error(f"Error generating 850 hPa dewpoint map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 850 hPa dewpoint map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def mAdv850(ctx):
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
        filename = f'mAdv850_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 850 hPa moisture advection map")
    except Exception as e:
        logging.error(f"Error generating 850 hPa moisture advection map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 850 hPa moisture advection map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def tAdv850(ctx):
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
        filename = f'tAdv850_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 850 hPa temperature advection map")
    except Exception as e:
        logging.error(f"Error generating 850 hPa temperature advection map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 850 hPa temperature advection map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def mslp_temp(ctx):
    await ctx.send('Generating MSLP with temperature gradient map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for MSLP with temperature gradient map")
        image_bytes, run_date = await loop.run_in_executor(None, generate_mslp_temp_map)
        if image_bytes is None or run_date is None:
            await ctx.send('Failed to generate the MSLP with temperature gradient map due to missing or invalid data.')
            return
        filename = f'mslp_temp_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent MSLP with temperature gradient map")
    except Exception as e:
        logging.error(f"Error generating MSLP with temperature gradient map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the MSLP with temperature gradient map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def divcon300(ctx):
    await ctx.send('Generating 300 hPa divergence map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        logging.info("Fetching data for 300 hPa divergence/convergence map")
        ds, run_date = await loop.run_in_executor(None, lambda: get_gfs_data_for_level(30000))
        if ds is None:
            await ctx.send('Failed to retrieve data for the 300 hPa divergence/convergence map.')
            return
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(
            ds, run_date, 30000, 'divergence', 'RdBu_r',
            '300-hPa Divergence/Convergence', 'Divergence (10^5 s^-1)'
        ))
        if image_bytes is None:
            await ctx.send('Failed to generate the 300 hPa divergence/convergence map due to missing or invalid data.')
            return
        filename = f'divcon300_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
        logging.info("Successfully generated and sent 300 hPa divergence/convergence map")
    except Exception as e:
        logging.error(f"Error generating 300 hPa divergence/convergence map: {e}")
        await ctx.send(f'An unexpected error occurred while generating the 300 hPa divergence/convergence map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

# Uncomment and add your bot token to run
# bot.run('YOUR_DISCORD_BOT_TOKEN')
