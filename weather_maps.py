import discord
from discord.ext import commands
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from siphon.catalog import TDSCatalog
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.units import units
import metpy.calc as mpcalc
from scipy import ndimage
import asyncio
import io
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Intents are required for Discord.py 1.5+
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='$', intents=intents)

# Helper function to fetch GFS data for a specific level
def get_gfs_data_for_level(level):
    now = datetime.utcnow()
    run_hours = [0, 6, 12, 18]
    run_time = max([hour for hour in run_hours if hour <= now.hour])
    run_date = now.replace(hour=run_time, minute=0, second=0, microsecond=0)
    if run_time > now.hour:
        run_date -= timedelta(days=1)

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
                    'Temperature_surface'),
    query.vertical_level([level])
    query.lonlat_box(north=50, south=25, east=-65, west=-125)

    data = ncss.get_data(query)
    ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
    ds = ds.metpy.parse_cf()
    return ds, run_date

# Helper function to add map features
def plot_background(ax):
    ax.set_extent([-125, -65, 25, 50], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidths=2, edgecolor='#750b7a')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', linewidths=2, edgecolor='#750b7a')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

# Generalized map generation function
def generate_map(ds, run_date, level, variable, cmap, title, cb_label, levels=None):
    # Access longitude and latitude directly from the dataset
    lon = ds['longitude'].values
    lat = ds['latitude'].values

    # Create 2D grids of latitude and longitude
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Extract data at the specific level
    heights = ds['Geopotential_height_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
    u_wind = ds['u-component_of_wind_isobaric'].sel(isobaric=level, method='nearest').squeeze().copy()
    v_wind = ds['v-component_of_wind_isobaric'].sel(isobaric=level, method='nearest').squeeze().copy()

    # Smooth the height data
    heights_smooth = ndimage.gaussian_filter(heights, sigma=3, order=0)

    # Calculate the variable to plot (e.g., wind speed, relative humidity)
    if variable == 'wind_speed':
        data = mpcalc.wind_speed(u_wind.metpy.quantify(), v_wind.metpy.quantify()).metpy.dequantify()
        data = data.metpy.convert_units('knots')
    elif variable == 'vorticity':
        dx, dy = mpcalc.lat_lon_grid_deltas(lon_2d, lat_2d)
        data = mpcalc.vorticity(u_wind.metpy.quantify(), v_wind.metpy.quantify(), dx=dx, dy=dy, latitude=lat_2d * units.degrees)
        data = data * 1e5  # Scale vorticity for visualization
    elif variable == 'relative_humidity':
        data = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().copy()
    else:
        raise ValueError('Unsupported variable type')

    # Define the map projection
    crs = ccrs.PlateCarree()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': crs})

    # Set figure background color
    fig.patch.set_facecolor('lightsteelblue')

    # Add background to the plot
    plot_background(ax)

        # Add logos
    logo_paths = [
        "/home/evanl/Documents/boxlogo2.png",  # Right logo
        "/home/evanl/Documents/uga_logo.png"   # Left logo
    ]

    # Calculate logo positions in axes coordinates
    logo_size = 1.00  # 3/4 inch
    logo_pad = 0.2  # 0.2 inches
    fig_width, fig_height = fig.get_size_inches()
    logo_size_axes = logo_size / fig_width
    logo_pad_axes = logo_pad / fig_width

    for i, logo_path in enumerate(logo_paths):
        logo_img = plt.imread(logo_path)
        imagebox = OffsetImage(logo_img, zoom=logo_size_axes)
        ab = AnnotationBbox(imagebox, (1 - logo_pad_axes if i == 0 else logo_pad_axes, 1 - logo_pad_axes),
                            xycoords='figure fraction',  # Changed to figure fraction!
                            box_alignment=(1 if i == 0 else 0, 1),
                            frameon=False)
        ax.add_artist(ab)

    # Plot the data
    if levels is None:
        cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs)
    else:
        cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs, levels=levels)
    c = ax.contour(lon_2d, lat_2d, heights_smooth, colors='black', linewidths=2, transform=crs)
    ax.clabel(c, fontsize=8, inline=1, fmt='%i')
    ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind[::5, ::5], v_wind[::5, ::5], transform=crs, length=6)
    ax.set_title(title, fontsize=16)
    cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
    cb.set_label(cb_label, size='large')

    # Adjust layout and add figure title
    fig.tight_layout()
    fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=24, y=1.02)

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf

@bot.command()
async def wind300(ctx):
    await ctx.send('Generating 300 hPa wind map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        ds, run_date = get_gfs_data_for_level(30000)
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(ds, run_date, 30000, 'wind_speed', 'cool', '300-hPa Wind Speeds and Heights', 'Wind Speed (knots)'))
        await ctx.send(file=discord.File(fp=image_bytes, filename='wind300.png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

@bot.command()
async def vort500(ctx):
    await ctx.send('Generating 500 hPa vorticity map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        ds, run_date = get_gfs_data_for_level(50000)
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(ds, run_date, 50000, 'vorticity', 'seismic', '500-hPa Absolute Vorticity and Heights', r'Vorticity ($10^{-5}$ s$^{-1}$)', levels=np.linspace(-20, 20, 41)))
        await ctx.send(file=discord.File(fp=image_bytes, filename='vort500.png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

@bot.command()
async def rh700(ctx):
    await ctx.send('Generating 700 hPa relative humidity map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        ds, run_date = get_gfs_data_for_level(70000)
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(ds, run_date, 70000, 'relative_humidity', 'BuGn', '700-hPa Relative Humidity and Heights', 'Relative Humidity (%)'))
        await ctx.send(file=discord.File(fp=image_bytes, filename='rh700.png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

@bot.command()
async def wind850(ctx):
    await ctx.send('Generating 850 hPa wind map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        ds, run_date = get_gfs_data_for_level(85000)
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(ds, run_date, 85000, 'wind_speed', 'YlOrBr', '850-hPa Wind Speeds and Heights', 'Wind Speed (knots)'))
        await ctx.send(file=discord.File(fp=image_bytes, filename='wind850.png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

@bot.command()
async def surfaceTemp(ctx):
    await ctx.send('Generating surface temperature map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        image_bytes = await loop.run_in_executor(None, generate_surface_temp_map)
        await ctx.send(file=discord.File(fp=image_bytes, filename='surfaceTemp.png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

def generate_surface_temp_map():
    ds, run_date = get_gfs_data_for_level(100000)

    # Access longitude and latitude directly from the dataset
    lon = ds['longitude'].values
    lat = ds['latitude'].values

    # Create 2D grids of latitude and longitude
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Extract surface temperature
    temp_surface = ds['Temperature_surface'].squeeze().copy()

    # Extract mean sea level pressure if available, or calculate it
    if 'Pressure_msl' in ds:
        pressure_msl = ds['Pressure_msl'].squeeze()
    else:
        # Fallback to geopotential height to estimate pressure (not ideal, but a placeholder)
        pressure_msl = ds['Geopotential_height_isobaric'].sel(isobaric=1000, method='nearest').squeeze() / 10  # Placeholder scaling

    # Extract wind components (U and V) from isobaric level closest to the surface
    u_wind = ds['u-component_of_wind_isobaric'].sel(isobaric=1000, method='nearest').squeeze()
    v_wind = ds['v-component_of_wind_isobaric'].sel(isobaric=1000, method='nearest').squeeze()

    # Convert units to degrees Fahrenheit
    temp_surface = temp_surface.metpy.convert_units('degF')

    # Define the map projection
    crs = ccrs.PlateCarree()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': crs})

    # Set figure background color
    fig.patch.set_facecolor('lightsteelblue')

    # Add background to the plot
    plot_background(ax)

    # Plot Surface Temperatures
    cf1 = ax.contourf(lon_2d, lat_2d, temp_surface, cmap='jet',
                             transform=crs, levels=np.linspace(temp_surface.min(), temp_surface.max(), 20))
    c1 = ax.contour(lon_2d, lat_2d, temp_surface, colors='#244731', linewidths=2, transform=crs)
    ax.clabel(c1, fontsize=15, inline=1, fmt='%.2f°F')
    ax.set_title('Surface Temperatures', fontsize=16)
    cb1 = fig.colorbar(cf1, ax=ax, orientation='horizontal', shrink=1.0, pad=0.05, extend='both')
    cb1.set_label('Temperature (°F)', size='large')

    # Plot Isobars (Pressure Contours) using actual pressure or estimated pressure
    isobars = ax.contour(lon_2d, lat_2d, pressure_msl, colors='black', linestyles='-.', linewidths=2.5, transform=crs)
    ax.clabel(isobars, fontsize=16, inline=1, fmt='%.2f hPa')

    # Plot Wind Barbs with reduced density
    skip = (slice(None, None, 5), slice(None, None, 5))  # Reduce density by plotting every 5th point
    ax.barbs(lon_2d[skip], lat_2d[skip], u_wind[skip], v_wind[skip], length=6, transform=crs, pivot='middle', barbcolor='#3f0345')

    # Adjust layout and add figure title
    fig.tight_layout()
    fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=24, y=1.02)

    # Add logos
    logo_paths = [
        "/home/evanl/Documents/boxlogo2.png",  # Right logo
        "/home/evanl/Documents/uga_logo.png"   # Left logo
    ]

    # Calculate logo positions in axes coordinates
    logo_size = 1.00  # 1 inch
    logo_pad = 0.2  # 0.2 inches
    fig_width, fig_height = fig.get_size_inches()
    logo_size_axes = logo_size / fig_width
    logo_pad_axes = logo_pad / fig_width

    for i, logo_path in enumerate(logo_paths):
        logo_img = plt.imread(logo_path)
        imagebox = OffsetImage(logo_img, zoom=logo_size_axes)
        ab = AnnotationBbox(imagebox, (1 - logo_pad_axes if i == 0 else logo_pad_axes, 1 - logo_pad_axes),
                            xycoords='figure fraction',  # Changed to figure fraction!
                            box_alignment=(1 if i == 0 else 0, 1),
                            frameon=False)
        ax.add_artist(ab)

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf
