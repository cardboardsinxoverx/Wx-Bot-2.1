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

# Intents are required for Discord.py 1.5+
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='$', intents=intents)

# Helper function to fetch GFS data
def get_gfs_data():
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
                    'Temperature_surface',
                    'Relative_humidity_isobaric')
    query.vertical_level([30000, 50000, 70000, 85000])
    query.lonlat_box(north=50, south=25, east=-65, west=-125)

    data = ncss.get_data(query)
    ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
    ds = ds.metpy.parse_cf()
    # Removed the unit conversion to keep 'isobaric' in Pascals (Pa)
    # ds = ds.assign_coords(isobaric=ds['isobaric'] / 100.0)
    return ds, run_date

# Helper function to add map features
def plot_background(ax):
    ax.set_extent([-125, -65, 25, 50], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

@bot.command()
async def wind300(ctx):
    await ctx.send('Generating 300 hPa wind map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        image_bytes = await loop.run_in_executor(None, generate_300wind_map)
        await ctx.send(file=discord.File(fp=image_bytes, filename='wind300.png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

def generate_300wind_map():
    ds, run_date = get_gfs_data()

    # Access longitude and latitude directly from the dataset
    lon = ds['longitude'].values
    lat = ds['latitude'].values

    # Create 2D grids of latitude and longitude
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Corrected the pressure level to use Pascals (Pa)
    heights_300 = ds['Geopotential_height_isobaric'].sel(isobaric=30000, method='nearest').squeeze().values.copy()
    u_wind_300 = ds['u-component_of_wind_isobaric'].sel(isobaric=30000, method='nearest').squeeze()
    v_wind_300 = ds['v-component_of_wind_isobaric'].sel(isobaric=30000, method='nearest').squeeze()

    # Calculate wind speed at 300 hPa
    wind_speed_300 = mpcalc.wind_speed(u_wind_300.metpy.quantify(),
                                       v_wind_300.metpy.quantify()).metpy.dequantify()

    # Convert units where necessary
    wind_speed_300 = wind_speed_300.metpy.convert_units('knots')

    # Smooth the height data
    heights_300_smooth = ndimage.gaussian_filter(heights_300, sigma=3, order=0)

    # Define the map projection
    crs = ccrs.PlateCarree()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': crs})

    # Add background to the plot
    plot_background(ax)

    # Plot 300-hPa Wind Speeds and Heights
    cf1 = ax.contourf(lon_2d, lat_2d, wind_speed_300, cmap='cool',
                             transform=crs)
    c1 = ax.contour(lon_2d, lat_2d, heights_300_smooth, colors='black', linewidths=1, transform=crs)
    ax.clabel(c1, fontsize=8, inline=1, fmt='%i')
    ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind_300[::5, ::5], v_wind_300[::5, ::5], transform=crs, length=6)
    ax.set_title('300-hPa Wind Speeds and Heights', fontsize=16)
    cb1 = fig.colorbar(cf1, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
    cb1.set_label('Wind Speed (knots)', size='large')

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
async def vort500(ctx):
    await ctx.send('Generating 500 hPa vorticity map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        image_bytes = await loop.run_in_executor(None, generate_500vort_map)
        await ctx.send(file=discord.File(fp=image_bytes, filename='vort500.png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

def generate_500vort_map():
    ds, run_date = get_gfs_data()

    # Access longitude and latitude directly from the dataset
    lon = ds['longitude'].values
    lat = ds['latitude'].values

    # Create 2D grids of latitude and longitude
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Corrected the pressure level to use Pascals (Pa)
    heights_500 = ds['Geopotential_height_isobaric'].sel(isobaric=50000, method='nearest').squeeze().values.copy()
    u_wind_500 = ds['u-component_of_wind_isobaric'].sel(isobaric=50000, method='nearest').squeeze()
    v_wind_500 = ds['v-component_of_wind_isobaric'].sel(isobaric=50000, method='nearest').squeeze()

    # Calculate absolute vorticity at 500 hPa
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    vort_500 = mpcalc.vorticity(u_wind_500.metpy.quantify(), v_wind_500.metpy.quantify(), dx=dx, dy=dy, latitude=lat_2d * units.degrees)

    # Convert units where necessary
    vort_500 = vort_500 * 1e5  # Scale vorticity for visualization

    # Smooth the height data
    heights_500_smooth = ndimage.gaussian_filter(heights_500, sigma=3, order=0)

    # Define the map projection
    crs = ccrs.PlateCarree()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': crs})

    # Add background to the plot
    plot_background(ax)

    # Plot 500-hPa Absolute Vorticity and Heights
    cf1 = ax.contourf(lon_2d, lat_2d, vort_500, cmap='BrBG',
                             transform=crs, levels=np.linspace(-20, 20, 41), extend='both')
    c1 = ax.contour(lon_2d, lat_2d, heights_500_smooth, colors='black', linewidths=1, transform=crs)
    ax.clabel(c1, fontsize=8, inline=1, fmt='%i')
    ax.set_title('500-hPa Absolute Vorticity and Heights', fontsize=16)
    cb1 = fig.colorbar(cf1, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
    cb1.set_label(r'Vorticity ($10^{-5}$ s$^{-1}$)', size='large')

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
async def rh700(ctx):
    await ctx.send('Generating 700 hPa relative humidity map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        image_bytes = await loop.run_in_executor(None, generate_700rh_map)
        await ctx.send(file=discord.File(fp=image_bytes, filename='rh700.png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

def generate_700rh_map():
    ds, run_date = get_gfs_data()

    # Access longitude and latitude directly from the dataset
    lon = ds['longitude'].values
    lat = ds['latitude'].values

    # Create 2D grids of latitude and longitude
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Corrected the pressure level to use Pascals (Pa)
    heights_700 = ds['Geopotential_height_isobaric'].sel(isobaric=70000, method='nearest').squeeze().values.copy()
    u_wind_700 = ds['u-component_of_wind_isobaric'].sel(isobaric=70000, method='nearest').squeeze()
    v_wind_700 = ds['v-component_of_wind_isobaric'].sel(isobaric=70000, method='nearest').squeeze()
    rh_700 = ds['Relative_humidity_isobaric'].sel(isobaric=70000, method='nearest').squeeze()

    # Smooth the height data
    heights_700_smooth = ndimage.gaussian_filter(heights_700, sigma=3, order=0)

    # Define the map projection
    crs = ccrs.PlateCarree()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': crs})

    # Add background to the plot
    plot_background(ax)

    # Plot 700-hPa Relative Humidity and Heights
    cf1 = ax.contourf(lon_2d, lat_2d, rh_700, cmap='BuGn', transform=crs)
    c1 = ax.contour(lon_2d, lat_2d, heights_700_smooth, colors='black', linewidths=1, transform=crs)
    ax.clabel(c1, fontsize=8, inline=1, fmt='%i')
    ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind_700[::5, ::5], v_wind_700[::5, ::5], transform=crs, length=6)
    ax.set_title('700-hPa Relative Humidity and Heights', fontsize=16)
    cb1 = fig.colorbar(cf1, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
    cb1.set_label('Relative Humidity (%)', size='large')

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
async def wind850(ctx):
    await ctx.send('Generating 850 hPa wind map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        image_bytes = await loop.run_in_executor(None, generate_850wind_map)
        await ctx.send(file=discord.File(fp=image_bytes, filename='wind850.png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

def generate_850wind_map():
    ds, run_date = get_gfs_data()

    # Access longitude and latitude directly from the dataset
    lon = ds['longitude'].values
    lat = ds['latitude'].values

    # Create 2D grids of latitude and longitude
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Extract the 850 hPa level data explicitly
    heights_850 = ds['Geopotential_height_isobaric'].sel(isobaric=85000, method='nearest').squeeze()
    u_wind_850 = ds['u-component_of_wind_isobaric'].sel(isobaric=85000, method='nearest').squeeze()
    v_wind_850 = ds['v-component_of_wind_isobaric'].sel(isobaric=85000, method='nearest').squeeze()

    # Ensure correct data shape (may need to be copied explicitly)
    heights_850 = heights_850.values.copy()
    u_wind_850 = u_wind_850.values.copy()
    v_wind_850 = v_wind_850.values.copy()

    # Calculate wind speed at 850 hPa
    wind_speed_850 = mpcalc.wind_speed(u_wind_850 * units('m/s'),
                                       v_wind_850 * units('m/s')).to('knots')

    # Smooth the height data
    heights_850_smooth = ndimage.gaussian_filter(heights_850, sigma=3, order=0)

    # Define the map projection
    crs = ccrs.PlateCarree()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': crs})

    # Add background to the plot
    plot_background(ax)

    # Plot 850-hPa Wind Speeds and Heights
    cf1 = ax.contourf(lon_2d, lat_2d, wind_speed_850, cmap='cool', transform=crs)
    c1 = ax.contour(lon_2d, lat_2d, heights_850_smooth, colors='black', linewidths=1, transform=crs)
    ax.clabel(c1, fontsize=8, inline=1, fmt='%i')
    ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind_850[::5, ::5], v_wind_850[::5, ::5], transform=crs, length=6)
    ax.set_title('850-hPa Wind Speeds and Heights', fontsize=16)
    cb1 = fig.colorbar(cf1, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
    cb1.set_label('Wind Speed (knots)', size='large')

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
async def surfaceTemp(ctx):
    await ctx.send('Generating surface temperature map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        image_bytes = await loop.run_in_executor(None, generate_surface_temp_map)
        await ctx.send(file=discord.File(fp=image_bytes, filename='surfaceTemp.png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

def generate_surface_temp_map():
    ds, run_date = get_gfs_data()

    # Access longitude and latitude directly from the dataset
    lon = ds['longitude'].values
    lat = ds['latitude'].values

    # Create 2D grids of latitude and longitude
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Extract surface temperature
    temp_surface = ds['Temperature_surface'].squeeze()

    # Convert units to degrees Fahrenheit
    temp_surface = temp_surface.metpy.convert_units('degF')

    # Define the map projection
    crs = ccrs.PlateCarree()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': crs})

    # Add background to the plot
    plot_background(ax)

    # Plot Surface Temperatures
    cf1 = ax.contourf(lon_2d, lat_2d, temp_surface, cmap='jet',
                             transform=crs, levels=np.linspace(temp_surface.min(), temp_surface.max(), 20))
    c1 = ax.contour(lon_2d, lat_2d, temp_surface, colors='black', linewidths=0.5, transform=crs)
    ax.clabel(c1, fontsize=8, inline=1, fmt='%i')
    ax.set_title('Surface Temperatures', fontsize=16)
    cb1 = fig.colorbar(cf1, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05, extend='both')
    cb1.set_label('Temperature (°F)', size='large')

    # Adjust layout and add figure title
    fig.tight_layout()
    fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=24, y=1.02)

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf
