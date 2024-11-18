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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os

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

def generate_surface_temp_map():
    try:
        # Create the figure and axes early to ensure 'ax' is always defined
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})

        # Set figure background color
        fig.patch.set_facecolor('lightsteelblue')

        # Get dataset
        ds, run_date = get_gfs_data_for_level(100000)

        # Print dataset variables to debug
        print("Dataset variables:", ds.data_vars)

        # Access longitude and latitude directly from the dataset
        lon = ds.get('longitude')
        lat = ds.get('latitude')

        if lon is None or lat is None:
            raise ValueError("Longitude or latitude data is missing from the dataset.")

        lon = lon.values
        lat = lat.values

        # Create 2D grids of latitude and longitude
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        # Extract surface temperature
        temp_surface = ds.get('Temperature_surface')
        if temp_surface is None:
            raise ValueError("Surface temperature data is missing from the dataset.")
        temp_surface = temp_surface.squeeze().copy()

        # Extract geopotential height at the 1000 hPa isobaric level
        if 'Geopotential_height_isobaric' in ds:
            geopotential_height = ds['Geopotential_height_isobaric'].sel(isobaric=1000, method='nearest').squeeze()
        else:
            raise ValueError("No valid geopotential height data found in the dataset.")

        # Debugging: Check geopotential height values
        print("Geopotential Height Min:", geopotential_height.min().values)
        print("Geopotential Height Max:", geopotential_height.max().values)

        # Use geopotential height to calculate pseudo pressure with more realistic distribution
        H = 7000  # Scale height in meters, used to adjust the distribution
        P0 = 1013  # Reference pressure at sea level in hPa

        # Calculate the temperature lapse rate dynamically based on the mean temperature
        mean_temp = temp_surface.mean().values  # Mean temperature in the region (in Kelvin)
        lapse_rate = 6.5 / 1000  # Standard lapse rate in K/m (can be refined)

        # Barometric formula to adjust pressure calculation with a dynamic lapse rate
        pseudo_pressure = P0 * np.exp(-geopotential_height / (H * (1 - lapse_rate * (geopotential_height / mean_temp))))

        # Debugging: Check pseudo pressure values
        print("Pseudo Pressure (Initial) Min:", pseudo_pressure.min().values)
        print("Pseudo Pressure (Initial) Max:", pseudo_pressure.max().values)

        # ----> FIX INTEGRATED HERE <----
        # Apply further scaling and normalization to better match surface pressures
        mean_pressure = 1013  # Mean sea-level pressure in hPa
        scaling_factor = -1.05  # Adjust this value if needed
        pseudo_pressure_adjusted = (pseudo_pressure - pseudo_pressure.mean()) * scaling_factor + mean_pressure

        # Adjust bias correction to align with the new scaling
        bias_correction = 2  # Adjust this value if needed
        pseudo_pressure_adjusted += bias_correction

        # Clip values to avoid unrealistic pressure extremes
        pseudo_pressure_adjusted = np.clip(pseudo_pressure_adjusted, 980, 1050)
        # ----> END OF FIX <----

        # Debugging: Check pseudo pressure values
        print("Pseudo Pressure (Adjusted) Min:", pseudo_pressure_adjusted.min().values)
        print("Pseudo Pressure (Adjusted) Max:", pseudo_pressure_adjusted.max().values)

        # Extract wind components (U and V) from isobaric level closest to the surface
        u_wind = ds.get('u-component_of_wind_isobaric')
        v_wind = ds.get('v-component_of_wind_isobaric')
        if u_wind is None or v_wind is None:
            raise ValueError("Wind data is missing from the dataset.")

        u_wind = u_wind.sel(isobaric=1000, method='nearest').squeeze()
        v_wind = v_wind.sel(isobaric=1000, method='nearest').squeeze()

        # Convert units to degrees Fahrenheit
        temp_surface = temp_surface.metpy.convert_units('degF')

        # Add background to the plot
        plot_background(ax)

        # Add logos
        logo_paths = [
            "/home/evanl/Documents/boxlogo2.png",  # Right logo
            "/home/evanl/Documents/uga_logo.png"   # Left logo
        ]

        # Calculate logo positions in axes coordinates
        logo_size = 1.00  # 3/4 inch
        logo_pad = 0.25  # 0.25 inches
        fig_width, fig_height = fig.get_size_inches()
        logo_size_axes = logo_size / fig_width
        logo_pad_axes = logo_pad / fig_width

        for i, logo_path in enumerate(logo_paths):
            logo_img = plt.imread(logo_path)
            imagebox = OffsetImage(logo_img, zoom=logo_size / fig_width)
            ab = AnnotationBbox(imagebox, (1 - logo_pad_axes if i == 0 else logo_pad_axes, 1 - logo_pad / fig_height),
                                xycoords='figure fraction',  # Changed to figure fraction!
                                box_alignment=(1 if i == 0 else 0, 1),
                                frameon=False)
            ax.add_artist(ab)

        # Plot Surface Temperatures
        cf1 = ax.contourf(lon_2d, lat_2d, temp_surface, cmap='gist_rainbow_r',
                        transform=ccrs.PlateCarree(), levels=np.linspace(temp_surface.min(), temp_surface.max(), 10))  # Changed to 10 levels
        c1 = ax.contour(lon_2d, lat_2d, temp_surface, colors='#2e2300', linewidths=2, linestyles='dashed', transform=ccrs.PlateCarree())
        ax.clabel(c1, fontsize=12, inline=1, fmt='%.2f°F')
        ax.set_title('Surface Temperatures (°F)', fontsize=16)  # Corrected title
        cb1 = fig.colorbar(cf1, ax=ax, orientation='horizontal', shrink=1.0, pad=0.05, extend='both')
        cb1.set_label('Temperature (°F)', size='large')

        # Define isobar levels between a realistic surface pressure range
        start_level = 980  # Start at 980 hPa
        end_level = 1050   # End at 1050 hPa
        levels = np.arange(start_level, end_level + 2, 2)  # Changed increment to 2

        # Plot Isobars (Pressure Contours) using the adjusted pseudo pressure data
        isobars = ax.contour(
            lon_2d, lat_2d, pseudo_pressure_adjusted, levels=levels,
            colors='black', linestyles='-', linewidths=2.5, transform=ccrs.PlateCarree()
        )
        ax.clabel(isobars, fontsize=12, inline=1, fmt='%.1f hPa')

        # Plot Wind Barbs with reduced density
        skip = (slice(None, None, 5), slice(None, None, 5))  # Reduce density by plotting every 5th point
        ax.barbs(lon_2d[skip], lat_2d[skip], u_wind[skip], v_wind[skip], length=6, transform=ccrs.PlateCarree(), pivot='middle', barbcolor='#3f0345')

        # Adjust layout and add figure title
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        fig.suptitle(run_date.strftime('%d %B %Y %H:%MZ'), fontsize=24, y=1.02)

        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf
    except Exception as e:
        print(f"An error occurred in generate_surface_temp_map: {e}")
        return None

def plot_background(ax):
    # Add geographical features
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

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
    logo_pad = 0.25  # 0.25 inches
    fig_width, fig_height = fig.get_size_inches()
    logo_size_axes = logo_size / fig_width
    logo_pad_axes = logo_pad / fig_width

    for i, logo_path in enumerate(logo_paths):
        logo_img = plt.imread(logo_path)
        imagebox = OffsetImage(logo_img, zoom=logo_size / fig_width)
        ab = AnnotationBbox(imagebox, (1 - logo_pad_axes if i == 0 else logo_pad_axes, 1 - logo_pad / fig_height),
                            xycoords='figure fraction',  # Changed to figure fraction!
                            box_alignment=(1 if i == 0 else 0, 1),
                            frameon=False)
        ax.add_artist(ab)

    # Plot the data
    if levels is None:
        cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs)
    else:
        cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs, levels=levels)
    c = ax.contour(lon_2d, lat_2d, heights_smooth, colors='black', linewidths=2.5, transform=crs)
    ax.clabel(c, fontsize=8, inline=1, fmt='%i')
    ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind[::5, ::5], v_wind[::5, ::5], transform=crs, length=6)
    ax.set_title(title, fontsize=16)
    cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
    cb.set_label(cb_label, size='large')

    # Adjust layout and add figure title
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
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
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(ds, run_date, 30000, 'wind_speed', 'YlGnBu', '300-hPa Wind Speeds and Heights', 'Wind Speed (knots)'))
        await ctx.send(file=discord.File(fp=image_bytes, filename='wind300_{run_date.strftime("%HZ".png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

@bot.command()
async def vort500(ctx):
    await ctx.send('Generating 500 hPa vorticity map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        ds, run_date = get_gfs_data_for_level(50000)
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(ds, run_date, 50000, 'vorticity', 'seismic', '500-hPa Absolute Vorticity and Heights', r'Vorticity ($10^{-5}$ s$^{-1}$)', levels=np.linspace(-20, 20, 41)))
        await ctx.send(file=discord.File(fp=image_bytes, filename='vort500_{run_date.strftime("%HZ").png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

@bot.command()
async def rh700(ctx):
    await ctx.send('Generating 700 hPa relative humidity map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        ds, run_date = get_gfs_data_for_level(70000)
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(ds, run_date, 70000, 'relative_humidity', 'BuGn', '700-hPa Relative Humidity and Heights', 'Relative Humidity (%)'))
        await ctx.send(file=discord.File(fp=image_bytes, filename='rh700_{run_date.strftime("%HZ").png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

@bot.command()
async def wind850(ctx):
    await ctx.send('Generating 850 hPa wind map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        ds, run_date = get_gfs_data_for_level(85000)
        image_bytes = await loop.run_in_executor(None, lambda: generate_map(ds, run_date, 85000, 'wind_speed', 'PuRd', '850-hPa Wind Speeds and Heights', 'Wind Speed (knots)'))
        await ctx.send(file=discord.File(fp=image_bytes, filename='wind850_{run_date.strftime("%HZ").png'))
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')

@bot.command()
async def surfaceTemp(ctx):
    await ctx.send('Generating surface temperature map, please wait...')
    loop = asyncio.get_event_loop()
    try:
        image_bytes = await loop.run_in_executor(None, generate_surface_temp_map)
        if image_bytes:
            await ctx.send(file=discord.File(fp=image_bytes, filename='surfaceTemp_{run_date.strftime("%HZ").png'))
        else:
            await ctx.send('Failed to generate surface temperature map.')
    except Exception as e:
        await ctx.send(f'An error occurred: {e}')
