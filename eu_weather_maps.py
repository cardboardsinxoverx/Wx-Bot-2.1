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

# Intents are required for Discord.py 1.5+
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='$', intents=intents)

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
                    'Temperature_surface')
    query.vertical_level([level])
    query.lonlat_box(north=75, south=25, east=70, west=-30)

    try:
        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
        ds = ds.metpy.parse_cf()
        return ds, run_date
    except Exception as e:
        print(f"Error fetching data for level {level}: {e}")
        return None, None

# Helper function to fetch GFS surface data
def get_gfs_surface_data():
    """
    Fetches GFS surface data including surface temperature, pressure, and 10m wind components.

    Returns:
        tuple: (dataset, run_date)
            - dataset: Xarray dataset with surface variables.
            - run_date: Datetime object representing the model run time.
    """
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
    query.variables('Temperature_surface', 'Pressure_surface',
                    'u-component_of_wind_height_above_ground', 'v-component_of_wind_height_above_ground')
    query.lonlat_box(north=75, south=25, east=70, west=-30)

    try:
        data = ncss.get_data(query)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
        ds = ds.metpy.parse_cf()
        return ds, run_date
    except Exception as e:
        print(f"Error fetching surface data: {e}")
        return None, None

# Helper function to add map features
def plot_background(ax):
    """
    Adds background features to the map axes.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to which features are added.
    """
    ax.set_extent([-30, 70, 25, 75], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidths=2.5, edgecolor='#750b7a')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', linewidths=2, edgecolor='#750b7a')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

# Helper function to generate maps for isobaric levels
def generate_map(ds, run_date, level, variable, cmap, title, cb_label, levels=None):
    """
    Generates a map for a specified isobaric level and variable.

    Parameters:
        ds (xarray.Dataset): The dataset containing the data.
        run_date (datetime): The model run date.
        level (int): The pressure level in Pa.
        variable (str): The variable to plot ('wind_speed', 'vorticity', 'relative_humidity').
        cmap (str): The colormap to use.
        title (str): The title of the map.
        cb_label (str): The colorbar label.
        levels (array-like, optional): The contour levels for the variable.

    Returns:
        io.BytesIO: A BytesIO object containing the map image, or None if an error occurs.
    """
    try:
        # Access longitude and latitude directly from the dataset
        lon = ds.get('longitude')
        lat = ds.get('latitude')

        if lon is None or lat is None:
            raise ValueError("Longitude or latitude data is missing from the dataset.")

        lon = lon.values
        lat = lat.values

        # Create 2D grids of latitude and longitude
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        # Extract data at the specific level
        if 'Geopotential_height_isobaric' in ds:
            heights = ds['Geopotential_height_isobaric'].sel(isobaric=level, method='nearest').squeeze().values.copy()
        else:
            raise ValueError("Geopotential height data is missing from the dataset.")

        if 'u-component_of_wind_isobaric' in ds and 'v-component_of_wind_isobaric' in ds:
            u_wind = ds['u-component_of_wind_isobaric'].sel(isobaric=level, method='nearest').squeeze().copy()
            v_wind = ds['v-component_of_wind_isobaric'].sel(isobaric=level, method='nearest').squeeze().copy()
        else:
            raise ValueError("Wind data (u or v component) is missing from the dataset.")

        # Smooth the height data
        heights_smooth = ndimage.gaussian_filter(heights, sigma=3, order=0)

        # Calculate the variable to plot
        if variable == 'wind_speed':
            data = mpcalc.wind_speed(u_wind.metpy.quantify(), v_wind.metpy.quantify()).metpy.dequantify()
            data = data.metpy.convert_units('knots')
        elif variable == 'vorticity':
            dx, dy = mpcalc.lat_lon_grid_deltas(lon_2d, lat_2d)
            data = mpcalc.vorticity(u_wind.metpy.quantify(), v_wind.metpy.quantify(), dx=dx, dy=dy, latitude=lat_2d * units.degrees)
            data = data * 1e5  # Scale vorticity for visualization
        elif variable == 'relative_humidity':
            if 'Relative_humidity_isobaric' in ds:
                data = ds['Relative_humidity_isobaric'].sel(isobaric=level, method='nearest').squeeze().copy()
            else:
                raise ValueError("Relative humidity data is missing from the dataset.")
        else:
            raise ValueError('Unsupported variable type')

        # Define the map projection
        crs = ccrs.PlateCarree()

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': crs})

        # Set figure background color
        fig.patch.set_facecolor('lightsteelblue')

        # Plot the data
        if levels is None:
            cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs)
        else:
            cf = ax.contourf(lon_2d, lat_2d, data, cmap=cmap, transform=crs, levels=levels)

        # Plot contour lines for heights
        c = ax.contour(lon_2d, lat_2d, heights_smooth, colors='black', linewidths=2, transform=crs)
        ax.clabel(c, fontsize=8, inline=1, fmt='%i')

        # Plot wind barbs
        ax.barbs(lon_2d[::5, ::5], lat_2d[::5, ::5], u_wind[::5, ::5], v_wind[::5, ::5], transform=crs, length=6)

        # Set plot title
        ax.set_title(f"{title} {run_date.strftime('%d %B %Y %H:%MZ')}", fontsize=16)

        # Add colorbar
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=1.0, pad=0.03)
        cb.set_label(cb_label, size='large')

        # Add map features
        plot_background(ax)

        # Add logos
        logo_paths = [
            "/home/evanl/Documents/uga_logo.png",
            "/home/evanl/Documents/boxlogo2.png"
        ]
        add_logos_to_figure(fig, logo_paths, logo_size=1.0, logo_pad=0.2)

        # Adjust layout
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)

        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except ValueError as e:
        print(f"Data extraction error in generate_map: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in generate_map: {e}")
        return None

def generate_surface_temp_map():
    """
    Generates a surface temperature map with temperature contours, surface pressure isobars,
    and 10m wind barbs.

    Returns:
        tuple: (image_bytes, run_date)
            - image_bytes: BytesIO object containing the map image.
            - run_date: Datetime object representing the model run time.
    """
    try:
        # Get surface dataset
        ds, run_date = get_gfs_surface_data()
        if ds is None:
            raise ValueError("Failed to retrieve surface data.")
        print("Dataset variables:", ds.data_vars)
        print("Dataset dimensions:", ds.dims)

        # Access longitude and latitude directly from the dataset
        lon = ds.get('longitude')
        lat = ds.get('latitude')
        if lon is None or lat is None:
            raise ValueError("Longitude or latitude data is missing from the dataset.")
        lon = lon.values
        lat = lat.values
        print("lon shape:", lon.shape, "lat shape:", lat.shape)

        # Create 2D grids of latitude and longitude as plain numpy arrays
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        print("lon_2d shape:", lon_2d.shape, "lat_2d shape:", lat_2d.shape)

        # Extract surface temperature
        temp_surface = ds.get('Temperature_surface')
        if temp_surface is None:
            raise ValueError("Surface temperature data is missing from the dataset.")
        temp_surface = temp_surface.isel(time=0).squeeze().copy()
        print("temp_surface shape:", temp_surface.shape)

        # Extract surface pressure
        pressure_surface = ds.get('Pressure_surface')
        if pressure_surface is None:
            raise ValueError("Surface pressure data is missing from the dataset.")
        pressure_surface = pressure_surface.isel(time=0).squeeze().copy() / 100  # Convert to hPa
        print("pressure_surface shape:", pressure_surface.shape)

        # Extract wind components at 10m
        u_wind = ds.get('u-component_of_wind_height_above_ground')
        v_wind = ds.get('v-component_of_wind_height_above_ground')
        if u_wind is None or v_wind is None:
            raise ValueError("Surface wind data is missing from the dataset.")

        if 'height_above_ground2' in u_wind.dims:
            u_wind = u_wind.sel(height_above_ground2=10, method='nearest').isel(time=0).squeeze().metpy.dequantify().copy()
        else:
            u_wind = u_wind.isel(time=0).squeeze().metpy.dequantify().copy()
        print("u_wind shape:", u_wind.shape)

        if 'height_above_ground2' in v_wind.dims:
            v_wind = v_wind.sel(height_above_ground2=10, method='nearest').isel(time=0).squeeze().metpy.dequantify().copy()
        else:
            v_wind = v_wind.isel(time=0).squeeze().metpy.dequantify().copy()
        print("v_wind shape:", v_wind.shape)

        # Convert temperature to Fahrenheit and ensure plain numpy array
        temp_surface = temp_surface.metpy.convert_units('degF').metpy.dequantify()
        print("temp_surface shape after dequantify:", temp_surface.shape)
        print("temp_surface type:", type(temp_surface))

        # Verify all inputs to contourf
        print("lon_2d type:", type(lon_2d), "lat_2d type:", type(lat_2d))
        print("Plotting temperature contourf...")

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        fig.patch.set_facecolor('lightsteelblue')

        # Plot Surface Temperatures
        cf1 = ax.contourf(lon_2d, lat_2d, temp_surface, cmap='jet',
                          transform=ccrs.PlateCarree(), levels=np.linspace(temp_surface.min(), temp_surface.max(), 10))
        print("Plotting temperature contour...")
        c1 = ax.contour(lon_2d, lat_2d, temp_surface, colors='#2e2300', linewidths=2, linestyles='dashed',
                        transform=ccrs.PlateCarree())
        ax.clabel(c1, fontsize=12, inline=1, fmt='%.2f°F')
        ax.set_title(f"Surface Temperatures (°F) {run_date.strftime('%d %B %Y %H:%MZ')}", fontsize=16)
        cb1 = fig.colorbar(cf1, ax=ax, orientation='horizontal', shrink=1.0, pad=0.05, extend='both')
        cb1.set_label('Temperature (°F)', size='large')

        # Plot Isobars (Pressure Contours)
        print("Plotting pressure isobars...")
        levels = np.arange(980, 1050, 2)
        isobars = ax.contour(lon_2d, lat_2d, pressure_surface, levels=levels,
                             colors='black', linestyles='-', linewidths=2.5, transform=ccrs.PlateCarree())
        ax.clabel(isobars, fontsize=12, inline=1, fmt='%.1f hPa')

        # Plot Wind Barbs with reduced density
        skip = (slice(None, None, 5), slice(None, None, 5))
        print("lon_2d[skip] shape:", lon_2d[skip].shape)
        print("lat_2d[skip] shape:", lat_2d[skip].shape)
        print("u_wind[skip] shape:", u_wind[skip].shape)
        print("v_wind[skip] shape:", v_wind[skip].shape)
        print("Plotting wind barbs...")
        ax.barbs(lon_2d[skip], lat_2d[skip], u_wind[skip], v_wind[skip],
                 length=6, transform=ccrs.PlateCarree(), pivot='middle', barbcolor='#3f0345')

        # Add map features
        plot_background(ax)

        # Add logos
        logo_paths = ["/home/evanl/Documents/uga_logo.png", "/home/evanl/Documents/boxlogo2.png"]
        add_logos_to_figure(fig, logo_paths, logo_size=1.0, logo_pad=0.2)

        # Adjust layout
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)

        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf, run_date

    except ValueError as e:
        print(f"Data extraction error in generate_surface_temp_map: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred in generate_surface_temp_map: {e}")
        return None, None

# Helper function to add logos to the figure
def add_logos_to_figure(fig, logo_paths, logo_size=1.0, logo_pad=0.2):
    """
    Adds logos to the figure at the top-left and top-right positions with a specified size and padding.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to which logos are added.
        logo_paths (list): List of file paths to the logos.
        logo_size (float): Size of the logos in inches.
        logo_pad (float): Padding from the edges in inches.
    """
    # Get figure dimensions in inches
    fig_width, fig_height = fig.get_size_inches()

    # Calculate the size of the logo in figure fraction
    logo_width = logo_size / fig_width
    logo_height = logo_size / fig_height

    # Calculate padding in figure fraction
    pad_width = logo_pad / fig_width
    pad_height = logo_pad / fig_height

    # Positions for the logos
    positions = [
        {
            'left': pad_width,                   # Left logo (top left)
            'bottom': 1 - pad_height - logo_height,
            'path': logo_paths[0]
        },
        {
            'left': 1 - pad_width - logo_width,  # Right logo (top right)
            'bottom': 1 - pad_height - logo_height,
            'path': logo_paths[1]
        }
    ]

    # Adding logos to the plot
    for pos in positions:
        try:
            # Create an inset axes at the specified position and size
            ax_logo = fig.add_axes([pos['left'], pos['bottom'], logo_width, logo_height])
            ax_logo.axis('off')  # Hide the axes frame and ticks

            # Read and display the logo image
            logo_img = plt.imread(pos['path'])
            ax_logo.imshow(logo_img)
        except FileNotFoundError:
            print(f"Logo file not found: {pos['path']}")
        except Exception as e:
            print(f"Error adding logo {pos['path']}: {e}")

# Discord bot commands
@bot.command()
async def eu_wind300(ctx):
    await ctx.send('Generating 300 hPa wind map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        ds, run_date = get_gfs_data_for_level(30000)
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
    except Exception as e:
        await ctx.send(f'An unexpected error occurred while generating the 300 hPa wind map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_vort500(ctx):
    await ctx.send('Generating 500 hPa vorticity map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        ds, run_date = get_gfs_data_for_level(50000)
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
    except Exception as e:
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
        ds, run_date = get_gfs_data_for_level(70000)
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
    except Exception as e:
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
        ds, run_date = get_gfs_data_for_level(85000)
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
    except Exception as e:
        await ctx.send(f'An unexpected error occurred while generating the 850 hPa wind map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()

@bot.command()
async def eu_surfaceTemp(ctx):
    await ctx.send('Generating surface temperature map, please wait...')
    loop = asyncio.get_event_loop()
    image_bytes = None
    try:
        image_bytes, run_date = await loop.run_in_executor(None, generate_surface_temp_map)
        if image_bytes is None or run_date is None:
            await ctx.send('Failed to generate the surface temperature map due to missing or invalid data.')
            return
        filename = f'eu_surfaceTemp_{run_date.strftime("%HZ")}.png'
        await ctx.send(file=discord.File(fp=image_bytes, filename=filename))
    except Exception as e:
        await ctx.send(f'An unexpected error occurred while generating the surface temperature map: {e}')
    finally:
        if image_bytes:
            image_bytes.close()
