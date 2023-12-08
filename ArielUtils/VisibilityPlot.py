import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from ArielUtils.Functions import *

def plot_sky_view(data:pd.DataFrame, 
                  title:str,
                  plot_ecliptic:bool=True, 
                  ecliptic_color:str or tuple='grey',
                  cmap:str="hot", 
                  save_path:str="Visibility Plots/",
                  savefig:bool=True, 
                  long_pts:int=360, 
                  dpi:int=300, 
                  size:int=1, 
                  marker:str='o', 
                  label:str='Stars', 
                  marker_color:str='blue', 
                  projection:str='aitoff',
                  fancybox:bool=False,
                  loc:str or tuple=(0.76, 0.94),
                  alpha:str=.8,
                  edgecolor:str='0.8',
                  style:str='default',
                  transparent=False):
    
    # setting style
    plt.style.use(style)
    
    # preparing latitude and longitude points
    lat_pts = long_pts//2

    # coordinates
    ra = np.linspace(-pi, pi, long_pts)
    dec = np.linspace(-pi/2, pi/2, lat_pts)
    
    # preparing ecliptic
    sun_path = pd.DataFrame(np.array([[80, 172, 264, 355], 
                                      [0, pi/2, pi, -pi/2], 
                                      [0, np.radians(23.5), 0, np.radians(-23.5)]]).transpose(), 
                            columns=['Day of year', 'Ra', 'Dec'])
    ecliptic = sun_path['Dec'][1]*np.sin(ra)
    if plot_ecliptic:
        eclip = "_withEcliptic"
    else:
        eclip = ""

    # preparing data for the pcolormesh to be plotted
    Lon, Lat = np.meshgrid(ra, dec)

    # density
    if long_pts == 360:
        rho = np.genfromtxt('ArielUtils/VisibilityDensity_360.csv', delimiter=',')
    elif long_pts == 720:
        rho = np.genfromtxt('ArielUtils/VisibilityDensity_720.csv', delimiter=',')
    else:
        rho = np.zeros(np.shape(Lon)) # array that will contain all the values for the pcolormesh (higher value = more visible throughout year)
        for n in range(len(ra)):
            for i, row in enumerate(dec):
                for j, col in enumerate(ra):
                    if dist_angle(row,col, ecliptic[n],ra[n]) > np.radians(MIN_ANGLE_TOWARDS_SUN) and dist_angle(row, col, ecliptic[n]+pi, ra[n]) > np.radians(MIN_ANGLE_AWAY_SUN):
                        rho[i][j] += 1
        rho /= long_pts # normalizes values such that they are all between 0 and 1. 

    # Preparing the plot
    fig = plt.figure(figsize=(10,5))
    plt.subplot(projection = projection)
    plt.grid(True, alpha=.6)
    
    # plotting the pcolormesh and its color bar
    plt.pcolormesh(Lon, Lat, rho, cmap=cmap)
    plt.colorbar(label="Fraction of the year where\na given location is visible")

    # plotting ecliptic path
    if plot_ecliptic:
        plt.plot(ra, ecliptic, label='Ecliptic', color=ecliptic_color, linewidth=1) # plots ecliptic
        
    
    # If data is empty, only plots background
    if data is not None:
        # preparing data
        starRA = np.radians(data['Star RA'])
        starRA = starRA.apply(lambda x: x-2*pi if x>pi else x) # ensures that RA values are between -pi and pi.
        starDEC = np.radians(data['Star Dec'])
        # plotting data
        plt.scatter(starRA, starDEC, s=size, marker=marker, label=label, color=marker_color) # plots the stars of interest

    plt.xticks([-pi/2, 0, pi/2], ['18h', '0h', '6h'], color='white')
    plt.yticks([-pi/3, -pi/6, 0, pi/6, pi/3])

    # adding title and labels
    plt.title(title, pad=20)
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.legend(loc=loc, fancybox=fancybox, edgecolor=edgecolor, framealpha=alpha)

    # saving figure to the "Visibility Plots" folder
    if savefig:
        if style == 'default':
            plt.savefig(f"{save_path}{title}{eclip}.png", bbox_inches='tight', dpi=dpi, transparent=transparent)
        else:
            plt.savefig(f"{save_path}{title}{eclip}{style}.png", bbox_inches='tight', dpi=dpi, transparent=transparent)
    plt.show()