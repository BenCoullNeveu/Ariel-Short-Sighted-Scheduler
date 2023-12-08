# Functions for ARIEL scheduling and visualizations.

# # # importing useful libraries # # #
from numpy import arccos, sin, cos, pi
import pandas as pd

from astropy.time import Time
from astropy.coordinates import get_sun

from ArielUtils.Constants import *
# # # # # #


# # # FUNCTIONS # # #

# Distance between two points on the celestial sphere.
def dist_angle(dec1:float, 
               ra1:float, 
               dec2:float, 
               ra2:float)->float:
    """Returns the distance (in radians) between two points on the celestial sphere.

    Args:
        dec1 (float): Declination of first point (in radians).
        ra1 (float): Right ascention of first point (in radians).
        dec2 (float): Declination of second point (in radians).
        ra2 (float): Right ascention of second point (in radians).

    Returns:
        float: Distance between two given points on the celestial sphere, in radians.
    """
    ra_angle = cos(ra2 - ra1)
    return arccos( 0.5 * (cos(dec2-dec1)*(ra_angle+1) + cos(dec2 + dec1)*(ra_angle-1) ))


# Finds every visibe target at a given time
def find_visible_targets(targets:pd.DataFrame, 
                         time:Time=Time.now(), 
                         sun_angle:float=MIN_ANGLE_TOWARDS_SUN, 
                         opp_sun_angle:float=MIN_ANGLE_AWAY_SUN,
                         check_method:bool=False,
                         tier:int=None,
                         continue_if_complete=False)->pd.DataFrame:
    """Returns a data frame of all the targets that are visible at a given time. 

    Args:
        targets (pd.DataFrame): Data frame containing the targets.
        time (Time, optional): given time, as Time object. Defaults to time.now().
        sun_angle (float, optional): Limiting angle towards Sun, in degrees. Defaults to MIN_ANGLE_TOWARDS_SUN.
        opp_sun_angle (float, optional): Limiting angle away from Sun, in degrees. Defaults to MIN_ANGLE_AWAY_SUN.
        check_method (bool, optional): Check preferred observation method. Defaults to False.
        tier (int, optional): Desired tier for objects. If None, default to only one observation. Defaults to None.
        continue_if_complete (bool, optional): If True, the scheduler will continue to observe targets that have already been completed (i.e., SNR >= 7). Defaults to False.

    Returns:
        pd.DataFrame: Data frame containing all the targets visible at a given time. 
    """
    # initializing the list of visible targets and the Sun's coordinates at the given time
    viewable_targets = []
    sunRA = np.radians(get_sun(time).ra.value) # Sun's right ascension, in radians
    sunDEC = np.radians(get_sun(time).dec.value) # Sun's declination, in radians
    
    # iterating through the targets data frame. _ just means that the index will not be used in the loop
    for _, target in targets.iterrows():
        distance = dist_angle(np.radians(target['Star Dec']), np.radians(target['Star RA']), sunDEC, sunRA)
        
        if tier is None:
            if target['Transits Observed'] >= 1 or target['Eclipses Observed'] >= 1:
                continue
        
        # checking preferred method
        if check_method:
            # for transits
            if target['Preferred Method'] == 'Transit':
                if not continue_if_complete and target['Transits Observed'] >= target[f"Tier {tier} Transits"]:
                    continue
            
            # for eclipses
            elif target['Preferred Method'] == 'Eclipse':
                if not continue_if_complete and target['Eclipses Observed'] >= target[f"Tier {tier} Eclipses"]:
                    continue
                        
            # for other
            else:
                raise Exception("Check that the preferred method for every target is either 'Transit' or 'Eclipse'.")
                
        # Finding if target is in viewable area
        if distance >= np.radians(sun_angle) and distance <= pi - np.radians(opp_sun_angle): # checks if a target is visible at the given time
            viewable_targets.append(target) # adding target to list since it is visible at the given time
            
    return pd.DataFrame(viewable_targets)



# Finds the closest target to the current pointing position.
def find_closest_target(targets:pd.DataFrame, 
                        currentRA:float=0, 
                        currentDEC:float=0)->pd.DataFrame:
    """Returns a data frame containing the nearest target.

    Args:
        targets (pd.DataFrame): Data frame containing targets
        currentRA (float, optional): Current right ascension. Defaults to 0.
        currentDEC (float, optional): Current declination. Defaults to 0.

    Returns:
        pd.DataFrame: Data frame containing the target nearest current location
    """
    # initializing necessary variables
    closest_target = None 
    closest_distance = None
    
    # iterating through target data frame. _ just means that the index will not be used in the loop.
    for _, target in targets.iterrows():
        distance = dist_angle(np.radians(currentDEC), np.radians(currentRA), np.radians(target['Star Dec']), np.radians(target['Star RA']))
        if closest_target is None or distance < closest_distance: # checking if the target is closer than the previously closest target.
            closest_target = target
            closest_distance = distance
    return pd.DataFrame(closest_target).transpose()



# Finds the total slewtime, given a distance to cover.
def find_slewtime_minutes(dec1:float, 
                          ra1:float, 
                          dec2:float, 
                          ra2:float, 
                          slewrate:float=SLEWRATE)->float:
    """Returns the slewtime in minutes given the current position and the final/target position.

    Args:
        dec1 (float): initial declination, in radians.
        ra1 (float): initial right ascension, in radians.
        dec2 (float): target declination, in radians.
        ra2 (float): target right ascension, in radians.
        slewrate (float, optional): telescope slew rate. Defaults to SLEWRATE.

    Returns:
        float: slew time, in minutes.
    """
    distance = dist_angle(dec1, ra1, dec2, ra2)
    return (distance / slewrate)



# Checks if calibration must be done
def check_calibration(time_elapsed:float, frequency:float, error:float, time_of_previous_calibration:float)->bool:
    """Returns whether calibration is due. 

    Args:
        time_elapsed (float): Time elapsed since beginning of mission.
        frequency (float): Frequency of calibrations.
        error (float): The range within which calibration should be done (frequency +- error).
        time_of_previous_calibration (float): time of previous calibration.

    Returns:
        bool: True if calibration is due, False if not. 
    """
    if (time_elapsed - time_of_previous_calibration) <= frequency-error:
        return False
    diff = time_elapsed % frequency
    if diff <= error or frequency-diff <= error:
        return True
    return False


# Fitness function
def fitness(distance:float,
            orbital_period:float, 
            time_till_event:float, 
            quality_metric:float=1, 
            slewrate:float=SLEWRATE, 
            settle_time:float=SETTLE_TIME, 
            baseline_duration:float=BASELINE_DURATION,
            SNR:float=-1):
    """Returns the fitness value a target given parameters, as well as the estimated time until the telescope has to move to observe the next event. 
       Note that units are not compensated for, so be sure to use the same units for consistency.

    Args:
        distance (float): Distance between current pointing position & target location. 
        event_duration (float): Event duration.
        orbital_period (float): Orbital period.
        time_till_event (float): Time until the next event.
        quality_metric (float, optional): The quality metric of the target. Left vague, since it is metric-independent, it just uses it for a relative value. Defaults to 1.
        slewrate (float, optional): Slewrate of telescope. Defaults to SLEWRATE.
        settle_time (float, optional): Settle time of telescope. Defaults to SETTLE_TIME.
        baseline_duration (float, optional): Baseline duration. Defaults to BASELINE_RATIO.
        SNR (float, optional): Signal-to-Noise ratio. If set to -1, then the SNR-independent fitness function will be applied. Defaults to -1.

    Returns:
        (float, float): Fitness, wait time (in minutes). 
    """
    wait_time = time_till_event - (distance/slewrate + settle_time + baseline_duration/2)
    if wait_time < 0:
        F = 0 # setting fitness to 0 if wait time is negative
    elif SNR == -1:
        F = orbital_period * quality_metric / (wait_time+1)
    else:
        F = orbital_period * quality_metric / (wait_time+1) * (SNR + 1) * np.e**((7 - SNR)/2)
    return F, wait_time


# Sorts targets by fitness
def sort_by_fitness(targets:pd.DataFrame, 
                    time:Time=Time.now(),
                    currentRA:float=0, 
                    currentDEC:float=0,
                    ETSM:str=None,
                    ATSM:str=None,
                    check_method:bool=False,
                    SNRfitness:bool=True)->pd.DataFrame:
    """Returns a data frame containing all the input targets, but sorted in order of fitness.

    Args:
        targets (pd.DataFrame): Data frame of targets-of-interest.
        time (Time, optional): Time of interest. Defaults to Time.now().
        currentRA (float, optional): Current right-ascension. Defaults to 0.
        currentDEC (float, optional): Current declination. Defaults to 0.
        ETSM (str, optional): Column name for emission quality metric for Ariel. Defaults to None.
        ATSM (str, optional): Column name for transit quailty metric for Ariel. Defaults to None.
        SNRfitness (bool, optional): If True, SNR-dependant fitness function will be used. If False, SNR-independant fitness function will be used. Defaults to True.

    Returns:
        pd.DataFrame: Data frame of all the targets in order of their fitness.
    """
    targets_copy = targets.copy(deep=True)
    time.format = 'mjd'
    for index, target in targets_copy.iterrows():
        dist = dist_angle(currentDEC, currentRA, target['Star Dec'], target['Star RA'])
        period = target['Planet Period [days]']
        
        # checking method
        if check_method:
            if target['Preferred Method'] == 'Transit':
                t_diff = time.value - target['Transit Reference [MJD]']
                if ATSM is None:
                   q_metric = 1
                else: 
                    q_metric = target[ATSM]
                    
            elif target['Preferred Method'] == 'Eclipse':
                t_diff = time.value - target['Eclipse Reference [MJD]']
                if ETSM is None:
                    q_metric = 1
                else:
                    q_metric = target[ETSM]
        else:
            t_diff = time.value - target['Transit Reference [MJD]'] # defaults to transit
            q_metric = target[ATSM]
            
        # checks if SNR-dependant fitness is to be used or not
        if SNRfitness:
            SNR = target['SNR']
        else:
            SNR = -1 # -1 for SNR-independant fitness function
            
        time_till_transit = (period - t_diff) % period
        fit_val, wait_time = fitness(dist, 
                                     period*24*60, 
                                     time_till_transit*24*60,
                                     quality_metric = q_metric,
                                     SNR = SNR
                                     )
        targets_copy.at[index, 'Fitness'] = fit_val
        targets_copy.at[index, 'Wait Time [min]'] = wait_time
    targets_copy.sort_values(by=['Fitness'], ascending=False, inplace=True)
    return targets_copy

def estimate_snr(n_obs:float, n_tier:float):
    return 7 * np.sqrt(n_obs / n_tier)
# # # # # #