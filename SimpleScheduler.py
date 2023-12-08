import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from numpy import pi, sin, cos, arccos
from astropy.coordinates import get_sun
from astropy.time import Time


import warnings
from astropy.utils.exceptions import ErfaWarning
warnings.simplefilter('ignore', category=ErfaWarning)

from ArielUtils.Constants import *
from ArielUtils.Functions import *
from ArielUtils.VisibilityPlot import plot_sky_view

import seaborn as sns
import matplotlib.pyplot as plt

from tkinter import filedialog, Tk



########## SCHEDULER FUNCTION ############

def SimpleScheduler(filename=None, initial_MJD_time=62137, commission_time=COMMISSION_TIME, initial_ra=0, initial_dec=0, 
                    tier=None, check_method=True, continue_if_complete=False,
                    print_results=True, pause=False, 
                    plot_results=False, show_plot=False,
                    animate_period_vs_wait=False, show_animation=False, filetype='.mp4', anim_dpi=300,
                    AESM='ESM', ATSM='ASM',
                    SNRfitness=True):
    """Simple scheduler.

    Args:
        filename (string, optional): filename (including path, if necessary). If None, file dialog will open. Defaults to None.
        initial_MJD_time (int, optional): Initial date in MJD. Defaults to 62137, which is January 1st 2029 in modified julian date format (MJD).
        commision_time (float, optional): Time dedicated for commissioning at start of mission. Default is 6*30*24*60, or 6 months (leaving 3.5 years for the mission itself).
        initial_ra (int, optional): initial right ascension, in radians. Defaults to 0.
        initial_dec (int, optional): initial declination, in radians. Defaults to 0.
        tier (int, optional): desired tier for targets. If None, all targets will be viewed once. Defaults to None.
        check_method (bool, optional): If True, function will observe events specified under 'Preferred Method' (i.e., eclipse or transit). If False, it will only observe transiting events. Defaults to True.
        continue_if_complete (bool, optional): If True, the scheduler will continue to observe targets that have already been completed (i.e., SNR >= 7). Defaults to False.
        print_results (bool, optional): Print out basic results. Defaults to True
        pause (bool, optional): Wait for user input to continue after printing results (only useful if print_results is set to True). Defaults to False.
        ETSM (str, optional): Column name for emission metric. Defaults to 'ESM'.
        ATSM (str, optional): Column name for transit metric. Defaults to 'ASM'.
        SNRfitness (bool, optional): If True, SNR-dependant fitness function will be used. If False, SNR-independant fitness function will be used. Defaults to True.

    Returns:
        tuple: Returns a tuple containing targets, event_timings, fitness_anim_data, idle_time, observing_time, slew_time, short_calibration_time, long_calibration_time.
    """

    if tier is None:
        print("WARNING: with tier set to NONE, no SNR estimates will be made.")
    
    # getting filename from dialog
    if filename is None:
        Tk().withdraw()
        filename = filedialog.askopenfilename()
        
    # importing target list as dataframe
    df = pd.read_csv(filename)

    # Preparing target dataframe
    # targets = df[['Star Name', 'Star RA', 'Star Dec', 'Planet Period [days]', 'Transit Duration [s]', 'Transit Mid Time [JD - 2450000]', 'Eclipse Mid Time [JD - 2450000]', 'Eclipse Duration [s]', 
    #             'Tier 1 Transits', 'Tier 2 Transits', 'Tier 3 Transits', 'Tier 1 Eclipses', 'Tier 2 Eclipses',	'Tier 3 Eclipses',
    #             'Preferred Method', 'Tier 1 Observations', 'Tier 2 Observations', 'Tier 3 Observations']]
    targets = df.copy(deep=True)
    
    # setting star name as index
    targets.set_index(['Star Name'], inplace=True)

    # converting transit reference times from JD to MJD
    targets['Transit Mid Time [JD - 2450000]'] = targets['Transit Mid Time [JD - 2450000]'].apply(lambda x: Time(x+2450000, format='jd').mjd)
    targets['Transit Mid Time [JD - 2450000]'] = targets['Transit Mid Time [JD - 2450000]'] - targets['Transit Duration [s]']/60/60/24
    targets.rename(columns={'Transit Mid Time [JD - 2450000]': 'Transit Reference [MJD]'}, inplace=True)

    # converting eclipse reference times from JD to MJD 
    targets['Eclipse Mid Time [JD - 2450000]'] = targets['Eclipse Mid Time [JD - 2450000]'].apply(lambda x: Time(x+2450000, format='jd').mjd)
    targets['Eclipse Mid Time [JD - 2450000]'] = targets['Eclipse Mid Time [JD - 2450000]'] - targets['Eclipse Duration [s]']/60/60/24
    targets.rename(columns={'Eclipse Mid Time [JD - 2450000]': 'Eclipse Reference [MJD]'}, inplace=True)

    # initializing event counter
    targets["Transits Observed"] = 0
    targets["Eclipses Observed"] = 0
    
    # initializing target SNR
    targets["SNR"] = 0
    
    # initilizing target completness (based on SNR)
    targets['Completed'] = False

    # initial coordinates
    current_ra = initial_ra
    current_dec = initial_dec

    # Preparing all necessary variables and such
    previous_long_calib = 0
    previous_short_calib = 0
    time_elapsed = 0.
    observed_events = 0
    new_observations = 0
    completed_targets = 0
    event_timings = np.array([[0,0,0,0]], dtype=object)
    fitness_anim_data = np.array([[0,0,0,0,0,0]], dtype=object)
    
    # initializing progress bar
    progress = 0    
    
    idle_time = 0
    observing_time = 0 # including baseline
    slew_time = 0 # including settle time
    short_calibration_time = 0
    long_calibration_time = 0


    ################# THE SCHEDULE LOOP####################
    time_elapsed += commission_time # adding commission time to beginning of mission
    
    while time_elapsed < MISSION_DURATION:
        
        # adding to the event array for plotting
        event_timings = np.concatenate( (event_timings, np.array([[time_elapsed, observed_events, new_observations, completed_targets]])) )
        
        # drastically increases time elapsed rate if targets have all been observed once if tier is None
        if tier is None and observed_events == len(targets):
            time_elapsed += 24*60
            idle_time += 24*60
            
        # printing scheduler progress bar
        if progress > 6:
            progress = 0
        print(" "*50, end='\r')
        print(f"Time Elapsed: {round(time_elapsed/60/24)} days / {round(MISSION_DURATION/60/24)} days. " + "#"*round(progress), end='\r')
        progress += 0.3
        
        # calibration checks
        if check_calibration(time_elapsed, LONG_CALIBRATION_FREQUENCY, LONG_CALIBRATION_ERR, previous_long_calib):
            time_elapsed += LONG_CALIBRATION_DURATION
            long_calibration_time += LONG_CALIBRATION_DURATION
            previous_long_calib = previous_short_calib = time_elapsed
            
        elif check_calibration(time_elapsed, SHORT_CALIBRATION_FREQUENCY, SHORT_CALIBRATION_ERR, previous_short_calib):
            time_elapsed += SHORT_CALIBRATION_DURATION
            short_calibration_time += SHORT_CALIBRATION_DURATION
            previous_short_calib = time_elapsed
            
        # setting current time
        current_time = Time(initial_MJD_time + time_elapsed/(60*24), 
                            format='mjd')
        
        # finding currently visible targets
        currently_visible = find_visible_targets(targets = targets, 
                                                time = current_time,
                                                check_method=check_method,
                                                tier=tier,
                                                continue_if_complete=continue_if_complete
                                                )
        
        if currently_visible.empty:
            time_elapsed += 20
            idle_time += 20 # adding 20 minutes of idle time
            continue
        
        # sorting the visible targets in order of fitness
        fitest_targets = sort_by_fitness(currently_visible, time=current_time, 
                                         currentRA=current_ra, currentDEC=current_dec, 
                                         ETSM=AESM, ATSM=ATSM, 
                                         check_method=check_method, SNRfitness=SNRfitness)
        
        # adding to animation array
        fitness_anim_data = np.concatenate( (fitness_anim_data, np.array([[time_elapsed, 
                                                                        fitest_targets['Wait Time [min]'].to_numpy(),
                                                                        fitest_targets['Planet Period [days]'].to_numpy(), 
                                                                        fitest_targets['Fitness'].to_numpy(),
                                                                        targets.to_numpy(),
                                                                        list(targets.columns)]], 
                                                                        dtype=object)) )

        # getting fitest target
        fitest_target = fitest_targets.iloc[0]

        # increasing transit observation counter
        if fitest_target["Preferred Method"] == "Transit":
            if fitest_target['Transits Observed'] == 0:
                new_observations += 1 # adding to NEW observation
            fitest_target['Transits Observed'] += 1
            targets["Transits Observed"].loc[fitest_target.name] = fitest_target['Transits Observed']
            if tier is not None:
                snr = estimate_snr(fitest_target['Transits Observed'], fitest_target[f"Tier {tier} Transits"]) # updating SNR for transits
                targets['SNR'].loc[fitest_target.name] = snr
            
        # increasing eclipse observation counter
        elif fitest_target["Preferred Method"] == "Eclipse":
            if fitest_target['Eclipses Observed'] == 0:
                new_observations += 1
            fitest_target['Eclipses Observed'] += 1
            targets["Eclipses Observed"].loc[fitest_target.name] = fitest_target['Eclipses Observed']
            if tier is not None:
                snr = estimate_snr(fitest_target['Eclipses Observed'], fitest_target[f"Tier {tier} Eclipses"]) # updating SNR for eclipses
                targets['SNR'].loc[fitest_target.name] = snr
        
        # increase number of observed events
        observed_events += 1
        
        # check if target is completed (SNR >= 7), and update if necessary
        if tier is not None and snr >= 7 and not fitest_target['Completed']:
            completed_targets += 1
            targets['Completed'].loc[fitest_target.name] = True
        
        
        # getting coordinates of target
        target_dec, target_ra = fitest_target['Star Dec'], fitest_target['Star RA']
        slewtime = find_slewtime_minutes(dec1 = np.radians(current_dec), 
                                        ra1 = np.radians(current_ra), 
                                        dec2 = np.radians(target_dec), 
                                        ra2 = np.radians(target_ra)
                                        )
        
        
        # updates current position
        current_dec, current_ra = target_dec, target_ra 
        
        ## UPDATING TIMES ##
        # increases time by slewtime and average event duration (+ baseline)
        time_elapsed += slewtime + SETTLE_TIME
        slew_time += slewtime + SETTLE_TIME
        
        time_elapsed += fitest_target['Wait Time [min]']
        idle_time += fitest_target['Wait Time [min]']
        
        # increasing time based on event duration
        ## SECONDARY CONDITION IS ONLY NECESSARY SINCE THE TARGET LIST INCLUDES ECLIPSE DURATIONS OF 0. THIS ASSUME ECLIPSE DURATION ~~ TRANSIT DURATION FOR MOST TARGETS (BEST WE CAN DO FOR NOW) ##
        if fitest_target['Preferred Method'] == 'Transit' or fitest_target['Eclipse Duration [s]'] == 0: 
            time_elapsed += fitest_target['Transit Duration [s]']/60 + BASELINE_DURATION
            observing_time += fitest_target['Transit Duration [s]']/60 + BASELINE_DURATION
        else:
            time_elapsed += fitest_target['Eclipse Duration [s]']/60 + BASELINE_DURATION
            observing_time += fitest_target['Eclipse Duration [s]']/60 + BASELINE_DURATION
                
    event_timings = event_timings.transpose()
    fitness_anim_data = fitness_anim_data[1:].transpose()
    
    # printing basic results of scheduler, if desired
    if print_results:
        print(" "*50, end='\r')
        print('Scheduler Completed!')
        print(f'Tier: {tier}; Check method: {check_method}')
        print(' ==> Total number of targets in list: ', len(targets))
        print(' ==> Number of targets observed: ', new_observations)
        print(' ==> Total numnber of observed events: ' , observed_events)
        print(' ==> Number of completed targets: ', completed_targets)
        print('\n')
        if pause:
            input("Press Enter to continue...")
            print('\n')
            
        #### END OF SCHEDULE LOOP ####
        
        
# function that adds useful information to title and/or filename
    def add_to_title(title, tier=tier, check_method=check_method, continue_if_complete=continue_if_complete, SNRfitness=SNRfitness):
        if not check_method and tier is not None:
            title += " (Transits Only, Observed as Many Times Possible)"
        elif not check_method:
            title += " (Transits Only, Each Object Observed Once)"
        elif tier is None:
            title += " (Each Target Observed Once)"
            
        elif continue_if_complete and SNRfitness:
            title += f" (Observed to Tier {tier}, with no SNR Limit, using SNR-Dependant Fitness Function)"
        elif continue_if_complete and not SNRfitness:
            title += f" (Observed to Tier {tier}, with no SNR Limit, using SNR-Independant Fitness Function)"
        elif not continue_if_complete and SNRfitness:
            title += f" (Observed to Tier {tier}, with SNR Limited to ~7, using SNR-Dependant Fitness Function)"
        else:
            title += f" (Observed to Tier {tier}, with SNR Limited to ~7, using SNR-Independant Fitness Function)"
            
        return title

    ###### making plot of results, if desired ######
    if plot_results:
        fig, ax = plt.subplots(figsize=(8,5))

        ax.axhline(len(targets), linestyle='--', linewidth=1, label=f"Total Number of Targets ({len(targets)})", zorder=4, color='blue', alpha=.4)
        ax.plot(event_timings[0]/(60*24), event_timings[1], label=f'Observations ({observed_events} total)', zorder=3, color='black')
        if tier is not None:
            ax.plot(event_timings[0]/(60*24), event_timings[2], label=f'Targets Observed ({new_observations} total)', zorder=3, color='orange')
            ax.plot(event_timings[0]/(60*24), event_timings[3], label=f'Targets Completed ({completed_targets} total)', zorder=3, color='red')
        ax.set_xlabel("Time Elapsed [days]")
        ax.set_ylabel("Observations")
        fig.suptitle("Observations over Time", fontsize='15')
        ax.set_title(add_to_title(""), fontsize='11')
        ax.legend(fancybox=False)
        ax.grid(which='major', alpha=.5, zorder=0)
        ax.grid(which='minor', alpha=.2, linestyle=':', zorder=0)
        ax.minorticks_on()

        fname = add_to_title("Figures/Observations vs Time plots/Observation over Time")

        plt.savefig(fname + ".png", bbox_inches='tight', dpi=300)
        
        if show_plot:
            plt.show()
            
            
        ##### making period vs wait time animation ######
        if animate_period_vs_wait:
            from matplotlib.animation import FuncAnimation
            from PIL import Image

            # angle calculation
            def rotation_angle(timeElapsed):
                year = 365 * 24 * 60 # year in minutes
                fraction_year = timeElapsed / year
                return 360 * fraction_year # rotation angle


            fitness_anim_data_copy = np.copy(fitness_anim_data)
            max_time = fitness_anim_data_copy[0][-1]//60 # in hours

            # importing image
            raw_image = Image.open('Photos/ArielSunImage.png')
            img = raw_image.resize((1000, 1000))

            # preparing plot
            fig, ax = plt.subplots(figsize=(6,4))
            
            # labels
            ax.set_ylabel("Orbital period [days]")
            ax.set_xlabel("Wait time [min]")
            
            # title
            fig.suptitle("Orbital Period vs Wait Time until Next Event", fontsize = 12)
            title = add_to_title("")
            ax.set_title(title, fontsize=7, loc='center')
            
            # grid
            ax.grid(True, which='major', alpha=.6, linewidth=.7, zorder=0)
            ax.grid(True, which='minor', alpha=.2, linestyle="--", linewidth=.7, zorder=0)
            ax.minorticks_on()
            
            # scales
            ax.set_yscale('log')
            ax.set_xscale('log')
            
            # limits
            ax.set_xlim(2e-1, 1e6)
            ax.set_ylim(2e-1, 2e3)
            
            # plotting
            scat = ax.scatter(fitness_anim_data_copy[1][0], fitness_anim_data_copy[2][0], c=fitness_anim_data_copy[3][0], s=1, zorder=3)
            
            # colorbar
            plt.colorbar(scat, norm='log', label="Fitness")

            # setting up image
            imgax = fig.add_axes([0.1,0.6,0.25,0.25], anchor='NE', zorder=3)
            image = imgax.imshow(img)
            imgax.axis('off')

            # initializing progress bar
            progress = 0
                
            # animation function
            def update(frame):
                nonlocal scat
                nonlocal image
                nonlocal img
                nonlocal progress
                scat.remove()
                scat = ax.scatter(fitness_anim_data_copy[1][frame], fitness_anim_data_copy[2][frame], c=fitness_anim_data_copy[3][frame], s=1, zorder=3)
                
                # rotating image
                angle = rotation_angle(fitness_anim_data_copy[0][frame])
                image.remove()
                image = imgax.imshow(img.rotate(angle))
                
                # printing progress
                if progress > 6:
                    progress = 0
                    print(' '*50, end='\r')
                print(f"Animation progress: {round(frame / len(fitness_anim_data_copy[0]) * 100, 1)} %." + ' #'*progress, end='\r')
                progress += 1
                
                return scat, image

            ani = FuncAnimation(fig, update, frames=np.arange(0, len(fitness_anim_data_copy[0]), 1), repeat=False, interval=30)
            print(' '*50, end='\r')
            print('Period vs Wait Time Animation Completed!')

            fileName = add_to_title('Animations/Orbital Period vs Wait Time Plots/Orbital Period vs Wait Time Animation')
            ani.save(fileName + filetype, writer='ffmpeg', dpi=anim_dpi)

            if show_animation:
                plt.show()
                
    
    return targets, [event_timings, len(targets), observed_events, new_observations, completed_targets], fitness_anim_data, observing_time, idle_time, slew_time, short_calibration_time, long_calibration_time, commission_time