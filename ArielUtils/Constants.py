import numpy as np
from astropy.time import Time

# Sun constraints
MIN_ANGLE_TOWARDS_SUN = 70 # in degrees
MIN_ANGLE_AWAY_SUN = 60 # in degrees

# Telescope constraints
SLEWRATE = np.radians(4.5) # radians per minute
SETTLE_TIME = 10 # minutes

# Mission constrants
MISSION_DURATION = 4*365*24*60 # minutes
COMMISSION_TIME = 6.*30*24*60 # minutes

# Calibration constraints
SHORT_CALIBRATION_DURATION = 1 *60 # minutes
SHORT_CALIBRATION_FREQUENCY = 36 *60 # minutes
SHORT_CALIBRATION_ERR = 12 *60 # minutes

LONG_CALIBRATION_DURATION = 6 *60 # minutes
LONG_CALIBRATION_FREQUENCY = 30 *24*60 # minutes
LONG_CALIBRATION_ERR = 10 *24*60 # minutes

# Baseline parameters
BASELINE_RATIO = 0.25 # ratio of baseline-to-event-duration on either side of the event. 
BASELINE_DURATION = 120 # baseline duration, on either side of the event

# Original Reference time, in case none is provided
T0 = Time(-240000, format='mjd')# MJD calendar (1201, Oct 12)