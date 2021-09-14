"""
Introduction to Radar Course

Authors
=======

Zachary Chance, Robert Freking, Victoria Helus
MIT Lincoln Laboratory
Lexington, MA 02421

Distribution Statement
======================

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the United States Air Force under Air 
Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or 
recommendations expressed in this material are those of the author(s) and do not 
necessarily reflect the views of the United States Air Force.

© 2021 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. 
Government rights in this work are defined by DFARS 252.227-7013 or 
DFARS 252.227-7014 as detailed above. Use of this work other than as specifically 
authorized by the U.S. Government may violate any copyrights that exist in this work.

RAMS ID: 1016938
"""

# Imports
import rad.const as cnst
from math import pi
import numpy as np
from scipy.special import j1
from scipy.optimize import linear_sum_assignment

def area2gain(area, freq):
    """
    Convert effective aperture area to gain.
    
    Inputs:
    - area [float]: Effective aperture area (m^2)
    - freq [float]: Transmit frequency (Hz)
    
    Outputs:
    - gain [float]: Gain
    """

    wlen = wavelen(freq)
    return 4*pi*area/(wlen**2)

def cw(freq_bins, freq=3E9, dr=0, integ_time=0.1):
    """
    Doppler profile with continuous wave radar for two targets.
    
    Inputs:
    - f [ArrayLike]: Frequency bins (Hz)
    - obs_time [float]: Observation time (s)
    
    Outputs:
    - prof [ArrayLike]: Doppler profile (V)
    """

    # Initialize output
    pat = np.zeros(freq_bins.shape)

    # Add target
    pat += cw_pat(freq_bins - dopp_shift(dr, freq), integ_time)
    
    return pat

def cw_pat(freq_bins, obs_time):
    """
    Doppler profile for continuous wave radar.
    
    Inputs:
    - f [ArrayLike]: Frequency bins (Hz)
    - obs_time [float]: Observation time (s)
    
    Outputs:
    - prof [ArrayLike]: Doppler profile (V)
    """

    return sinc(freq_bins, obs_time)

def deg2rad(x):
    """
    Convert from degrees to radians.
    
    Inputs:
    - x [float]: Input data (deg)
    
    Outputs:
    - y [float]: Output data (rad)
    """

    return (pi/180)*x

def detect(
    az_mesh, 
    range_mesh, 
    snr, 
    thresh: float
    ):
    """
    Constant threshold detection.
    
    Inputs:
    - az_mesh [ArrayLike]: Mesh of azimuth points (deg)
    - range_mesh [ArrayLike]: Mesh of range points (m)
    - snr [ArrayLike]: Signal-to-noise ratio values
    - thresh [float]: Detection threshold
    
    Outputs:
    - az_det [ArrayLike]: Azimuth values of detections (deg)
    - range_det [ArrayLike]: Range values of detections (m)
    """

    det_ix = (snr >= thresh)
    return az_mesh[det_ix], range_mesh[det_ix]

def dish_beamw(radius: float, freq: float):
    """
    Dish radar beamwidth.
    
    Inputs:
    - radius [float]: Dish radius (m)
    - freq [float]: Transmit frequency (Hz)
    
    Outputs:
    - beamw [float]: Beamwidth (deg)
    """

    return np.minimum(360.0, 70.0*wavelen(freq)/2/radius)

def dish_cross_range(
    xr_bins, 
    freq: float = 3E9, 
    r: float = 100E3, 
    radius: float = 3.0, 
    xr: float = 100
    ):
    """
    Angle cut of two targets in dish radar pattern.
    
    Inputs:
    - xr_bins [ArrayLike]: Cross-range bins (m)
    - freq [float]: Transmit frequency (Hz); default 3E9 Hz
    - r [float]: Target range (m); default 100E3 m
    - radius [float]: Dish radius (m); default 3 m
    - xr [float]: Target cross-range separation (m); default 100 m
    
    Outputs:
    - pat [ArrayLike]: Radar pattern (V)
    """

    # Initialize output
    xr_pat = np.zeros(xr_bins.shape)
    
    # Bins
    sin_theta = xr_bins/np.sqrt(r**2 + (xr_bins/2)**2)
    
    # Wavelength
    wlen = wavelen(freq)
    
    # First target
    sin_theta1 = -xr/2/np.sqrt(r**2 + (xr/2)**2)
    xr_pat += dish_pat((radius/wlen)*(sin_theta - sin_theta1))
    
    # Second target
    sin_theta2 = xr/2/np.sqrt(r**2 + (xr/2)**2)
    xr_pat += dish_pat((radius/wlen)*(sin_theta - sin_theta2))
        
    return xr_pat

def dish_gain(radius, freq):
    """
    Dish radar gain.
    
    Inputs:
    - radius [float]: Dish radius (m)
    - freq [float]: Transmit frequency (Hz)
    
    Outputs:
    - g: Gain
    """

    return 4*pi**2*radius**2/wavelen(freq)**2

def dish_pat(u):
    """
    Dish radar radiation pattern.
    
    Inputs:
    - u [float]: Normalized direction sine, u = (a/wlen)*sin(theta)
    
    Outputs:
    - f: Radiation vector amplitude
    """

    uz = np.abs(u) > 1E-4
    pat = np.zeros(u.shape)
    pat[uz] = 2*j1(pi*u[uz])/pi/u[uz]
    pat[np.logical_not(uz)] = 1.0
    return pat

def dopp_shift(dr, freq):
    """
    Doppler shift.
    
    Inputs:
    - dr [float]: Range rate (m/s)
    - freq [float]: Transmit frequency (Hz)
    
    Outputs:
    - fd [float]: Doppler shift (Hz)
    """
        
    return -2*dr*freq/cnst.c

def en2ra(x_en):
    """
    Convert from East-North to range-azimuth.
    
    Inputs:
    - x_en [ArrayLike]: East-North state vector
    
    Outputs:
    - x_ra [ArrayLike]: Range-azimuth state vector
    """

    x_ra = np.zeros((2))
    x_ra[0] = np.sqrt(x_en[0]**2 + x_en[1]**2)
    x_ra[1] = np.arctan2(x_en[0], x_en[1])
    
    return x_ra
    
def en2ra_jac(x_en):
    """
    Trasformation Jacobian from East-North to range-azimuth.
    
    Inputs:
    - x_en [ArrayLike]: East-North state vector
    
    Outputs:
    - jac_ra [ArrayLike]: Jacobian matrix
    """

    r = np.sqrt(x_en[0]**2 + x_en[1]**2)
    rho2 = x_en[0]**2 + x_en[1]**2
    
    jac_ra = np.zeros((2, 4))
    jac_ra[0, 0] = x_en[0]/r
    jac_ra[0, 1] = x_en[1]/r
    jac_ra[1, 0] = x_en[1]/rho2
    jac_ra[1, 1] = -x_en[0]/rho2
    
    return jac_ra

def flat_prof(r, res):
    """
    Range profile for sinc pulse.
    
    Inputs:
    - r [ArrayLike]: Range bins (m)
    - res [float]: Range resolution (m)
    
    Outputs:
    - prof [ArrayLike]: Range profile (V)
    """

    return sinc(r, 1/res)

def friis(
    r: float, 
    power: float, 
    area: float = 1,
    gain: float = 1, 
    loss: float = 1
    ):
    """
    Received power using Friis transmission equation.
    
    Inputs:
    - r [float]: Range (m)
    - power [float]: Transmit power (W)
    - area [float]: Receive aperture area (m^2); default 1 m^2
    - gain [float]: Transmit gain; default 1
    - loss [float]: Receive loss; default 1
    
    Outputs:
    - rx_power [float]: Received power (W)
    """
    
    numer = power*gain*area
    denom = 4*pi*(r**2)*loss
    return numer/denom

def from_db(x):
    """
    Convert from decibels to original units.
    
    Inputs:
    - x [float]: Input data (dB)
    
    Outputs:
    - y [float]: Output data
    """
    
    return 10**(0.1*x)

def gain2area(gain: float, freq: float):
    """
    Convert gain to effective aperture area.
    
    Inputs:
    - gain [float]: Gain
    - freq [float]: Transmit frequency (Hz)
    
    Outputs:
    - area [float]: Effective aperture area (m^2)
    """
    
    wlen = wavelen(freq)
    return gain*(wlen**2)/4/pi

def gnn(track_x, track_y, num_track, det_x, det_y, num_det):
    """
    Global nearest neighbor data association.
    
    Inputs:
    - track_x [ArrayLike]: Track state x components (m)
    - track_y [ArrayLike]: Track state y components (m)
    - num_track [int]: Number of input tracks
    - det_x [ArrayLike]: Detection x components (m)
    - det_x [ArrayLike]: Detection y components (m)
    - num_det [int]: Number of input detections
    
    Outputs:
    - assoc [ArrayLike]: Assignment vector
    """

    assoc = -1*np.ones((num_det), dtype='int')
    
    dist = np.zeros((num_det, num_track))
    for ii in range(num_det):
        for jj in range(num_track):
            dist[ii, jj] = np.sqrt((det_x[ii] - track_x[jj])**2 + (det_y[ii] - track_y[jj])**2)
            
    row_ind, col_ind = linear_sum_assignment(dist)
    assoc[row_ind] = col_ind
            
    return assoc

def lfm_prof(r, res: float):
    """
    Range profile for linear frequency-modulated waveform (LFM).
    
    Inputs:
    - r [ArrayLike]: Range bins (m)
    - res [float]: Range resolution (m)
    
    Outputs:
    - prof [ArrayLike]: Range profile (V)
    """
    
    return sinc(r, 1/res)

def nearest(track_x, track_y, num_track, det_x, det_y, num_det):
    """
    Global nearest neighbor data association.
    
    Inputs:
    - track_x [ArrayLike]: Track state x components (m)
    - track_y [ArrayLike]: Track state y components (m)
    - num_track [int]: Number of input tracks
    - det_x [ArrayLike]: Detection x components (m)
    - det_x [ArrayLike]: Detection y components (m)
    - num_det [int]: Number of input detections
    
    Outputs:
    - assoc [ArrayLike]: Assignment vector
    """

    assoc = -1*np.ones((num_det), dtype='int')
    
    dist = np.zeros((num_track))
    for ii in range(num_det):
        if (ii < num_track):
            for jj in range(num_track):
                if jj not in assoc:
                    dist[jj] = np.sqrt((det_x[ii] - track_x[jj])**2 + (det_y[ii] - track_y[jj])**2)
                else:
                    dist[jj] = np.Inf

            assoc[ii] = dist.argmin()

    return assoc

def noise_energy(
    noise_temp: float
    ):
    """
    Thermal noise energy.
    
    Inputs:
    - noise_temp [float]: System noise temperature (K)
    
    Outputs:
    - noise_energy [float]: System noise energy (J)
    """
        
    return cnst.k*noise_temp

def price(
    bandw: float = 100,
    bandw_cost: float = 100,
    energy: float = 100,
    energy_cost: float = 20,
    freq: float = 1E3,
    freq_cost: float = 10,
    num_integ: int = 1,
    integ_cost: float = 100,
    noise_temp: float = 800,
    noise_temp_cost: float = 50,
    radius: float = 1,
    radius_cost: float = 10000,
    scan_rate: float = 0,
    scan_rate_cost: float = 500
    ):
    """
    Notional price model.
    
    Inputs:
    - bandw [float]: Transmit bandwidth (MHz); default 100 MHz
    - bandw_cost [float]: Transmit bandwidth cost ($/MHz); default $100/MHz
    - energy [float]: Transmit energy (mJ); default 100 mJ
    - energy_cost [float]: Transmit energy cost ($/mJ); default $20/mJ
    - freq [float]: Transmit frequency (MHz); default 1E3 MHz
    - freq_cost [float]: Transmit frequency cost ($/MHz); default $10/MHz
    - noise_temp [float]: System noise temperature (K); default 1000 K
    - noise_temp_cost [float]: System noise temperature cost ($/K); default $50/K
    - num_integ [int]: Number of integrated pulses; default 1 pulse
    - integ_cost [float]: Integration cost ($/pulse); default $100/pulse
    - radius [float]: Dish radius (m): default 0.3 m
    - radius_cost [float]: Dish radius cost ($/m): default $10,000/m
    - scan_rate [float]: Scan rate (scans/min); default 5 scans/min
    - scan_rate_cost [float]: Scan rate cost ($/scans/min); default 500 $/scans/min
    
    Outputs:
    (none)
    """
    
    return bandw_cost*bandw + energy*energy_cost + freq*freq_cost + \
           (num_integ - 1)*integ_cost + np.maximum(1500 - noise_temp, 0)*noise_temp_cost + \
           radius*radius_cost + scan_rate*scan_rate_cost

def prop_cv(dt):
    """
    State transition matrix for constant velocity target.
    
    Inputs:
    - dt [float]: Propagation time (s).
    
    Outputs:
    - phi [ArrayLike]: State transition matrix
    """
    phi = np.eye(4)
    phi[0, 2] = dt
    phi[1, 3] = dt
    return phi

def rad2deg(x):
    """
    Convert from radians to degrees.
    
    Inputs:
    - x [float]: Input data (rad)
    
    Outputs:
    - y [float]: Output data (deg)
    """

    return (180/pi)*x

def range_res(bandw):
    """
    Range resolution of transmit waveform.
    
    Inputs:
    - bandw [float]: Transmit bandwidth (Hz)
    
    Outputs:
    - res [float]: Range resolution (m)
    """

    return cnst.c/2/bandw

def rdi(
    range_mesh, 
    dopp_mesh, 
    r, 
    dr, 
    bandw, 
    freq, 
    num_pulse, 
    prf
    ):
    """
    Range-Doppler image
    
    Inputs:
    - range_mesh [ArrayLike]: Mesh of range points (m)
    - dopp_mesh [ArrayLike]: Mesh of Doppler points (Hz)
    - r [float]: Target range (m)
    - dr [float]: Target range rate (m/s)
    - bandw [float]: Transmit bandwidth (Hz)
    - freq [float]: Transmit frequency (Hz)
    - num_pulse [int]: Number of pulses in burst
    - prf [float]: Pulse repetition frequency (Hz)
    
    Outputs:
    - img [ArrayLike]: Range-Doppler image (V)
    """

    range_pat = flat_prof(range_mesh - r, range_res(bandw))
    dopp_pat = sinc(dopp_mesh - dopp_shift(dr, freq), num_pulse/prf)
    return range_pat*dopp_pat

def rect_beamw(h, w, freq):
    """
    Rectangular radar beamwidth.
    
    Inputs:
    - h [float]: Aperture height (m)
    - w [float]: Aperture width (m)
    - freq [float]: Transmit frequency (Hz)
    
    Outputs:
    - beamw_h [float]: Horizontal beamwidth (deg)
    - beamw_v [float]: Vertical beamwidth (deg)
    """
    wlen = wavelen(freq)
    beamw_h = np.minimum(360.0, rad2deg(wlen/w))
    beamw_v = np.minimum(360.0, rad2deg(wlen/h))
    return (beamw_h, beamw_v)

def rect_pat(u, v):
    """
    Rectangular radar radiation pattern.
    
    Inputs:
    - u [float]: Normalized horizontal direction sine, u = -(w/wlen)*sin(az)*cos(el)
    - v [float]: Normalized vertical direction sine, v = -(h/wlen)*sin(el)
    
    Outputs:
    - f: Radiation vector amplitude
    """

    return 2*sinc(u, 1)*sinc(v, 1)

def rx_power(
    r: float, 
    power: float, 
    freq: float, 
    gain: float = 1, 
    loss: float = 1,
    rcs: float = 1
    ):
    """
    Received power from radar transmission.
    
    Inputs:
    - r [float]: Target range (m)
    - power [float]: Transmit power (W)
    - freq [float]: Transmit frequency (Hz)
    - gain [float]: Total transmit and receive gain; default 1
    - loss [float]: Total transmit and receive loss; default 1
    - rcs [float]: Target radar cross section (m^2); default 1 m^2
    
    Outputs:
    - power [float]: Received power (W)
    """
    
    wlen = wavelen(freq)
    numer = power*(wlen**2)*gain*rcs
    denom = ((4*pi)**3)*(r**4)*loss
    return numer/denom

def rx_energy(
    r: float, 
    energy: float, 
    freq: float, 
    gain: float = 1, 
    loss: float = 1,
    rcs: float = 1
    ):
    """
    Received energy from radar transmission.
    
    Inputs:
    - r [float]: Target range (m)
    - energy [float]: Transmit energy (W)
    - freq [float]: Transmit frequency (Hz)
    - gain [float]: Total transmit and receive gain; default 1
    - loss [float]: Total transmit and receive loss; default 1
    - rcs [float]: Target radar cross section (m^2); default 1 m^2
    
    Outputs:
    - energy [float]: Received energy (J)
    """
        
    wlen = wavelen(freq)
    numer = energy*(wlen**2)*gain*rcs
    denom = ((4*pi)**3)*(r**4)*loss
    return numer/denom

def rx_snr(
    r: float, 
    energy: float, 
    freq: float, 
    noise_temp: float,
    gain: float = 1, 
    loss: float = 1,
    rcs: float = 1
    ):
    """
    Signal-to-noise ratio from radar transmission.
    
    Inputs:
    - r [float]: Target range (m)
    - energy [float]: Transmit energy (W)
    - freq [float]: Transmit frequency (Hz)
    - noise_temp [float]: System noise temperature (K)
    - gain [float]: Total transmit and receive gain; default 1
    - loss [float]: Total transmit and receive loss; default 1
    - rcs [float]: Target radar cross section (m^2); default 1 m^2
    
    Outputs:
    - snr [float]: Signal-to-noise ratio
    """
        
    wlen = wavelen(freq)
    numer = energy*(wlen**2)*gain*rcs
    denom = ((4*pi)**3)*(r**4)*loss*noise_energy(noise_temp)
    return numer/denom
    
def sinc(x, w):
    """
    Sinc function.
    
    Inputs:
    - x [ArrayLike]: Input data
    - w [float]: Inverse mainlobe width
    
    Outputs:
    - y [ArrayLike]: Output data
    """
        
    xz = np.abs(x) > 1E-4
    pat = np.zeros(x.shape)
    pat[xz] = np.sin(pi*w*x[xz])/pi/x[xz]/w
    pat[np.logical_not(xz)] = 1.0
    return pat
    
def snrconv(
    snr0: float, 
    r0: float, 
    rcs0: float, 
    r: float, 
    rcs: float
    ):
    """
    Signal-to-noise ratio (SNR) from reference.
    
    Inputs:
    - snr0 [float]: Reference SNR
    - r0 [float]: Reference range (m)
    - rcs0 [float]: Reference radar cross section (m^2)
    - r [float]: Target range (m)
    - rcs [float]: Target radar cross section (m^2)
    
    Outputs:
    - snr [float]: SNR
    """
    
    return snr0*((r0/r)**4)*(rcs/rcs0)
    
class Target():
    """
    Generic target container.
    """

    def __init__(self):
        self.pos = None
        self.rcs = None
        self.route = None
        
    def get_pos(self, t):
        if self.pos.size == 2:
            return self.pos
        elif self.pos.size == 4:
            return self.pos[0:2] + t*self.pos[2:4]

    def get_raz(self, t):
        pos = self.get_pos(t)
        raz = np.zeros((2))
        raz[0] = np.sqrt(pos[0]**2 + pos[1]**2)
        raz[1] = np.pi/2 - np.arctan2(pos[1], pos[0])
        
        return raz
        
def to_db(x):
    """
    Convert from original units to decibels.
    
    Inputs:
    - x [float]: Input data
    
    Outputs:
    - y [float]: Output data (dB)
    """
    
    if np.isscalar(x):
        if np.abs(x) < 1E-40:
            return -400
    else:
        x[np.abs(x) < 1E-40] = 1E-40
    return 10*np.log10(np.abs(x))
        
def wavelen(freq, propvel=cnst.c):
    """
    Calculate wavelength.
    
    Inputs:
    - freq [float]: Transmit frequency (Hz)
    - propvel [float]: Proagation velocity (m/s); default 3E8 m/s
    
    Outputs:
    - wlen [float]: Wavelength (m)
    """
    
    return propvel/freq