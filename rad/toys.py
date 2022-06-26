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
from IPython.display import display, HTML
import ipywidgets as wdg
import matplotlib.patches as ptch
import matplotlib.pyplot as pyp
import rad.plot as plt
import rad.const as cnst
import rad.radar as rd
import math
import numpy as np
from scipy import special

# Constants
AZ_DEG_LABEL = 'Azimuth (deg)'
BANDW_MHZ_LABEL = 'Bandwidth (MHz)'
BEAMW_DEG_LABEL = 'Beamwidth (deg)'
BOX_LAYOUT = wdg.Layout(justify_items='flex-start')
CONTROL_BLOCK_LABEL = f"<b><font color='black'>Controls</b>"
CROSS_RANGE_M_LABEL = 'Cross Range (m)'
DELAY_US_LABEL = 'Delay (µs)'
DISH_RADIUS_LABEL = 'Dish Radius (m)'
DISP_BLOCK_LABEL = f"<b><font color='black'>Display</b>"
EAST_M_LABEL = 'East (m)'
ENERGY_MJ_LABEL = 'Transmit Energy (mJ)'
FRAME_LABEL = 'Frame'
FREQ_HZ_LABEL = 'Frequency (Hz)'
FREQ_MHZ_LABEL = 'Frequency (MHz)'
LABEL_STYLE = {'description_width': 'initial'}
NOISE_TEMP_LABEL = 'Noise Temperature (°K)'
NORM_ENERGY_DBJ_LABEL = 'Normalized Energy (dBJ)'
NORTH_M_LABEL = 'North (m)'
OBS_TIME_MS_LABEL = 'Observation Time (ms)'
RADAR_BLOCK_LABEL = f"<b><font color='black'>Radar</b>"
RANGE_M_LABEL = 'Range (m)'
RANGE_KM_LABEL = 'Range (km)'
RANGE_RATE_M_S_LABEL = 'Range Rate (m/s)'
RCS_DBSM_LABEL = 'RCS (dBsm)'
REL_FREQ_HZ_LABEL = 'Relative Frequency (Hz)'
RUN_LABEL = 'Run'
RUN_BLOCK_LABEL = f"<b><font color='black'>Run</b>"
RX_AZ_DEG_LABEL = 'Receive Azimuth (deg)'
SENSOR_BLOCK_LABEL = f"<b><font color='black'>Sensor</b>"
SNR_DB_LABEL = 'SNR (dB)'
TARGET_BLOCK_LABEL = f"<b><font color='black'>Target</b>"
TIME_S_LABEL = 'Time (s)'
TIME_MS_LABEL = 'Time (ms)'
TIME_NS_LABEL = 'Time (ns)'
TIME_US_LABEL = 'Time (µs)'
TX_AZ_DEG_LABEL = 'Transmit Azimuth (deg)'
TX_BANDW_MHZ_LABEL = 'Transmit Bandwidth (MHz)'
TX_BLOCK_LABEL = f"<b><font color='black'>Transmission</b>"
TX_FREQ_HZ_LABEL = 'Transmit Frequency (Hz)'
TX_FREQ_MHZ_LABEL = 'Transmit Frequency (MHz)'
TX_GAIN_DB_LABEL = 'Transmit Gain (dB)'
TX_POW_W_LABEL = 'Transmit Power (W)'
WAVE_LABEL = 'Wave'
WAVEFORM_V_LABEL = 'Waveform (V)'
WDG_LAYOUT = wdg.Layout(grid_template_columns="repeat(3, 350px)")

def array(
    dx: float = 3,
    interval: float = 75,
    max_range: float = 150,
    num_elem: int = 10,
    num_step: int = 150,
    propvel: float = 1000,
    rx_az: float = 0.0,
    tgt_hide: bool = False,
    tgt_x: float = 50, 
    tgt_y: float = 150,
    tx_az: float = 0.0,
    widgets = ['run', 'tx_az', 'x', 'y'],
    xlim =[-100, 100],
    ylim =[0, 200]):
    """
    Delay steering animation for linear array.
    
    Inputs:
    - dx [float]: Element spacing (m); default 3 m
    - interval [float]: Time between animation steps (ms); default 75 ms
    - max_range [float]: Maximum range for plotting (m); default 150
    - num_elem [int]: Number of elements; default 10 elements
    - num_step [int]: Number of animation steps; default 150 steps
    - propvel [float]: Propagation velocity (m/s); default 1000 m/s
    - rx_az [float]: Receive steering direction (deg); default 0 deg
    - tgt_hide [bool]: Flag for hidden target; default False
    - tgt_x [float]: Target East coordinate (m); default 50 m
    - tgt_y [float]: Target North coordinate (m); default 150 m
    - tx_az [float]: Transmit steering direction (deg); default 0 deg
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (m)
    
    Outputs:
    (none)
    """

    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(EAST_M_LABEL)
    ax1.set_ylabel(NORTH_M_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # Maximum range
    max_plot_range = calc_max_range(xlim, ylim)
    if not max_range:
        max_range = max_plot_range
    
    # Time vector
    t0 = 0
    t1 = 2*max_range/propvel
    dt = (t1 - t0)/(num_step - 1)

    # Timestamp 
    timestamp = ax1.text(xlim[1] + 5, ylim[1] - 12, f"Time: {t0*1000:.2f} ms", size=12.0)

    # Element markers
    x_elem = []
    elems = []
    for ii in range(num_elem):
        xii = -dx*(num_elem - 1)/2 + dx*ii
        elemii = ptch.Circle((xii, 0), 2, color='blue')
        x_elem.append(xii)
        elems.append(elemii)
        ax1.add_patch(elemii)
    
    # Target marker
    if not tgt_hide:
        ct = ptch.Circle((tgt_x, tgt_y), 3, color='red')
        ax1.add_patch(ct)
        echoes = []
        for ii in range(num_elem):
            echoii = ptch.Circle((tgt_x, tgt_y), 0, fill=False, color='red', linewidth=2.0, alpha=0.4)
            echoes.append(echoii)
            ax1.add_patch(echoii)
    
    # Transmit beam
    tx_beamw = 2
    tx_theta1 = 90 - tx_az - tx_beamw/2
    tx_theta2 = 90 - tx_az + tx_beamw/2
    tx_beam = ptch.Wedge((0, 0), max_plot_range, tx_theta1, tx_theta2, color='gray', alpha=0.2)
    ax1.add_patch(tx_beam)
    
    # Transmit impulses
    pulses = []
    for ii in range(num_elem):
        pulseii = ptch.Circle((x_elem[ii], 0), 0, fill=False, color='blue', linewidth=2.0, alpha=0.4)
        pulses.append(pulseii)
        ax1.add_patch(pulseii)
        
    # Control widgets
    controls_box = []
        
    # Sensor control widgets
    sensor_controls = []
    
    # Boresight azimuths
    if ('rx_az' in widgets):
        rx_az_wdg = wdg.FloatSlider(
            min=0, 
            max=360, 
            step=1, 
            value=rx_az, 
            description=RX_AZ_DEG_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sensor_controls.append(rx_az_wdg)
    else:
        rx_az_wdg = wdg.fixed(rx_az)
        
    if ('tx_az' in widgets):
        tx_az_wdg = wdg.FloatSlider(
            min=-90, 
            max=90, 
            step=1, 
            value=tx_az, 
            description=TX_AZ_DEG_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sensor_controls.append(tx_az_wdg)
    else:
        tx_az_wdg = wdg.fixed(tx_az)
    
    sensor_box = []
    if sensor_controls:
        sensor_title = [wdg.HTML(value = SENSOR_BLOCK_LABEL)]
        sensor_box = wdg.VBox(sensor_title + sensor_controls, layout=BOX_LAYOUT)
        controls_box.append(sensor_box)
        
    # Target control widgets
    target_controls = []
    
    # Target x position
    if ('x' in widgets):
        # Build widget
        x_wdg = wdg.FloatSlider(
            min=xlim[0], 
            max=xlim[1], 
            step=(xlim[1] - xlim[0])/200, 
            value=tgt_x,
            description=EAST_M_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        target_controls.append(x_wdg)
    else:
        x_wdg = wdg.fixed(tgt_x)
        
    # Target y position
    if ('y' in widgets):
        # Build widget
        y_wdg = wdg.FloatSlider(
            min=ylim[0], 
            max=ylim[1], 
            step=(ylim[1] - ylim[0])/200, 
            value=tgt_y,
            description=NORTH_M_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        target_controls.append(y_wdg)
    else:
        y_wdg = wdg.fixed(tgt_y)
        
    target_box = []
    if target_controls:
        target_title = [wdg.HTML(value = TARGET_BLOCK_LABEL)]
        target_box = wdg.VBox(target_title + target_controls, layout=BOX_LAYOUT)
        controls_box.append(target_box)
    
    # Run widgets
    run_controls = []
    if ('run' in widgets):
        play_wdg = wdg.Play(
            interval=interval,
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=RUN_LABEL,
            disabled=False
        )
        run_controls.append(play_wdg)
        slider_wdg = wdg.IntSlider(
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=FRAME_LABEL,
            readout=False,
            disabled=False
        )
        wdg.jslink((play_wdg, 'value'), (slider_wdg, 'value'))
        run_controls.append(slider_wdg)
        
    run_box = []
    if run_controls:
        run_title = [wdg.HTML(value = RUN_BLOCK_LABEL)]
        run_box = wdg.VBox(run_title + run_controls, layout=BOX_LAYOUT)
        controls_box.append(run_box)   
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
        
    # Plot
    def animate(frame, rx_az, tx_az, tgt_x, tgt_y):

        # Time
        t = t0 + (frame - 1)*dt

        # Update timestamp
        timestamp.set_text(f"Time: {t*1000:.2f} ms")
        
        # Update target
        if not tgt_hide:
            ct.center = tgt_x, tgt_y
            for ii in range(num_elem):
                echoes[ii].center = tgt_x, tgt_y
                   
        # Input conversion
        if rx_az:
            rx_az = rx_az*(np.pi/180)
        if tx_az:
            tx_az = tx_az*(np.pi/180)
        
        # Update beams
        tx_beam.set_theta1((180/np.pi)*(np.pi/2 - tx_az) - tx_beamw/2)
        tx_beam.set_theta2((180/np.pi)*(np.pi/2 - tx_az) + tx_beamw/2)
        
        # Element ranges/delays
        r_elem = np.zeros((num_elem,))
        delays = np.zeros((num_elem,))
        for ii in range(num_elem):
            r_elem[ii] = np.sqrt((x_elem[ii] - tgt_x)**2 + tgt_y**2)
            delays[ii] = -dx*np.sin(tx_az)*ii/propvel
        
        # Shift delays
        delays -= np.max(delays)
        
        # Update pulses
        for ii in range(num_elem):
            tii = t + delays[ii]
            if (propvel*tii > 0):
                pulses[ii].set_radius(propvel*tii)
            else:
                pulses[ii].set_radius(0.0)
        
        # Update echo
        if not tgt_hide:
            for ii in range(num_elem):
                tii = t + delays[ii]
                drii = propvel*tii - r_elem[ii]
                if (drii > 0):
                    echoes[ii].set_radius(drii)
                else:
                    echoes[ii].set_radius(0.0)  
        
        # Disable controls during play
        if (frame > 1):
            for w in sensor_controls:
                if not w.disabled:
                    w.disabled = True
            for w in target_controls:
                if not w.disabled:
                    w.disabled = True
        elif (frame == 1):
            for w in sensor_controls:
                if w.disabled:
                    w.disabled = False
            for w in target_controls:
                if w.disabled:
                    w.disabled = False
    
    # Add interaction
    wdg.interactive(
        animate, 
        frame=slider_wdg,
        rx_az=rx_az_wdg, 
        tx_az=tx_az_wdg, 
        tgt_x=x_wdg,
        tgt_y=y_wdg
    )

def sine(
    amp: float = 0,
    freq: float = 0,
    num_step: int = 500,
    phase: float = 0,
    widgets = ['amp', 'freq', 'phase'],
    xlim = [0, 10],
    ylim = [-10, 10]
    ):
    """
    Sine wave display.
    
    Inputs:
    - amp [float]: Wave amplitude (V); default 0 V
    - freq [float]: Wave frequency (Hz); default 0 Hz
    - num_step [int]: Number of animation steps; default 500
    - phase [float]: Wave phase (deg); default 0 deg
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (s)
    - ylim [List[float]]: y-axis limits for plotting (V)
    
    Outputs:
    (none)
    """

    # Initialize plot
    fig1, ax1 = plt.new_plot()
    ax1.set_xlabel(TIME_S_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # Time vector
    t0 = 0
    t1 = 10
    dt = (t1 - t0)/(num_step - 1)

    # Wave
    xvec = np.linspace(xlim[0], xlim[1], num_step)
    ax1.plot(xlim, [0, 0], color='gray', alpha=0.2)
    wave = ax1.plot(xvec, amp*np.sin(2*np.pi*freq*xvec + (np.pi/180)*phase), color='red', linewidth=1.0)[0]
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Amplitude
    if ('amp' in widgets):
        amp_wdg = wdg.FloatSlider(
            min=0, 
            max=10, 
            step=0.5, 
            value=amp, 
            description="Amplitude", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(amp_wdg)
    else:
        amp_wdg = wdg.fixed(amp)
        
    # Frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=0, 
            max=2, 
            step=0.05, 
            value=freq, 
            description=FREQ_HZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    # Phase
    if ('phase' in widgets):
        phase_wdg = wdg.FloatSlider(
            min=0, 
            max=360, 
            step=1, 
            value=phase, 
            description="Initial Phase (deg)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(phase_wdg)
    else:
        phase_wdg = wdg.fixed(phase)
        
    wave_box = []
    if wave_controls:
        wave_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        wave_box = wdg.VBox(wave_title + wave_controls, layout=BOX_LAYOUT)
        controls_box.append(wave_box)
               
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def animate(amp, freq, phase):
        wave.set_data(xvec, amp*np.sin(2*np.pi*freq*xvec + (np.pi/180)*phase))
    
    # Add interaction
    wdg.interactive(
        animate, 
        amp=amp_wdg,
        freq=freq_wdg, 
        phase=phase_wdg
    )
    
def radar_wave(
    interval: float = 50,
    num_step: int = 500,
    play_lock: bool = False,
    widgets = ['freq', 'run'],
    xlim = [0, 4],
    ylim = [-2, 2]
    ):
    """
    Wavelength display for radar.
    
    Inputs:
    - interval [float]: Time between animation steps (ms); default 50 ms
    - num_step [int]: Number of animation steps; default 500 steps
    - play_lock [bool]: Flag for locking widgets while playing; default False
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (V)
    
    Outputs:
    (none)
    """

    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(RANGE_M_LABEL)
    ax1.set_ylabel(WAVE_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
 
    # Propagation velocity
    propvel = 3E8
    
    # Time vector
    t0 = 0
    t1 = xlim[1]/propvel
    dt = (t1 - t0)/(num_step - 1)
    
    # Timestamp 
    timestamp = ax1.text(xlim[1] + 0.1, ylim[1] - 0.2, f"Time: {t0*1E9:.2f} ns", size=12.0)

    # Wave value marker
    wave_val = ptch.Circle((0, 0), 0.1, color='pink')
    ax1.add_patch(wave_val)
    
    # Wave
    xvec = np.linspace(xlim[0], xlim[1], num_step)
    wave = ax1.plot(xvec, np.zeros((num_step,)), color='red', linewidth=3.0)[0]
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Boresight azimuths
    if ('freq' in widgets):
        freq_wdg = wdg.FloatLogSlider(
            min=1, 
            max=4, 
            step=0.01, 
            value=2, 
            description=FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    wave_box = []
    if wave_controls:
        wave_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        wave_box = wdg.VBox(wave_title + wave_controls, layout=BOX_LAYOUT)
        controls_box.append(wave_box)
        
    # Run widgets
    run_controls = []
    if ('run' in widgets):
        play_wdg = wdg.Play(
            interval=interval,
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=RUN_LABEL,
            disabled=False
        )
        run_controls.append(play_wdg)
        slider_wdg = wdg.IntSlider(
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=FRAME_LABEL,
            readout=False,
            disabled=False
        )
        wdg.jslink((play_wdg, 'value'), (slider_wdg, 'value'))
        run_controls.append(slider_wdg)
        
    run_box = []
    if run_controls:
        run_title = [wdg.HTML(value = RUN_BLOCK_LABEL)]
        run_box = wdg.VBox(run_title + run_controls, layout=BOX_LAYOUT)
        controls_box.append(run_box)   
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def animate(frame, freq):

        # Time
        t = t0 + (frame - 1)*dt

        # Update timestamp
        timestamp.set_text(f"Time: {t*1E9:.2f} ns")
 
        wave_amp = np.cos(2*np.pi*(1E6*freq/propvel)*(xvec - propvel*t))
        wave_amp[xvec > propvel*t] = 0.0

        wave_val.center = 0, wave_amp[0]
        wave.set_ydata(wave_amp)
        
        # Disable controls during play
        if (play_lock):
            if (frame > 1):
                for w in wave_controls:
                    if not w.disabled:
                        w.disabled = True
            elif (frame == 1):
                for w in wave_controls:
                    if w.disabled:
                        w.disabled = False
    
    # Add interaction
    wdg.interactive(
        animate, 
        frame=slider_wdg,
        freq=freq_wdg
    )    
    
def calc_max_range(xlim, ylim):
    """
    Maximum plot range.
    
    Inputs:
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (m)
    
    Outputs:
    - r [float]: Maximum range (m)
    """

    r = np.sqrt(xlim[0]**2 + ylim[0]**2)
    r = np.maximum(r, np.sqrt(xlim[1]**2 + ylim[0]**2))
    r = np.maximum(r, np.sqrt(xlim[0]**2 + ylim[1]**2))
    r = np.maximum(r, np.sqrt(xlim[1]**2 + ylim[1]**2))
    return r
    
def cross_range(
    beamw: float = 0.5,
    num_bins: int = 500,
    r: float = 100.0,
    xr: float = 1000.0,
    widgets=['beamw', 'r', 'xr'],
    xlim=[-5E3, 5E3],
    ylim=[-40, 10]
    ):
    """
    Angle cut with two targets with variable cross range.
    
    Inputs:
    - beamw [float]: Transmit beamwidth (deg); deault 0.5 deg
    - num_bins [int]: Number of cross range bins; default 500 bins
    - r [float]: Target range (km); default 100 km
    - xr [float]: Target cross-range separation (m); default 1000 m
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (dB)
    
    Outputs:
    (none)
    """

    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(CROSS_RANGE_M_LABEL)
    ax1.set_ylabel(NORM_ENERGY_DBJ_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
   
    # Cross range bins
    xr_bins = np.linspace(xlim[0], xlim[1], num_bins)

    # Initial plot
    freq = 3E9
    wavelen = rd.wavelen(freq)
    radius = (70/beamw)*wavelen
    plot_line = ax1.plot(xr_bins, 2*rd.to_db(rd.dish_cross_range(xr_bins, freq=freq, radius=radius, r=r*1E3, xr=xr)), color='red', linewidth=3.0)[0]

    # Target lines
    tgt1_line = ax1.plot([-xr/2, -xr/2], [ylim[0], ylim[1]], color='black', linestyle='dashed', linewidth=2.0)[0]
    tgt2_line = ax1.plot([xr/2, xr/2], [ylim[0], ylim[1]], color='black', linestyle='dashed', linewidth=2.0)[0]
    
    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    sub_controls1 = []
   
    # Beamwidth
    if ('beamw' in widgets):
        beamw_wdg = wdg.FloatSlider(
            min=0.01, 
            max=3.0, 
            step=0.01, 
            value=beamw, 
            description=BEAMW_DEG_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(beamw_wdg)
    else:
        beamw_wdg = wdg.fixed(beamw)

    # Range
    if ('r' in widgets):
        r_wdg = wdg.FloatSlider(
            min=1, 
            max=300, 
            step=1, 
            value=r, 
            description=RANGE_KM_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(r_wdg)
    else:
        r_wdg = wdg.fixed(r)
        
    # Cross-range
    if ('xr' in widgets):
        xr_wdg = wdg.FloatSlider(
            min=100, 
            max=5000, 
            step=100, 
            value=xr, 
            description=CROSS_RANGE_M_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(xr_wdg)
    else:
        xr_wdg = wdg.fixed(xr)
        
    sub_controls1_box = []
    if sub_controls1:
        sub_controls1_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        sub_controls1_box = wdg.VBox(sub_controls1_title + sub_controls1, layout=BOX_LAYOUT)
        controls_box.append(sub_controls1_box)
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # Plot
    def plot(beamw, r, xr):
        plot_line.set_ydata(2*rd.to_db(rd.dish_cross_range(xr_bins, freq=freq, radius=(70/beamw)*wavelen, r=r*1E3, xr=xr)))
        tgt1_line.set_xdata([-xr/2, -xr/2])
        tgt2_line.set_xdata([xr/2, xr/2])
   
    # Add interaction
    wdg.interactive(
        plot, 
        beamw=beamw_wdg,
        r=r_wdg,
        xr=xr_wdg
    )
    
def cw(
    freq: float = 1.0E3,
    integ_time: float = 5,
    max_freq: float = 1E3,
    min_freq: float = -1E3,
    num_bins: int = 1000,
    dr: float = 0.0,
    targ_line: bool = True,
    widgets = ['dr', 'freq', 'integ_time'],
    ylim = [-40, 0]
    ):
    """
    Continuous wave radar response.
    
    Inputs:
    - freq [float]: Transmit frequency (MHz); default 1E3 MHz
    - integ_time [float]: Observation time (ms); default 5 ms
    - max_freq [float]: Maximum frequency for plotting (Hz); default 1000 Hz
    - max_freq [float]: Minimum frequency for plotting (Hz); default -1000 Hz
    - num_bins [int]: Number of frequency bins; default 1000 bins
    - dr [float]: Target range rate (m/s); default 0 m/s
    - targ_line [bool]: Flag for target line; default True
    - widgets [List[str]]: List of desired widgets
    - ylim [List[float]]: y-axis limits for plotting (dB)
    
    Outputs:
    (none)
    """

    # x axis limits
    xlim = [min_freq, max_freq]

    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(REL_FREQ_HZ_LABEL)
    ax1.set_ylabel(NORM_ENERGY_DBJ_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
   
    # Frequency bins
    freq_bins = np.linspace(xlim[0], xlim[1], num_bins)

    # Initial plot
    plot_line = ax1.plot(freq_bins, 2*rd.to_db(rd.cw(freq_bins, freq=freq*1E6, integ_time=integ_time*1E-3, dr=dr)), color='red', linewidth=3.0)[0]

    # Target lines
    if targ_line:
        dopp0 = rd.dopp_shift(dr, freq*1E6)
        tgt1_line = ax1.plot([dopp0, dopp0], [ylim[0], ylim[1]], color='black', linestyle='dashed', linewidth=2.0)[0]
    
    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    sub_controls1 = []
   
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=3E3, 
            step=100, 
            value=freq, 
            description=TX_FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)

    # Range rate
    if ('dr' in widgets):
        dr_wdg = wdg.FloatSlider(
            min=-100, 
            max=100, 
            step=1, 
            value=dr, 
            description=RANGE_RATE_M_S_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(dr_wdg)
    else:
        dr_wdg = wdg.fixed(dr)
        
    # Integration time
    if ('integ_time' in widgets):
        integ_time_wdg = wdg.FloatSlider(
            min=1, 
            max=20, 
            step=0.1, 
            value=integ_time, 
            description=OBS_TIME_MS_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(integ_time_wdg)
    else:
        integ_time_wdg = wdg.fixed(integ_time)
        
    sub_controls1_box = []
    if sub_controls1:
        sub_controls1_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        sub_controls1_box = wdg.VBox(sub_controls1_title + sub_controls1, layout=BOX_LAYOUT)
        controls_box.append(sub_controls1_box)
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # Plot
    def plot(dr, freq, integ_time):
        dopp = rd.dopp_shift(dr, freq*1E6)
        plot_line.set_ydata(2*rd.to_db(rd.cw(freq_bins, freq=freq*1E6, integ_time=integ_time*1E-3, dr=dr)))
        if targ_line:
            tgt1_line.set_xdata([dopp, dopp])
   
    # Add interaction
    wdg.interactive(
        plot, 
        dr=dr_wdg,
        freq=freq_wdg,
        integ_time=integ_time_wdg
    ) 
    
def delay_steer(
    az: float = 0,
    freq: float = 5E2,
    num_cycle: int = 4,
    num_bins: int = 500,
    r: float = 2.5,
    widgets = ['az', 'delays', 'freq', 'range'],
    ):
    """
    Delay steering demonstration.
    
    Inputs:
    - az [float]: Target azimuth (deg); default 0 deg
    - freq [float]: Transmit frequency (MHz); default 5E2 MHz
    - num_bins [int]: Number of time steps in plot; default 500 steps
    - num_cycle [int]: Number of wave cycles for plotting; default 4 cycles
    - r [float]: Target range (km); default 2.5 km
    - widgets [List[str]]: List of desired widgets
    
    Outputs:
    (none)
    """

    # Minimum frequency
    min_freq = 500E6
    
    # x limits
    xlim = [-0.6E9*num_cycle/min_freq, 0.6E9*num_cycle/min_freq]
    
    # Initialize plot
    fig, axs = plt.new_plot2()
    wave_ax = axs[0]
    total_ax = axs[1]
    total_ax.set_xlabel(TIME_NS_LABEL)
    wave_ax.set_ylabel('Waves')
    total_ax.set_ylabel('Total')
    wave_ax.set_xlim(xlim)
    wave_ax.set_ylim([-1, 1])
    total_ax.set_xlim(xlim)
    total_ax.set_ylim([-5, 5])
    
    # Timestamp
    rel_energy = fig.text(0.82, 0.9, "Energy Loss: ---", size=12.0)
    
    # Number of elements
    num_elem = 5
    
    # Time vector
    xvec = np.linspace(xlim[0], xlim[1], num_bins)
    dx = (xvec[1] - xvec[0])/1E9
    
    # Element locations
    wlen = rd.wavelen(min_freq)
    elem_x = np.linspace(-wlen*(num_elem - 1)/2, wlen*(num_elem - 1)/2, num_elem)
    
    # Target location
    targ_x = (r*1E3)*np.cos(np.pi/2 - rd.deg2rad(az))
    targ_y = (r*1E3)*np.sin(np.pi/2 - rd.deg2rad(az))
    
    # Reference delay
    ref_delay = np.sqrt(targ_x**2 + targ_y**2)/cnst.c
    
    # Waves
    wave_lines = []
    total_y = np.zeros(xvec.shape)
    for ii in range(num_elem):
        delayii = np.sqrt((targ_x - elem_x[ii])**2 + targ_y**2)/cnst.c - ref_delay
        yii = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - delayii))
        zeroix = np.abs(xvec/1E9 - delayii) > (num_cycle/2/freq/1E6)
        yii[zeroix] = 0.0
        total_y += yii
        if (ii == 2):
            waveii = wave_ax.plot(xvec, yii, linewidth=3.0, color='black', linestyle='dashed')[0]
        else:
            waveii = wave_ax.plot(xvec, yii, linewidth=2.0)[0]
        wave_lines.append(waveii)
    
    # Total
    total_line = total_ax.plot(xvec, total_y, linewidth=3.0, color='red')[0]
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=500, 
            max=1000, 
            step=10, 
            value=freq, 
            description=FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    wave_title = [wdg.HTML(value = TX_BLOCK_LABEL)]
        
    # Target controls
    targ_controls = []
        
    # Azimuth
    if ('az' in widgets):
        az_wdg = wdg.FloatSlider(
            min=-60, 
            max=60, 
            step=1, 
            value=az, 
            description=AZ_DEG_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        targ_controls.append(az_wdg)
    else:
        az_wdg = wdg.fixed(r)
    
        
    # Range
    if ('range' in widgets):
        range_wdg = wdg.FloatSlider(
            min=0.01, 
            max=5, 
            step=0.01, 
            value=r, 
            description=RANGE_KM_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        targ_controls.append(range_wdg)
    else:
        range_wdg = wdg.fixed(r)
        
    # Transmission + target controls
    targ_title = [wdg.HTML(value = TARGET_BLOCK_LABEL)]
    targ_box = wdg.VBox(wave_title + wave_controls + targ_title + targ_controls, layout=BOX_LAYOUT)
    controls_box.append(targ_box)
        
    # Element controls
    elem_controls = []
        
    # Delays
    delay1_wdg = wdg.FloatSlider(
        min=-5, 
        max=5, 
        step=0.01,
        value=0.0, 
        description="Delay #1 (ns)", 
        style=LABEL_STYLE, 
        readout_format='.2f'
    )
    elem_controls.append(delay1_wdg)
    
    delay2_wdg = wdg.FloatSlider(
        min=-5, 
        max=5, 
        step=0.01,
        value=0.0, 
        description="Delay #2 (ns)", 
        style=LABEL_STYLE, 
        readout_format='.2f'
    )
    elem_controls.append(delay2_wdg)
    
    delay4_wdg = wdg.FloatSlider(
        min=-5, 
        max=5, 
        step=0.01,
        value=0.0, 
        description="Delay #4 (ns)", 
        style=LABEL_STYLE, 
        readout_format='.2f'
    )
    elem_controls.append(delay4_wdg)
    
    delay5_wdg = wdg.FloatSlider(
        min=-5, 
        max=5, 
        step=0.01,
        value=0.0, 
        description="Delay #5 (ns)", 
        style=LABEL_STYLE, 
        readout_format='.2f'
    )
    elem_controls.append(delay5_wdg)
        
    reset_btn = wdg.Button(description="Reset")
    elem_controls.append(reset_btn)
        
    elem_box = []
    if elem_controls:
        elem_title = [wdg.HTML(value = f"<b><font color='black'>Elements</b>")]
        elem_box = wdg.VBox(elem_title + elem_controls, layout=BOX_LAYOUT)
        controls_box.append(elem_box)
        
    show_controls = []
        
    # Show widgets
    show1_wdg = wdg.ToggleButton(
        value=True,
        description='Element #1'
    )
    show_controls.append(show1_wdg)
    
    show2_wdg = wdg.ToggleButton(
        value=True,
        description='Element #2'
    )
    show_controls.append(show2_wdg)
    
    show4_wdg = wdg.ToggleButton(
        value=True,
        description='Element #4'
    )
    show_controls.append(show4_wdg)
    
    show5_wdg = wdg.ToggleButton(
        value=True,
        description='Element #5'
    )
    show_controls.append(show5_wdg)
        
    show_box = []
    if show_controls:
        show_title = [wdg.HTML(value = DISP_BLOCK_LABEL)]
        show_box = wdg.VBox(show_title + show_controls, layout=BOX_LAYOUT)
        controls_box.append(show_box)
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def plot(
        az, 
        freq, 
        r, 
        delay1, 
        delay2, 
        delay4, 
        delay5,
        show1,
        show2,
        show4,
        show5
        ):
        
        # Target location
        targ_x = (r*1E3)*np.cos(np.pi/2 - rd.deg2rad(az))
        targ_y = (r*1E3)*np.sin(np.pi/2 - rd.deg2rad(az))
        
        # Reference delay
        ref_delay = np.sqrt(targ_x**2 + targ_y**2)/cnst.c
        
        # Initialize total
        total_y = np.zeros(xvec.shape)
        
        # Element #1
        true_delay1 = np.sqrt((targ_x - elem_x[0])**2 + targ_y**2)/cnst.c - ref_delay
        y1 = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - true_delay1 - delay1*1E-9))
        zeroix = np.abs(xvec/1E9 - true_delay1 - delay1*1E-9) > (num_cycle/2/freq/1E6)
        y1[zeroix] = 0.0
        wave_lines[0].set_ydata(y1)
        wave_lines[0].set_visible(show1)
        total_y += y1
        
        # Element #2
        true_delay2 = np.sqrt((targ_x - elem_x[1])**2 + targ_y**2)/cnst.c - ref_delay
        y2 = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - true_delay2 - delay2*1E-9))
        zeroix = np.abs(xvec/1E9 - true_delay2 - delay2*1E-9) > (num_cycle/2/freq/1E6)
        y2[zeroix] = 0.0
        wave_lines[1].set_ydata(y2)
        wave_lines[1].set_visible(show2)
        total_y += y2
        
        # Element #3
        true_delay3 = np.sqrt((targ_x - elem_x[2])**2 + targ_y**2)/cnst.c - ref_delay
        y3 = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - true_delay3))
        zeroix = np.abs(xvec/1E9 - true_delay3) > (num_cycle/2/freq/1E6)
        y3[zeroix] = 0.0
        wave_lines[2].set_ydata(y3)
        total_y += y3
        
        # Element #4
        true_delay4 = np.sqrt((targ_x - elem_x[3])**2 + targ_y**2)/cnst.c - ref_delay
        y4 = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - true_delay4 - delay4*1E-9))
        zeroix = np.abs(xvec/1E9 - true_delay4 - delay4*1E-9) > (num_cycle/2/freq/1E6)
        y4[zeroix] = 0.0
        wave_lines[3].set_ydata(y4)
        wave_lines[3].set_visible(show4)
        total_y += y4
        
        # Element #5
        true_delay5 = np.sqrt((targ_x - elem_x[4])**2 + targ_y**2)/cnst.c - ref_delay
        y5 = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - true_delay5 - delay5*1E-9))
        zeroix = np.abs(xvec/1E9 - true_delay5 - delay5*1E-9) > (num_cycle/2/freq/1E6)
        y5[zeroix] = 0.0
        wave_lines[4].set_ydata(y5)
        wave_lines[4].set_visible(show5)
        total_y += y5

        # Total
        total_line.set_ydata(total_y)

        # Update relative energy
        e0 = (num_elem**2/2)*num_cycle/freq/1E6
        rel_e = dx*np.sum(np.abs(total_y)**2)/e0
        rel_energy.set_text(f"Energy Loss: {rd.to_db(rel_e):.2f} dB")
        
    # Reset
    def reset_delays(btn):
        delay1_wdg.value = 0.0
        delay2_wdg.value = 0.0
        delay4_wdg.value = 0.0
        delay5_wdg.value = 0.0
        
    # Add interaction
    wdg.interactive(
        plot, 
        az=az_wdg,
        freq=freq_wdg,
        r=range_wdg,
        delay1=delay1_wdg,
        delay2=delay2_wdg,
        delay4=delay4_wdg,
        delay5=delay5_wdg,
        show1=show1_wdg,
        show2=show2_wdg,
        show4=show4_wdg,
        show5=show5_wdg
    )
    
    reset_btn.on_click(reset_delays)

def detect_game(
    n = 40, 
    snr = 10
    ):  
    """
    A game in which the user picks pulse categories return by return.
    
    Inputs:
    - n [int]: Number of data points; default 40 points
    - snr [float]: Target signal-to-noise ratio (dB); default 10 dB
    
    Outputs:
    (none)
    """
    
    # Minimum energy value
    y_min = -40
    
    # Initial values
    noiseAmp = 1
    time = np.arange(0, n).repeat(3)
    values = y_min*np.ones(len(time))
    positions = np.zeros((n), dtype='bool')
    
    # Figure/axes
    fig, ax = plt.new_plot()
    ax.set_xlabel(DELAY_US_LABEL)
    ax.set_ylabel(SNR_DB_LABEL)
    ax.set_ylim([y_min, y_min + 80])
    
    # Accuracy texts
    pd_text = fig.text(0.72, 0.85, "Prob. of Detection: ---", size=12.0)
    pfa_text = fig.text(0.72, 0.8, "Prob. of False Alarm: ---", size=12.0)
    
    # Samples
    pulse = ax.plot(time, values, 'k')[0]
    
    # Current guess
    highlight = ax.plot([time[0], time[0]], [y_min, y_min], 'k', linewidth=5.0)[0]
    
    # Results
    results = []
    for ii in range(n):
        results.append(ax.plot([time[3*ii], time[3*ii]], [y_min, y_min], color='k', linewidth=4.0)[0])
    
    controls_box = []
    
    game_controls = []
    
    # Start button
    start_btn = wdg.Button(description="Start")
    game_controls.append(start_btn)
    
    # Reset button
    reset_btn = wdg.Button(description="Reset")
    game_controls.append(reset_btn)
    
    game_controls.append(wdg.HTML(value = TARGET_BLOCK_LABEL))
    
    # Signal-to-noise ratio
    snr_wdg = wdg.FloatSlider(
            min=0, 
            max=30, 
            step=1, 
            value=snr, 
            description="Target SNR (dB)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
    game_controls.append(snr_wdg)
    
    # Display game controls
    game_box = []
    if game_controls:
        game_title = [wdg.HTML(value = f"<b><font color='black'>Game</b>")]
        game_box = wdg.VBox(game_title + game_controls, layout=BOX_LAYOUT)
        controls_box.append(game_box)   
    
    # Detection controls
    det_controls = []
    
    # Det/No Det buttons
    det_btn = wdg.Button(description="Detection")
    det_controls.append(det_btn)
    nodet_btn = wdg.Button(description="No Detection")
    det_controls.append(nodet_btn)
    
    # Detection control box
    det_box = []
    if det_controls:
        det_title = [wdg.HTML(value = f"<b><font color='black'>Detection</b>")]
        det_box = wdg.VBox(det_title + det_controls, layout=BOX_LAYOUT)
        controls_box.append(det_box)
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
    
    # Counters
    guess = [0]
    dets = [0]
    echoes = [0]
    noises = [0]
    alarms = [0]
    
    # Started
    started = [False]
    
    def start_game(btn):

        # Draw samples
        echoAmp = np.sqrt(rd.from_db(snr_wdg.value))*noiseAmp
        positions[:] = np.random.randint(0, 5, n) >= 4
        values[1::3] = 2*rd.to_db(echoAmp * positions + noiseAmp * np.random.randn(n))
        
        # Initialize
        pulse.set_ydata(values)
        highlight.set_ydata([y_min, values[1]])
        
        # Mark as started
        started[0] = True
        
        # Disable start, SNR
        start_btn.disabled = True
        snr_wdg.disabled = True
        
    def reset_game(btn):
        
        # Reset values
        guess[0] = 0
        dets[0] = 0
        echoes[0] = 0
        noises[0] = 0
        alarms[0] = 0
        values[:] = y_min*np.ones(len(time))
        
        # Reset results
        for res in results:
            res.set_ydata([y_min, y_min])
            res.set_color('k')
        
        # Initialize
        pulse.set_ydata(values)
        highlight.set_data([time[1], time[1]], [y_min, values[1]])
        
        # Mark as not started
        started[0] = False
        
        # Allow start, SNR
        start_btn.disabled = False
        snr_wdg.disabled = False
        
        # Update text
        update_text()
    
    def det_input(btn):
        
        if (guess[0] < n) and started[0]:

            if positions[guess[0]]:
                dets[0] += 1
                echoes[0] += 1
                result('det')
            else:
                noises[0] += 1
                alarms[0] += 1
                result('alarm')

            guess[0] += 1
            
            move_highlight()    
            update_text()
    
    def nodet_input(btn):
        
        if (guess[0] < n) and started[0]:

            if positions[guess[0]]:
                echoes[0] += 1
                result('miss')
            else:
                noises[0] += 1
                result('rej')
                
            guess[0] += 1
            
            move_highlight()
            update_text()
        
    def move_highlight():
        ix = 1 + guess[0]*3
        highlight.set_data([time[ix], time[ix]], [y_min, values[ix]])
        
    def result(s):
        
        if s == 'det':
            color = 'g'
        elif s == 'miss':
            color = 'm'
        elif s == 'alarm':
            color = 'r'
        elif s == 'rej':
            color = 'b'
            
        results[guess[0]].set_ydata([y_min, values[1 + 3*guess[0]]])
        results[guess[0]].set_color(color)
        
    def update_text():
        
        if echoes[0] > 0:
            pd_text.set_text(f"Prob. of Detection: {dets[0]/echoes[0]:.3f}")
        else:
            pd_text.set_text(f"Prob. of Detection: ---")
        
        if noises[0] > 0:
            pfa_text.set_text(f"Prob. of False Alarm: {alarms[0]/noises[0]:.3f}")
        else:
            pfa_text.set_text(f"Prob. of False Alarm: ---")
    
    # Button interactions
    start_btn.on_click(start_game)
    reset_btn.on_click(reset_game)
    det_btn.on_click(det_input)
    nodet_btn.on_click(nodet_input)

def design(
    bandw=1,
    bandw_lim=[0.1, 3],
    coherent=False,
    energy=100,
    energy_lim=[100, 2000],
    freq=1E2,
    freq_lim=[100, 10000],
    metrics=['beamw', 'price', 'range', 'rcs', 'snr'],
    max_beamw=1,
    max_price=1E5,
    max_rcs=-5,
    min_beamw=0,
    min_range=100,
    min_snr=15,
    min_snr_lim=[5, 30],
    noise_temp=1200,
    noise_temp_lim=[500, 1200],
    num_integ=1,
    num_integ_lim=[1, 32],
    r=10,
    r_lim=[10, 500],
    radius=0.5,
    radius_lim=[0.1, 4],
    rcs=10,
    rcs_lim=[-30, 10],
    scan_rate=0,
    scan_rate_lim=[1, 10],
    widgets=['bandw', 'coherent', 'freq', 'energy', 'noise_temp', 'num_integ', 'r', 'radius', 'rcs']
    ):
    """
    Notional radar design.
    
    Inputs:
    - bandw [float]: Transmit bandwidth (MHz); default 1 MHz
    - bandw_lim [List[float]]: Transmit bandwidth limits (MHz)
    - coherent [bool]: Flag for coherent integration; default False
    - energy [float]: Transmit energy (mJ); default 100 mJ
    - energy_lim [List[float]]: Transmit energy limits (mJ)
    - freq [float]: Transmit frequency (MHz); default 1E2 MHz
    - freq_lim [List[float]]: Transmit frequency limits (MHz)
    - metrics [List[str]]: List of desired metrics
    - max_beamw [float]: Maximum allowable beamwidth (deg); default 1 deg
    - max_price [float]: Maximum allowable price ($); default $100,000
    - max_rcs [float]: Maximum observable radar cross section (dBsm); default -5 dBsm
    - min_beamw [float]: Minimum allowable beamwidth (deg); default 0 deg
    - min_range [float]: Minimum observable range (km); default 100 km
    - min_snr [float]: Minimum signal-to-noise ratio (dB); default 15 dB
    - min_snr_lim [List[float]]: Minimum signal-to-noise ratio limits (dB)
    - noise_temp [float]: System noise temperature (K); default 1200
    - noise_temp_lim [List[float]]: System noise temperature limits (K)
    - num_integ [int]: Number of integrated pulses; default 1 pulse
    - num_integ_lim [List[float]]: Number of integrated pulses limits
    - r [float]: Target range (km); default 10 km
    - r_lim [List[float]]: Target range limits (km)
    - radius [float]: Dish radius (m): default 0.5 m
    - radius_lim [List[float]]: Dish radius limits (m)
    - rcs [float]: Target radar cross section (dBsm); default 10 dBsm
    - rcs_lim [List[float]]: Target radar cross section limits (dBsm)
    - scan_rate [float]: Scan rate (scans/min); default 0 scans/min
    - scan_rate_lim [List[float]]: Scan rate limits (scans/min)
    - widgets [List[str]]: List of desired widgets
    
    Outputs:
    (none)
    """

    # Control widgets
    controls_box = []
       
    # Block 1
    block1 = []
       
    # Radar controls
    if ('noise_temp' in widgets) or ('radius' in widgets) or ('scan_rate' in widgets):
        
        # Title
        block1.append(wdg.HTML(value = RADAR_BLOCK_LABEL))

        # Dish radius
        if ('radius' in widgets):
            radius_wdg = wdg.FloatSlider(
                min=radius_lim[0], 
                max=radius_lim[1], 
                step=(radius_lim[1] - radius_lim[0])/200, 
                value=radius, 
                description=DISH_RADIUS_LABEL, 
                style=LABEL_STYLE, 
                readout_format='.2f'
            )
            block1.append(radius_wdg)
        else:
            radius_wdg = wdg.fixed(radius)
        
        # Noise temperature
        if ('noise_temp' in widgets):
            noise_temp_wdg = wdg.FloatSlider(
                min=noise_temp_lim[0], 
                max=noise_temp_lim[1], 
                step=(noise_temp_lim[1] - noise_temp_lim[0])/200, 
                value=noise_temp, 
                description=NOISE_TEMP_LABEL, 
                style=LABEL_STYLE, 
                readout_format='.2f'
            )
            block1.append(noise_temp_wdg)
        else:
            noise_temp_wdg = wdg.fixed(noise_temp)

        # Scan rate
        if ('scan_rate' in widgets):
            scan_rate_wdg = wdg.FloatSlider(
                min=scan_rate_lim[0], 
                max=scan_rate_lim[1], 
                step=(scan_rate_lim[1] - scan_rate_lim[0])/200, 
                value=scan_rate, 
                description="Scan Rate (scans/min)", 
                style=LABEL_STYLE, 
                readout_format='.2f'
            )
            block1.append(scan_rate_wdg)
        else:
            scan_rate_wdg = wdg.fixed(scan_rate)
            
    else:
        radius_wdg = wdg.fixed(radius)
        noise_temp_wdg = wdg.fixed(noise_temp)    
        scan_rate_wdg = wdg.fixed(scan_rate)
        
    # Add to controls
    if block1:
        controls_box.append(wdg.VBox(block1))
        
    # Block 2
    block2 = []
        
    # Transmission controls
    if ('bandw' in widgets) or ('energy' in widgets) or ('freq' in widgets):
        
        # Title    
        block2.append(wdg.HTML(value = TX_BLOCK_LABEL))

        # Bandwidth
        if ('bandw' in widgets):
            bandw_wdg = wdg.FloatSlider(
                min=bandw_lim[0], 
                max=bandw_lim[1], 
                step=(bandw_lim[1] - bandw_lim[0])/200, 
                value=energy, 
                description=BANDW_MHZ_LABEL, 
                style=LABEL_STYLE, 
                readout_format='.2f'
            )
            block2.append(bandw_wdg)
        else:
            bandw_wdg = wdg.fixed(bandw)

        # Energy
        if ('energy' in widgets):
            energy_wdg = wdg.FloatSlider(
                min=energy_lim[0], 
                max=energy_lim[1], 
                step=(energy_lim[1] - energy_lim[0])/200, 
                value=energy, 
                description="Energy (mJ)", 
                style=LABEL_STYLE, 
                readout_format='.2f'
            )
            block2.append(energy_wdg)
        else:
            energy_wdg = wdg.fixed(energy)

        if ('freq' in widgets):
            freq_wdg = wdg.FloatSlider(
                min=freq_lim[0], 
                max=freq_lim[1], 
                step=(freq_lim[1] - freq_lim[0])/200, 
                value=freq, 
                description=FREQ_MHZ_LABEL, 
                style=LABEL_STYLE, 
                readout_format='.2f'
            )
            block2.append(freq_wdg)
        else:
            freq_wdg = wdg.fixed(freq)
            
    else:
        bandw_wdg = wdg.fixed(bandw)
        energy_wdg = wdg.fixed(energy)
        freq_wdg = wdg.fixed(freq)
       
    # Processing controls
    if ('num_integ' in widgets) or ('coherent' in widgets):
    
        # Title
        block2.append(wdg.HTML(value = f"<b><font color='black'>Processing</b>"))

        # Bandwidth
        if ('num_integ' in widgets):
            num_integ_wdg = wdg.IntSlider(
                min=num_integ_lim[0], 
                max=num_integ_lim[1], 
                value=num_integ, 
                description="Integrated Pulses", 
                style=LABEL_STYLE, 
                readout_format='d'
            )
            block2.append(num_integ_wdg)
        else:
            num_integ_wdg = wdg.fixed(num_integ)

        if ('coherent' in widgets):
            coh_wdg = wdg.Checkbox(
                value=False,
                description='Coherent Integration',
                disabled=False
                )
            block2.append(coh_wdg)
        else:
            coh_wdg = wdg.fixed(coherent)
            
    else:
        num_integ_wdg = wdg.fixed(num_integ)
        coh_wdg = wdg.fixed(coherent)
        
    # Add to controls
    if block2:
        controls_box.append(wdg.VBox(block2))
    
    # Block 3
    block3 = []
    
    # Target controls
    if ('r' in widgets) or ('rcs' in widgets):
    
        # Title
        block3.append(wdg.HTML(value = TARGET_BLOCK_LABEL))
    
        # Range
        if ('r' in widgets):
            r_wdg = wdg.FloatSlider(
                min=r_lim[0], 
                max=r_lim[1], 
                step=(r_lim[1] - r_lim[0])/200, 
                value=r, 
                description= RANGE_KM_LABEL, 
                style=LABEL_STYLE, 
                readout_format='.2f'
            )
            block3.append(r_wdg)
        else:
            r_wdg = wdg.fixed(r)

        # Radar cross section
        if ('rcs' in widgets):
            rcs_wdg = wdg.FloatSlider(
                min=rcs_lim[0], 
                max=rcs_lim[1], 
                step=(rcs_lim[1] - rcs_lim[0])/200, 
                value=rcs, 
                description= "RCS (dBsm)", 
                style=LABEL_STYLE, 
                readout_format='.2f'
            )
            block3.append(rcs_wdg)
        else:
            rcs_wdg = wdg.fixed(rcs)
            
    else:
        r_wdg = wdg.fixed(r)
        rcs_wdg = wdg.fixed(rcs)
    
    # Objectives
    if ('min_snr' in widgets):
    
        # Title
        block3.append(wdg.HTML(value = f"<b><font color='black'>Objective</b>"))
    
        # Range
        if ('min_snr' in widgets):
            min_snr_wdg = wdg.FloatSlider(
                min=min_snr_lim[0], 
                max=min_snr_lim[1], 
                step=(min_snr_lim[1] - min_snr_lim[0])/200, 
                value=min_snr, 
                description= "SNR (dB)", 
                style=LABEL_STYLE, 
                readout_format='.2f'
            )
            block3.append(min_snr_wdg)
        else:
            min_snr_wdg = wdg.fixed(min_snr)
            
    else:
        min_snr_wdg = wdg.fixed(min_snr)
    
    # Add to controls
    if block3:
        controls_box.append(wdg.VBox(block3))
    
    metric_wdg = []
    
    pad_wdg = wdg.HTML(value="<h2> | </h2>")
    
    # Price
    if 'price' in metrics:
        
        # Coherent processing
        integ_cost = 40
        if coherent:
            integ_cost = 100
        
        price0 = rd.price(
            bandw=bandw,
            energy=energy,
            freq=freq,
            integ_cost=integ_cost,
            noise_temp=noise_temp,
            num_integ=num_integ,
            radius=radius,
            scan_rate=scan_rate
        )
        price_wdg = wdg.HTML(value=f"")
        if price0 <= max_price:
            price_wdg.value = f"<font color=\"DarkGreen\"><h2>Price: ${price0:.2f}</h2></font>"
        else:
            price_wdg.value = f"<font color=\"Red\"><h2>Price: ${price0:.2f}</h2></font>"
        metric_wdg.append(price_wdg)
        
    # SNR
    if 'snr' in metrics:
        integ_gain = 0.5
        if coherent:
            integ_gain = 1.0
        snr0 = rd.to_db(rd.rx_snr(
            r*1E3,
            energy*1E-3,
            freq*1E6,
            noise_temp,
            gain=rd.dish_gain(radius, freq*1E6)**2,
            rcs=rd.from_db(rcs)
        )) + integ_gain*rd.to_db(num_integ)
        snr_wdg = wdg.HTML(value=f"")
        if snr0 >= min_snr:
            snr_wdg.value = f"<font color=\"DarkGreen\"><h2>SNR: {snr0:.2f} dB</h2></font>"
        else:
            snr_wdg.value = f"<font color=\"Red\"><h2>SNR: {snr0:.2f} dB</h2></font>"
        
        # Pad
        if metric_wdg:
            metric_wdg.append(pad_wdg)
        metric_wdg.append(snr_wdg)
        
    # Beamwidth
    if 'beamw' in metrics:
        beamw0 = rd.dish_beamw(radius, freq*1E6)
        beamw_wdg = wdg.HTML(value=f"")
        if beamw0 <= max_beamw:
            beamw_wdg.value = f"<font color=\"DarkGreen\"><h2>Beamwidth: {beamw0:.2f} deg</h2></font>"
        else:
            beamw_wdg.value = f"<font color=\"Red\"><h2>Beamwidth: {beamw0:.2f} deg</h2></font>"
        
        # Pad
        if metric_wdg:
            metric_wdg.append(pad_wdg)
        metric_wdg.append(beamw_wdg)
        
    # RCS
    if 'rcs' in metrics:
        rcs_metric_wdg = wdg.HTML(value=f"")
        if rcs <= max_rcs:
            rcs_metric_wdg.value = f"<font color=\"DarkGreen\"><h2>RCS: {rcs:.2f} dBsm</h2></font>"
        else:
            rcs_metric_wdg.value = f"<font color=\"Red\"><h2>RCS: {rcs:.2f} dBsm</h2></font>"
                
        # Pad
        if metric_wdg:
            metric_wdg.append(pad_wdg)
        metric_wdg.append(rcs_metric_wdg)
        
    # Range
    if 'range' in metrics:
        range_metric_wdg = wdg.HTML(value=f"")
        if r >= min_range:
            range_metric_wdg.value = f"<font color=\"DarkGreen\"><h2>Range: {r:.2f} km</h2></font>"
        else:
            range_metric_wdg.value = f"<font color=\"Red\"><h2>Range: {r:.2f} km</h2></font>"
        
        # Pad
        if metric_wdg:
            metric_wdg.append(pad_wdg)
        metric_wdg.append(range_metric_wdg)
        
    # Display metrics
    if metric_wdg:
        metric_wdg_box = wdg.HBox(metric_wdg)

    # Display widgets
    if controls_box:
        if metric_wdg:
            display(wdg.VBox([metric_wdg_box, wdg.GridBox(controls_box, layout=WDG_LAYOUT)]))
        else:
            display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
    
    # Price display
    def update_budget(bandw, coherent, energy, freq, min_snr, noise_temp, num_integ, radius, r, rcs, scan_rate):
                
        # Coherent processing
        integ_cost = 40
        if coherent:
            integ_cost = 100
            
        # Calculate price
        price = rd.price(
            bandw=bandw,
            energy=energy,
            freq=freq,
            integ_cost=integ_cost,
            noise_temp=noise_temp,
            num_integ=num_integ,
            radius=radius,
            scan_rate=scan_rate
        )
        
        # SNR
        integ_gain = 0.5
        if coherent:
            integ_gain = 1.0
        snr = rd.to_db(rd.rx_snr(
            r*1E3,
            energy*1E-3,
            freq*1E6,
            noise_temp,
            gain=rd.dish_gain(radius, freq*1E6)**2,
            rcs=rd.from_db(rcs)
        )) + integ_gain*rd.to_db(num_integ)
                
        # Beamwidth
        beamw = rd.dish_beamw(radius, freq*1E6)
            
        # Update display price
        if 'price' in metrics:
            if price <= max_price:
                price_wdg.value = f"<font color=\"DarkGreen\"><h2>Price: ${price:.2f}</h2></font>"
            else:
                price_wdg.value = f"<font color=\"Red\"><h2>Price: ${price:.2f}</h2></font>"
            
        # Update SNR
        if 'snr' in metrics:
            if snr >= min_snr:
                snr_wdg.value = f"<font color=\"DarkGreen\"><h2>SNR: {snr:.2f} dB</h2></font>"
            else:
                snr_wdg.value = f"<font color=\"Red\"><h2>SNR: {snr:.2f} dB</h2></font>"
            
        # Update beamwidth
        if 'beamw' in metrics:
            if beamw <= max_beamw:
                beamw_wdg.value = f"<font color=\"DarkGreen\"><h2>Beamwidth: {beamw:.2f} deg</h2></font>"
            else:
                beamw_wdg.value = f"<font color=\"Red\"><h2>Beamwidth: {beamw:.2f} deg</h2></font>"
            
        # Range
        if 'range' in metrics:
            if r >= min_range:
                range_metric_wdg.value = f"<font color=\"DarkGreen\"><h2>Range: {r:.2f} km</h2></font>"
            else:
                range_metric_wdg.value = f"<font color=\"Red\"><h2>Range: {r:.2f} km</h2></font>"
            
        # Radar cross section
        if 'rcs' in metrics:
            if rcs <= max_rcs:
                rcs_metric_wdg.value = f"<font color=\"DarkGreen\"><h2>RCS: {rcs:.2f} dBsm</h2></font>"
            else:
                rcs_metric_wdg.value = f"<font color=\"Red\"><h2>RCS: {rcs:.2f} dBsm</h2></font>"
   
    # Add interaction
    wdg.interactive(
        update_budget,
        bandw=bandw_wdg,
        coherent=coh_wdg,
        energy=energy_wdg,
        freq=freq_wdg,
        min_snr=min_snr_wdg,
        noise_temp=noise_temp_wdg,
        num_integ=num_integ_wdg,
        r=r_wdg,
        radius=radius_wdg,
        rcs=rcs_wdg,
        scan_rate=scan_rate_wdg
    )

def dish_pat(
    show_beamw: bool = False,
    xlim = [-5, 5],
    ylim = [0, 70]
    ):
    """
    Dish radar gain pattern.
    
    Inputs:
    - show_beamw [bool]: Flag for displaying beamwidth; default False
    - xlim [List[float]]: x-axis limits for plotting (deg)
    - ylim [List[float]]: y-axis limits for plotting (dB)
    
    Outputs:
    (none)
    """

    # Angle
    thetalim = [-(5*np.pi/180), (5*np.pi/180)]

    # Number of samples
    num_samp = 1000
    
    # Initialize plot
    _, ax_dp = plt.new_plot()
    ax_dp.set_xlabel('Angle (deg)')
    ax_dp.set_ylabel(TX_GAIN_DB_LABEL)
    ax_dp.set_xlim(xlim)
    ax_dp.set_ylim(ylim)
    
    # Live text
    if show_beamw:
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        beamw0 = rd.dish_beamw(1.0, 1E9)
        text1 = ax_dp.text(xlim[1] + 0.025*dx, ylim[1] - 0.07*dy, f"Beamwidth: {beamw0:.2f} deg", size=12.0)
    
    # Initialize angle vector
    theta = np.linspace(thetalim[0], thetalim[1], num_samp)
    stheta = np.sin(theta)
    
    # Plot update
    def plot(r, freq):

        # Convert dimensions
        freq = freq*1E6

        # Wavelength
        wavelen = rd.wavelen(freq)
        
        # Beamwidth
        if show_beamw:
            beamw = rd.dish_beamw(r, freq)
            text1.set_text(f"Beamwidth: {beamw:.2f} deg")
        
        # Range range equation
        u = (r/wavelen)*stheta
        pat = 20*np.log10(np.abs(rd.dish_pat(u))) + 10*np.log10(rd.dish_gain(r, freq))
        
        # Clear plot
        [l.remove() for l in ax_dp.lines]
        
        # Plot new
        ax_dp.plot((180/np.pi)*theta, pat, color='red')
    
    # Add interaction
    wdg.interact(plot,
    r=wdg.FloatSlider(min=0.1, max=30, step=0.1, value=1, description=DISH_RADIUS_LABEL, style=LABEL_STYLE, readout_format='.2f'),
    freq=wdg.FloatLogSlider(base=10, min=2, max=4, step=0.01, value=1E3, description=FREQ_MHZ_LABEL, style=LABEL_STYLE, readout_format='.2f'))    

def doppler(
    dr: float = -300,
    freq: float = 50,
    interval: float = 75,
    max_range: float = None,
    num_cycle: int = 5,
    num_step: int = 200,
    propvel: float = 1000,
    r: float = 100,
    tx_az: float = 0.0,
    tx_beamw: float = 20.0,
    tx_omni: bool = False,
    widgets = ['freq', 'run', 'r', 'dr'],
    xlim = [-100, 100],
    ylim = [-10, 190]
    ):
    """
    Doppler effect demonstration.
    
    Inputs:
    - dr [float]: Target range rate (m/s); default -300 m/s
    - freq [float]: Transmit frequency (Hz); default 50 Hz
    - interval [float]: Time between animation steps (ms); default 75 ms
    - max_range [float]: Maximum range for plotting (m); default None
    - num_cycle [int]: Number of wave cycles for plotting; default 5 cycles
    - num_step [int]: Number of animation steps; default 200 steps
    - propvel [float]: Propagation velocity (m/s); default 1000 m/s
    - r [float]: Target range (m); default 100 m
    - tx_az [float]: Transmit steering direction (deg); default 0 deg
    - tx_beamw [float]: Transmit beamwidth (deg); default 20 deg
    - tx_omni [bool]: Flag for omnidirectional transmit; default False
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (m)
    
    Outputs:
    (none)
    """

    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(EAST_M_LABEL)
    ax1.set_ylabel(NORTH_M_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    
    # Maximum range
    max_plot_range = calc_max_range(xlim, ylim)
    if not max_range:
        max_range = max_plot_range
    
    # Time vector
    t0 = 0
    t1 = 2*max_range/propvel
    dt = (t1 - t0)/(num_step - 1)

    # Timestamp 
    timestamp = ax1.text(xlim[1] + 5, ylim[1] - 12, f"Time: {t0*1E3:.2f} ms", size=12.0)

    # Sensor marker
    cs = ptch.Circle((0, 0), 5, color='blue')
    ax1.add_patch(cs)
    
    # Target marker
    ct = ptch.Circle((0, r), 5, color='red')
    ax1.add_patch(ct)

    # Initial beam limits
    tx_theta1 = 90 - tx_az - tx_beamw/2
    tx_theta2 = 90 - tx_az + tx_beamw/2
    
    # Transmit beam
    tx_peaks = []
    tx_valleys = []
    for ii in range(num_cycle):
        if tx_omni:
            tx_peakii = ptch.Circle((0, 0), 0, fill=False, color='blue', linewidth=2.0, alpha=1.0)
            tx_valleyii = ptch.Circle((0, 0), 0, fill=False, color='blue', linewidth=1.0, alpha=1.0)
        else:
            tx_peakii = ptch.Arc((0, 0), 0, 0, theta1=tx_theta1, theta2=tx_theta2, color='blue', linewidth=2.0, alpha=1.0)
            tx_valleyii = ptch.Arc((0, 0), 0, 0, theta1=tx_theta1, theta2=tx_theta2, color='blue', linewidth=1.0, alpha=1.0)
        tx_peaks.append(tx_peakii)
        tx_valleys.append(tx_valleyii)
        ax1.add_patch(tx_peakii)
        ax1.add_patch(tx_valleyii)
        
    # Transmit beam
    if not tx_omni:
        tx_beam = ptch.Wedge((0, 0), max_plot_range, tx_theta1, tx_theta2, color='gray', alpha=0.2)
        ax1.add_patch(tx_beam)
        
    # Echoes
    echo_peaks = []
    echo_valleys = []
    for ii in range(num_cycle):
        echo_peakii = ptch.Circle((0, r), 0, fill=False, color='red', linewidth=2.0, alpha=1.0)
        echo_valleyii = ptch.Circle((0, r), 0, fill=False, color='red', linewidth=1.0, alpha=1.0)
        echo_peaks.append(echo_peakii)
        echo_valleys.append(echo_valleyii)
        ax1.add_patch(echo_peakii)
        ax1.add_patch(echo_valleyii)
        
    # Control widgets
    controls_box = []
        
    # Sensor control widgets
    sensor_controls = []
    
    # Frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=50, 
            max=500, 
            step=10, 
            value=freq, 
            description=TX_FREQ_HZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sensor_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    sensor_box = []
    if sensor_controls:
        sensor_title = [wdg.HTML(value = TX_BLOCK_LABEL)]
        sensor_box = wdg.VBox(sensor_title + sensor_controls, layout=BOX_LAYOUT)
        controls_box.append(sensor_box)
        
    # Target control widgets
    target_controls = []
    
    # Target range
    if ('r' in widgets):
        # Build widget
        r_wdg = wdg.FloatSlider(
            min=0, 
            max=100, 
            step=1, 
            value=r,
            description="Initial Range (m)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        target_controls.append(r_wdg)
    else:
        r_wdg = wdg.fixed(r)
        
    # Target range rate
    if ('dr' in widgets):
        # Build widget
        dr_wdg = wdg.FloatSlider(
            min=-500, 
            max=500, 
            step=10, 
            value=dr,
            description=RANGE_RATE_M_S_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        target_controls.append(dr_wdg)
    else:
        dr_wdg = wdg.fixed(dr)
        
    target_box = []
    if target_controls:
        target_title = [wdg.HTML(value = TARGET_BLOCK_LABEL)]
        target_box = wdg.VBox(target_title + target_controls, layout=BOX_LAYOUT)
        controls_box.append(target_box)
    
    # Run widgets
    run_controls = []
    if ('run' in widgets):
        play_wdg = wdg.Play(
            interval=interval,
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=RUN_LABEL,
            disabled=False
        )
        run_controls.append(play_wdg)
        slider_wdg = wdg.IntSlider(
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=FRAME_LABEL,
            readout=False,
            disabled=False
        )
        wdg.jslink((play_wdg, 'value'), (slider_wdg, 'value'))
        run_controls.append(slider_wdg)
        
    run_box = []
    if run_controls:
        run_title = [wdg.HTML(value = RUN_BLOCK_LABEL)]
        run_box = wdg.VBox(run_title + run_controls, layout=BOX_LAYOUT)
        controls_box.append(run_box)   
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
        
    # Plot
    r_peak = np.zeros((num_cycle))
    r_valley = np.zeros((num_cycle))
    def animate(frame, freq, r, dr):

        # Time
        t = t0 + (frame - 1)*dt

        # Update timestamp
        timestamp.set_text(f"Time: {t*1E3:.2f} ms")
        
        # Target range, azimuth
        tgt_r = r + t*dr

        # Update target
        ct.center = 0, tgt_r
        
        # Frequency
        freq = freq# *1E6
                   
        # Update transmit pulse and echo
        for ii in range(num_cycle):
            
            # Transmit peaks/valleys
            peak_rii = np.maximum(propvel*(t - ii/freq), 0.0)
            valley_rii = np.maximum(propvel*(t - ii/freq - 1/2/freq), 0.0)
            if tx_omni:
                tx_peaks[ii].set_radius(peak_rii)
                tx_peaks[ii].set_alpha(np.maximum(1 - peak_rii/max_plot_range, 0.0))
                tx_valleys[ii].set_radius(valley_rii)
                tx_valleys[ii].set_alpha(np.maximum(1 - valley_rii/max_plot_range, 0.0))
            else:
                tx_peaks[ii].set_width(2*peak_rii)
                tx_peaks[ii].set_height(2*peak_rii)
                tx_peaks[ii].set_alpha(np.maximum(1 - peak_rii/max_plot_range, 0.0))
                tx_valleys[ii].set_width(2*valley_rii)
                tx_valleys[ii].set_height(2*valley_rii)
                tx_valleys[ii].set_alpha(np.maximum(1 - valley_rii/max_plot_range, 0.0))      
        
            # Impinging range
            r_peak[ii] = propvel*(r + dr*ii/freq)/(propvel - dr)
            r_valley[ii] = propvel*(r + dr*(ii/freq + 1/freq/2))/(propvel - dr)
        
            # Update echoes
            if (peak_rii >= r_peak[ii]):
                echo_peaks[ii].center = 0, r_peak[ii]
                echo_peaks[ii].set_radius(peak_rii - r_peak[ii])
            else:
                echo_peaks[ii].center = 0, r
                echo_peaks[ii].set_radius(0.0)
                echo_peaks[ii].set_alpha(1.0)

            if (valley_rii >= r_valley[ii]):
                echo_valleys[ii].center = 0, r_valley[ii]
                echo_valleys[ii].set_radius(valley_rii - r_valley[ii])
            else:
                echo_valleys[ii].center = 0, r
                echo_valleys[ii].set_radius(0.0)
                echo_valleys[ii].set_alpha(1.0)
        
        # Disable controls during play
        if (frame > 1):
            for w in sensor_controls:
                if not w.disabled:
                    w.disabled = True
            for w in target_controls:
                if not w.disabled:
                    w.disabled = True
        elif (frame == 1):
            for w in sensor_controls:
                if w.disabled:
                    w.disabled = False
            for w in target_controls:
                if w.disabled:
                    w.disabled = False
          
    # Add interaction
    wdg.interactive(
        animate, 
        frame=slider_wdg,
        freq=freq_wdg, 
        r=r_wdg,
        dr=dr_wdg
    )

def ekf(
    bandw=1,
    dx=10,
    dy=10,
    energy=5,
    freq=1.5E3,
    noise_temp=700,
    r=25,
    radius=2,
    widgets=['bandw', 'dx', 'dy', 'energy', 'freq', 'noise_temp', 'r', 'radius'],
    xlim=[0, 50],
    ylim=[-200, 200]
    ):
    """
    Extended Kalman filter demonstration.
    
    Inputs:
    - bandw [float]: Transmit bandwidth (MHz); default 1 MHz
    - dx [float]: Initial target velocity in x component (m/s); default 10 m/s
    - dy [float]: Initial target velocity in y component (m/s); default 10 m/s
    - energy [float]: Transmit energy (mJ); default 5 mJ
    - freq [float]: Transmit frequency (MHz); default 1.5E3 MHz
    - noise_temp [float]: System noise temperature (K); default 700
    - r [float]: Target range (km); default 25 km
    - radius [float]: Dish radius (m): default 2 m
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (s)
    - ylim [List[float]]: y-axis limits for plotting (m)
    
    Outputs:
    (none)
    """

    # Generate detections
    def make_dets(
        bandw,
        dx, 
        dy,
        energy,
        freq,
        r,
        radius,
        t
        ):
        
        num_det = t.size
        z = np.zeros((2, num_det))
        R = np.zeros((2, 2, num_det))
        
        targ_en = np.zeros((2))
        for ii in range(num_det):
            
            # Target state
            targ_en[0] = r*1E3 + dx*t[ii]
            targ_en[1] = dy*t[ii]
            targ_ra = rd.en2ra(targ_en)
            ri = np.sqrt(targ_en[0]**2 + targ_en[1]**2)
            
            # Gain
            gain = rd.dish_gain(radius, freq*1E6)
            
            # SNR
            snr = rd.rx_snr(ri, energy*1E-3, freq*1E6, noise_temp, gain=gain**2)
            
            # Accuracies
            range_acc = rd.range_res(bandw*1E6)/np.sqrt(snr)
            az_acc = rd.deg2rad(rd.dish_beamw(radius, freq*1E6))/np.sqrt(snr)
            
            # Measurement
            z[0, ii] = targ_ra[0] + range_acc*np.random.randn()
            z[1, ii] = targ_ra[1] + az_acc*np.random.randn()
            
            # Covariance
            R[0, 0, ii] = range_acc**2
            R[1, 1, ii] = az_acc**2
            
        return z, R   
        
    # Generate truth
    def make_truth(dx, dy, r, t):
        
        num_det = t.size
        x = np.zeros((4, num_det))
        
        for ii in range(num_det):
            x[0, ii] = r*1E3 + dx*t[ii]
            x[1, ii] = dy*t[ii]
            x[2, ii] = dx
            x[3, ii] = dy
            
        return x


    # Run EKF
    def run_ekf(dx, dy, r, z, R, t):
        
        # Initialize state estimates
        num_det = t.size
        x = np.zeros((4, num_det))
        P = np.zeros((4, 4, num_det))
        
        # Initial state
        x0 = np.zeros((4, 1))
        x0[0] = r*1E3
        x0[1] = 0
        x0[2] = dx
        x0[3] = dy
        
        # Initial covariance
        P0 = np.zeros((4, 4))
        P0[0, 0] = 1E5
        P0[1, 1] = 1E5
        P0[2, 2] = 1E4
        P0[3, 3] = 1E4
        
        # Perturb initial state
        x0 = x0 + np.matmul(np.linalg.cholesky(P0), np.random.randn(4, 1))
        
        # EKF
        for ii in range(num_det):
            
            # Predict
            if (ii > 0):
                phi = rd.prop_cv(t[ii] - t[ii - 1])
                x_pred = np.matmul(phi, x_past)
                P_pred = np.matmul(np.matmul(phi, P_past), phi.transpose())
            else:
                x_pred = x0
                P_pred = P0
                
            # Measurement Jacobian
            H = rd.en2ra_jac(x_pred)
            
            # Residual covariance
            S = np.matmul(np.matmul(H, P_pred), H.transpose()) + R[:, :, ii]
            
            # Gain
            K = np.matmul(np.matmul(P_pred, H.transpose()), np.linalg.inv(S))
            
            # Residual
            residual = z[:, ii].reshape((2, 1)) - rd.en2ra(x_pred).reshape((2, 1))
            
            # Update state and covariance
            x_corr = np.matmul(K, residual)
            x[:, ii] = (x_pred + x_corr).squeeze()
            P[:, :, ii] = np.matmul((np.eye(4) - np.matmul(K, H)), P_pred)
            
            # Store as past
            x_past = x[:, ii].reshape((4, 1))
            P_past = P[:, :, ii]
            
        return x, P
            
    # Initialize plot
    _, axs = plt.new_plot2()
    ax_east = axs[0]
    ax_north = axs[1]
    ax_north.set_xlabel(TIME_S_LABEL)
    ax_east.set_ylabel('East Error (m)')
    ax_north.set_ylabel('North Error (m)')
    ax_east.set_xlim(xlim)
    ax_east.set_ylim(ylim)
    ax_north.set_xlim(xlim)
    ax_north.set_ylim(ylim)

    # Time vector
    t = np.arange(xlim[0], xlim[1] + 1, 1)
    
    # Measurements
    z, R = make_dets(
        bandw,
        dx, 
        dy,
        energy,
        freq,
        r,
        radius,
        t
        )
    
    # EKF
    x, P = run_ekf(dx, dy, r, z, R, t)
    
    # Error
    err = x - make_truth(dx, dy, r, t)
    
    # Plot errors
    east_err = ax_east.plot(t, err[0, :], color='r', linewidth=2.0)[0]
    east_cov_up = ax_east.plot(t, 3*np.sqrt(P[0, 0, :]), color='r', linewidth=2.0, linestyle='dashed')[0]
    east_cov_down = ax_east.plot(t, 3*-np.sqrt(P[0, 0, :]), color='r', linewidth=2.0, linestyle='dashed')[0]
    
    north_err = ax_north.plot(t, err[1, :], color='r', linewidth=2.0)[0]
    north_cov_up = ax_north.plot(t, 3*np.sqrt(P[1, 1, :]), color='r', linewidth=2.0, linestyle='dashed')[0]
    north_cov_down = ax_north.plot(t, 3*-np.sqrt(P[1, 1, :]), color='r', linewidth=2.0, linestyle='dashed')[0]
    
    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    block1 = []
   
    # Radar block
    block1.append(wdg.HTML(value = RADAR_BLOCK_LABEL))

    # Noise temperature
    if ('noise_temp' in widgets):
        noise_temp_wdg = wdg.FloatSlider(
            min=50, 
            max=2000, 
            step=10, 
            value=noise_temp, 
            description=NOISE_TEMP_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block1.append(noise_temp_wdg)
    else:
        noise_temp_wdg = wdg.fixed(noise_temp)
        
    # Radius
    if ('radius' in widgets):
        radius_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=radius, 
            description=DISH_RADIUS_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block1.append(radius_wdg)
    else:
        radius_wdg = wdg.fixed(radius)
    
    # Transmission block
    block1.append(wdg.HTML(value = TX_BLOCK_LABEL))
    
    # Transmit bandwidth
    if ('bandw' in widgets):
        bandw_wdg = wdg.FloatSlider(
            min=0.1, 
            max=10, 
            step=0.1, 
            value=bandw, 
            description=TX_BANDW_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block1.append(bandw_wdg)
    else:
        bandw_wdg = wdg.fixed(bandw)

    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=3E3, 
            step=100, 
            value=freq, 
            description=TX_FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block1.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)

    # Transmit power
    if ('energy' in widgets):
        energy_wdg = wdg.FloatSlider(
            min=1, 
            max=100, 
            step=1, 
            value=energy, 
            description=ENERGY_MJ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block1.append(energy_wdg)
    else:
        energy_wdg = wdg.fixed(energy)
        
    # Add to controls
    controls_box.append(wdg.VBox(block1))
    
    # Second block
    block2 = []
    
    # Target block
    block2.append(wdg.HTML(value = TARGET_BLOCK_LABEL))
    
    # Range
    if ('r' in widgets):
        r_wdg = wdg.FloatSlider(
            min=1, 
            max=100, 
            step=1, 
            value=r, 
            description="Initial Range (km)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block2.append(r_wdg)
    else:
        r_wdg = wdg.fixed(r)

    # East rate
    if ('dx' in widgets):
        dx_wdg = wdg.FloatSlider(
            min=-100, 
            max=100, 
            step=1, 
            value=dx, 
            description="East Rate (m/s)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block2.append(dx_wdg)
    else:
        dx_wdg = wdg.fixed(dx)
        
    # North rate
    if ('dy' in widgets):
        dy_wdg = wdg.FloatSlider(
            min=-100, 
            max=100, 
            step=1, 
            value=dx, 
            description="North Rate (m/s)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block2.append(dy_wdg)
    else:
        dy_wdg = wdg.fixed(dy)
        
    # Filter block
    block2.append(wdg.HTML(value = f"<b><font color='black'>Filter</b>"))
        
    new_btn = wdg.Button(description="New")
    block2.append(new_btn)
    
    # Add to controls
    controls_box.append(wdg.VBox(block2))
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
        
    def new_run(btn):
        
        # Measurements
        z, R = make_dets(
            bandw_wdg.value,
            dx_wdg.value, 
            dy_wdg.value,
            energy_wdg.value,
            freq_wdg.value,
            r_wdg.value,
            radius_wdg.value,
            t
            )

        # EKF
        x, P = run_ekf(dx_wdg.value, dy_wdg.value, r_wdg.value, z, R, t)

        # Error
        err = x - make_truth(dx_wdg.value, dy_wdg.value, r_wdg.value, t)

        # Plot errors
        east_err.set_ydata(err[0, :])
        east_cov_up.set_ydata(3*np.sqrt(P[0, 0, :]))
        east_cov_down.set_ydata(3*-np.sqrt(P[0, 0, :]))

        north_err.set_ydata(err[1, :])
        north_cov_up.set_ydata(3*np.sqrt(P[1, 1, :]))
        north_cov_down.set_ydata(3*-np.sqrt(P[1, 1, :]))
        
    new_btn.on_click(new_run)
    
def friis(
    freq=1E3,
    num_area=100,
    num_range=100,
    power=1E3,
    radius=1,
    widgets=['freq', 'power', 'radius'],
    xlim=[1, 500],
    ylim=[-30, 10]
    ):
    """
    Friis transmission equation demonstration.
    
    Inputs:
    - freq [float]: Transmit frequency (MHz); default 1E3 MHz
    - num_area [int]: Number of receive area bins; default 100 bins
    - num_range [int]: Number of range bins; default 100 bins
    - power [float]: Transmit power (kW); default 1E3 kW
    - radius [float]: Dish radius (m): default 1 m
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (dBsm)
    
    Outputs:
    (none)
    """

    # Initialize plot
    fig1, ax1 = plt.new_plot(axes_width=0.5)
    ax1.set_xlabel(RANGE_KM_LABEL)
    ax1.set_ylabel('Receive Aperture Area (dBsm)')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # Range bins
    range_bin = np.linspace(xlim[0], xlim[1], num_range)
    
    # Area bins
    area_bin = np.linspace(ylim[0], ylim[1], num_area)
    
    # Mesh
    range_mesh, area_mesh = np.meshgrid(range_bin, area_bin)
    range_mesh_m = range_mesh*1E3
    area_mesh_lin = rd.from_db(area_mesh)
    
    # Initial plot
    img = rd.to_db(rd.friis(range_mesh_m, power*1E3, gain=rd.dish_gain(radius, freq*1E6), area=area_mesh_lin))
    pc = ax1.contourf(range_bin, area_bin, img, np.linspace(-80, 0, 9), cmap='inferno', extend='both')
    
    # Colorbar
    cbar = pyp.colorbar(pc, ax=ax1)
    cbar.ax.set_ylabel('Received Power (dBW)', size=12, name=plt.DEF_SANS)
    
    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    sub_controls1 = []
   
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=3E3, 
            step=100, 
            value=freq, 
            description=TX_FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)

    # Transmit power
    if ('power' in widgets):
        power_wdg = wdg.FloatSlider(
            min=1E2, 
            max=1E4, 
            step=1E2, 
            value=power, 
            description="Transmit Power (kW)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(power_wdg)
    else:
        power_wdg = wdg.fixed(power)
        
    # Radius
    if ('radius' in widgets):
        radius_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=radius, 
            description=DISH_RADIUS_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(radius_wdg)
    else:
        radius_wdg = wdg.fixed(radius)
        
    sub_controls1_box = []
    if sub_controls1:
        sub_controls1_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        sub_controls1_box = wdg.VBox(sub_controls1_title + sub_controls1, layout=BOX_LAYOUT)
        controls_box.append(sub_controls1_box)
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # Plot
    def plot(freq, power, radius):
        img = rd.to_db(rd.friis(range_mesh_m, power*1E3, gain=rd.dish_gain(radius, freq*1E6), area=area_mesh_lin))
        for art in ax1.artists:
            art.remove()
        ax1.contourf(range_bin, area_bin, img, np.linspace(-80, 0, 9), cmap='inferno', extend='both')
        fig1.canvas.draw()
   
    # Add interaction
    wdg.interactive(
        plot, 
        freq=freq_wdg,
        power=power_wdg,
        radius=radius_wdg
    )

def gnn(
    det_acc: float = 10,
    num_det: int = 2,
    num_track: int = 3,
    track_acc: float = 5,
    widgets = ['det_acc', 'num_det', 'num_track', 'track_acc'],
    xlim = [-100, 100],
    ylim = [-100, 100]
    ):
    """
    Global nearest neighbor data association demonstration.
    
    Inputs:
    - det_acc [float]: Detection accuracy (m); default 10 m
    - num_det [int]: Number of detections; default 2
    - num_track [int]: Number of tracks; default 3
    - track_acc [float]: Track accuracy (m); default 5 m
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (m)
    
    Outputs:
    (none)
    """

    # Draw scene
    def draw_scene(
        det_acc, 
        num_det, 
        num_track, 
        track_acc, 
        xlim, 
        ylim
        ):
        
        # Shrink
        shrink = 0.5
        
        # Draw track states
        track_x = np.random.uniform(low=shrink*xlim[0], high=shrink*xlim[1], size=(num_track))
        track_y = np.random.uniform(low=shrink*ylim[0], high=shrink*ylim[1], size=(num_track))
        
        # Initialize detection states
        det_x = np.zeros((num_det))
        det_y = np.zeros((num_det))
        
        # Random association
        num_comm = np.minimum(num_det, num_track)
        assoc = np.random.permutation(num_comm)
        for ii in range(num_comm):
            det_x[ii] = track_x[assoc[ii]] + np.sqrt(det_acc**2 + track_acc**2)*np.random.randn()
            det_y[ii] = track_y[assoc[ii]] + np.sqrt(det_acc**2 + track_acc**2)*np.random.randn()
            
        # Leftover detections
        if (num_det > num_track):
            det_x[num_track:] = np.random.uniform(low=shrink*xlim[0], high=shrink*xlim[1], size=(num_det - num_track))
            det_y[num_track:] = np.random.uniform(low=shrink*ylim[0], high=shrink*ylim[1], size=(num_det - num_track))
            
        return det_x, det_y, track_x, track_y
    
    # Maximum number of detections/tracks
    max_det = 5
    
    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(EAST_M_LABEL)
    ax1.set_ylabel(NORTH_M_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
   
    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    sub_controls1 = []
   
    # Detection uncertainty
    if ('det_acc' in widgets):
        det_acc_wdg = wdg.FloatSlider(
            min=1, 
            max=30, 
            step=0.1, 
            value=det_acc, 
            description="Detection Accuracy (m)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(det_acc_wdg)
    else:
        det_acc_wdg = wdg.fixed(det_acc)

    # Number of detections
    if ('num_det' in widgets):
        num_det_wdg = wdg.IntSlider(
            min=1, 
            max=max_det, 
            value=num_det, 
            description="Detections", 
            style=LABEL_STYLE, 
            readout_format='d'
        )
        sub_controls1.append(num_det_wdg)
    else:
        num_det_wdg = wdg.fixed(num_det)

    # Number of tracks
    if ('num_track' in widgets):
        num_track_wdg = wdg.IntSlider(
            min=1, 
            max=max_det, 
            value=num_track, 
            description="Tracks", 
            style=LABEL_STYLE, 
            readout_format='d'
        )
        sub_controls1.append(num_track_wdg)
    else:
        num_track_wdg = wdg.fixed(num_track)
        
    # Track uncertainty
    if ('track_acc' in widgets):
        track_acc_wdg = wdg.FloatSlider(
            min=1, 
            max=30, 
            step=0.1, 
            value=track_acc, 
            description="Track Accuracy (m)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(track_acc_wdg)
    else:
        track_acc_wdg = wdg.fixed(track_acc)
        
    # New scene
    new_btn = wdg.Button(description="New")
    sub_controls1.append(new_btn)
        
    sub_controls1_box = []
    if sub_controls1:
        sub_controls1_title = [wdg.HTML(value = f"<b><font color='black'>Scene</b>")]
        sub_controls1_box = wdg.VBox(sub_controls1_title + sub_controls1, layout=BOX_LAYOUT)
        controls_box.append(sub_controls1_box)
       
    assoc_controls = []
    
    nn_btn = wdg.Button(description="Nearest")
    assoc_controls.append(nn_btn)
    gnn_btn = wdg.Button(description="Global Nearest")
    assoc_controls.append(gnn_btn)
    
    assoc_controls_box = []
    if assoc_controls:
        assoc_controls_title = [wdg.HTML(value = f"<b><font color='black'>Association</b>")]
        assoc_controls_box = wdg.VBox(assoc_controls_title + assoc_controls, layout=BOX_LAYOUT)
        controls_box.append(assoc_controls_box)
    
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # Initialize states
    det_x = np.zeros((max_det))
    det_y = np.zeros((max_det))
    track_x = np.zeros((max_det))
    track_y = np.zeros((max_det))
    
    # Draw track states
    det_xi, det_yi, track_xi, track_yi = draw_scene(
        det_acc, 
        num_det, 
        num_track, 
        track_acc, 
        xlim, 
        ylim
    )
    det_x[:num_det] = det_xi
    det_y[:num_det] = det_yi
    track_x[:num_track] = track_xi
    track_y[:num_track] = track_yi
        
    # Plot states
    det_pt = [ax1.scatter(det_x[:num_det], det_y[:num_det], 20.0, color="red")]
    track_pt = [ax1.scatter(track_x[:num_track], track_y[:num_track], 20.0, color="blue")]
    
    # Plot covariances
    det_cov = []
    for ii in range(num_track):
        det_cov.append(ptch.Circle((track_x[ii], track_y[ii]), track_acc, color="blue", fill=False))
        ax1.add_patch(det_cov[-1])
    track_cov = []
    for ii in range(num_det):
        track_cov.append(ptch.Circle((det_x[ii], det_y[ii]), det_acc, color="red", fill=False))
        ax1.add_patch(track_cov[-1])
    
    # Plot links
    links = []
    for ii in range(max_det):
        links.append(ax1.plot([det_x[ii], det_x[ii]], [det_y[ii], det_y[ii]], color='k', linewidth=3.0)[0])
    
    def update_links(assoc):
        for ii in range(max_det):
            if (ii < num_det_wdg.value):
                if not (assoc[ii] == -1):
                    links[ii].set_data([det_x[ii], track_x[assoc[ii]]], [det_y[ii], track_y[assoc[ii]]])
                else:
                    links[ii].set_data([det_x[ii], det_x[ii]], [det_y[ii], det_y[ii]])
            else:
                links[ii].set_data([det_x[ii], det_x[ii]], [det_y[ii], det_y[ii]])
    
    def nn_click(btn):
        assoc = rd.nearest(track_x, track_y, num_track_wdg.value, det_x, det_y, num_det_wdg.value)
        update_links(assoc)

    def gnn_click(btn):
        assoc = rd.gnn(track_x, track_y, num_track_wdg.value, det_x, det_y, num_det_wdg.value)
        update_links(assoc)
        
    # New scene
    def new_scene(btn):
        
        # Remove states
        det_pt[0].remove()
        track_pt[0].remove()
        
        # Remove covariances
        for cov in det_cov:
            cov.remove()
        
        for cov in track_cov:
            cov.remove()
            
        # Clear lists
        det_pt.clear()
        track_pt.clear()
        det_cov.clear()
        track_cov.clear()

        # Draw track states
        det_xi, det_yi, track_xi, track_yi = draw_scene(
            det_acc_wdg.value, 
            num_det_wdg.value, 
            num_track_wdg.value, 
            track_acc_wdg.value, 
            xlim, 
            ylim
        )
        det_x[:num_det_wdg.value] = det_xi
        det_y[:num_det_wdg.value] = det_yi
        track_x[:num_track_wdg.value] = track_xi
        track_y[:num_track_wdg.value] = track_yi

        # Plot states
        det_pt.append(ax1.scatter(det_x[:num_det_wdg.value], det_y[:num_det_wdg.value], 20.0, color="red"))
        track_pt.append(ax1.scatter(track_x[:num_track_wdg.value], track_y[:num_track_wdg.value], 20.0, color="blue"))

        # Plot covariances
        for ii in range(num_track_wdg.value):
            det_cov.append(ptch.Circle((track_x[ii], track_y[ii]), track_acc_wdg.value, color="blue", fill=False))
            ax1.add_patch(det_cov[-1])
        for ii in range(num_det_wdg.value):
            track_cov.append(ptch.Circle((det_x[ii], det_y[ii]), det_acc_wdg.value, color="red", fill=False))
            ax1.add_patch(track_cov[-1])
            
        update_links(-1*np.ones((max_det), dtype='int'))
      
    # Buttons
    new_btn.on_click(new_scene)
    nn_btn.on_click(nn_click)
    gnn_btn.on_click(gnn_click)
 
def lfm(
    energy: float = 50,
    num_bins: int = 1000,
    pulsewidth: float = 5,
    start_freq: float = 1,
    stop_freq: float = 5,
    widgets = ['energy', 'pulsewidth', 'start_freq', 'stop_freq'],
    ):
    """
    Linear frequency-modulated (LFM) waveform demonstration.
    
    Inputs:
    - energy [float]: Transmit energy (mJ); default 50 mJ
    - num_bins [int]: Number of time bins; default 1000 bins
    - pulsewidth [float]: Transmit pulsewidth (µs); default 5 µs
    - start_freq [float]: Transmit start frequency (MHz); default 1 MHz
    - stop_freq [float]: Transmit stop frequency (MHz); default 5 MHz
    - widgets [List[str]]: List of desired widgets
    
    Outputs:
    (none)
    """

    # Midpoint frequency
    max_pulsewidth = 10
    
    # x limits
    xlim = [0, max_pulsewidth]
    
    # Initialize plot
    fig, ax = plt.new_plot()
    ax.set_xlabel(TIME_US_LABEL)
    ax.set_ylabel(WAVEFORM_V_LABEL)
    ax.set_xlim(xlim)
    ax.set_ylim([-500, 500])
    
    # Time vector
    xvec = np.linspace(xlim[0], xlim[1], num_bins)
    
    # Waves
    amp = np.sqrt(2*(energy/1E3)/(pulsewidth/1E6))
    slope = (stop_freq - start_freq)*1E6/(pulsewidth/1E6)
    yii = amp*np.sin(2*np.pi*(start_freq*1E6)*(xvec/1E6) + np.pi*slope*(xvec/1E6)**2)
    zeroix = (xvec >= pulsewidth)
    yii[zeroix] = 0.0
    wave_line = ax.plot(xvec, yii, linewidth=2.0, color='red')[0]
         
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Transmit energy
    if ('energy' in widgets):
        energy_wdg = wdg.FloatSlider(
            min=1, 
            max=100, 
            step=1, 
            value=energy, 
            description=ENERGY_MJ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(energy_wdg)
    else:
        energy_wdg = wdg.fixed(energy)
    
    # Start frequency
    if ('start_freq' in widgets):
        start_freq_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=start_freq, 
            description="Start Frequency (MHz)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(start_freq_wdg)
    else:
        start_freq_wdg = wdg.fixed(start_freq)
        
    # Stop frequency
    if ('stop_freq' in widgets):
        stop_freq_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=stop_freq, 
            description="Stop Frequency (MHz)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(stop_freq_wdg)
    else:
        stop_freq_wdg = wdg.fixed(stop_freq)
        
    # Transmit pulsewidth
    if ('pulsewidth' in widgets):
        pulsewidth_wdg = wdg.FloatSlider(
            min=0.1, 
            max=10, 
            step=0.1, 
            value=pulsewidth, 
            description="Pulsewidth (µs)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(pulsewidth_wdg)
    else:
        pulsewidth_wdg = wdg.fixed(pulsewidth)
        
    wave_box = []
    if wave_controls:
        wave_title = [wdg.HTML(value = TX_BLOCK_LABEL)]
        wave_box = wdg.VBox(wave_title + wave_controls, layout=BOX_LAYOUT)
        controls_box.append(wave_box)
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def plot(energy, start_freq, stop_freq, pulsewidth):
        
        # Waves
        amp = np.sqrt(2*(energy/1E3)/(pulsewidth/1E6))
        slope = (stop_freq - start_freq)*1E6/(pulsewidth/1E6)
        yii = amp*np.sin(2*np.pi*(start_freq*1E6)*(xvec/1E6) + np.pi*slope*(xvec/1E6)**2)
        zeroix = (xvec >= pulsewidth)
        yii[zeroix] = 0.0
        wave_line.set_ydata(yii)

    # Add interaction
    wdg.interactive(
        plot, 
        energy=energy_wdg,
        start_freq=start_freq_wdg,
        stop_freq=stop_freq_wdg,
        pulsewidth=pulsewidth_wdg
    )

def matched_filter(
    delay: float = 0,
    num_bins: int = 1000,
    pulsewidth: float = 3,
    start_freq: float = 1,
    stop_freq: float = 1,
    widgets = ['delay', 'pulsewidth', 'start_freq', 'stop_freq'],
    ):
    """
    Matched filter demonstration.
    
    Inputs:
    - delay [float]: Transmit delay (µs); default 0 µs
    - num_bins [int]: Number of delay bins; default 1000 bins
    - pulsewidth [float]: Transmit pulsewidth (µs); default 5 µs
    - start_freq [float]: Transmit start frequency (MHz); default 1 MHz
    - stop_freq [float]: Transmit stop frequency (MHz); default 1 MHz
    - widgets [List[str]]: List of desired widgets
    
    Outputs:
    (none)
    """

    # Maximum pulsewidth
    max_pulsewidth = 10
    
    # x limits
    xlim = [-max_pulsewidth/2, max_pulsewidth/2]
    
    # Initialize plot
    fig, axs = plt.new_plot2()
    recv_ax = axs[0]
    filt_ax = axs[1]
    filt_ax.set_xlabel(DELAY_US_LABEL)
    recv_ax.set_ylabel('Received Signal (V)')
    filt_ax.set_ylabel('Filter Output (dB)')
    recv_ax.set_xlim(xlim)
    recv_ax.set_ylim([-1.5, 1.5])
    filt_ax.set_xlim(xlim)
    filt_ax.set_ylim([-20, 5])
    
    # Time vector
    xvec = np.linspace(xlim[0], xlim[1], num_bins)

    # Waves
    slope = (stop_freq - start_freq)*1E6/(pulsewidth/1E6)
    yii = np.sin(2*np.pi*(start_freq*1E6)*((xvec + pulsewidth/2)/1E6) + np.pi*slope*((xvec + pulsewidth/2)/1E6)**2)
    zeroix = np.logical_or(xvec < -pulsewidth/2, xvec > pulsewidth/2)
    yii[zeroix] = 0.0
    testii = np.sin(2*np.pi*(start_freq*1E6)*(((xvec - delay) + pulsewidth/2)/1E6) + np.pi*slope*(((xvec - delay) + pulsewidth/2)/1E6)**2)
    zeroix = np.logical_or((xvec - delay) < -pulsewidth/2, (xvec - delay) > pulsewidth/2)
    testii[zeroix] = 0.0
    truth_line = recv_ax.plot(xvec, yii, linewidth=2.0, color='blue')[0]
    test_line = recv_ax.plot(xvec, testii, linewidth=2.0, color='red')[0]
    
    # Filter output
    E = np.sum(yii*yii)
    filt_val = 2*rd.to_db(np.sum(yii*testii)/E)
    pulse = np.correlate(yii, yii, mode='same')/E
    corr_line = filt_ax.plot(xvec, 2*rd.to_db(pulse), color='black', linewidth=2.0, linestyle='dashed')[0]
    test_point = filt_ax.scatter(delay, filt_val, c='red', s=50.0)
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Test delay
    if ('delay' in widgets):
        delay_wdg = wdg.FloatSlider(
            min=-max_pulsewidth/2, 
            max=max_pulsewidth/2, 
            step=0.01, 
            value=delay, 
            description="Delay Hypothesis (µs)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(delay_wdg)
    else:
        delay_wdg = wdg.fixed(delay)
    
    # Start frequency
    if ('start_freq' in widgets):
        start_freq_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=start_freq, 
            description="Start Frequency (MHz)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(start_freq_wdg)
    else:
        start_freq_wdg = wdg.fixed(start_freq)
        
    # Stop frequency
    if ('stop_freq' in widgets):
        stop_freq_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=stop_freq, 
            description="Stop Frequency (MHz)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(stop_freq_wdg)
    else:
        stop_freq_wdg = wdg.fixed(stop_freq)
        
    # Transmit pulsewidth
    if ('pulsewidth' in widgets):
        pulsewidth_wdg = wdg.FloatSlider(
            min=0.1, 
            max=10, 
            step=0.1, 
            value=pulsewidth, 
            description="Pulsewidth (µs)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(pulsewidth_wdg)
    else:
        pulsewidth_wdg = wdg.fixed(pulsewidth)
        
    wave_box = []
    if wave_controls:
        wave_title = [wdg.HTML(value = f"<b><font color='black'>Waveform</b>")]
        wave_box = wdg.VBox(wave_title + wave_controls, layout=BOX_LAYOUT)
        controls_box.append(wave_box)
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def plot(delay, start_freq, stop_freq, pulsewidth):
        
        # Waves
        slope = (stop_freq - start_freq)*1E6/(pulsewidth/1E6)
        yii = np.sin(2*np.pi*(start_freq*1E6)*((xvec + pulsewidth/2)/1E6) + np.pi*slope*((xvec + pulsewidth/2)/1E6)**2)
        zeroix = np.logical_or(xvec < -pulsewidth/2, xvec > pulsewidth/2)
        yii[zeroix] = 0.0
        testii = np.sin(2*np.pi*(start_freq*1E6)*(((xvec - delay) + pulsewidth/2)/1E6) + np.pi*slope*(((xvec - delay) + pulsewidth/2)/1E6)**2)
        zeroix = np.logical_or((xvec - delay) < -pulsewidth/2, (xvec - delay) > pulsewidth/2)
        testii[zeroix] = 0.0
        truth_line.set_ydata(yii)
        test_line.set_ydata(testii)
        
        # Filter output
        E = np.sum(yii*yii)
        filt_val = 2*rd.to_db(np.sum(yii*testii)/E)
        pulse = np.correlate(yii, yii, mode='same')/E
        corr_line.set_ydata(2*rd.to_db(pulse))
        test_point.set_offsets([delay, filt_val])

    # Add interaction
    wdg.interactive(
        plot, 
        delay=delay_wdg,
        start_freq=start_freq_wdg,
        stop_freq=stop_freq_wdg,
        pulsewidth=pulsewidth_wdg
    )

def phase_steer(
    az: float = 0,
    freq: float = 5E2,
    num_bins: int = 500,
    num_cycle: int = 4,
    r: float = 2.5,
    widgets = ['az', 'phases', 'freq', 'range'],
    ):
    """
    Phase steering demonstration.
    
    Inputs:
    - az [float]: Target azimuth (deg); default 0 deg
    - freq [float]: Transmit frequency (MHz); default 5E2 MHz
    - num_bins [int]: Number of time steps in plot; default 500 steps
    - num_cycle [int]: Number of wave cycles for plotting; default 4 cycles
    - r [float]: Target range (km); default 2.5 km
    - widgets [List[str]]: List of desired widgets
    
    Outputs:
    (none)
    """
    
    # Midpoint frequency
    mid_freq = 750E6
    min_freq = 500E6
    
    # x limits
    xlim = [-0.6E9*num_cycle/min_freq, 0.6E9*num_cycle/min_freq]
    
    # Initialize plot
    fig, axs = plt.new_plot2()
    wave_ax = axs[0]
    total_ax = axs[1]
    total_ax.set_xlabel(TIME_NS_LABEL)
    wave_ax.set_ylabel('Waves')
    total_ax.set_ylabel('Total')
    wave_ax.set_xlim(xlim)
    wave_ax.set_ylim([-1, 1])
    total_ax.set_xlim(xlim)
    total_ax.set_ylim([-5, 5])
    
    # Timestamp
    rel_energy = fig.text(0.82, 0.9, "Energy Loss: ---", size=12.0)
    
    # Number of elements
    num_elem = 5
    
    # Time vector
    xvec = np.linspace(xlim[0], xlim[1], num_bins)
    dx = (xvec[1] - xvec[0])/1E9
    
    # Element locations
    wlen = rd.wavelen(mid_freq)
    elem_x = np.linspace(-wlen*(num_elem - 1)/2, wlen*(num_elem - 1)/2, num_elem)
    
    # Target location
    targ_x = (r*1E3)*np.cos(np.pi/2 - rd.deg2rad(az))
    targ_y = (r*1E3)*np.sin(np.pi/2 - rd.deg2rad(az))
    
    # Reference delay
    ref_delay = np.sqrt(targ_x**2 + targ_y**2)/cnst.c
    
    # Waves
    wave_lines = []
    total_y = np.zeros(xvec.shape)
    for ii in range(num_elem):
        delayii = np.sqrt((targ_x - elem_x[ii])**2 + targ_y**2)/cnst.c - ref_delay
        yii = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - delayii))
        zeroix = np.abs(xvec/1E9 - delayii) > (num_cycle/2/freq/1E6)
        yii[zeroix] = 0.0
        total_y += yii
        if (ii == 2):
            waveii = wave_ax.plot(xvec, yii, linewidth=3.0, color='black', linestyle='dashed')[0]
        else:
            waveii = wave_ax.plot(xvec, yii, linewidth=2.0)[0]
        wave_lines.append(waveii)
    
    # Total
    total_line = total_ax.plot(xvec, total_y, linewidth=3.0, color='red')[0]
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=500, 
            max=1000, 
            step=10, 
            value=freq, 
            description=FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    wave_title = [wdg.HTML(value = TX_BLOCK_LABEL)]
        
    # Target controls
    targ_controls = []
        
    # Azimuth
    if ('az' in widgets):
        az_wdg = wdg.FloatSlider(
            min=-60, 
            max=60, 
            step=1, 
            value=az, 
            description=AZ_DEG_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        targ_controls.append(az_wdg)
    else:
        az_wdg = wdg.fixed(r)
    
        
    # Range
    if ('range' in widgets):
        range_wdg = wdg.FloatSlider(
            min=0.01, 
            max=5, 
            step=0.01, 
            value=r, 
            description=RANGE_KM_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        targ_controls.append(range_wdg)
    else:
        range_wdg = wdg.fixed(r)
        
    # Transmission + target controls
    targ_title = [wdg.HTML(value = TARGET_BLOCK_LABEL)]
    targ_box = wdg.VBox(wave_title + wave_controls + targ_title + targ_controls, layout=BOX_LAYOUT)
    controls_box.append(targ_box)
        
    # Element controls
    elem_controls = []
        
    # Phase shifts
    phase1_wdg = wdg.FloatSlider(
        min=-180, 
        max=180, 
        step=1,
        value=0.0, 
        description="Phase #1 (deg)", 
        style=LABEL_STYLE, 
        readout_format='.2f'
    )
    elem_controls.append(phase1_wdg)
    
    phase2_wdg = wdg.FloatSlider(
        min=-180, 
        max=180, 
        step=1,
        value=0.0, 
        description="Phase #2 (deg)", 
        style=LABEL_STYLE, 
        readout_format='.2f'
    )
    elem_controls.append(phase2_wdg)
    
    phase4_wdg = wdg.FloatSlider(
        min=-180, 
        max=180, 
        step=1,
        value=0.0, 
        description="Phase #4 (deg)", 
        style=LABEL_STYLE, 
        readout_format='.2f'
    )
    elem_controls.append(phase4_wdg)
    
    phase5_wdg = wdg.FloatSlider(
        min=-180, 
        max=180, 
        step=1,
        value=0.0, 
        description="Phase #5 (deg)", 
        style=LABEL_STYLE, 
        readout_format='.2f'
    )
    elem_controls.append(phase5_wdg)
        
    reset_btn = wdg.Button(description="Reset")
    elem_controls.append(reset_btn)
        
    elem_box = []
    if elem_controls:
        elem_title = [wdg.HTML(value = f"<b><font color='black'>Elements</b>")]
        elem_box = wdg.VBox(elem_title + elem_controls, layout=BOX_LAYOUT)
        controls_box.append(elem_box)
        
    show_controls = []
        
    # Show widgets
    show1_wdg = wdg.ToggleButton(
        value=True,
        description='Element #1'
    )
    show_controls.append(show1_wdg)
    
    show2_wdg = wdg.ToggleButton(
        value=True,
        description='Element #2'
    )
    show_controls.append(show2_wdg)
    
    show4_wdg = wdg.ToggleButton(
        value=True,
        description='Element #4'
    )
    show_controls.append(show4_wdg)
    
    show5_wdg = wdg.ToggleButton(
        value=True,
        description='Element #5'
    )
    show_controls.append(show5_wdg)
        
    show_box = []
    if show_controls:
        show_title = [wdg.HTML(value = DISP_BLOCK_LABEL)]
        show_box = wdg.VBox(show_title + show_controls, layout=BOX_LAYOUT)
        controls_box.append(show_box)
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def plot(
        az, 
        freq, 
        r, 
        phase1, 
        phase2, 
        phase4, 
        phase5,
        show1,
        show2,
        show4,
        show5
        ):
        
        # Target location
        targ_x = (r*1E3)*np.cos(np.pi/2 - rd.deg2rad(az))
        targ_y = (r*1E3)*np.sin(np.pi/2 - rd.deg2rad(az))
        
        # Reference delay
        ref_delay = np.sqrt(targ_x**2 + targ_y**2)/cnst.c
        
        # Initialize total
        total_y = np.zeros(xvec.shape)
        
        # Element #1
        true_delay1 = np.sqrt((targ_x - elem_x[0])**2 + targ_y**2)/cnst.c - ref_delay
        y1 = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - true_delay1) - rd.deg2rad(phase1))
        zeroix = np.abs(xvec/1E9 - true_delay1) > (num_cycle/2/freq/1E6)
        y1[zeroix] = 0.0
        wave_lines[0].set_ydata(y1)
        wave_lines[0].set_visible(show1)
        total_y += y1
        
        # Element #2
        true_delay2 = np.sqrt((targ_x - elem_x[1])**2 + targ_y**2)/cnst.c - ref_delay
        y2 = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - true_delay2) - rd.deg2rad(phase2))
        zeroix = np.abs(xvec/1E9 - true_delay2) > (num_cycle/2/freq/1E6)
        y2[zeroix] = 0.0
        wave_lines[1].set_ydata(y2)
        wave_lines[1].set_visible(show2)
        total_y += y2
        
        # Element #3
        true_delay3 = np.sqrt((targ_x - elem_x[2])**2 + targ_y**2)/cnst.c - ref_delay
        y3 = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - true_delay3))
        zeroix = np.abs(xvec/1E9 - true_delay3) > (num_cycle/2/freq/1E6)
        y3[zeroix] = 0.0
        wave_lines[2].set_ydata(y3)
        total_y += y3
        
        # Element #4
        true_delay4 = np.sqrt((targ_x - elem_x[3])**2 + targ_y**2)/cnst.c - ref_delay
        y4 = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - true_delay4) - rd.deg2rad(phase4))
        zeroix = np.abs(xvec/1E9 - true_delay4) > (num_cycle/2/freq/1E6)
        y4[zeroix] = 0.0
        wave_lines[3].set_ydata(y4)
        wave_lines[3].set_visible(show4)
        total_y += y4
        
        # Element #5
        true_delay5 = np.sqrt((targ_x - elem_x[4])**2 + targ_y**2)/cnst.c - ref_delay
        y5 = np.sin(2*np.pi*(freq*1E6)*(xvec/1E9 - true_delay5) - rd.deg2rad(phase5))
        zeroix = np.abs(xvec/1E9 - true_delay5) > (num_cycle/2/freq/1E6)
        y5[zeroix] = 0.0
        wave_lines[4].set_ydata(y5)
        wave_lines[4].set_visible(show5)
        total_y += y5

        # Total
        total_line.set_ydata(total_y)

        # Update relative energy
        e0 = (num_elem**2/2)*num_cycle/freq/1E6
        rel_e = dx*np.sum(np.abs(total_y)**2)/e0
        rel_energy.set_text(f"Energy Loss: {rd.to_db(rel_e):.2f} dB")
        
    # Reset
    def reset_phases(btn):
        phase1_wdg.value = 0.0
        phase2_wdg.value = 0.0
        phase4_wdg.value = 0.0
        phase5_wdg.value = 0.0
        
    # Add interaction
    wdg.interactive(
        plot, 
        az=az_wdg,
        freq=freq_wdg,
        r=range_wdg,
        phase1=phase1_wdg,
        phase2=phase2_wdg,
        phase4=phase4_wdg,
        phase5=phase5_wdg,
        show1=show1_wdg,
        show2=show2_wdg,
        show4=show4_wdg,
        show5=show5_wdg
    )    
    
    # Reset button
    reset_btn.on_click(reset_phases)

def pol(
    freq: float = 500.0,
    imag: float = 0.0,
    interval: float = 50,
    num_step: int = 500,
    play_lock: bool = False,
    vh: float = 0.0,
    widgets = ['freq', 'imag', 'run', 'vh']
    ):
    """
    Radar polarization demonstration.
    
    Inputs:
    - freq [float]: Transmit frequency (MHz); default 500.0 MHz
    - imag [float]: Phase angle (deg); default 0.0 deg
    - interval [float]: Time between animation steps (ms); default 50 ms
    - num_step [int]: Number of animation steps; default 500 steps
    - play_lock [bool]: Flag for locking widgets while playing; default False
    - vh [float]: Tilt angle (deg); default 0.0 deg
    - widgets [List[str]]: List of desired widgets
    
    Outputs:
    (none)
    """

    # Axis limits
    xlim = [0, 4]
    ylim = [-2, 2]
    zlim = [-2, 2]
    
    # Initialize plot
    fig1, ax1 = plt.new_plot(
        axes_width=0.55, 
        axes_height=0.90, 
        projection='3d'
    )
    ax1.set_xlabel(RANGE_M_LABEL)
    ax1.set_ylabel('Horizontal (m)')
    ax1.set_zlabel('Vertical (m)')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim)
    
    # Maximum range
    max_plot_range = xlim[1]
    
    # Propagation velocity
    propvel = 3E8
    
    # Time vector
    t0 = 0
    t1 = max_plot_range/propvel
    dt = (t1 - t0)/(num_step - 1)
    
    # Timestamp
    timestamp = fig1.text(0.8, 0.8, f"Time: {t0*1E9:.2f} ns", size=12)
    
    # Wave
    r0 = np.linspace(xlim[0], xlim[1], num_step)
    hdata = np.zeros((3*num_step))
    hdata[2::3] = np.NAN
    vdata = np.zeros((3*num_step))
    vdata[2::3] = np.NAN
    rdata = np.zeros((3*num_step))
    rdata[0::3] = r0
    rdata[1::3] = r0
    rdata[2::3] = r0
    wave = ax1.plot(rdata, hdata, vdata, color='red', marker='.', markersize=2.0, markeredgecolor='None', markerfacecolor='black', linewidth=1.0)[0]

    # Zero
    ax1.plot(r0, np.zeros((num_step)), np.zeros((num_step)), color='black', linewidth=1.0)
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=1000, 
            step=1, 
            value=2, 
            description=FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    # VH angle
    if ('vh' in widgets):
        vh_wdg = wdg.FloatSlider(
            min=0, 
            max=180, 
            step=1, 
            value=vh, 
            description="Orientation Angle (deg)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(vh_wdg)
    else:
        vh_wdg = wdg.fixed(vh)
        
    # Imaginary
    if ('imag' in widgets):
        imag_wdg = wdg.FloatSlider(
            min=0, 
            max=359, 
            step=1, 
            value=imag, 
            description="Phase Difference (deg)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(imag_wdg)
    else:
        imag_wdg = wdg.fixed(imag)
        
    wave_box = []
    if wave_controls:
        wave_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        wave_box = wdg.VBox(wave_title + wave_controls, layout=BOX_LAYOUT)
        controls_box.append(wave_box)
        
    # Run widgets
    run_controls = []
    if ('run' in widgets):
        play_wdg = wdg.Play(
            interval=interval,
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=RUN_LABEL,
            disabled=False
        )
        run_controls.append(play_wdg)
        slider_wdg = wdg.IntSlider(
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=FRAME_LABEL,
            readout=False,
            disabled=False
        )
        wdg.jslink((play_wdg, 'value'), (slider_wdg, 'value'))
        run_controls.append(slider_wdg)
        
    run_box = []
    if run_controls:
        run_title = [wdg.HTML(value = RUN_BLOCK_LABEL)]
        run_box = wdg.VBox(run_title + run_controls, layout=BOX_LAYOUT)
        controls_box.append(run_box)   
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def animate(frame, vh, imag, freq):

        # Time
        t = t0 + (frame - 1)*dt

        # Update timestamp
        timestamp.set_text(f"Time: {t*1E9:.2f} ns")
 
        # Amplitude
        wave_h = np.cos(2*np.pi*(1E6*freq/propvel)*(r0 - propvel*t))*np.cos(rd.deg2rad(vh))
        wave_h[r0 > propvel*t] = 0.0
        wave_v = np.cos(2*np.pi*(1E6*freq/propvel)*(r0 - propvel*t) + rd.deg2rad(imag))*np.sin(rd.deg2rad(vh))
        wave_v[r0 > propvel*t] = 0.0
        
        # y-z data
        hdata[1::3] = wave_h
        vdata[1::3] = wave_v

        # Build wave
        wave.set_xdata(rdata)
        wave.set_ydata(hdata)
        wave.set_3d_properties(vdata)
        
        # Disable controls during play
        if (play_lock):
            if (frame > 1):
                for w in wave_controls:
                    if not w.disabled:
                        w.disabled = True
            elif (frame == 1):
                for w in wave_controls:
                    if w.disabled:
                        w.disabled = False
    
    # Add interaction
    wdg.interactive(
        animate, 
        frame=slider_wdg,
        freq=freq_wdg,
        vh=vh_wdg,
        imag=imag_wdg
    )    
    
def prop_loss(
    freq: float = 100,
    interval: float = 100,
    max_range: float = None,
    num_cycle: int = 4,
    num_step: int = 150,
    propvel: float = 1000,
    widgets = ['energy', 'freq', 'run'],
    xlim = [-100, 100],
    ylim = [-100, 100]
    ):
    """
    Radar polarization demonstration.
    
    Inputs:
    - freq [float]: Transmit frequency (Hz); default 100 Hz
    - interval [float]: Time between animation steps (ms); default 100 ms
    - max_range [float]: Maximum range for plotting (m); default None
    - num_cycle [int]: Number of wave cycles for plotting; default 4 cycles
    - num_step [int]: Number of animation steps; default 150 steps
    - propvel [float]: Propagation velocity (m/s); default 1000 m/s
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (m)
    
    Outputs:
    (none)
    """
    
    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(EAST_M_LABEL)
    ax1.set_ylabel(NORTH_M_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    
    # Maximum range
    max_plot_range = calc_max_range(xlim, ylim)
    if not max_range:
        max_range = max_plot_range
    
    # Time vector
    t0 = 0
    t1 = 2*max_range/propvel
    dt = (t1 - t0)/(num_step - 1)
    
    # Timestamp 
    timestamp = ax1.text(xlim[1] + 5, ylim[1] - 12, f"Time: {t0*1000:.2f} ms", size=12.0)

    # Sensor marker
    cs = ptch.Circle((0, 0), 5, color='blue')
    ax1.add_patch(cs)
    
    # Transmit beam
    peaks = []
    valleys = []
    for ii in range(num_cycle):
        peakii = ptch.Circle((0, 0), 0, fill=False, color='blue', linewidth=2.0, alpha=1.0)
        valleyii = ptch.Circle((0, 0), 0, fill=False, color='blue', linewidth=1.0, alpha=1.0)
        peaks.append(peakii)
        valleys.append(valleyii)
        ax1.add_patch(peakii)
        ax1.add_patch(valleyii)

    # Control widgets
    controls_box = []
        
    # Sensor control widgets
    sensor_controls = []
    
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=1, 
            max=200, 
            step=10, 
            value=freq, 
            description=TX_FREQ_HZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sensor_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    sensor_box = []
    if sensor_controls:
        sensor_title = [wdg.HTML(value = SENSOR_BLOCK_LABEL)]
        sensor_box = wdg.VBox(sensor_title + sensor_controls, layout=BOX_LAYOUT)
        controls_box.append(sensor_box)
 
    # Run widgets
    run_controls = []
    if ('run' in widgets):
        play_wdg = wdg.Play(
            interval=interval,
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=RUN_LABEL,
            disabled=False
        )
        run_controls.append(play_wdg)
        slider_wdg = wdg.IntSlider(
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=FRAME_LABEL,
            readout=False,
            disabled=False
        )
        wdg.jslink((play_wdg, 'value'), (slider_wdg, 'value'))
        run_controls.append(slider_wdg)
        
    run_box = []
    if run_controls:
        run_title = [wdg.HTML(value = RUN_BLOCK_LABEL)]
        run_box = wdg.VBox(run_title + run_controls, layout=BOX_LAYOUT)
        controls_box.append(run_box)   
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
    
    # Plot
    def animate(frame, freq):

        # Time
        t = t0 + (frame - 1)*dt

        # Update timestamp
        timestamp.set_text(f"Time: {t*1000:.2f} ms")
        
        # Update wavefront
        for ii in range(num_cycle):
            peak_range = propvel*(t - (1/freq)*ii)
            peak_alpha = np.maximum(1 - 2*peak_range/max_plot_range, 0.1)
            valley_range = propvel*(t - (1/freq)*ii - 1/freq/2)
            valley_alpha = np.maximum(1 - 2*valley_range/max_plot_range, 0.1)
            if peak_range > 0:
                peaks[ii].set_radius(peak_range)
                peaks[ii].set_alpha(peak_alpha)
            else:
                peaks[ii].set_radius(0)
                peaks[ii].set_alpha(1.0)
            if valley_range > 0:
                valleys[ii].set_radius(valley_range)
                valleys[ii].set_alpha(valley_alpha)
            else:
                valleys[ii].set_radius(0)
                valleys[ii].set_alpha(1.0)

        # Disable controls during play
        if (frame > 1):
            for w in sensor_controls:
                if not w.disabled:
                    w.disabled = True
        elif (frame == 1):
            for w in sensor_controls:
                if w.disabled:
                    w.disabled = False
    
    # Add interaction
    wdg.interactive(
        animate, 
        frame=slider_wdg,
        freq=freq_wdg
    )    

def propagation(
    dx: float = 10,
    dy: float = 10,
    interval: float = 75,
    model: str = 'Gravity',
    num_step: int = 200,
    widgets = ['dx', 'dy', 'model', 'run', 'x', 'y'],
    x: float = -90,
    y: float = 0,
    xlim = [-100, 100],
    ylim = [0, 200]
    ):
    """
    Dynamic model propagation demonstration.
    
    Inputs:
    - dx [float]: Initial target velocity in x component (m/s); default 10 m/s
    - dy [float]: Initial target velocity in y component (m/s); default 10 m/s
    - interval [float]: Time between animation steps (ms); default 75 ms
    - model [str]: Dynamic model ['Gravity', 'No Gravity']; default 'Gravity'
    - num_step [int]: Number of animation steps; default 150 steps
    - widgets [List[str]]: List of desired widgets
    - x [float]: Initial target position in x component (m); default -90 m/s
    - y [float]: Initial target position in y component (m); default 0 m/s
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (m)
    
    Outputs:
    (none)
    """

    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(RANGE_M_LABEL)
    ax1.set_ylabel('Altitude (m)')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
   
    # Time vector
    t0 = 0
    t1 = 20
    dt = (t1 - t0)/(num_step - 1)

    # Timestamp 
    timestamp = ax1.text(xlim[1] + 5, ylim[1] - 12, f"Time: {t0:.2f} s", size=12.0)
   
    # Target position
    targ_pt = ax1.scatter(x, y, 50.0, color='red')

    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    sub_controls1 = []
   
    # Range
    if ('x' in widgets):
        x_wdg = wdg.FloatSlider(
            min=-100, 
            max=100, 
            step=1, 
            value=x, 
            description="Initial Range (m)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(x_wdg)
    else:
        x_wdg = wdg.fixed(x)

    # Range rate
    if ('dx' in widgets):
        dx_wdg = wdg.FloatSlider(
            min=0, 
            max=100, 
            step=1, 
            value=dx, 
            description="Initial Range Rate (m/s)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(dx_wdg)
    else:
        dx_wdg = wdg.fixed(dx)
        
    # Altitude
    if ('y' in widgets):
        y_wdg = wdg.FloatSlider(
            min=0, 
            max=100, 
            step=1, 
            value=x, 
            description="Initial Altitude (m)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(y_wdg)
    else:
        y_wdg = wdg.fixed(y)
        
    # Altitude rate
    if ('dy' in widgets):
        dy_wdg = wdg.FloatSlider(
            min=0, 
            max=100, 
            step=1, 
            value=dx, 
            description="Initial Altitude Rate (m/s)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(dy_wdg)
    else:
        dy_wdg = wdg.fixed(dy)
        
    # Dynamic model
    if ('model' in widgets):
        model_wdg = wdg.RadioButtons(
            options=['Gravity', 'No Gravity'],
            value=model,
            description='Dynamic Model:',
            style=LABEL_STYLE
        )
        sub_controls1.append(model_wdg)
    else:
        model_wdg = wdg.fixed(model)
        
    sub_controls1_box = []
    if sub_controls1:
        sub_controls1_title = [wdg.HTML(value = f"<b><font color='black'>State</b>")]
        sub_controls1_box = wdg.VBox(sub_controls1_title + sub_controls1, layout=BOX_LAYOUT)
        controls_box.append(sub_controls1_box)
       
    # Run widgets
    run_controls = []
    if ('run' in widgets):
        play_wdg = wdg.Play(
            interval=interval,
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=RUN_LABEL,
            disabled=False
        )
        run_controls.append(play_wdg)
        slider_wdg = wdg.IntSlider(
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=FRAME_LABEL,
            readout=False,
            disabled=False
        )
        wdg.jslink((play_wdg, 'value'), (slider_wdg, 'value'))
        run_controls.append(slider_wdg)
        
    run_box = []
    if run_controls:
        run_title = [wdg.HTML(value = RUN_BLOCK_LABEL)]
        run_box = wdg.VBox(run_title + run_controls, layout=BOX_LAYOUT)
        controls_box.append(run_box)   

    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # Plot
    def animate(frame, dx, dy, model, x, y):

        # Time
        t = t0 + (frame - 1)*dt

        # Update text
        timestamp.set_text(f"Time: {t:.2f} s")

        # State
        if model_wdg.value == 'Gravity':
            xi = x + dx*t
            yi = y + dy*t - 0.5*9.8*t**2
        elif model_wdg.value == 'No Gravity':
            xi = x + dx*t
            yi = y + dy*t
            
        if (yi >= 0):    
            targ_pt.set_offsets([xi, yi])    
        
        # Disable controls during play
        if (frame > 1):
            for w in sub_controls1:
                if not w.disabled:
                    w.disabled = True
        elif (frame == 1):
            for w in sub_controls1:
                if w.disabled:
                    w.disabled = False
   
    # Add interaction
    wdg.interactive(
        animate, 
        frame=play_wdg,
        dx=dx_wdg,
        dy=dy_wdg,
        model=model_wdg,
        x=x_wdg,
        y=y_wdg
    )
    
def radar_range_det(
    energy=1E2,
    freq=1E3,
    num_samp=500,
    highlight=True,
    noise_temp=400,
    r=100,
    radius=1,
    rcs=0,
    widgets=['energy', 'freq', 'noise_temp', 'radius', 'r', 'rcs'],
    xlim=[50, 150],
    ylim=[-250, -100]
    ):
    """
    Radar range equation, detection.
    
    Inputs:
    - energy [float]: Transmit energy (J); default 1E2 J
    - freq [float]: Transmit frequency (MHz); default 1E3 MHz
    - num_samp [int]: Number of datat samples; default 500 samples
    - highlight [bool]: Flag for highlighting true echo sample
    - r [float]: Target range (km); default 100 km
    - radius [float]: Dish radius (m): default 1 m
    - rcs [float]: Target radar cross section (dBsm); default 0 dBsm
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (ms)
    - ylim [List[float]]: y-axis limits for plotting (dBJ)
    
    Outputs:
    (none)
    """

    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel('Delay (ms)')
    ax1.set_ylabel('Energy (dBJ)')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
   
    # Range bins
    range_bin = np.linspace(xlim[0], xlim[1], num_samp)
    dr = range_bin[1] - range_bin[0]
    
    # Control widgets
    controls_box = []
       
    # Radar
    radar_controls = []
    
    # Noise temperature
    if ('noise_temp' in widgets):
        noise_temp_wdg = wdg.FloatSlider(
            min=50, 
            max=2000, 
            step=10, 
            value=noise_temp, 
            description=NOISE_TEMP_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        radar_controls.append(noise_temp_wdg)
    else:
        noise_temp_wdg = wdg.fixed(noise_temp)
    
    # Radius
    if ('radius' in widgets):
        radius_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=radius, 
            description=DISH_RADIUS_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        radar_controls.append(radius_wdg)
    else:
        radius_wdg = wdg.fixed(radius)
        
    radar_controls_box = []
    if radar_controls:
        radar_controls_title = [wdg.HTML(value = RADAR_BLOCK_LABEL)]
        radar_controls_box = wdg.VBox(radar_controls_title + radar_controls, layout=BOX_LAYOUT)
        controls_box.append(radar_controls_box)
        
    # Transmission
    tx_controls = []
   
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=3E3, 
            step=100, 
            value=freq, 
            description=TX_FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        tx_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)

    # Transmit power
    if ('energy' in widgets):
        energy_wdg = wdg.FloatSlider(
            min=10, 
            max=1E3, 
            step=10, 
            value=energy, 
            description="Transmit Energy (J)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        tx_controls.append(energy_wdg)
    else:
        energy_wdg = wdg.fixed(energy)
        
    tx_controls_box = []
    if tx_controls:
        tx_controls_title = [wdg.HTML(value = TX_BLOCK_LABEL)]
        tx_controls_box = wdg.VBox(tx_controls_title + tx_controls, layout=BOX_LAYOUT)
        controls_box.append(tx_controls_box)
        
    # Target controls
    targ_controls = []
        
    # Range
    if ('r' in widgets):
        r_wdg = wdg.FloatSlider(
            min=range_bin[0], 
            max=range_bin[-1], 
            step=dr, 
            value=r, 
            description= RANGE_KM_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        targ_controls.append(r_wdg)
    else:
        r_wdg = wdg.fixed(r)
        
    # Radar cross section
    if ('rcs' in widgets):
        rcs_wdg = wdg.FloatSlider(
            min=-30, 
            max=10, 
            step=0.1, 
            value=rcs, 
            description= "RCS (dBsm)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        targ_controls.append(rcs_wdg)
    else:
        rcs_wdg = wdg.fixed(rcs)
        
    targ_controls_box = []
    if targ_controls:
        targ_controls_title = [wdg.HTML(value = TARGET_BLOCK_LABEL)]
        targ_controls_box = wdg.VBox(targ_controls_title + targ_controls, layout=BOX_LAYOUT)
        controls_box.append(targ_controls_box)
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # First plot
    noise_energy = rd.noise_energy(noise_temp)
    signal_energy = rd.rx_energy(r*1E3, energy, freq*1E6, gain=rd.dish_gain(radius, freq*1E6)**2, rcs=rd.from_db(rcs))
    r_bin = (np.abs(range_bin - r)).argmin()
    y_noise = np.sqrt(noise_energy)*np.random.randn(num_samp)
    y_sig = np.zeros((num_samp))
    y_sig[r_bin] = np.sqrt(signal_energy)
    y = y_sig + y_noise
    
    recv_line = ax1.plot(range_bin, 2*rd.to_db(y), color='red')[0]
    
    if highlight:
        echo_line = ax1.plot([r, r], [ylim[0], 2*rd.to_db(y[r_bin])], color='black', linewidth=3.0)[0]
        
    
    # Live text
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    snr0 = signal_energy/noise_energy
    snr0_db = 10*np.log10(snr0)
    text1 = ax1.text(xlim[1] + 0.025*dx, ylim[1] - 0.07*dy, f"SNR: {snr0_db:.2f} dB", size=12.0)
    
    # Plot
    def plot(freq, energy, radius, noise_temp, rcs, r):
        noise_energy = rd.noise_energy(noise_temp)
        signal_energy = rd.rx_energy(r*1E3, energy, freq*1E6, gain=rd.dish_gain(radius, freq*1E6)**2, rcs=rd.from_db(rcs))
        r_bin = (np.abs(range_bin - r)).argmin()
        y_noise = np.sqrt(noise_energy)*np.random.randn(num_samp)
        y_sig = np.zeros((num_samp))
        y_sig[r_bin] = np.sqrt(signal_energy)
        y = y_sig + y_noise
        
        recv_line.set_ydata(2*rd.to_db(y))
        
        if highlight:
            echo_line.set_data([r, r], [ylim[0], 2*rd.to_db(y[r_bin])])
        
        snr0 = signal_energy/noise_energy
        snr0_db = 10*np.log10(snr0)
        text1.set_text("SNR: {val_db:.2f} dB".format(val_db=snr0_db))
   
    # Add interaction
    wdg.interactive(
        plot, 
        freq=freq_wdg,
        energy=energy_wdg,
        radius=radius_wdg,
        noise_temp=noise_temp_wdg,
        r=r_wdg,
        rcs=rcs_wdg
    )    
    
def radar_range_energy(
    energy=1E2,
    freq=1E3,
    num_range=100,
    num_rcs=100,
    radius=1,
    widgets=['energy', 'freq', 'radius'],
    xlim=[1, 500],
    ylim=[-30, 10]
    ):
    """
    Radar range equation, received energy demonstration.
    
    Inputs:
    - energy [float]: Transmit energy (J); default 1E2 J
    - freq [float]: Transmit frequency (MHz); default 1E3 MHz
    - num_range [int]: Number of range bins; default 100 bins
    - num_rcs [int]: Number of radar cross section bins; default 100 bins
    - radius [float]: Dish radius (m): default 1 m
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (dBsm)
    
    Outputs:
    (none)
    """

    # Initialize plot
    fig1, ax1 = plt.new_plot(axes_width=0.5)
    ax1.set_xlabel(RANGE_KM_LABEL)
    ax1.set_ylabel(RCS_DBSM_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # Range bins
    range_bin = np.linspace(xlim[0], xlim[1], num_range)
    
    # RCS bins
    rcs_bin = np.linspace(ylim[0], ylim[1], num_rcs)
    
    # Mesh
    range_mesh, rcs_mesh = np.meshgrid(range_bin, rcs_bin)
    range_mesh_m = range_mesh*1E3
    rcs_mesh_lin = rd.from_db(rcs_mesh)
    
    # Initial plot
    img = rd.to_db(rd.rx_energy(range_mesh_m, energy, freq*1E6, gain=rd.dish_gain(radius, freq*1E6)**2, rcs=rcs_mesh_lin))
    pc = ax1.contourf(range_bin, rcs_bin, img, np.linspace(-230, -90, 8), cmap='inferno', extend='both')
    
    # Colorbar
    cbar = pyp.colorbar(pc, ax=ax1)
    cbar.ax.set_ylabel('Received Energy (dBJ)', size=12, name=plt.DEF_SANS)
    
    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    sub_controls1 = []
   
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=3E3, 
            step=100, 
            value=freq, 
            description=TX_FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)

    # Transmit power
    if ('energy' in widgets):
        energy_wdg = wdg.FloatSlider(
            min=10, 
            max=1E3, 
            step=10, 
            value=energy, 
            description="Transmit Energy (J)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(energy_wdg)
    else:
        energy_wdg = wdg.fixed(energy)
        
    # Radius
    if ('radius' in widgets):
        radius_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=radius, 
            description=DISH_RADIUS_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(radius_wdg)
    else:
        radius_wdg = wdg.fixed(radius)
        
    sub_controls1_box = []
    if sub_controls1:
        sub_controls1_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        sub_controls1_box = wdg.VBox(sub_controls1_title + sub_controls1, layout=BOX_LAYOUT)
        controls_box.append(sub_controls1_box)
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # Plot
    def plot(freq, energy, radius):
        img = rd.to_db(rd.rx_energy(range_mesh_m, energy, freq*1E6, gain=rd.dish_gain(radius, freq*1E6)**2, rcs=rcs_mesh_lin))
        for art in ax1.artists:
            art.remove()
        ax1.contourf(range_bin, rcs_bin, img, np.linspace(-230, -90, 8), cmap='inferno', extend='both')
        fig1.canvas.draw()
   
    # Add interaction
    wdg.interactive(
        plot, 
        freq=freq_wdg,
        energy=energy_wdg,
        radius=radius_wdg
    )
    
def radar_range_power(
    freq=1E3,
    num_range=100,
    num_rcs=100,
    power=1E3,
    radius=1,
    widgets=['freq', 'power', 'radius'],
    xlim=[1, 500],
    ylim=[-30, 10]
    ):
    """
    Radar range equation, received energy demonstration.
    
    Inputs:
    - freq [float]: Transmit frequency (MHz); default 1E3 MHz
    - num_range [int]: Number of range bins; default 100 bins
    - num_rcs [int]: Number of radar cross section bins; default 100 bins
    - power [float]: Transmit power (kW); default 1E3 kW
    - radius [float]: Dish radius (m): default 1 m
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (dBsm)
    
    Outputs:
    (none)
    """

    # Initialize plot
    fig1, ax1 = plt.new_plot(axes_width=0.5)
    ax1.set_xlabel(RANGE_KM_LABEL)
    ax1.set_ylabel(RCS_DBSM_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # Range bins
    range_bin = np.linspace(xlim[0], xlim[1], num_range)
    
    # RCS bins
    rcs_bin = np.linspace(ylim[0], ylim[1], num_rcs)
    
    # Mesh
    range_mesh, rcs_mesh = np.meshgrid(range_bin, rcs_bin)
    range_mesh_m = range_mesh*1E3
    rcs_mesh_lin = rd.from_db(rcs_mesh)
    
    # Initial plot
    img = rd.to_db(rd.rx_power(range_mesh_m, power*1E3, freq*1E6, gain=rd.dish_gain(radius, freq*1E6)**2, rcs=rcs_mesh_lin))
    pc = ax1.contourf(range_bin, rcs_bin, img, np.linspace(-200, -60, 8), cmap='inferno', extend='both')
    
    # Colorbar
    cbar = pyp.colorbar(pc, ax=ax1)
    cbar.ax.set_ylabel('Received Power (dBW)', size=12, name=plt.DEF_SANS)
    
    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    sub_controls1 = []
   
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=3E3, 
            step=100, 
            value=freq, 
            description=TX_FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)

    # Transmit power
    if ('power' in widgets):
        power_wdg = wdg.FloatSlider(
            min=1E2, 
            max=1E4, 
            step=1E2, 
            value=power, 
            description="Transmit Power (kW)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(power_wdg)
    else:
        power_wdg = wdg.fixed(power)
        
    # Radius
    if ('radius' in widgets):
        radius_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=radius, 
            description=DISH_RADIUS_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(radius_wdg)
    else:
        radius_wdg = wdg.fixed(radius)
        
    sub_controls1_box = []
    if sub_controls1:
        sub_controls1_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        sub_controls1_box = wdg.VBox(sub_controls1_title + sub_controls1, layout=BOX_LAYOUT)
        controls_box.append(sub_controls1_box)
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # Plot
    def plot(freq, power, radius):
        img = rd.to_db(rd.rx_power(range_mesh_m, power*1E3, freq*1E6, gain=rd.dish_gain(radius, freq*1E6)**2, rcs=rcs_mesh_lin))
        for art in ax1.artists:
            art.remove()
        ax1.contourf(range_bin, rcs_bin, img, np.linspace(-200, -60, 8), cmap='inferno', extend='both')
        fig1.canvas.draw()
   
    # Add interaction
    wdg.interactive(
        plot, 
        freq=freq_wdg,
        power=power_wdg,
        radius=radius_wdg
    )
    
def radar_range_snr(
    energy=1E2,
    freq=1E3,
    noise_temp=400,
    num_range=100,
    num_rcs=100,
    radius=1,
    widgets=['energy', 'freq', 'noise_temp', 'radius'],
    xlim=[1, 500],
    ylim=[-30, 10]
    ):
    """
    Radar range equation, received energy demonstration.
    
    Inputs:
    - energy [float]: Transmit energy (J); default 1E2 J
    - freq [float]: Transmit frequency (MHz); default 1E3 MHz
    - noise_temp [float]: System noise temperature (K); default 400
    - num_range [int]: Number of range bins; default 100 bins
    - num_rcs [int]: Number of radar cross section bins; default 100 bins
    - radius [float]: Dish radius (m): default 1 m
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (dBsm)
    
    Outputs:
    (none)
    """

    # Initialize plot
    fig1, ax1 = plt.new_plot(axes_width=0.5)
    ax1.set_xlabel(RANGE_KM_LABEL)
    ax1.set_ylabel(RCS_DBSM_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # Range bins
    range_bin = np.linspace(xlim[0], xlim[1], num_range)
    
    # RCS bins
    rcs_bin = np.linspace(ylim[0], ylim[1], num_rcs)
    
    # Mesh
    range_mesh, rcs_mesh = np.meshgrid(range_bin, rcs_bin)
    range_mesh_m = range_mesh*1E3
    rcs_mesh_lin = rd.from_db(rcs_mesh)
    
    # Initial plot
    img = rd.to_db(rd.rx_snr(range_mesh_m, energy, freq*1E6, noise_temp, gain=rd.dish_gain(radius, freq*1E6)**2, rcs=rcs_mesh_lin))
    pc = ax1.contourf(range_bin, rcs_bin, img, np.linspace(-10, 60, 8), cmap='inferno', extend='both')
    
    # Colorbar
    cbar = pyp.colorbar(pc, ax=ax1)
    cbar.ax.set_ylabel(SNR_DB_LABEL, size=12, name=plt.DEF_SANS)
    
    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    sub_controls1 = []
   
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=3E3, 
            step=100, 
            value=freq, 
            description=TX_FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)

    # Transmit power
    if ('energy' in widgets):
        energy_wdg = wdg.FloatSlider(
            min=10, 
            max=1E3, 
            step=10, 
            value=energy, 
            description="Transmit Energy (J)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(energy_wdg)
    else:
        energy_wdg = wdg.fixed(energy)
        
    # Noise temperature
    if ('noise_temp' in widgets):
        noise_temp_wdg = wdg.FloatSlider(
            min=50, 
            max=2000, 
            step=10, 
            value=noise_temp, 
            description=NOISE_TEMP_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(noise_temp_wdg)
    else:
        noise_temp_wdg = wdg.fixed(noise_temp)
        
    # Radius
    if ('radius' in widgets):
        radius_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=radius, 
            description=DISH_RADIUS_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(radius_wdg)
    else:
        radius_wdg = wdg.fixed(radius)
        
    sub_controls1_box = []
    if sub_controls1:
        sub_controls1_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        sub_controls1_box = wdg.VBox(sub_controls1_title + sub_controls1, layout=BOX_LAYOUT)
        controls_box.append(sub_controls1_box)
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # Plot
    def plot(freq, energy, radius, noise_temp):
        img = rd.to_db(rd.rx_snr(range_mesh_m, energy, freq*1E6, noise_temp, gain=rd.dish_gain(radius, freq*1E6)**2, rcs=rcs_mesh_lin))
        for art in ax1.artists:
            art.remove()
        ax1.contourf(range_bin, rcs_bin, img, np.linspace(-10, 60, 8), cmap='inferno', extend='both')
        fig1.canvas.draw()
   
    # Add interaction
    wdg.interactive(
        plot, 
        freq=freq_wdg,
        energy=energy_wdg,
        radius=radius_wdg,
        noise_temp=noise_temp_wdg
    )    
    
def range_res(
    num_bins: int = 1000,
    pulsewidth: float = 5,
    r: float = 100,
    start_freq: float = 1,
    stop_freq: float = 1,
    widgets = ['range', 'pulsewidth', 'start_freq', 'stop_freq'],
    ):
    """
    Range resolution with LFM demonstration.
    
    Inputs:
    - num_bins [int]: Number of delay bins; default 1000 bins
    - pulsewidth [float]: Transmit pulsewidth (µs); default 5 µs
    - r [float]: Target range separation (m); default 100 m
    - start_freq [float]: Transmit start frequency (MHz); default 1 MHz
    - stop_freq [float]: Transmit stop frequency (MHz); default 1 MHz
    - widgets [List[str]]: List of desired widgets
    
    Outputs:
    (none)
    """

    # Maximum pulsewidth
    max_pulsewidth = 10
    max_range = cnst.c*max_pulsewidth/1E6/2
    
    # x limits
    xlim = [-max_pulsewidth, max_pulsewidth]
    
    # y limits
    ylim = [-20, 5]
    
    # Initialize plot
    fig, ax = plt.new_plot()
    ax.set_xlabel(DELAY_US_LABEL)
    ax.set_ylabel('Filter Output (dB)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Time vector
    xvec = np.linspace(xlim[0], xlim[1], num_bins)

    # Waves
    slope = (stop_freq - start_freq)*1E6/(pulsewidth/1E6)
    replica = np.sin(2*np.pi*(start_freq*1E6)*((xvec + pulsewidth/2)/1E6) + np.pi*slope*((xvec + pulsewidth/2)/1E6)**2)
    zeroix = np.logical_or(xvec < -pulsewidth/2, xvec > pulsewidth/2)
    replica[zeroix] = 0.0
    delay1 = -2E6*r/cnst.c
    targ1 = np.sin(2*np.pi*(start_freq*1E6)*(((xvec - delay1) + pulsewidth/2)/1E6) + np.pi*slope*(((xvec - delay1) + pulsewidth/2)/1E6)**2)
    zeroix = np.logical_or((xvec - delay1) < -pulsewidth/2, (xvec - delay1) > pulsewidth/2)
    targ1[zeroix] = 0.0
    delay2 = 2E6*r/cnst.c
    targ2 = np.sin(2*np.pi*(start_freq*1E6)*(((xvec - delay2) + pulsewidth/2)/1E6) + np.pi*slope*(((xvec - delay2) + pulsewidth/2)/1E6)**2)
    zeroix = np.logical_or((xvec - delay2) < -pulsewidth/2, (xvec - delay2) > pulsewidth/2)
    targ2[zeroix] = 0.0
    E = np.sum(replica*replica)
    pulse = np.correlate(targ1 + targ2, replica, mode='same')/E
    test_line = ax.plot(xvec, 2*rd.to_db(pulse), linewidth=2.0, color='red')[0]
    
    # Target lines
    targ1_line = ax.plot([delay1, delay1], ylim, color='black', linestyle='dashed', linewidth=2.0)[0]
    targ2_line = ax.plot([delay2, delay2], ylim, color='black', linestyle='dashed', linewidth=2.0)[0]
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Test delay
    if ('range' in widgets):
        range_wdg = wdg.FloatSlider(
            min=-max_range/2, 
            max=max_range/2, 
            step=0.1, 
            value=r, 
            description="Range Separation (m)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(range_wdg)
    else:
        range_wdg = wdg.fixed(r)
    
    # Start frequency
    if ('start_freq' in widgets):
        start_freq_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=start_freq, 
            description="Start Frequency (MHz)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(start_freq_wdg)
    else:
        start_freq_wdg = wdg.fixed(start_freq)
        
    # Stop frequency
    if ('stop_freq' in widgets):
        stop_freq_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=stop_freq, 
            description="Stop Frequency (MHz)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(stop_freq_wdg)
    else:
        stop_freq_wdg = wdg.fixed(stop_freq)
        
    # Transmit pulsewidth
    if ('pulsewidth' in widgets):
        pulsewidth_wdg = wdg.FloatSlider(
            min=0.1, 
            max=10, 
            step=0.1, 
            value=pulsewidth, 
            description="Pulsewidth (µs)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(pulsewidth_wdg)
    else:
        pulsewidth_wdg = wdg.fixed(pulsewidth)
        
    wave_box = []
    if wave_controls:
        wave_title = [wdg.HTML(value = f"<b><font color='black'>Waveform</b>")]
        wave_box = wdg.VBox(wave_title + wave_controls, layout=BOX_LAYOUT)
        controls_box.append(wave_box)
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def plot(r, start_freq, stop_freq, pulsewidth):
        
        # Waves
        slope = (stop_freq - start_freq)*1E6/(pulsewidth/1E6)
        replica = np.sin(2*np.pi*(start_freq*1E6)*((xvec + pulsewidth/2)/1E6) + np.pi*slope*((xvec + pulsewidth/2)/1E6)**2)
        zeroix = np.logical_or(xvec < -pulsewidth/2, xvec > pulsewidth/2)
        replica[zeroix] = 0.0
        delay1 = -2E6*r/cnst.c
        targ1 = np.sin(2*np.pi*(start_freq*1E6)*(((xvec - delay1) + pulsewidth/2)/1E6) + np.pi*slope*(((xvec - delay1) + pulsewidth/2)/1E6)**2)
        zeroix = np.logical_or((xvec - delay1) < -pulsewidth/2, (xvec - delay1) > pulsewidth/2)
        targ1[zeroix] = 0.0
        delay2 = 2E6*r/cnst.c
        targ2 = np.sin(2*np.pi*(start_freq*1E6)*(((xvec - delay2) + pulsewidth/2)/1E6) + np.pi*slope*(((xvec - delay2) + pulsewidth/2)/1E6)**2)
        zeroix = np.logical_or((xvec - delay2) < -pulsewidth/2, (xvec - delay2) > pulsewidth/2)
        targ2[zeroix] = 0.0
        
        E = np.sum(replica*replica)
        pulse = np.correlate(targ1 + targ2, replica, mode='same')/E
        test_line.set_ydata(2*rd.to_db(pulse))
        
        targ1_line.set_xdata([delay1, delay1])
        targ2_line.set_xdata([delay2, delay2])

    # Add interaction
    wdg.interactive(
        plot, 
        r=range_wdg,
        start_freq=start_freq_wdg,
        stop_freq=stop_freq_wdg,
        pulsewidth=pulsewidth_wdg
    )

def ranging(
    interval=75,
    max_range=None,
    num_step=150,
    propvel=1000,
    rx_az=0.0,
    rx_beamw=1.0,
    rx_omni=True,
    tgt_hide=False,
    tgt_x=50, 
    tgt_y=50,
    tx_az=0.0,
    tx_beamw=1.0,
    tx_omni=False,
    widgets=['dets', 'rx_az', 'rx_beamw', 'run', 'tx_az', 'tx_beamw', 'x', 'y'],
    xlim=[-100, 100],
    ylim=[-100, 100]):
    """
    Generic ranging demonstration.
    
    Inputs:
    - interval [float]: Time between animation steps (ms); default 75 ms
    - max_range [float]: Maximum range for plotting (m); default 150
    - num_step [int]: Number of animation steps; default 150 steps
    - propvel [float]: Propagation velocity (m/s); default 1000 m/s
    - rx_az [float]: Receive steering direction (deg); default 0 deg
    - rx_beam [float]: Receive beamwidth (deg); default 1.0 deg
    - rx_omni [bool]: Flag for omnidirectional receive
    - tgt_hide [bool]: Flag for hidden target; default False
    - tgt_x [float]: Target East coordinate (m); default 50 m
    - tgt_y [float]: Target North coordinate (m); default 50 m
    - tx_az [float]: Transmit steering direction (deg); default 0 deg
    - tx_beam [float]: Transmit beamwidth (deg); default 1.0 deg
    - tx_omni [bool]: Flag for omnidirectional transmit
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (m)
    
    Outputs:
    (none)
    """
    
    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(EAST_M_LABEL)
    ax1.set_ylabel(NORTH_M_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    
    # Timestamp
    timestamp = ax1.text(xlim[1] + 5, ylim[1] - 12, "Time: ---", size=12.0)
    
    # Maximum range
    max_plot_range = calc_max_range(xlim, ylim)
    if not max_range:
        max_range = max_plot_range
    
    # Time vector
    t0 = 0
    t1 = 2*max_range/propvel
    dt = (t1 - t0)/(num_step - 1)
    tvec = np.arange(t0, t1, dt)
    
    # Sensor marker
    cs = ptch.Circle((0, 0), 5, color='blue')
    ax1.add_patch(cs)
    
    # Target marker
    if not tgt_hide:
        ct = ptch.Circle((tgt_x, tgt_y), 5, color='red')
        ce = ptch.Circle((tgt_x, tgt_y), 0, fill=False, color='red', linewidth=2.0, alpha=0.4)
        ax1.add_patch(ct)
        ax1.add_patch(ce)
    
    # Transmit beam
    if tx_omni:
        cp = ptch.Circle((0, 0), 0, fill=False, color='blue', linewidth=2.0, alpha=0.4)
        ax1.add_patch(cp)
    else:
        tx_theta1 = 90 - tx_az - tx_beamw/2
        tx_theta2 = 90 - tx_az + tx_beamw/2
        wave_arc = ptch.Arc((0, 0), 0, 0, theta1=tx_theta1, theta2=tx_theta2, color='blue', linewidth=2.0, alpha=0.4)
        tx_beam = ptch.Wedge((0, 0), max_plot_range, tx_theta1, tx_theta2, color='gray', alpha=0.2)
        ax1.add_patch(tx_beam)
        ax1.add_patch(wave_arc)

    # Receive beam
    if not rx_omni:
        rx_theta1 = 90 - rx_az - rx_beamw/2
        rx_theta2 = 90 - rx_az + rx_beamw/2
        rx_beam = ptch.Wedge((0, 0), max_plot_range, rx_theta1, rx_theta2, color='gray', alpha=0.2)
        ax1.add_patch(rx_beam)
        
    # Control widgets
    controls_box = []
        
    # Sensor control widgets
    sensor_controls = []
    
    # Boresight azimuths
    if (not rx_omni) and ('rx_az' in widgets):
        rx_az_wdg = wdg.FloatSlider(
            min=0, 
            max=360, 
            step=1, 
            value=rx_az, 
            description=RX_AZ_DEG_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sensor_controls.append(rx_az_wdg)
    else:
        rx_az_wdg = wdg.fixed(rx_az)
        
    if (not tx_omni) and ('tx_az' in widgets):
        tx_az_wdg = wdg.FloatSlider(
            min=0, 
            max=360, 
            step=1, 
            value=tx_az, 
            description=TX_AZ_DEG_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sensor_controls.append(tx_az_wdg)
    else:
        tx_az_wdg = wdg.fixed(tx_az)
    
    # Beamwidths
    if (not rx_omni) and ('rx_beamw' in widgets):
        rx_beamw_wdg = wdg.FloatSlider(
            min=1, 
            max=90, 
            step=1,
            value=rx_beamw, 
            description="Receive Beamwidth (deg)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sensor_controls.append(rx_beamw_wdg)
    else:
        rx_beamw_wdg = wdg.fixed(rx_beamw)
        
    if (not tx_omni) and ('tx_beamw' in widgets):
        tx_beamw_wdg = wdg.FloatSlider(
            min=1, 
            max=90, 
            step=1, 
            value=tx_beamw, 
            description="Transmit Beamwidth (deg)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sensor_controls.append(tx_beamw_wdg)
    else:
        tx_beamw_wdg = wdg.fixed(tx_beamw)
        
    sensor_box = []
    if sensor_controls:
        sensor_title = [wdg.HTML(value = SENSOR_BLOCK_LABEL)]
        sensor_box = wdg.VBox(sensor_title + sensor_controls, layout=BOX_LAYOUT)
        controls_box.append(sensor_box)
        
    # Target control widgets
    target_controls = []
    
    # Target x position
    if ('x' in widgets):
        # Build widget
        x_wdg = wdg.FloatSlider(
            min=xlim[0], 
            max=xlim[1], 
            step=(xlim[1] - xlim[0])/200, 
            value=tgt_x,
            description=EAST_M_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        target_controls.append(x_wdg)
    else:
        x_wdg = wdg.fixed(tgt_x)
        
    # Target y position
    if ('y' in widgets):
        # Build widget
        y_wdg = wdg.FloatSlider(
            min=ylim[0], 
            max=ylim[1], 
            step=(ylim[1] - ylim[0])/200, 
            value=tgt_y,
            description=NORTH_M_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        target_controls.append(y_wdg)
    else:
        y_wdg = wdg.fixed(tgt_y)
        
    target_box = []
    if target_controls:
        target_title = [wdg.HTML(value = TARGET_BLOCK_LABEL)]
        target_box = wdg.VBox(target_title + target_controls, layout=BOX_LAYOUT)
        controls_box.append(target_box)
    
    # Output widgets
    outputs = []
    
    # Detection list
    dets_show = False
    if 'dets' in widgets:
        dets_show = True
        dets_wdg = wdg.Select(description='Detections:', rows=2)
        outputs.append(dets_wdg)
        
    output_box = []
    if outputs:
        output_title = [wdg.HTML(value = f"<b><font color='black'>Output</b>")]
        output_box = wdg.VBox(output_title + outputs, layout=BOX_LAYOUT)
        controls_box.append(output_box)
    
    # Run widgets
    run_controls = []
    if ('run' in widgets):
        play_wdg = wdg.Play(
            interval=interval,
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=RUN_LABEL,
            disabled=False
        )
        run_controls.append(play_wdg)
        slider_wdg = wdg.IntSlider(
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=FRAME_LABEL,
            readout=False,
            disabled=False
        )
        wdg.jslink((play_wdg, 'value'), (slider_wdg, 'value'))
        run_controls.append(slider_wdg)
        
    run_box = []
    if run_controls:
        run_title = [wdg.HTML(value = RUN_BLOCK_LABEL)]
        run_box = wdg.VBox(run_title + run_controls, layout=BOX_LAYOUT)
        controls_box.append(run_box)   
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
        
    # Initialize detections
    det_list = []
    
    # Plot
    def animate(frame, rx_az, rx_beamw, tx_az, tx_beamw, tgt_x, tgt_y):

        # Time
        t = t0 + (frame - 1)*dt

        # Update timestamp
        timestamp.set_text("Time: {ti:.2f} ms".format(ti=t*1000))
        
        # Target range, azimuth
        tgt_r = np.sqrt(tgt_x**2 + tgt_y**2)
        tgt_az = np.pi/2 - np.arctan2(tgt_y, tgt_x)

        # Update target
        if not tgt_hide:
            ct.center = tgt_x, tgt_y
            ce.center = tgt_x, tgt_y
                   
        # Input conversion
        if rx_az:
            rx_az = rx_az*(np.pi/180)
        if tx_az:
            tx_az = tx_az*(np.pi/180)
        if rx_beamw:
            rx_beamw = rx_beamw*(np.pi/180)
        if tx_beamw:
            tx_beamw = tx_beamw*(np.pi/180)
        
        # Update transmit beam and pulse
        if tx_omni:
            # Update wavefront
            cp.set_radius(propvel*t)
            cp.set_alpha(np.maximum(1 - propvel*t/max_plot_range, 0))
        else:
            # Update wavefront
            wave_arc.theta1 = (180/np.pi)*(np.pi/2 - tx_az - tx_beamw/2)
            wave_arc.theta2 = (180/np.pi)*(np.pi/2 - tx_az + tx_beamw/2)
            wave_arc.set_width(2*propvel*t)
            wave_arc.set_height(2*propvel*t)
            wave_arc.set_alpha(np.maximum(1 - propvel*t/max_plot_range, 0.1))

            # Update beams
            tx_beam.set_theta1((180/np.pi)*(np.pi/2 - tx_az - tx_beamw/2))
            tx_beam.set_theta2((180/np.pi)*(np.pi/2 - tx_az + tx_beamw/2))
        
        # Update receive beam
        if not rx_omni:
            # Update beams
            rx_beam.set_theta1((180/np.pi)*(np.pi/2 - rx_az - rx_beamw/2))
            rx_beam.set_theta2((180/np.pi)*(np.pi/2 - rx_az + rx_beamw/2))
        
        # Check for in-beam
        in_tx_beam = True
        in_rx_beam = True
        if not tx_omni:
            in_tx_beam = np.abs(np.angle(np.exp(1j*(tgt_az - tx_az)))) <= tx_beamw/2
        if not rx_omni:
            in_rx_beam = np.abs(np.angle(np.exp(1j*(tgt_az - rx_az)))) <= rx_beamw/2
        
        # Update echo
        if not tgt_hide and in_tx_beam:
            if (propvel*t > tgt_r):
                echo_range = propvel*t - tgt_r
                ce.set_radius(echo_range)
                ce.set_alpha(np.maximum(1 - (tgt_r + echo_range)/max_plot_range, 0.1))
            else:
                ce.set_radius(0.0)
                ce.set_alpha(1.0)
        
        # Detection marker
        if (np.abs(propvel*t - 2*tgt_r) <= propvel*3E-3) and in_tx_beam and in_rx_beam:
            cs.set_facecolor('orange')
        else:
            cs.set_facecolor('blue')
            
        # Detection list
        if dets_show:
            det_frame = np.argmin(np.abs(tvec - 2*tgt_r/propvel))
            if (frame == det_frame) and in_tx_beam and in_rx_beam:
                det_str = "Time: {det_time:.2f} ms".format(det_time=2000*tgt_r/propvel)
                det_list.append(det_str)
                dets_wdg.options = det_list
            elif (frame == 1):
                det_list.clear()
                dets_wdg.options = []
            
        # Disable controls during play
        if (frame > 1):
            for w in sensor_controls:
                if not w.disabled:
                    w.disabled = True
            for w in target_controls:
                if not w.disabled:
                    w.disabled = True
        elif (frame == 1):
            for w in sensor_controls:
                if w.disabled:
                    w.disabled = False
            for w in target_controls:
                if w.disabled:
                    w.disabled = False
    
    # Add interaction
    wdg.interactive(
        animate, 
        frame=slider_wdg,
        rx_az=rx_az_wdg, 
        rx_beamw=rx_beamw_wdg,
        tx_az=tx_az_wdg, 
        tx_beamw=tx_beamw_wdg,
        tgt_x=x_wdg,
        tgt_y=y_wdg
    )
    
def rdi(
    bandw: float = 1.0,
    dr: float = 0.0,
    freq: float = 1.0E3,
    max_freq: float = 5E2,
    max_range: float = 500.0,
    min_freq: float = -5E2,
    min_range: float = -500.0,
    num_dopp: int = 150,
    num_pulse: int = 32,
    num_range: int = 100,
    prf: float = 1E3,
    r: float = 0.0,
    widgets=['bandw', 'dr', 'freq', 'num_pulse', 'prf', 'r'],
    ):
    """
    Range-Doppler image demonstration.
    
    Inputs:
    - bandw [float]: Transmit bandwidth (MHz); default 1 MHz
    - dr [float]: Target range rate (m/s); default 0 m/s
    - freq [float]: Transmit frequency (MHz); default 1E3 MHz
    - max_freq [float]: Maximum frequency for plotting (Hz); default 500 Hz
    - max_range [float]: Maximum range for plotting (m); default 500 m
    - min_freq [float]: Minimum frequency for plotting (Hz); default -500 Hz
    - min_range [float]: Minimum range for plotting (m); default -500 m
    - num_dopp [int]: Number of Doppler bins; default 150 bins
    - num_pulse [int]: Number of pulses in burst; default 32 pulses
    - num_range [int]: Number of range bins; default 100 bins
    - prf [float]: Pule repetition frequency (Hz); default 1000 Hz
    - r [float]: Target range (m); default 100 m
    - widgets [List[str]]: List of desired widgets
    
    Outputs:
    (none)
    """

    # Axis limits
    xlim = [min_freq, max_freq]
    ylim = [min_range, max_range]

    # Initialize plot
    fig1, ax1 = plt.new_plot()
    ax1.set_xlabel(REL_FREQ_HZ_LABEL)
    ax1.set_ylabel('Relative Range (m)')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
   
    # Frequency, range bins
    dopp_bins = np.linspace(xlim[0], xlim[1], num_dopp)
    range_bins = np.linspace(ylim[0], ylim[1], num_range)
    
    # Meshes
    dopp_mesh, range_mesh = np.meshgrid(dopp_bins, range_bins)

    # Initial plot
    img = 2*rd.to_db(rd.rdi(range_mesh, dopp_mesh, r, dr, bandw*1E6, freq*1E6, num_pulse, prf))
    pc = ax1.pcolormesh(dopp_bins, range_bins, img, shading='gouraud', cmap='inferno')
    pc.set_clim(-40, 0)

    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    block1 = []
   
    # Transmission
    block1.append(wdg.HTML(value = TX_BLOCK_LABEL))

    # Bandwidth
    if ('bandw' in widgets):
        bandw_wdg = wdg.FloatSlider(
            min=0.1, 
            max=3, 
            step=0.05, 
            value=bandw, 
            description=BANDW_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block1.append(bandw_wdg)
    else:
        bandw_wdg = wdg.fixed(bandw)

    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=3E3, 
            step=100, 
            value=freq, 
            description=TX_FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block1.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    # Pulse repetition frequency
    if ('prf' in widgets):
        prf_wdg = wdg.FloatSlider(
            min=10, 
            max=1000, 
            step=10, 
            value=prf, 
            description="Pulse Repetition Frequency (Hz)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block1.append(prf_wdg)
    else:
        prf_wdg = wdg.fixed(prf)
        
    # Number of pulses
    if ('num_pulse' in widgets):
        num_pulse_wdg = wdg.IntSlider(
            min=1, 
            max=100, 
            step=1, 
            value=num_pulse, 
            description="Number of Pulses", 
            style=LABEL_STYLE, 
            readout_format='d'
        )
        block1.append(num_pulse_wdg)
    else:
        num_pulse_wdg = wdg.fixed(num_pulse)
        
    # Add block
    controls_box.append(wdg.VBox(block1))
       
    # Target block
    block2 = []
    
    # Title
    block2.append(wdg.HTML(value = TARGET_BLOCK_LABEL))
        
    # Range
    if ('r' in widgets):
        r_wdg = wdg.FloatSlider(
            min=min_range, 
            max=max_range, 
            step=(max_range - min_range)/100, 
            value=r, 
            description="Range (m)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block2.append(r_wdg)
    else:
        r_wdg = wdg.fixed(r)
        
        
    # Range rate
    if ('dr' in widgets):
        dr_wdg = wdg.FloatSlider(
            min=-100, 
            max=100, 
            step=1, 
            value=dr, 
            description=RANGE_RATE_M_S_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        block2.append(dr_wdg)
    else:
        dr_wdg = wdg.fixed(dr)
    
    # Add block
    controls_box.append(wdg.VBox(block2))
    
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # Plot
    def plot(r, dr, bandw, freq, num_pulse, prf):
        img = 2*rd.to_db(rd.rdi(range_mesh, dopp_mesh, r, dr, bandw*1E6, freq*1E6, num_pulse, prf))
        pc.set_array(img.ravel())
        ax1.set_xlim([-prf/2, prf/2])
        fig1.canvas.draw()
   
    # Add interaction
    wdg.interactive(
        plot, 
        bandw=bandw_wdg,
        r=r_wdg,
        dr=dr_wdg,
        freq=freq_wdg,
        num_pulse=num_pulse_wdg,
        prf=prf_wdg
    )    

def rect_pat(
    freq: float = 1.0E3,
    height: float = 1.0,
    num_az: int = 100,
    num_el: int = 100,
    show_beamw: bool = False,
    width: float = 1.0,
    widgets=['freq', 'height', 'width'],
    xlim=[-60, 60],
    ylim=[-60, 60]
    ):
    """
    Rectangular radar gain pattern.
    
    Inputs:
    - freq [float]: Transmit frequency (MHz); default 1E3 MHz
    - height [float]: Aperture height (m); default 1.0 m
    - num_az [int]: Number of azimuth bins; default 100 bins
    - num_el [int]: Number of elevation bins; default 100 bins
    - show_beamw [bool]: Flag for displaying beamwidth; default False
    - widgets [List[str]]: List of desired widgets
    - width [float]: Aperture width (m); default 1.0 m
    - xlim [List[float]]: x-axis limits for plotting (deg)
    - ylim [List[float]]: y-axis limits for plotting (deg)
    
    Outputs:
    (none)
    """

    # Initialize plot
    fig1, ax1 = plt.new_plot()
    ax1.set_xlabel('Horizontal (deg)')
    ax1.set_ylabel('Vertical (deg)')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # az, el bins
    az_bins = np.linspace(xlim[0], xlim[1], num_az)
    el_bins = np.linspace(ylim[0], ylim[1], num_el)
    
    # Meshes
    az_mesh, el_mesh = np.meshgrid(rd.deg2rad(az_bins), rd.deg2rad(el_bins))
    u_mesh0 = np.sin(az_mesh)*np.cos(el_mesh)
    v_mesh0 = np.sin(el_mesh)

    # Initial plot
    wlen = rd.wavelen(freq*1E6)
    u_mesh = -(width/wlen)*u_mesh0
    v_mesh = (height/wlen)*v_mesh0
    img = 2*rd.to_db(rd.rect_pat(u_mesh, v_mesh))
    pc = ax1.pcolormesh(az_bins, el_bins, img, shading='gouraud', cmap='inferno')
    pc.set_clim(-40, 0)

    # Live text
    if show_beamw:
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        h_beamw0, v_beamw0 = rd.rect_beamw(height, width, freq*1E6)
        text1 = ax1.text(xlim[1] + 0.025*dx, ylim[1] - 0.07*dy, f"Horiz. Beamwidth: {h_beamw0:.2f} deg", size=12.0)
        text2 = ax1.text(xlim[1] + 0.025*dx, ylim[1] - 0.14*dy, f"Vert. Beamwidth: {v_beamw0:.2f} deg", size=12.0)
    
    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    sub_controls1 = []
   
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=3E3, 
            step=100, 
            value=freq, 
            description=TX_FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)

    # Height
    if ('height' in widgets):
        height_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=height, 
            description="Aperture Height (m)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(height_wdg)
    else:
        height_wdg = wdg.fixed(height)
        
    # Width
    if ('width' in widgets):
        width_wdg = wdg.FloatSlider(
            min=0.1, 
            max=5, 
            step=0.1, 
            value=width, 
            description="Aperture Width (m)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(width_wdg)
    else:
        width_wdg = wdg.fixed(width)
        
    sub_controls1_box = []
    if sub_controls1:
        sub_controls1_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        sub_controls1_box = wdg.VBox(sub_controls1_title + sub_controls1, layout=BOX_LAYOUT)
        controls_box.append(sub_controls1_box)
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # Plot
    def plot(freq, height, width):
        h_beamw, v_beamw = rd.rect_beamw(height, width, freq*1E6)
        text1.set_text(f"Horiz. Beamwidth: {h_beamw:.2f} deg")
        text2.set_text(f"Vert. Beamwidth: {v_beamw:.2f} deg")
        wlen = rd.wavelen(freq*1E6)
        u_mesh = -(width/wlen)*u_mesh0
        v_mesh = (height/wlen)*v_mesh0
        img = 2*rd.to_db(rd.rect_pat(u_mesh, v_mesh))
        pc.set_array(img.ravel())
        fig1.canvas.draw()
   
    # Add interaction
    wdg.interactive(
        plot, 
        freq=freq_wdg,
        height=height_wdg,
        width=width_wdg
    )    
    
def roc(
    num_samp: int = 1000,
    snr: float = 10,
    widgets = ['snr'],
    xlim = [0, 1],
    ylim = [0, 1]
    ):
    """
    Receiver operating characteristic curve demonstration.
    
    Inputs:
    - num_samp [int]: Number of data points; default 1000 points
    - snr [float]: Target signal-to-noise ratio (dB); default 10 dB
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting
    - ylim [List[float]]: y-axis limits for plotting
    
    Outputs:
    (none)
    """

    def qfun(x):
        return 0.5 - 0.5*special.erf(x/np.sqrt(2))
    
    def qfuninv(x):
        return np.sqrt(2)*special.erfinv(1 - 2*x)
    
    # Initialize plot
    fig1, ax1 = plt.new_plot()
    ax1.set_xlabel(r'$P_{FA}$')
    ax1.set_ylabel(r'$P_{D}$')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # False alarm bins
    pfa_bin = np.linspace(xlim[0], xlim[1], num_samp)
    
    # Initial plot
    pd = qfun(qfuninv(pfa_bin) - np.sqrt(rd.from_db(snr)))
    roc_line = ax1.plot(pfa_bin, pd, color='red', linewidth=3.0)[0]
    ax1.plot([0, 1], [0, 1], color='k', linestyle='dashed', linewidth=2.0)
    
    # Controls box
    controls_box = []
    
    # ROC controls
    roc_controls = []
    
    # Signal-to-noise ratio
    snr_wdg = wdg.FloatSlider(
            min=-20, 
            max=30, 
            step=1, 
            value=snr, 
            description="SNR (dB)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
    roc_controls.append(snr_wdg)
    
    # Display game controls
    roc_box = []
    if roc_controls:
        roc_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        roc_box = wdg.VBox(roc_title + roc_controls, layout=BOX_LAYOUT)
        controls_box.append(roc_box) 
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))    

    # Plot
    def plot(snr):
        pd = qfun(qfuninv(pfa_bin) - np.sqrt(rd.from_db(snr)))
        roc_line.set_ydata(pd)
   
    # Add interaction
    wdg.interactive(
        plot, 
        snr=snr_wdg
    )
    
def sine_prop(
    freq: float = 50,
    interval: float = 75,
    num_step: int = 200,
    play_lock: bool = False,
    power: float = 500,
    propvel: float = 3E8,
    time_scale: int = -3,
    wave_label: str = 'V/m',
    widgets = ['freq', 'power', 'run'],
    xlim = [1, 50]):
    """
    Propagation loss with sine wave.
    
    Inputs:
    - freq [float]: Transmit frequency (MHz); default 50 MHz
    - interval [float]: Time between animation steps (ms); default 75 ms
    - num_step [int]: Number of animation steps; default 200 steps
    - play_lock [bool]: Flag for locking widgets while playing; default False
    - power [float]: Transmit power (W); default 500 W
    - propvel [float]: Propagation velocity (m/s); default 1000 m/s
    - time_scale [int]: Time scale, e.g., -1 = milli; default -3
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (V)
    
    Outputs:
    (none)
    """

    # y axis limits
    ylim = [-25, 25]
    
    # Axis spans
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    
    # Wave amplitude label
    if wave_label:
        wave_str = 'Wave (' + wave_label + ')'
    else:
        wave_str = 'Wave'

    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(RANGE_M_LABEL)
    ax1.set_ylabel(wave_str)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    
    # Time vector
    t0 = xlim[0]/propvel
    t1 = xlim[1]/propvel
    dt = (t1 - t0)/(num_step - 1)
    
    # Time scale
    if (time_scale == -1):
        time_label = 'ms'
        time_scalar = 1E3
    elif (time_scale == -2):
        time_label = 'µs'
        time_scalar = 1E6
    elif (time_scale == -3):
        time_label = 'ns'
        time_scalar = 1E9

    # Timestamp
    timestamp = ax1.text(xlim[1] + 0.02*x_span, ylim[1] - 0.03*y_span, f"Time: {t0*time_scalar:.2f} {time_label}", size=12.0)

    # Wave
    xvec = np.linspace(xlim[0], xlim[1], num_step)
    y_wave = np.zeros(xvec.shape)
    wave = ax1.plot(xvec, y_wave, color='red', linewidth=3.0)[0]
        
    # Truth
    a0 = np.sqrt(cnst.eta*power)
    y_truth = a0*np.sqrt(1/4/np.pi/xvec**2)
    truth1 = ax1.plot(xvec, y_truth, color='gray', linewidth=2.0, linestyle='dashed')[0]
    truth2 = ax1.plot(xvec, -y_truth, color='gray', linewidth=2.0, linestyle='dashed')[0]
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=10, 
            max=500, 
            step=10, 
            value=freq, 
            description=TX_FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
    
    # Power
    if ('power' in widgets):
        power_wdg = wdg.FloatSlider(
            min=0, 
            max=1000, 
            step=10, 
            value=power, 
            description=TX_POW_W_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(power_wdg)
    else:
        power_wdg = wdg.fixed(power)
        
    wave_box = []
    if wave_controls:
        wave_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        wave_box = wdg.VBox(wave_title + wave_controls, layout=BOX_LAYOUT)
        controls_box.append(wave_box)
        
    # Run widgets
    run_controls = []
    if ('run' in widgets):
        play_wdg = wdg.Play(
            interval=interval,
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=RUN_LABEL,
            disabled=False
        )
        run_controls.append(play_wdg)
        slider_wdg = wdg.IntSlider(
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=FRAME_LABEL,
            readout=False,
            disabled=False
        )
        wdg.jslink((play_wdg, 'value'), (slider_wdg, 'value'))
        run_controls.append(slider_wdg)
        
    run_box = []
    if run_controls:
        run_title = [wdg.HTML(value = RUN_BLOCK_LABEL)]
        run_box = wdg.VBox(run_title + run_controls, layout=BOX_LAYOUT)
        controls_box.append(run_box)   
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def animate(frame, freq, power):

        # Time
        t = t0 + (frame - 1)*dt

        # Update timestamp
        timestamp.set_text(f"Time: {t*time_scalar:.2f} {time_label}")
 
        # Update truth
        a0 = np.sqrt(cnst.eta*power)
        y_truth = a0*np.sqrt(1/4/np.pi/xvec**2)
        truth1.set_ydata(y_truth)
        truth2.set_ydata(-y_truth)

        # Build impulse
        y_wave = y_truth*np.sin(2*np.pi*(1E6*freq/propvel)*(xvec - propvel*t))
        y_wave[xvec > propvel*t] = 0.0
        wave.set_ydata(y_wave)
        
        # Disable controls during play
        if (play_lock):
            if (frame > 1):
                for w in wave_controls:
                    if not w.disabled:
                        w.disabled = True
            elif (frame == 1):
                for w in wave_controls:
                    if w.disabled:
                        w.disabled = False
    
    # Add interaction
    wdg.interactive(
        animate, 
        frame=slider_wdg,
        freq=freq_wdg,
        power=power_wdg
    ) 
    
def sine_prop_generic(
    freq=200,
    interval=75,
    num_step=200,
    play_lock=False,
    power=500,
    propvel=1E3,
    time_scale=-1,
    widgets=['freq', 'power', 'run'],
    xlim=[1, 50]):
    """
    Propagation loss with sine wave.
    
    Inputs:
    - freq [float]: Transmit frequency (Hz); default 200 Hz
    - interval [float]: Time between animation steps (ms); default 75 ms
    - num_step [int]: Number of animation steps; default 200 steps
    - play_lock [bool]: Flag for locking widgets while playing; default False
    - power [float]: Transmit power (W); default 500 W
    - propvel [float]: Propagation velocity (m/s); default 1000 m/s
    - time_scale [int]: Time scale, e.g., -1 = milli; default -1
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (V)
    
    Outputs:
    (none)
    """
    
    # y axis limits
    ylim = [-25, 25]
    
    # Axis spans
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    
    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(RANGE_M_LABEL)
    ax1.set_ylabel(WAVE_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    
    # Timestamp
    timestamp = ax1.text(xlim[1] + 0.02*x_span, ylim[1] - 0.03*y_span, "Time: ---", size=12.0)

    # Time vector
    t0 = xlim[0]/propvel
    t1 = xlim[1]/propvel
    dt = (t1 - t0)/(num_step - 1)
    
    # Wave
    xvec = np.linspace(xlim[0], xlim[1], num_step)
    y_wave = np.zeros(xvec.shape)
    wave = ax1.plot(xvec, y_wave, color='red', linewidth=3.0)[0]
        
    # Truth
    a0 = np.sqrt(cnst.eta*power)
    y_truth = a0*np.sqrt(1/4/np.pi/xvec**2)
    truth1 = ax1.plot(xvec, y_truth, color='gray', linewidth=2.0, linestyle='dashed')[0]
    truth2 = ax1.plot(xvec, -y_truth, color='gray', linewidth=2.0, linestyle='dashed')[0]
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=100, 
            max=500, 
            step=10, 
            value=freq, 
            description=TX_FREQ_HZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
    
    # Power
    if ('power' in widgets):
        power_wdg = wdg.FloatSlider(
            min=0, 
            max=1000, 
            step=10, 
            value=power, 
            description=TX_POW_W_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(power_wdg)
    else:
        power_wdg = wdg.fixed(power)
        
    wave_box = []
    if wave_controls:
        wave_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        wave_box = wdg.VBox(wave_title + wave_controls, layout=BOX_LAYOUT)
        controls_box.append(wave_box)
        
    # Run widgets
    run_controls = []
    if ('run' in widgets):
        play_wdg = wdg.Play(
            interval=interval,
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=RUN_LABEL,
            disabled=False
        )
        run_controls.append(play_wdg)
        slider_wdg = wdg.IntSlider(
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=FRAME_LABEL,
            readout=False,
            disabled=False
        )
        wdg.jslink((play_wdg, 'value'), (slider_wdg, 'value'))
        run_controls.append(slider_wdg)
        
    run_box = []
    if run_controls:
        run_title = [wdg.HTML(value = RUN_BLOCK_LABEL)]
        run_box = wdg.VBox(run_title + run_controls, layout=BOX_LAYOUT)
        controls_box.append(run_box)   
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def animate(frame, freq, power):

        # Time
        t = t0 + (frame - 1)*dt
        
        # Update timestamp
        if (time_scale == -1):
            timestamp.set_text("Time: {ti:.2f} ms".format(ti=t*1E3))
        elif (time_scale == -2):
            timestamp.set_text("Time: {ti:.2f} µs".format(ti=t*1E6))
        elif (time_scale == -3):
            timestamp.set_text("Time: {ti:.2f} ns".format(ti=t*1E9))
 
        # Update truth
        a0 = np.sqrt(cnst.eta*power)
        y_truth = a0*np.sqrt(1/4/np.pi/xvec**2)
        truth1.set_ydata(y_truth)
        truth2.set_ydata(-y_truth)

        # Build impulse
        y_wave = y_truth*np.sin(2*np.pi*(freq/propvel)*(xvec - propvel*t))
        y_wave[xvec > propvel*t] = 0.0
        wave.set_ydata(y_wave)
        
        # Disable controls during play
        if (play_lock):
            if (frame > 1):
                for w in wave_controls:
                    if not w.disabled:
                        w.disabled = True
            elif (frame == 1):
                for w in wave_controls:
                    if w.disabled:
                        w.disabled = False
    
    # Add interaction
    wdg.interactive(
        animate, 
        frame=slider_wdg,
        freq=freq_wdg,
        power=power_wdg
    ) 
    
def sine_pulse(
    energy: float = 50,
    freq: float = 3,
    num_bins: int = 1000,
    prf: float = 200,
    pulsewidth: float = 1,
    show_duty: bool = False,
    show_peak: bool = True,
    show_pri: bool = False,
    widgets=['energy', 'freq', 'prf', 'pulsewidth'],
    ):
    """
    Pulsed sine wave demonstration.
    
    Inputs:
    - energy [float]: Transmit energy (mJ); default 50 mJ
    - freq [float]: Transmit frequency (MHz); default 3 MHz
    - num_bins [int]: Number of time bins; default 1000 steps
    - prf [float]: Pule repetition frequency (Hz); default 1000 Hz
    - pulsewidth [float]: Transmit pulsewidth (µs); default 5 µs
    - show_duty [bool]: Flag for duty cycle display; default False
    - show_peak [bool]: Flag for peak power display; default True
    - show_pri [bool]: Flag for pulse repetition interval display; default False
    - widgets [List[str]]: List of desired widgets
    
    Outputs:
    (none)
    """

    # Midpoint frequency
    max_pulsewidth = 10
    
    # x limits
    xlim = [0, max_pulsewidth]
    
    # Initialize plot
    fig, axs = plt.new_plot2()
    wave_ax = axs[0]
    power_ax = axs[1]
    power_ax.set_xlabel(TIME_US_LABEL)
    wave_ax.set_ylabel(WAVEFORM_V_LABEL)
    power_ax.set_ylabel('Power (kW)')
    wave_ax.set_xlim(xlim)
    wave_ax.set_ylim([-500, 500])
    power_ax.set_xlim(xlim)
    power_ax.set_ylim([0, 100])
    
    # Time vector
    xvec = np.linspace(xlim[0], xlim[1], num_bins)
    
    # Waves
    pri = 1/prf/1E3
    amp = np.sqrt(2*(energy/1E3)/(np.minimum(pulsewidth/1E6, pri)))
    yii = amp*np.sin(2*np.pi*(freq*1E6)*(xvec/1E6))
    num_pulse = np.ceil((max_pulsewidth/1E6)/pri).astype('int')
    for ii in range(num_pulse):
        zeroix = np.logical_and(xvec >= (ii*pri*1E6 + pulsewidth), xvec < (ii+1)*pri*1E6)
        yii[zeroix] = 0.0
    wave_line = wave_ax.plot(xvec, yii, linewidth=2.0, color='red')[0]
    
    # Power
    power = (np.abs(yii)**2)/2
    power_line = power_ax.plot(xvec, power/1E3, linewidth=2.0, color='red')[0]
    
    # Timestamp
    if show_peak:
        peak = amp**2/2
        peak_power = fig.text(0.82, 0.9, f"Peak Power: {peak/1E3:.2f} kW", size=12)
    if show_pri:
        pri_text = fig.text(0.82, 0.85, f"PRI: {pri*1E6:.2f} µs", size=12)
    if show_duty:
        duty_cycle = np.minimum(pulsewidth/1E6/pri, 1.0)
        duty_text = fig.text(0.82, 0.8, f"Duty Cycle: {duty_cycle:.2f}", size=12)
        
    # Peak power
    peak_line = power_ax.plot(
        [xlim[0], xlim[1]], 
        [amp**2/2/cnst.eta/1E3, amp**2/2/cnst.eta/1E3], 
        linewidth=1.0, 
        color='black', 
        linestyle='dashed'
        )[0]
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Transmit energy
    if ('energy' in widgets):
        energy_wdg = wdg.FloatSlider(
            min=1, 
            max=100, 
            step=1, 
            value=energy, 
            description=ENERGY_MJ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(energy_wdg)
    else:
        energy_wdg = wdg.fixed(energy)
    
    # Transmit frequency
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=1, 
            max=5, 
            step=0.1, 
            value=freq, 
            description=FREQ_MHZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    # Pulse repetition frequency
    if ('prf' in widgets):
        prf_wdg = wdg.FloatSlider(
            min=100, 
            max=500, 
            step=1, 
            value=prf, 
            description="PRF (kHz)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(prf_wdg)
    else:
        prf_wdg = wdg.fixed(prf)
        
    # Transmit pulsewidth
    if ('pulsewidth' in widgets):
        pulsewidth_wdg = wdg.FloatSlider(
            min=0.1, 
            max=10, 
            step=0.1, 
            value=pulsewidth, 
            description="Pulsewidth (µs)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(pulsewidth_wdg)
    else:
        pulsewidth_wdg = wdg.fixed(pulsewidth)
        
    wave_box = []
    if wave_controls:
        wave_title = [wdg.HTML(value = TX_BLOCK_LABEL)]
        wave_box = wdg.VBox(wave_title + wave_controls, layout=BOX_LAYOUT)
        controls_box.append(wave_box)
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def plot(energy, freq, prf, pulsewidth):
        
        # Waves
        pri = 1/prf/1E3
        amp = np.sqrt(2*(energy/1E3)/(np.minimum(pulsewidth/1E6, pri)))
        yii = amp*np.sin(2*np.pi*(freq*1E6)*(xvec/1E6))
        num_pulse = np.ceil((max_pulsewidth/1E6)/pri).astype('int')
        for ii in range(num_pulse):
            zeroix = np.logical_and(xvec >= (ii*pri*1E6 + pulsewidth), xvec < (ii+1)*pri*1E6)
            yii[zeroix] = 0.0
        wave_line.set_ydata(yii)

        # Power
        power = (np.abs(yii)**2)/2
        peak = amp**2/2
        power_line.set_ydata(power/1E3)
        
        # Peak power
        if show_peak:
            peak_power.set_text(f"Peak Power: {peak/1E3:.2f} kW")
            
        # Peak power
        if show_pri:
            pri_text.set_text(f"PRI: {pri*1E6:.2f} µs")
            
        # Duty cycle
        if show_duty:
            duty_cycle = np.minimum(pulsewidth/1E6/pri, 1.0)
            duty_text.set_text(f"Duty Cycle: {duty_cycle:.2f}")
            
        # Peak power line
        peak_line.set_ydata([amp**2/2/1E3, amp**2/2/1E3])
        
    # Add interaction
    wdg.interactive(
        plot, 
        energy=energy_wdg,
        freq=freq_wdg,
        prf=prf_wdg,
        pulsewidth=pulsewidth_wdg
    )    

def snr(
    noise_energy: float = -20,
    num_samp: int = 1000,
    show_snr: bool = True,
    signal_energy: float = 10,
    widgets = ['noise', 'signal'],
    xlim = [0, 100],
    ylim = [-30, 30]
    ):
    """
    Pulsed sine wave demonstration.
    
    Inputs:
    - noise_energy [float]: Noise energy (dBJ); default -20 dBJ
    - num_samp [int]: Number of data samples; default 1000 samples
    - signal_energy [float]: Signal energy (dBJ); default 10 dBJ
    - show_snr [bool]: Flag for signal-to-noise ratio display; default True
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (ms)
    - ylim [List[float]]: y-axis limits for plotting (dBJ)

    Outputs:
    (none)
    """

    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Energy (dBJ)')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
   
    # Live text
    if show_snr:
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        snr0_db = signal_energy - noise_energy
        text1 = ax1.text(xlim[1] + 0.025*dx, ylim[1] - 0.07*dy, "SNR: {val_db:.2f} dB".format(val_db=snr0_db), size=12.0)

    # Control widgets
    controls_box = []
       
    # Subcontrol widgets
    sub_controls1 = []
   
    # Noise energy
    if ('noise' in widgets):
        noise_wdg = wdg.FloatSlider(
            min=-20, 
            max=20, 
            step=0.5, 
            value=noise_energy, 
            description="Noise Energy (dBJ)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(noise_wdg)
    else:
        noise_wdg = wdg.fixed(noise_energy)

    # Signal energy
    if ('signal' in widgets):
        signal_wdg = wdg.FloatSlider(
            min=-20, 
            max=20, 
            step=0.5, 
            value=signal_energy, 
            description="Signal Energy (dBJ)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        sub_controls1.append(signal_wdg)
    else:
        signal_wdg = wdg.fixed(signal_energy)
        
    sub_controls1_box = []
    if sub_controls1:
        sub_controls1_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        sub_controls1_box = wdg.VBox(sub_controls1_title + sub_controls1, layout=BOX_LAYOUT)
        controls_box.append(sub_controls1_box)
       
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
       
    # First plot
    tvec = np.linspace(0, 100, num_samp)
    noise_line = ax1.plot(tvec, 2*rd.to_db(np.sqrt(rd.from_db(noise_energy))*np.random.randn(num_samp)), color='red')[0]
    echo_line = ax1.plot([50, 50], [ylim[0], signal_energy], color='black', linewidth=3.0)[0]
    
    # Plot
    def plot(noise_energy, signal_energy):
        if show_snr:
            snr0_db = signal_energy - noise_energy
            text1.set_text("SNR: {val_db:.2f} dB".format(val_db=snr0_db))
        noise_line.set_ydata(2*rd.to_db(np.sqrt(rd.from_db(noise_energy))*np.random.randn(num_samp)))
        echo_line.set_ydata([ylim[0], signal_energy])
   
    # Add interaction
    wdg.interactive(
        plot, 
        noise_energy=noise_wdg,
        signal_energy=signal_wdg
    )
      
def threshold(
    n: int = 50,
    thresh: float = 0,
    snr: float = 10
    ):
    """
    Constant threshold detection demonstration.
    
    Inputs:
    - n [int]: Number of data points; default 50 points
    - thresh [float]: Detection threshold (dB); default 0 dB
    - snr [float]: Target signal-to-noise ratio (dB); default 10 dB
    
    Outputs:
    (none)
    """
    
    def update_results(values, results, positions, thresh):
        exceed = values >= thresh
        num_echo = np.sum(positions)
        num_noise = np.sum(~positions)
        pd = 0.0
        pfa = 0.0
        for ii in range(n):
            if positions[ii] and exceed[ii]:
                pd += 1/num_echo
                color = 'g'
            elif positions[ii] and not exceed[ii]:
                color = 'm'
            elif not positions[ii] and exceed[ii]:
                pfa += 1/num_noise
                color = 'r'
            elif not positions[ii] and not exceed[ii]:
                color = 'b'

            results[ii].set_ydata([y_min, values[ii]])
            results[ii].set_color(color)
            
        return pd, pfa
    
    def update_text(pd, pfa):
        pd_text.set_text(f"Prob. of Detection: {pd:.3f}")
        pfa_text.set_text(f"Prob. of False Alarm: {pfa:.3f}")
    
    # Minimum energy value
    y_min = -40
    
    # Initial values
    noiseAmp = 1E0
    time = np.arange(0, n)
    values = y_min*np.ones(len(time))
    positions = np.zeros((n), dtype='bool')
    
    # Figure/axes
    fig, ax = plt.new_plot()
    ax.set_xlabel(DELAY_US_LABEL)
    ax.set_ylabel(SNR_DB_LABEL)
    ax.set_ylim([y_min, y_min + 80])
    
    # Draw samples
    echoAmp = np.sqrt(rd.from_db(snr))*noiseAmp
    positions[:] = np.random.randint(0, 5, n) >= 4
    values[:] = 2*rd.to_db(echoAmp * positions + noiseAmp * np.random.randn(n))
    
    # Threshold
    thresh_line = ax.plot([0, n], [thresh, thresh], '--k')[0]
    
    # Results
    results = []
    for ii in range(n):
        results.append(ax.plot([time[ii], time[ii]], [y_min, y_min], color='k', linewidth=2.0)[0])
        
    # Update colors
    pd, pfa = update_results(values, results, positions, thresh)
    
    # Accuracy texts
    pd_text = fig.text(0.72, 0.85, f"Prob. of Detection: {pd:.3f}", size=12.0)
    pfa_text = fig.text(0.72, 0.8, f"Prob. of False Alarm: {pfa:.3f}", size=12.0)
    
    # Controls box
    controls_box = []
    
    # Data controls
    data_controls = []
    
    # New button
    new_btn = wdg.Button(description="New")
    data_controls.append(new_btn)
    
    data_controls.append(wdg.HTML(value = TARGET_BLOCK_LABEL))
    
    # Signal-to-noise ratio
    snr_wdg = wdg.FloatSlider(
            min=0, 
            max=30, 
            step=1, 
            value=snr, 
            description="Target SNR (dB)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
    data_controls.append(snr_wdg)
    
    # Data control box
    data_box = []
    if data_controls:
        data_title = [wdg.HTML(value = f"<b><font color='black'>Data</b>")]
        data_box = wdg.VBox(data_title + data_controls, layout=BOX_LAYOUT)
        controls_box.append(data_box)
        
    # Detection controls
    det_controls = []
    
    # Threshold
    thresh_wdg = wdg.FloatSlider(
            min=-40, 
            max=40, 
            step=1, 
            value=thresh, 
            description="Threshold (dB)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
    det_controls.append(thresh_wdg)
    
    # Detection control box
    det_box = []
    if det_controls:
        det_title = [wdg.HTML(value = f"<b><font color='black'>Detection</b>")]
        det_box = wdg.VBox(det_title + det_controls, layout=BOX_LAYOUT)
        controls_box.append(det_box)
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))
    
    def new_data(btn):
        
        # Draw samples
        echoAmp = np.sqrt(rd.from_db(snr_wdg.value))*noiseAmp
        positions[:] = np.random.randint(0, 5, n) >= 4
        values[:] = 2*rd.to_db(echoAmp * positions + noiseAmp * np.random.randn(n))
        
        # Update display
        pd, pfa = update_results(values, results, positions, thresh)
        update_text(pd, pfa)
    
    def threshold_update(thresh):
        thresh_line.set_ydata([thresh, thresh])
        pd, pfa = update_results(values, results, positions, thresh)
        update_text(pd, pfa)
    
    wdg.interactive(threshold_update, thresh=thresh_wdg)
    
    new_btn.on_click(new_data)
    
def wave(
    freq=1000,
    interval=50,
    max_freq=5000,
    min_freq=100,
    num_step=250,
    play_lock=False,
    propvel=1000,
    widgets=['freq', 'propvel', 'run'],
    xlim=[0, 4],
    ylim=[-2, 2]):
    """
    Radiated wave demonstration.
    
    Inputs:
    - freq [float]: Transmit frequency (Hz); default 1000 Hz
    - interval [float]: Time between animation steps (ms); default 50 ms
    - max_freq [float]: Maximum frequency (Hz); default 5000
    - min_freq [float]: Minimum frequency (Hz); default 100
    - num_step [int]: Number of animation steps; default 250 steps
    - propvel [float]: Propagation velocity (m/s); default 1000 m/s
    - widgets [List[str]]: List of desired widgets
    - xlim [List[float]]: x-axis limits for plotting (m)
    - ylim [List[float]]: y-axis limits for plotting (m)
    
    Outputs:
    (none)
    """

    # Initialize plot
    _, ax1 = plt.new_plot()
    ax1.set_xlabel(RANGE_M_LABEL)
    ax1.set_ylabel(WAVE_LABEL)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # Time vector
    t0 = 0
    t1 = 2/500
    dt = (t1 - t0)/(num_step - 1)
    
    # Timestamp
    timestamp = ax1.text(xlim[1] + 0.1, ylim[1] - 0.2, f"Time: {t0*1E3:.2f} ms", size=12.0)

    # Wave value marker
    wave_val = ptch.Circle((0, 0), 0.1, color='pink')
    ax1.add_patch(wave_val)
    
    # Wave
    xvec = np.linspace(xlim[0], xlim[1], num_step)
    wave = ax1.plot(xvec, np.zeros((num_step,)), color='red', linewidth=3.0)[0]
        
    # Control widgets
    controls_box = []
        
    # Wave control widgets
    wave_controls = []
    
    # Boresight azimuths
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=min_freq, 
            max=max_freq, 
            step=100, 
            value=freq, 
            description=FREQ_HZ_LABEL, 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    if ('propvel' in widgets):
        propvel_wdg = wdg.FloatSlider(
            min=500, 
            max=2500, 
            step=100, 
            value=propvel, 
            description="Propagation Velocity (m/s)", 
            style=LABEL_STYLE, 
            readout_format='.2f'
        )
        wave_controls.append(propvel_wdg)
    else:
        propvel_wdg = wdg.fixed(propvel)
        
    wave_box = []
    if wave_controls:
        wave_title = [wdg.HTML(value = CONTROL_BLOCK_LABEL)]
        wave_box = wdg.VBox(wave_title + wave_controls, layout=BOX_LAYOUT)
        controls_box.append(wave_box)
        
    # Run widgets
    run_controls = []
    if ('run' in widgets):
        play_wdg = wdg.Play(
            interval=interval,
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=RUN_LABEL,
            disabled=False
        )
        run_controls.append(play_wdg)
        slider_wdg = wdg.IntSlider(
            value=1,
            min=1,
            max=num_step,
            step=1,
            description=FRAME_LABEL,
            readout=False,
            disabled=False
        )
        wdg.jslink((play_wdg, 'value'), (slider_wdg, 'value'))
        run_controls.append(slider_wdg)
        
    run_box = []
    if run_controls:
        run_title = [wdg.HTML(value = RUN_BLOCK_LABEL)]
        run_box = wdg.VBox(run_title + run_controls, layout=BOX_LAYOUT)
        controls_box.append(run_box)   
        
    # Display widgets
    if controls_box:
        display(wdg.GridBox(controls_box, layout=WDG_LAYOUT))

    # Plot
    def animate(frame, freq, propvel):

        # Time
        t = t0 + (frame - 1)*dt

        # Update timestamp
        timestamp.set_text("Time: {ti:.2f} ms".format(ti=t*1000))
 
        wave_amp = np.cos(2*np.pi*(freq/propvel)*(xvec - propvel*t))
        wave_amp[xvec > propvel*t] = 0.0

        wave_val.center = 0, wave_amp[0]
        wave.set_ydata(wave_amp)
        
        # Disable controls during play
        if (play_lock):
            if (frame > 1):
                for w in wave_controls:
                    if not w.disabled:
                        w.disabled = True
            elif (frame == 1):
                for w in wave_controls:
                    if w.disabled:
                        w.disabled = False
    
    # Add interaction
    wdg.interactive(
        animate, 
        frame=slider_wdg,
        freq=freq_wdg, 
        propvel=propvel_wdg
    )
