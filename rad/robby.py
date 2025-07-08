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
from IPython.display import display
import ipywidgets as wdg
import matplotlib.pyplot as pyp
import numpy as np
from numpy.random import randn
import rad.plot as pl
import rad.radar as rd
import rad.const as cnst
from rad.radar import Target

def pulse(
    bandw: float, 
    beamw: float, 
    coherent: bool,
    energy: float,
    freq: float,
    gain: float,
    min_range: float,
    noise_power: float,
    num_integ: int,
    targets: list[Target],
    time: float,
    az: float,
    range_bins
    ):
    """
    Generate a single pulse response from a set of Targets.
    
    Inputs:
    - bandw [float]: Transmit bandwidth (Hz)
    - beamw [float]: Transmit azimuth beamwidth (rad)
    - coherent [bool]: Flag for coherent integration
    - energy [float]: Transmit energy (J)
    - freq [float]: Transmit frequency (Hz)
    - gain [float]: Total transmit and receive gain
    - min_range [float]: Minimum observable range (m)
    - noise_power [float]: System noise power (W)
    - num_integ [int]: Number of integrated pulses
    - targets [list[Target]]: List of Targets
    - time [float]: Time (s)
    - az [float]: Transmit azimuth (rad)
    - range_bins [ArrayLike]: Range bins (m)
    
    Outputs:
    - pulse [ArrayLike]: Pulse (V)
    """
    
    # Range resolution
    res = rd.range_res(bandw)
    
    # Generate pulse
    pulse = np.zeros((range_bins.size,))
    for tgt in targets:
        
        # Target range/az
        tgt_raz = tgt.get_raz(time)
        
        # Check for blanking
        if (tgt_raz[0] >= min_range):
        
            # Centered range bins
            dr = range_bins - tgt_raz[0]

            # Distance from boresight (rad)
            daz = np.angle(np.exp(1j*(tgt_raz[1] - az)))

            # Beamshape
            beamshape = rd.dish_pat((0.61/beamw)*np.sin(daz))

            # One-way
            beamshape[np.abs(daz) > np.pi/2] = 0.0

            # Received energy (J)
            rx_energy = rd.rx_energy(tgt_raz[0], energy, freq, gain=gain, rcs=rd.from_db(tgt.rcs))
            
            # Integration gain
            if (num_integ > 1):
                if coherent:
                    integ_gain = num_integ
                else:
                    integ_gain = np.sqrt(num_integ)
                rx_energy *= integ_gain

            # Add to pulse
            pulse += np.sqrt(rx_energy)*beamshape*rd.flat_prof(dr, res)

    # Add noise
    pulse += np.sqrt(noise_power)*randn((range_bins.size))

    return pulse    

def robby(
    bandw: float = 1,
    coherent: bool = True,
    det_thresh: float = 12,
    dets: bool = False,
    energy: float = 0.3,
    freq: float = 2E3,
    max_price: float = 1E5,
    max_range: float = 10E3,
    min_range: float = 2E3,
    noise_temp: float = 1000,
    num_integ: int = 1,
    num_targ: int = 1,
    pulses: bool = True,
    radius: float = 0.3,
    reset: bool = False,
    scan_rate: float = 5,
    show_price: bool = False,
    targets: list[Target] | None = None,
    widgets = ['bandw', 'coherent', 'det_thresh', 'freq', 'energy', 'noise_temp', 'num_integ', 'prf', 'radius', 'scan_rate'] # 'test_targ'
    ):
    """
    Create test radar display.
    
    Inputs:
    - bandw [float]: Transmit bandwidth (MHz); default 1 MHz
    - coherent [bool]: Flag for coherent integration; default True
    - det_thresh [float]: Detection threshold (dB); default 12 dB
    - dets [bool]: Flag for detection display; default False
    - energy [float]: Transmit energy (mJ); default 0.3 mJ
    - freq [float]: Transmit frequency (MHz); default 2E3 MHz
    - max_price [float]: Maximum allowable price ($); default $100,000
    - max_range [float]: Maximum observable range (m); default 10E3
    - min_range [float]: Minimum observable range (m); default 2E3
    - noise_temp [float]: System noise temperature (K); default 1000
    - num_integ [int]: Number of integrated pulses; default 1 pulse
    - num_targ [int]: Number of test targets; default 1 target
    - pulses [bool]: Flag for pulse display; default True
    - radius [float]: Dish radius (m): default 0.3 m
    - reset [bool]: Flag for reset button; default False
    - scan_rate [float]: Scan rate (scans/min); default 5 scans/min
    - show_price [bool]: Flat for price display; default False
    - targets [list[Target]]: list of Targets; default None
    - widgets [list[str]]: list of desired widgets
    
    Outputs:
    (none)
    """
        
    # Widget style
    style = {'description_width': 'initial'}
    box_layout = wdg.Layout(justify_items='flex-start')
    wdg_layout = wdg.Layout(grid_template_columns="repeat(3, 350px)")    

    # Build axes
    fig_scan, ax_scan, fig_pulse, ax_pulse = pl.new_rad_plot()
    
    # Pulse axes
    ax_pulse.set_ylabel('SNR (dB)')
    ax_pulse.set_xlabel('Range (m)')
    
    # Remove radius labels
    ax_scan.set_yticklabels([])
    
    # Live text
    text1 = fig_scan.text(0, 0, "Time: ---", 
        family='monospace', 
        size=9,
        weight='normal'
    )
    
    # Azimuth bins (rad)
    num_az = 500
    az_bins = np.linspace(0, 2*np.pi, num_az)
    daz = az_bins[1] - az_bins[0]
    
    # Range bins
    num_range = 500
    range_bins = np.linspace(min_range, max_range, num_range)
    dr = range_bins[1] - range_bins[0]
    
    # Mesh for detections
    az_mesh, range_mesh = np.meshgrid(az_bins, range_bins)
    
    # Targets
    if not targets:
        targets = []
        for ii in range(num_targ):
            targii = rd.Target()
            targii.pos = np.array([0, min_range])
            targii.rcs = 0
            targets.append(targii)
    
    # Control widgets
    controls_box = []
       
    # Radar controls
    rad_controls = []
    
    # Noise temperature
    if ('noise_temp' in widgets):
        noise_temp_wdg = wdg.FloatSlider(
            min=600, 
            max=1200, 
            step=10, 
            value=noise_temp, 
            description="Noise Temperature (°K)", 
            style=style, 
            readout_format='.2f'
        )
        rad_controls.append(noise_temp_wdg)
    else:
        noise_temp_wdg = wdg.fixed(noise_temp)
    
    # Dish radius
    if ('radius' in widgets):
        radius_wdg = wdg.FloatSlider(
            min=0.1, 
            max=1,
            step=0.05, 
            value=radius, 
            description="Dish Radius (m)", 
            style=style, 
            readout_format='.2f'
        )
        rad_controls.append(radius_wdg)
    else:
        radius_wdg = wdg.fixed(radius)
        
    if ('scan_rate' in widgets):
        scan_rate_wdg = wdg.FloatSlider(
            min=1, 
            max=10, 
            step=0.1, 
            value=scan_rate, 
            description="Scan Rate (scans/min)", 
            style=style, 
            readout_format='.2f'
        )
        rad_controls.append(scan_rate_wdg)
    else:
        scan_rate_wdg = wdg.fixed(scan_rate)
        
    if ('freq' in widgets):
        freq_wdg = wdg.FloatSlider(
            min=500, 
            max=5000, 
            step=50, 
            value=freq, 
            description="Transmit Frequency (MHz)", 
            style=style, 
            readout_format='.2f'
        )
        rad_controls.append(freq_wdg)
    else:
        freq_wdg = wdg.fixed(freq)
        
    rad_controls_box = []
    if rad_controls:
        rad_controls_title = [wdg.HTML(value = "<b><font color='black'>Radar</b>")]
        rad_controls_box = wdg.VBox(rad_controls_title + rad_controls, layout=box_layout)
        controls_box.append(rad_controls_box)
        
    # Transmission controls
    tx_controls = []
    
    # Bandwidth
    if ('bandw' in widgets):
        bandw_wdg = wdg.FloatSlider(
            min=0.1, 
            max=3, 
            step=0.05, 
            value=bandw, 
            description="Bandwidth (MHz)", 
            style=style, 
            readout_format='.2f'
        )
        tx_controls.append(bandw_wdg)
    else:
        bandw_wdg = wdg.fixed(bandw)

    # Energy
    if ('energy' in widgets):
        energy_wdg = wdg.FloatSlider(
            min=0.1, 
            max=1,
            step=0.01, 
            value=energy, 
            description="Transmit Energy (mJ)", 
            style=style, 
            readout_format='.2f'
        )
        tx_controls.append(energy_wdg)
    else:
        energy_wdg = wdg.fixed(energy)
        
    tx_controls_box = []
    if tx_controls:
        tx_controls_title = [wdg.HTML(value = "<b><font color='black'>Transmission</b>")]
        tx_controls_box = wdg.VBox(tx_controls_title + tx_controls, layout=box_layout)
       
    # Processing controls
    proc_controls = []
    
    if dets and ('det_thresh' in widgets):
        det_thresh_wdg = wdg.FloatSlider(
            min=0, 
            max=30, 
            step=1, 
            value=det_thresh, 
            description="Detection Threshold (dB)", 
            style=style, 
            readout_format='d'
        )
        proc_controls.append(det_thresh_wdg)
    else:
        det_thresh_wdg = wdg.fixed(det_thresh)
    
    if ('num_integ' in widgets):
        num_integ_wdg = wdg.FloatSlider(
            min=1, 
            max=25, 
            step=1, 
            value=num_integ, 
            description="Integrated Pulses", 
            style=style, 
            readout_format='d'
        )
        proc_controls.append(num_integ_wdg)
    else:
        num_integ_wdg = wdg.fixed(num_integ)
    
    if ('coherent' in widgets):
        coh_wdg = wdg.Checkbox(
            value=False,
            description='Coherent Integration'
        )
        proc_controls.append(coh_wdg)
    else:
        coh_wdg = wdg.fixed(coherent)    

    proc_controls_box = []
    if proc_controls:
        proc_controls_title = [wdg.HTML(value = "<b><font color='black'>Processing</b>")]
        proc_controls_box = wdg.VBox(proc_controls_title + proc_controls, layout=box_layout)
        if tx_controls:
            controls_box.append(wdg.VBox([tx_controls_box, proc_controls_box]))
        else:
            controls_box.append(proc_controls_box)
    
    # Operation
    oper_controls = []
    
    scan_wdg = wdg.Button(description="Scan")
    oper_controls.append(scan_wdg)
    
    if reset:
        reset_wdg = wdg.Button(description="Reset")
        oper_controls.append(reset_wdg)

    oper_controls_box = []
    if oper_controls:
        oper_controls_title = [wdg.HTML(value = "<b><font color='black'>Operation</b>")]
        oper_controls_box = wdg.VBox(oper_controls_title + oper_controls, layout=box_layout)
        controls_box.append(oper_controls_box)
    
    # Pulse display slider
    az_pulse_wdg = wdg.FloatSlider(
        min=0, 
        max=360, 
        step=360/(num_az - 1), 
        value=0, 
        description="Azimuth (deg)", 
        style=style, 
        readout_format='.2f'
    )
    az_pulse_title = wdg.HTML(value = "<b><font color='black'>Pulse Display</b>")
    pulse_disp_box = [az_pulse_title, az_pulse_wdg]
    
    # Test targets
    target_controls = []
    targ_update = False
    if ('test_targ' in widgets):
        targ_update = True
        targ_r_wdg = []
        targ_az_wdg = []
        targ_rcs_wdg = []
        for ii in range(num_targ):
            targ_raz = targets[ii].get_raz(0)
            targ_r_wdg.append(wdg.FloatSlider(
                min=min_range, 
                max=max_range, 
                step=dr, 
                value=targ_raz[0], 
                description="Target #" + str(ii) + " Range (m)", 
                style=style, 
                readout_format='.2f'
            ))
            targ_az_wdg.append(wdg.FloatSlider(
                min=0, 
                max=360, 
                step=360/(num_az - 1),
                value=rd.rad2deg(targ_raz[1]), 
                description="Target #" + str(ii) + " Azimuth (deg)", 
                style=style, 
                readout_format='.2f'
            ))
            targ_rcs_wdg.append(wdg.FloatSlider(
                min=-20, 
                max=20, 
                step=1,
                value=targets[ii].rcs,
                description="Target #" + str(ii) + " RCS (dBsm)", 
                style=style, 
                readout_format='.2f'
            ))
            target_controls.append(wdg.VBox([targ_r_wdg[-1], targ_az_wdg[-1], targ_rcs_wdg[-1]]))
        
    target_controls_box = []
    if target_controls:
        target_controls_title = [wdg.HTML(value = "<b><font color='black'>Targets</b>")]
        target_controls_box = wdg.VBox(target_controls_title + target_controls, layout=box_layout)
        controls_box.append(target_controls_box)
        
    # Display widgets
    if controls_box:
        
        controls_wdg = wdg.GridBox(controls_box, layout=wdg_layout)
        if show_price:
            
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
            price_wdg = wdg.HTML(value="")
            if price0 <= max_price:
                price_wdg.value = f"<font color=\"DarkGreen\"><h2>Price: ${price0:.2f}</h2></font>"
            else:
                price_wdg.value = f"<font color=\"Red\"><h2>Price: ${price0:.2f}</h2></font>"
                
            controls_box.insert(0, price_wdg)
            
        display(
            wdg.VBox([
                wdg.AppLayout(
                center=fig_scan.canvas, 
                right_sidebar=wdg.VBox(controls_box),
                pane_widths=[0, '820px', '350px']
                ),
                wdg.AppLayout(
                center=fig_pulse.canvas, 
                right_sidebar=wdg.VBox(pulse_disp_box),
                pane_widths=[0, '820px', '350px']
                ),
            ])      
        )
       
    # Initialize frame
    frame_wdg = wdg.IntSlider(
        value=1,
        min=1,
        max=20,
        step=1,
        description="Frame",
        readout=False,
        disabled=False
    )
    
    # SNR range
    min_scan_snr = -10
    max_scan_snr = 30
    min_pulse_snr = -5
    max_pulse_snr = 30
    
    # Initialize SNR
    snr = min_scan_snr*np.ones((num_range, num_az))
    
    # Pulse radial plot
    pc = ax_scan.pcolormesh(az_bins, range_bins, snr, shading='gouraud', cmap='inferno')
    pc.set_clim(min_scan_snr, max_scan_snr)

    # Colorbar
    cbar = fig_scan.colorbar(pc, ax=ax_scan, shrink=0.5, pad=0.1)
    cbar.ax.set_ylabel('Signal-to-Noise Ratio (dB)', size=12, name=pl.DEF_SANS)
        
    # Detections
    if dets:
        
        # Detection plot
        dets_range = []
        dets_az = []
        dets_line = ax_scan.plot(
            dets_az, 
            dets_range, 
            fillstyle='full', 
            markeredgecolor='white', 
            markerfacecolor='white', 
            marker='.', 
            linestyle='None'
        )[0]
    
    # Initial single pulse
    ax_pulse.set_xlim([min_range, max_range])
    ax_pulse.set_ylim([min_pulse_snr, max_pulse_snr])
    line_pulse = ax_pulse.plot(range_bins, min_scan_snr*np.ones((num_range,)), linewidth=2.0, color='red')[0]
    
    # Scan
    def plot(btn):
    
        # Time (s)
        t = (frame_wdg.value - 1)*(60/scan_rate_wdg.value)
    
        # Update text
        text1.set_text("Time: {ti:.2f} s".format(ti=t))
    
        # Coherent operation
        coherent = coh_wdg.value
        
        # Detection threshold
        det_thresh = det_thresh_wdg.value
        
        # Integration
        num_integ = num_integ_wdg.value*1.0
    
        # Frequency (Hz)
        freq = freq_wdg.value*1E6
        
        # Wavelength (m)
        wavelen = rd.wavelen(freq)

        # Dish radius (m)
        radius = radius_wdg.value
        
        # Dish beamwidth (rad)
        beamw = 1.22*wavelen/2/radius

        # Dish gain
        tx_gain = rd.dish_gain(radius_wdg.value, freq)
        rx_gain = tx_gain
        gain = tx_gain*rx_gain

        # Transmit energy
        tx_energy = energy_wdg.value*1E-3
        
        # Transmit bandwidth (Hz)
        bandw = bandw_wdg.value*1E6
        
        # Noise temperature (°K)
        noise_temp = noise_temp_wdg.value
        
        # Noise power (W)
        noise_power = cnst.k*noise_temp
        
        # Update targets
        if targ_update:
            for ii in range(num_targ):
                rii = targ_r_wdg[ii].value
                azii = np.pi/2 - rd.deg2rad(targ_az_wdg[ii].value)
                rcsii = targ_rcs_wdg[ii].value
                targets[ii].pos[0] = rii*np.cos(azii)
                targets[ii].pos[1] = rii*np.sin(azii)
                targets[ii].rcs = rcsii
        
        # Generate pulses        
        for ii in range(num_az):
            snr[:, ii] = 20*np.log10(np.abs(pulse(
                bandw, 
                beamw,
                coherent,
                tx_energy,
                freq, 
                gain, 
                min_range,
                noise_power,
                num_integ,
                targets, 
                t,
                az_bins[ii], 
                range_bins
            ))) - 10*np.log10(noise_power)
            
        # Update pulses
        if pulses:
            pc.set_array(snr.ravel())
            
        # Update detections
        if dets:
            dets_az, dets_range = rd.detect(az_mesh, range_mesh, snr, det_thresh)
            dets_line.set_data(dets_az, dets_range)
        
        # Update pulse
        update_line(az_pulse_wdg.value)
        
        # Increment frame
        frame_wdg.value += 1
        
        # Grid
        ax_scan.grid(color=[0.8, 0.8, 0.8], linestyle=':', linewidth=1.0)

    def reset_plot(btn):
        frame_wdg.value = 1
        plot(btn)
        
    def update_line(az):
        
        # Update pulse line
        az = rd.deg2rad(az)
        az_bin = np.round(az/daz).astype('int')
        line_pulse.set_ydata(snr[:, az_bin])
        
    def update_price(bandw, coherent, energy, freq, noise_temp, num_integ, radius, scan_rate):

        if show_price:
        
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

            # Update display price
            if price <= max_price:
                price_wdg.value = f"<font color=\"DarkGreen\"><h2>Price: ${price:.2f}</h2></font>"
            else:
                price_wdg.value = f"<font color=\"Red\"><h2>Price: ${price:.2f}</h2></font>"
    
    # Add interactions
    scan_wdg.on_click(plot)
    if reset:
        reset_wdg.on_click(reset_plot)
    wdg.interactive(update_line, az=az_pulse_wdg)
    wdg.interactive(
        update_price,
        bandw=bandw_wdg,
        coherent=coh_wdg,
        energy=energy_wdg,
        freq=freq_wdg,
        noise_temp=noise_temp_wdg,
        num_integ=num_integ_wdg,
        radius=radius_wdg,
        scan_rate=scan_rate_wdg
    )
    


    
    
    
    
    
        
    
    