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

from IPython.display import display, HTML
import ipywidgets as wdg
import matplotlib.patches as ptch
import matplotlib.pyplot as pyp
import rad.air as air
import rad.plot as plt
import rad.const as cnst
import rad.radar as rd
import rad.robby as rby
import rad.toys as ts
import math
import numpy as np
 
#-------Lab 1.1: Introduction to Labs-------

# Example 1.1.1
def ex_1_1_1():
    ts.sine(
        amp = 1,
        freq = 1,
        phase = 0
    )

# Example 1.1.2
def ex_1_1_2():
    ts.sine(
        amp=3,
        freq=2,
        phase=0,
        widgets=[]
    )
    
#-------Lab 1.2: Introduction to Radar-------

# Example 1.2.1
def ex_1_2_1():
    ts.wave()

# Example 1.2.2
def ex_1_2_2():
    ts.prop_loss()
    
# Example 1.2.3
def ex_1_2_3():
    ts.sine_prop_generic()
    
# Example 1.2.4
def ex_1_2_4():
    ts.ranging(
        rx_omni=True,
        tx_omni=True,
        tgt_hide=False,
        tgt_x = 50,
        tgt_y = 50,
        widgets=['run', 'x', 'y']
    )

# Example 1.2.5
def ex_1_2_5():
    ts.ranging(
        rx_omni=True,
        tx_omni=True,
        max_range=400,
        tgt_hide=True,
        tgt_x=75,
        tgt_y=-100,
        widgets=['dets', 'run']
    )
    
# Example 1.2.6
def ex_1_2_6():
    ts.ranging(
        rx_omni=True,
        tx_omni=False,
        tgt_hide=False,
        tgt_x=50,
        tgt_y=50,
        widgets=['tx_az', 'tx_beamw', 'dets', 'run', 'x', 'y']
    )

# Example 1.2.7
def ex_1_2_7():
    ts.ranging(
        rx_omni=True,
        tx_omni=False,
        tgt_hide=True,
        tgt_x=50,
        tgt_y=50,
        widgets=['tx_az', 'tx_beamw', 'dets', 'run']
    )
    
# Example 1.2.8
def ex_1_2_8():
    ts.ranging(
        rx_omni=False,
        tx_omni=True,
        tgt_hide=False,
        tgt_x=50,
        tgt_y=50,
        widgets=['rx_az', 'rx_beamw', 'dets', 'run', 'x', 'y']
    )
    
# Example 1.2.9
def ex_1_2_9():
    ts.ranging(
        rx_omni=False,
        tx_omni=True,
        tgt_hide=True,
        tgt_x=40,
        tgt_y=-20,
        widgets=['rx_az', 'rx_beamw', 'dets', 'run']
    )    

# Example 1.2.10
def ex_1_2_10():
    ts.dish_pat()
    
# Example 1.2.11
def ex_1_2_11():
    ts.array(
        num_elem=7,
        dx=4,
        widgets=['tx_az', 'run', 'x', 'y']
    )    
    
# Example 1.2.12
def ex_1_2_12():
    ts.doppler()
    
# Example 1.2.13
def ex_1_2_13():
    ts.radar_wave()
    
#-------Lab 2.1: Radar Range Equation-------

# Example 2.1.1
def ex_2_1_1a():
    ts.snr(
        noise_energy=-20,
        show_snr=False,
        signal_energy=10,
        widgets=[]
    )

# Example 2.1.2
def ex_2_1_1b():
    ts.snr(
        noise_energy=0,
        show_snr=False,
        signal_energy=0,
        widgets=[]
    )

# Example 2.1.3
def ex_2_1_2():
    ts.snr()
    
# Example 2.1.4
def ex_2_1_3():
    ts.sine_prop()
    
# Example 2.1.5
def ex_2_1_4():
    ts.friis()
    
# Example 2.1.6
def ex_2_1_5():
    ts.radar_range_power()
    
# Example 2.1.7
def ex_2_1_6():
    ts.radar_range_energy()
    
# Example 2.1.8
def ex_2_1_7():
    ts.radar_range_snr()

# Example 2.1.9
def ex_2_1_8():
    ts.radar_range_det()
    
#-------Lab 2.2: Basic Radar Design-------

# Example 2.2.1
def ex_2_2_1():
    ts.radar_range_det()

# Example 2.2.2
def ex_2_2_2():
    ts.design(
        max_rcs=-5,
        min_range=100,
        min_snr=15,
        metrics=['price', 'range', 'rcs', 'snr'],
        widgets=['freq', 'energy', 'noise_temp', 'r', 'radius', 'rcs']
    )

# Example 2.2.3
def ex_2_2_3():
    ts.dish_pat(show_beamw=True)

# Example 2.2.4
def ex_2_2_4():
    ts.rect_pat(show_beamw=True)

# Example 2.2.5
def ex_2_2_5():
    ts.design(
        max_beamw=3,
        max_price=110000,
        max_rcs=-10,
        min_range=120,
        min_snr=14,
        metrics=['beamw', 'price', 'range', 'rcs', 'snr'],
        widgets=['freq', 'energy', 'noise_temp', 'r', 'radius', 'rcs']
    )

# Example 2.2.6
def ex_2_2_6():
    
    test_tgt = rd.Target()
    test_tgt.rcs = 5
    r = 6E3
    az = math.pi/2 - math.pi/4
    test_tgt.pos = np.array([r*math.cos(az), r*math.sin(az)])
    
    rby.robby(targets=[test_tgt], reset=False, widgets=['freq', 'energy', 'noise_temp', 'radius'])
    
#-------Lab 3.1: Radar Transmissions and Receptions-------

# Example 3.1.1
def ex_3_1_1():
    ts.sine_pulse(
        freq=1,
        prf=0.1,
        widgets=['energy', 'freq', 'pulsewidth']
    )

# Example 3.1.2
def ex_3_1_2():
    ts.sine_pulse(
        show_duty=True,
        show_pri=True,
        widgets=['energy', 'freq', 'prf', 'pulsewidth']
    )
    
# Example 3.1.3
def ex_3_1_3():
    ts.lfm()
    
# Example 3.1.4
def ex_3_1_4():
    ts.dish_pat()

# Example 3.1.5
def ex_3_1_5():
    ts.array()
    
# Example 3.1.6
def ex_3_1_6():
    ts.delay_steer()
    
# Example 3.1.7
def ex_3_1_7():
    ts.phase_steer()
    
# Example 3.1.8
def ex_3_1_8():
    ts.pol()
    
# Example 3.1.9
def ex_3_1_9():
    ts.matched_filter(
        start_freq=1, 
        stop_freq=1, 
        widgets=['delay', 'pulsewidth']
    )
    
# Example 3.1.10
def ex_3_1_10():
    ts.range_res(
        start_freq=1,
        stop_freq=1,
        widgets=['range', 'pulsewidth']
    )  
    
# Example 3.1.11
def ex_3_1_11():
    ts.matched_filter()
    
# Example 3.1.12
def ex_3_1_12():
    ts.range_res()
    
# Example 3.1.13
def ex_3_1_13():
    test_tgt = rd.Target()
    test_tgt.rcs = 5
    r = 6E3
    az = math.pi/2 - math.pi/4
    test_tgt.pos = np.array([r*math.cos(az), r*math.sin(az)])
    
    rby.robby(targets=[test_tgt], reset=False, widgets=['bandw', 'coherent', 'freq', 'energy', 'noise_temp', 'num_integ', 'radius'])
    
#-------Lab 3.2: Detection-------

# Example 3.2.1
def ex_3_2_1():
    ts.radar_range_det()
    
# Example 3.2.2
def ex_3_2_2():
    ts.radar_range_det(highlight=False)
    
# Example 3.2.3
def ex_3_2_3():
    ts.detect_game()
    
# Example 3.2.4
def ex_3_2_4():
    ts.threshold()
    
# Example 3.2.5
def ex_3_2_5():
    ts.roc()
    
# Example 3.2.6
def ex_3_2_6():
    test_tgt = rd.Target()
    test_tgt.rcs = 5
    r = 6E3
    az = math.pi/2 - math.pi/4
    test_tgt.pos = np.array([r*math.cos(az), r*math.sin(az)])
    
    rby.robby(
        targets=[test_tgt], 
        dets=True, 
        reset=False, 
        widgets=['bandw', 'coherent', 'det_thresh', 'freq', 'energy', 'noise_temp', 'num_integ', 'radius']
    )
    
#-------Lab 4.1: Target Parameter Estimation-------

# Example 4.1.1
def ex_4_1_1():
    ts.dish_pat()

# Example 4.1.2
def ex_4_1_2():

    test_tgt = rd.Target()
    test_tgt.rcs = 10
    r = 5E3
    az = math.pi/2 - math.pi/4
    test_tgt.pos = np.array([r*math.cos(az), r*math.sin(az)])
    
    rby.robby(
        targets=[test_tgt], 
        energy=0.8,
        freq=4E3,
        radius=0.8,
        reset=False, 
        widgets=[]
    )

# Example 4.1.3
def ex_4_1_3():
    ts.cross_range()

# Example 4.1.4
def ex_4_1_4():
    ts.doppler()

# Example 4.1.5
def ex_4_1_5():
    ts.cw()

# Example 4.1.6
def ex_4_1_6():
    ts.cw(
        freq=2E3,
        dr=55,
        integ_time=15,
        targ_line=False,
        widgets=[]
    )

# Example 4.1.7
def ex_4_1_7():
    ts.rdi()
    
#-------Lab 4.2: Target Tracking-------

# Example 4.2.1
def ex_4_2_1():
    
    # Test route
    test_route = air.Route()
    test_route.start = np.array([-10E3, 0])
    test_route.end = np.array([0, 10E3])
    test_route.lifetime = 100
    test_route.vel = (test_route.end - test_route.start)/test_route.lifetime
    test_route.speed = np.sqrt(test_route.vel[0]**2 + test_route.vel[1]**2)
    test_route.max_range = 10E3
    
    # Plot and return
    rby.robby(
        targets=air.to_target([test_route]),
        radius=1.0,
        freq=5E3,
        reset=True, 
        dets=True,
        scan_rate=8,
        bandw=2,
        widgets=['det_thresh']
    )

# Example 4.2.2
def ex_4_2_2():
    ts.propagation()

# Example 4.2.3
def ex_4_2_3():
    ts.gnn()

# Example 4.2.4
def ex_4_2_4():
    ts.ekf()

# Example 4.2.5
def ex_4_2_5():
    
    # Test route
    routes = air.routes(6, 10E3, 200)
    targets = air.to_target(routes)
    
    # Plot and return
    rby.robby(
        targets=targets,
        radius=1.0,
        freq=5E3,
        reset=True, 
        dets=True,
        pulses=True,
        scan_rate=8,
        bandw=2
    )
    
#-------Lab 5.1: Radar Design Revisited-------

# Example 5.1.1
def ex_5_1_1():
    routes = air.routes(6, 10E3, 150)
    air.plot_routes(routes)
    targets = air.to_target(routes)
    
    return routes, targets

# Example 5.1.2
def ex_5_1_2():
    ts.design(
        bandw=0.5,
        bandw_lim=[0.1, 3],
        energy=1,
        energy_lim=[1, 10],
        freq=500,
        freq_lim=[100, 3000],
        max_price=50E3,
        noise_temp=1000,
        noise_temp_lim=[500, 1200],
        num_integ=1,
        num_integ_lim=[1, 10],
        r=1,
        r_lim=[1, 20],
        radius=0.1,
        radius_lim=[0.1, 0.5],
        rcs=10,
        rcs_lim=[-10, 25],
        scan_rate=1,
        scan_rate_lim=[1, 10],
        metrics=['price'],
        widgets=['bandw', 'coherent', 'freq', 'energy', 'noise_temp', 'num_integ', 'radius', 'scan_rate']
        )

# Example 5.1.3
def ex_5_1_2b():
    ts.design(
        bandw=0.5,
        bandw_lim=[0.1, 3],
        energy=1,
        energy_lim=[1, 10],
        freq=500,
        freq_lim=[100, 3000],
        max_price=50E3,
        min_snr=5,
        noise_temp=1000,
        noise_temp_lim=[500, 1200],
        num_integ=1,
        num_integ_lim=[1, 10],
        r=1,
        r_lim=[1, 20],
        radius=0.1,
        radius_lim=[0.1, 0.5],
        rcs=10,
        rcs_lim=[-10, 25],
        scan_rate=1,
        scan_rate_lim=[1, 10],
        metrics=['price', 'snr'],
        widgets=['bandw', 'coherent', 'freq', 'energy', 'min_snr', 'noise_temp', 'num_integ', 'r', 'radius', 'rcs', 'scan_rate']
        )

# Example 5.1.4
def ex_5_1_3a():
    
    # Test route
    test_route = air.Route()
    test_route.start = np.array([-10E3, 0])
    test_route.end = np.array([0, 10E3])
    test_route.lifetime = 100
    test_route.vel = (test_route.end - test_route.start)/test_route.lifetime
    test_route.speed = np.sqrt(test_route.vel[0]**2 + test_route.vel[1]**2)
    test_route.max_range = 10E3
    
    # Plot and return
    air.plot_routes([test_route])
    return air.to_target([test_route])

# Example 5.1.5
def ex_5_1_3b(test_tgt):
    rby.robby(targets=test_tgt, max_price=50E3, reset=True, show_price=True)

# Example 5.1.6
def ex_5_1_4a(routes, targets):
    air.plot_routes(routes)

# Example 5.1.7
def ex_5_1_4b(routes, targets):
    rby.robby(targets=targets, max_price=50E3, reset=True, show_price=True)