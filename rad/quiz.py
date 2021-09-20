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

from IPython.display import display
import ipywidgets as wdg
import math
import numpy as np
from rad.const import k, c
import rad.radar as rd

# jupyter_intro

# Simple quiz with scalar value
def new_quiz(prompt, val, abs_tol=None, rel_tol=1E-2):
    """
    Generate quiz answer box.
    
    Inputs:
    - prompt [str]: Question prompt
    - val [float]: True answer
    - abs_tol [float]: Absolute tolerance; if None, uses rel_tol
    - rel_tol [float]: Relative tolerance
    
    Outputs:
    (none)
    """
    title = wdg.HTML(value = f"<p><font color='black'>{prompt}</p>")
    answer = wdg.FloatText()
    submit = wdg.Button(description="Submit")
    result = wdg.HTML(value = f"<b><font color='black'>Ready</b>")
    ansbox = wdg.VBox([wdg.HBox([title, answer, result]), submit])
    display(ansbox)

    def check_ans(b):
        if abs_tol and (abs(answer.value - val) < abs_tol):
            result.value = f"<b><font color='green'>Correct!</b>"
        elif rel_tol and ((abs(answer.value - val)/abs(val)) < rel_tol):
            result.value = f"<b><font color='green'>Correct!</b>"
        else:
            result.value = f"<b><font color='red'>Incorrect.</b>"
        
    submit.on_click(check_ans)

#-------Lab 1.1: Introduction to Labs-------
    
# Q1.1.1
def quiz_1_1_1():
    prompt = 'Enter answer:'
    val = 1.0
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)

# Q1.1.2
def quiz_1_1_2():
    prompt = 'Enter answer:'
    val = (math.log10(2.72**3) + math.cos(4*math.pi/7))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.1.3
def quiz_1_1_3():
    prompt = 'Enter answer:'
    val = 7.76E4/5.1E-3
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.1.4a
def quiz_1_1_4a():
    prompt = 'Enter answer (in dB):'
    val = rd.to_db(1.5E5)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.1.4b
def quiz_1_1_4b():
    prompt = 'Enter answer (in dB):'
    val = rd.to_db(7.2E-7)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 

# Q1.1.5a
def quiz_1_1_5a():
    prompt = 'Enter answer:'
    val = rd.from_db(51.2)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.1.5b
def quiz_1_1_5b():
    prompt = 'Enter answer:'
    val = rd.from_db(-20.1)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 
    
# Q1.1.6
def quiz_1_1_6():
    prompt = 'Enter answer (in dB):'
    val = rd.to_db(3.7**5)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.1.7a
def quiz_1_1_7a():
    prompt = 'Enter answer:'
    val = 3
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.1.7b
def quiz_1_1_7b():
    prompt = 'Enter answer (in s, within ±0.05 s):'
    val = 0.5
    tol = 0.05
    new_quiz(prompt, val, abs_tol=tol)
    
# Q1.1.7c
def quiz_1_1_7c():
    prompt = 'Enter answer (in Hz, within ±0.2 Hz):'
    val = 2
    tol = 0.2
    new_quiz(prompt, val, abs_tol=tol) 
    
#-------Lab 1.2: Introduction to Radar-------
    
# Q1.2.1a
def quiz_1_2_1a():
    prompt = 'Enter wavelength (in m):'
    val = rd.wavelen(1000, propvel=1000)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.1b
def quiz_1_2_1b():
    prompt = 'Enter wavelength (in m):'
    val = rd.wavelen(1000, propvel=2000)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)    

# Q1.2.1c
def quiz_1_2_1c():
    prompt = 'Enter wavelength (in m):'
    val = rd.wavelen(500, propvel=1000)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)    
    
# Q1.2.2a
def quiz_1_2_2a():
    prompt = 'Enter wavelength (in m):'
    val = rd.wavelen(500E6, propvel=3E8)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.2b
def quiz_1_2_2b():
    prompt = 'Enter frequency (in Hz):'
    val = 1500/5
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.3a
def quiz_1_2_3a():
    prompt = 'Enter time (in ms):'
    val = math.sqrt(75**2 + 80**2)
    tol = 3
    new_quiz(prompt, val, abs_tol=tol)
    
# Q1.2.3b
def quiz_1_2_3b():
    prompt = 'Enter time (in ms):'
    val = 2*math.sqrt(75**2 + 80**2)
    tol = 3
    new_quiz(prompt, val, abs_tol=tol)
    
# Q1.2.4a
def quiz_1_2_4a():
    prompt = 'Enter time (in s):'
    val = 2*1000E3/3E8
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)

# Q1.2.4b
def quiz_1_2_4b():
    prompt = 'Enter range (in m):'
    val = 3E8*0.0057/2
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.5
def quiz_1_2_5():
    prompt = 'Enter range (in m):'
    val = math.sqrt(75**2 + 100**2)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.6a
def quiz_1_2_6a():
    prompt = 'Enter range (in m):'
    val = math.sqrt(50**2 + 50**2)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.6b
def quiz_1_2_6b():
    prompt = 'Enter azimuth (in deg, within +/- 3 deg):'
    val = 90 - rd.rad2deg(math.atan2(50, 50))
    tol = 3
    new_quiz(prompt, val, abs_tol=tol)
    
# Q1.2.7a
def quiz_1_2_7a():
    prompt = 'Enter range (in m):'
    val = math.sqrt(40**2 + 20**2)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.7b
def quiz_1_2_7b():
    prompt = 'Enter azimuth (in deg, within +/- 3 deg):'
    val = 90 - rd.rad2deg(math.atan2(-20, 40))
    tol = 3
    new_quiz(prompt, val, abs_tol=tol)
    
# Q1.2.8a
def quiz_1_2_8a():
    prompt = 'Enter beamwidth (in deg):'
    val = 70*0.12/3.8
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.8b
def quiz_1_2_8b():
    prompt = 'Enter diameter (in m):'
    val = 70*rd.wavelen(10E9, propvel=3E8)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.9
def quiz_1_2_9():
    prompt = 'Enter transmit gain (in dB):'
    val = rd.to_db(4*math.pi*2.1*3.3/rd.wavelen(15E3, propvel=2E3)**2)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.10a
def quiz_1_2_10a():
    prompt = 'Enter wavelength (in m):'
    val = rd.wavelen(5E9)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.10b
def quiz_1_2_10b():
    prompt = 'Enter beamwidth (in deg):'
    val = 70*rd.wavelen(5E9)/4.4
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q1.2.10c
def quiz_1_2_10c():
    prompt = 'Enter transmit gain (in dB):'
    val = rd.to_db(4*math.pi*(math.pi*2.2**2)/rd.wavelen(5E9)**2)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)

#-------Lab 2.1: Radar Range Equation-------
    
# Q2.1.1a
def quiz_2_1_1a():
    prompt = 'Enter SNR (in dB):'
    val = rd.to_db(100/5.3)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q2.1.1b
def quiz_2_1_1b():
    prompt = 'Enter signal energy (in J):'
    val = rd.from_db(15)*2.2
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)  

# Q2.1.2a
def quiz_2_1_2a():
    prompt = 'Enter received power (in W):'
    val = rd.friis(50E3, 10E3, area=10, gain=rd.from_db(35.0))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 

# Q2.1.2b
def quiz_2_1_2b():
    prompt = 'Enter dish radius (in m):'
    val = np.sqrt(rd.gain2area((1/100E3/5)*4*np.pi*(10E3)**2, 3.5E9)/np.pi)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 
    
# Q2.1.3a
def quiz_2_1_3a():
    prompt = 'Enter incident power (in W):'
    val = rd.friis(50E3, 150E3, area=rd.from_db(5), gain=rd.from_db(30))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 
    
# Q2.1.3b
def quiz_2_1_3b():
    prompt = 'Enter range (in m):'
    val = math.sqrt(500E3*1*rd.from_db(25)/4/math.pi/0.1E-3)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 
    
# Q2.1.4a
def quiz_2_1_4a():
    prompt = 'Enter received power (in W):'
    val = rd.rx_power(100E3, 600E3, 3E9, rcs=rd.from_db(5), gain=rd.from_db(80))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 
    
# Q2.1.4b
def quiz_2_1_4b():
    prompt = 'Enter effective aperture area (in m^2):'
    val = rd.gain2area(rd.from_db(24.3), 1.3E9)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 
    
# Q2.1.4c
def quiz_2_1_4c():
    prompt = 'Enter RCS (in dBsm):'
    wlen = rd.wavelen(1.5E9)
    val = rd.to_db(10E-12*(10E3**4)*(4*np.pi)**3/30/rd.from_db(80)/wlen**2)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 
    
# Q2.1.5
def quiz_2_1_5():
    prompt = 'Enter noise energy (in J):'
    val = k*722
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 

# Q2.1.6
def quiz_2_1_6():
    prompt = 'Enter SNR (in dB):'
    val = rd.to_db(rd.rx_snr(200E3, 50, 5E9, 750, rcs=rd.from_db(-5), gain=rd.from_db(70)))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 
    
# Q2.1.7
def quiz_2_1_7():
    prompt = 'Enter SNR (in dB):'
    val = rd.to_db(rd.snrconv(rd.from_db(17), 75E3, rd.from_db(-10), 50E3, rd.from_db(-15)))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol) 

#-------Lab 2.2: Basic Radar Design-------
    
#-------Lab 3.1: Radar Transmissions and Receptions-------
    
# Q3.1.1
def quiz_3_1_1():
    prompt = 'Enter pulsewidth (in µs):'
    val = 7.2
    tol = 0.1
    new_quiz(prompt, val, abs_tol=tol)
    
# Q3.1.2a
def quiz_3_1_2a():
    prompt = 'Enter PRI (in s):'
    val = 1/10E3
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q3.1.2b
def quiz_3_1_2b():
    prompt = 'Enter duty cycle:'
    val = 150E-6*1E3
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q3.1.3a
def quiz_3_1_3a():
    prompt = 'Enter delay (in ns):'
    val = -2.83
    tol = 0.01
    new_quiz(prompt, val, abs_tol=tol)
    
# Q3.1.3b
def quiz_3_1_3b():
    prompt = 'Enter delay (in ns):'
    val = -2.83
    tol = 0.01
    new_quiz(prompt, val, abs_tol=tol)
    
# Q3.1.4a
def quiz_3_1_4a():
    prompt = 'Enter phase (in deg):'
    val = 168
    tol = 1
    new_quiz(prompt, val, abs_tol=tol)
    
# Q3.1.4b
def quiz_3_1_4b():
    prompt = 'Enter phase (in deg):'
    val = 120
    tol = 1
    new_quiz(prompt, val, abs_tol=tol)
    
# Q3.1.5
def quiz_3_1_5():
    prompt = 'Enter bandwidth (in Hz):'
    val = c/2/10
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
#-------Lab 3.2: Detection-------
    
# Q3.2.1a
def quiz_3_2_1a():
    prompt = 'Enter prob. of detection:'
    val = 0.69
    tol = 0.02
    new_quiz(prompt, val, abs_tol=tol)
    
# Q3.2.1b
def quiz_3_2_1b():
    prompt = 'Enter prob. of false alarm:'
    val = 0.17
    tol = 0.02
    new_quiz(prompt, val, abs_tol=tol)
    
#-------Lab 4.1: Target Parameter Estimation-------
    
# Q4.1.1a
def quiz_4_1_1a():
    prompt = 'Enter range (in m):'
    val = c*0.00333/2
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.1b
def quiz_4_1_1b():
    prompt = 'Enter range resolution (in m):'
    val = rd.range_res(30E6)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.1c
def quiz_4_1_1c():
    prompt = 'Enter range accuracy (in m):'
    val = rd.range_res(30E6)/np.sqrt(rd.from_db(11))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.2a
def quiz_4_1_2a():
    prompt = 'Enter angle accuracy (in deg):'
    val = rd.dish_beamw(2, 3.5E9)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.2b
def quiz_4_1_2b():
    prompt = 'Enter angle accuracy (in deg):'
    val = rd.dish_beamw(2, 3.5E9)/np.sqrt(rd.from_db(14))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.3
def quiz_4_1_3():
    prompt = 'Enter cross-range resoluation (in m):'
    val = 50E3*rd.wavelen(3.5E9)/2.1
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.4a
def quiz_4_1_4a():
    prompt = 'Enter Doppler shift (in Hz):'
    val = rd.dopp_shift(500, 1.5E9)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.4b
def quiz_4_1_4b():
    prompt = 'Enter range rate (in m/s):'
    val = -c*10E3/2/3.0E9
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.5
def quiz_4_1_5():
    prompt = 'Enter range rate (in m/s):'
    val = 55.0
    tol = 1.0
    new_quiz(prompt, val, abs_tol=tol)
    
# Q4.1.6a
def quiz_4_1_6a():
    prompt = 'Enter range rate resolution (in m/s):'
    val = c/2/5E9/50E-3
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.6b
def quiz_4_1_6b():
    prompt = 'Enter range rate (in m/s):'
    val = c*500/4/5E9
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.7a
def quiz_4_1_7a():
    prompt = 'Enter RCS (in dBsm):'
    val = rd.to_db(rd.from_db(0)*(rd.from_db(13)/rd.from_db(15))*(65E3/50E3)**4)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q4.1.7b
def quiz_4_1_7b():
    prompt = 'Enter RCS (in dBsm):'
    val = rd.to_db(rd.from_db(5)*(rd.from_db(13)/rd.from_db(7))*(100E3/500E3)**4)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.1a
def quiz_5_1_1a():
    prompt = 'Enter wavelength (in m):'
    val = rd.wavelen(10E9)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.1b
def quiz_5_1_1b():
    prompt = 'Enter beamwidth (in deg):'
    val = 70*rd.wavelen(10E9)/4
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.1c
def quiz_5_1_1c():
    prompt = 'Enter transmit gain (in dB):'
    val = rd.to_db(rd.dish_gain(4, 10E9))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.2a
def quiz_5_1_2a():
    prompt = 'Enter received energy (in J):'
    val = rd.rx_energy(200E3, 10, 2E9, gain=rd.area2gain(10, 2E9)**2, rcs=rd.from_db(-5))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.2b
def quiz_5_1_2b():
    prompt = 'Enter SNR (in dB):'
    val = rd.to_db(rd.rx_energy(200E3, 10, 2E9, gain=rd.area2gain(10, 2E9)**2, rcs=rd.from_db(-5))) - \
        rd.to_db(k*500)
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.3a
def quiz_5_1_3a():
    prompt = 'Enter duty cycle:'
    val = 100E-6*2E3
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.3b
def quiz_5_1_3b():
    prompt = 'Enter number of pulses:'
    val = math.ceil(rd.from_db(12))
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.4a
def quiz_5_1_4a():
    prompt = 'Enter bandwidth (in Hz):'
    val = c/2/1.5
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.4b
def quiz_5_1_4b():
    prompt = 'Enter cross range resolution (in m):'
    beamw = 57.3*rd.wavelen(5E9)/5.2
    val = 90E3*beamw/57.3
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.4c
def quiz_5_1_4c():
    prompt = 'Enter range rate (in m/s):'
    val = c*700/4/3E9
    tol = 0.01
    new_quiz(prompt, val, rel_tol=tol)
    
# Q5.1.5
def quiz_5_1_5(targets):
    for ii in range(len(targets)): 
        prompt = f'Enter route number for target with RCS {targets[ii].rcs:.2f} dBsm:'
        val = targets[ii].route
        tol = 0.01
        new_quiz(prompt, val, abs_tol=tol)
