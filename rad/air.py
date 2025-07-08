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

from matplotlib.patches import Circle
import numpy as np
from numpy.random import permutation, rand, randint
from numpy.typing import ArrayLike, NDArray
import rad.plot as plt
import rad.radar as rd
from typing import List

N_COMP = 4
N_SIDE = 4

class Route():
    """
    Container class for air traffic route.
    """
    def __init__(
        self,
        start: ArrayLike,
        end: ArrayLike,
        lifetime: float,
        max_range: float,
        ):
        self.start: NDArray = np.asanyarray(start)
        self.end: NDArray = np.asanyarray(end)
        self.lifetime: float = lifetime
        self.max_range: float = max_range

    @property
    def speed(self) -> float:
        """Air speed."""
        vel_curr = self.vel
        return float(np.sqrt(vel_curr[0]**2 + vel_curr[1]**2))

    @property
    def vel(self) -> NDArray:
        """Velocity vector."""
        return (self.end - self.start)/self.lifetime

def plot_routes(
    routes: List[Route]
    ):
    """
    Plot air traffic routes.
    
    Inputs:
    - routes [List[Route]]: List of Routes
    
    Outputs:
    (none)
    """

    # Maximum range
    max_range = routes[0].max_range
    
    fig, ax = plt.new_plot()
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    
    radar_view = Circle((0, 0), max_range, color='black', linestyle='dashed', fill=False)
    ax.add_patch(radar_view)
    
    index = 0
    for rt in routes:
        line = ax.plot([rt.start[0], rt.end[0]], [rt.start[1], rt.end[1]])[0]
        rt_angle = np.arctan2(rt.vel[1], rt.vel[0])
        dv = rt_angle + np.pi/2
        dx = 5*(max_range/100)*np.cos(dv)
        dy = 5*(max_range/100)*np.sin(dv)
        alpha = np.random.uniform(low=0.4, high=0.6)
        ax.text(rt.start[0] + rt.vel[0]*rt.lifetime*alpha + dx, rt.start[1] + rt.vel[1]*rt.lifetime*alpha + dy, str(index), color=line.get_color())
        index += 1

def routes(n, max_range, avg_speed):
    """
    Generate random air traffic routes
    
    Inputs:
    - n [int]: Number of routes
    - max_range [float]: Maximum range of route (m)
    - avg_speed [float]: Average air speed (m/s)
    
    Outputs:
    - routes [List[Route]]: Output Routes
    """

    # Initailize output 
    out = []
    
    # Draw routes
    for _ in range(n):
        
        # Draw starting side
        start_side = randint(low=0, high=N_SIDE)
        
        # Draw offset for end side
        offset = randint(low=1, high=N_SIDE)
        
        # Ending side
        end_side = np.mod(start_side + offset, N_SIDE)
        
        # Draw start/end point
        start_alpha = 0.3 + 0.4*rand()
        end_alpha = 0.3 + 0.4*rand()
        
        # Build start point
        if (start_side == 0):
            start_x = start_alpha*2*max_range - max_range
            start_y = max_range
        elif (start_side == 1):
            start_x = max_range
            start_y = start_alpha*2*max_range - max_range
        elif (start_side == 2):
            start_x = start_alpha*2*max_range - max_range
            start_y = -max_range
        elif (start_side == 3):
            start_x = -max_range
            start_y = start_alpha*2*max_range - max_range
        else:
            start_x = 0
            start_y = 0
        
        # Build end point
        if (end_side == 0):
            end_x = end_alpha*2*max_range - max_range
            end_y = max_range
        elif (end_side == 1):
            end_x = max_range
            end_y = end_alpha*2*max_range - max_range
        elif (end_side == 2):
            end_x = end_alpha*2*max_range - max_range
            end_y = -max_range
        elif (end_side == 3):
            end_x = -max_range
            end_y = end_alpha*2*max_range - max_range
        else:
            end_x = 0
            end_y = 0
            
        # Make route
        lifetime = np.sqrt((start_x - end_x)**2 + (start_y - end_y)**2)/avg_speed
        rt = Route(
            start=np.array([start_x, start_y]), 
            end=np.array([end_x, end_y]), 
            lifetime=lifetime, 
            max_range=max_range
            )
   
        # Add to output
        out.append(rt)

    # Output
    return out

def to_target(routes, min_rcs=-5, max_rcs=5):
    """
    Convert Routes to Targets.
    
    Inputs:
    - routes [List[Route]]: Input Routes
    - min_rcs [float]: Minimum radar cross section (dBsm)
    - max_rcs [float]: Maximum radar cross section (dBsm)

    
    Outputs:
    - targets [List[Target]]: Output Targets
    """

    # Shuffled RCS values
    rcs = permutation(np.linspace(min_rcs, max_rcs, len(routes)))
    
    # Initialize output
    targets = []
    
    # Build targets
    index = 0
    for rt in routes:
        
        tgt = rd.Target()
        tgt.pos = np.array([rt.start[0], rt.start[1], rt.vel[0], rt.vel[1]])
        tgt.rcs = rcs[index]
        tgt.route = index
        index += 1
        
        targets.append(tgt)
        
    targets.sort(key=lambda x: x.rcs)
        
    # Output    
    return targets