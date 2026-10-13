"""Plot measured simulation traces with explicit units."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).parent / 'results'
air = np.genfromtxt(root/'air_10turns'/'trace.csv',delimiter=',',names=True)
ground = np.genfromtxt(root/'ground'/'trace.csv',delimiter=',',names=True)
half = np.genfromtxt(root/'ground_half_dt'/'trace.csv',delimiter=',',names=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axs = plt.subplots(2,2,figsize=(11,7),layout='constrained')
fig.suptitle('Torque-driven crank + passive two-axis gimbal | MuJoCo 3.3.7',fontsize=14)
ax=axs[0,0]
ax.plot(air['time_s'],air['crank_rad']/(2*np.pi),color='#6c4ab6')
ax.set(xlabel='Time [s]',ylabel='Crank revolutions [turn]',title='10 continuous steering revolutions')
ax.grid(alpha=.2)
ax=axs[0,1]
ax.plot(air['time_s'],air['gimbal_x_deg'],label='Gimbal X')
ax.plot(air['time_s'],air['gimbal_y_deg'],label='Gimbal Y')
ax.plot(air['time_s'],air['tilt_deg'],label='Axle tilt',color='#444444',linestyle='--')
ax.set(xlabel='Time [s]',ylabel='Angle [deg]',title='Passive joints remain bounded')
ax.legend(loc='lower right');ax.grid(alpha=.2)
ax=axs[1,0]
ax.plot(air['time_s'],air['closure_m']*1000,label='Air, 10 turns')
ax.plot(ground['time_s'],ground['closure_m']*1000,label='Ground')
ax.set(xlabel='Time [s]',ylabel='Rear pin separation [mm]',title='Soft equality constraint residual')
ax.legend();ax.grid(alpha=.2)
ax=axs[1,1]
ax.plot(ground['base_x_m']*1000,ground['base_y_m']*1000,label='Ground, dt = 1 ms')
ax.plot(half['base_x_m']*1000,half['base_y_m']*1000,'--',label='Ground, dt = 0.5 ms')
ax.scatter([0],[0],color='#333333',s=25,label='Start',zorder=5)
ax.set(xlabel='Carrier X [mm]',ylabel='Carrier Y [mm]',title='Steering sweep with wheel speed control')
ax.set_aspect('equal',adjustable='datalim');ax.legend();ax.grid(alpha=.2)
fig.savefig(root/'verification.png',dpi=160)
plt.close(fig)
print(root/'verification.png')
