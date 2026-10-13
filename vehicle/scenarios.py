"""Reference trajectories in the world frame: pose(t) = (x, y, yaw).

All references are smooth (minimum-jerk segments or ramped circles) and known in
advance, as in the 'model control' concept: the controller derives steering and wheel
commands from them and only adds small feedback corrections.
"""
from __future__ import annotations

import math
import numpy as np


def minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u**3*(10 - 15*u + 6*u*u)


class Piecewise:
    """Sequence of (duration, target pose) min-jerk moves; pose = (x, y, yaw)."""

    def __init__(self, start, moves, lead=1.0):
        self.knots = [(0.0, np.array(start, float))]
        t = lead
        self.knots.append((t, np.array(start, float)))
        for dur, target in moves:
            t += dur
            self.knots.append((t, np.array(target, float)))
        self.T = t

    def pose(self, t):
        k = self.knots
        if t <= k[0][0]:
            return k[0][1].copy()
        for (t0, p0), (t1, p1) in zip(k[:-1], k[1:]):
            if t <= t1:
                if t1 - t0 < 1e-9:
                    return p1.copy()
                return p0 + (p1 - p0)*minjerk((t - t0)/(t1 - t0))
        return k[-1][1].copy()


class Circle:
    """Circle of radius R about `center`; angular position follows a ramped rate.
    yaw(t) = yaw0 + yaw_ratio * (travelled angle)."""

    def __init__(self, R=0.4, laps=3, period=8.0, ramp=1.5, lead=1.0, yaw_ratio=0.0,
                 center=(0.0, 0.4)):
        self.R, self.laps, self.period, self.ramp, self.lead = R, laps, period, ramp, lead
        self.w = 2*math.pi/period
        self.yaw_ratio = yaw_ratio
        self.center = np.array(center)
        total = 2*math.pi*laps
        # travel = w*(T_c - ramp) for a symmetric linear ramp
        self.Tc = total/self.w + ramp
        self.T = lead + self.Tc

    def angle(self, t):
        t = t - self.lead
        w, r, Tc = self.w, self.ramp, self.Tc
        if t <= 0:
            return 0.0
        if t < r:
            return w*t*t/(2*r)
        if t <= Tc - r:
            return w*(t - r/2)
        if t < Tc:
            return 2*math.pi*self.laps - w*(Tc - t)**2/(2*r)
        return 2*math.pi*self.laps

    def pose(self, t):
        a = self.angle(t) - math.pi/2       # start at the bottom of the circle -> +x motion
        return np.array([self.center[0] + self.R*math.cos(a),
                         self.center[1] + self.R*math.sin(a),
                         self.yaw_ratio*(a + math.pi/2)])


class Reference:
    def __init__(self, path, name, flip=True, tail=1.5, notes=""):
        self.path, self.name, self.flip, self.notes = path, name, flip, notes
        self.T = path.T + tail

    def pose(self, t):
        return self.path.pose(t)

    def vel(self, t, h=1e-3):
        return (self.path.pose(t + h) - self.path.pose(t - h))/(2*h)

    def acc(self, t, h=2e-3):
        return (self.path.pose(t + h) - 2*self.path.pose(t) + self.path.pose(t - h))/(h*h)


def square(side=0.6, laps=2, move=2.6, pause=0.9):
    moves, x, y = [], 0.0, 0.0
    for _ in range(laps):
        for dx, dy in ((side, 0), (0, side), (-side, 0), (0, -side)):
            x, y = x + dx, y + dy
            moves.append((move, (x, y, 0.0)))
            moves.append((pause, (x, y, 0.0)))
    return Reference(Piecewise((0, 0, 0), moves), "square", flip=False,
                     notes="yaw held at 0; every corner steers +90 deg, no wheel reversal")


def star(dist=0.35, move=1.8, pause=0.8):
    moves = []
    for k in range(8):
        a = k*math.pi/4
        moves.append((move, (dist*math.cos(a), dist*math.sin(a), 0.0)))
        moves.append((pause, (dist*math.cos(a), dist*math.sin(a), 0.0)))
        moves.append((move, (0.0, 0.0, 0.0)))
        moves.append((pause, (0.0, 0.0, 0.0)))
    return Reference(Piecewise((0, 0, 0), moves), "star", flip=True,
                     notes="8 directions out and back, yaw held; wheel reversal allowed")


def spin(turns=1.0, dur=6.0):
    return Reference(Piecewise((0, 0, 0), [(dur, (0, 0, 2*math.pi*turns))]), "spin",
                     flip=True, notes="turn in place")


def circle(laps=3, period=8.0):
    return Reference(Circle(R=0.4, laps=laps, period=period), "circle", flip=False,
                     notes="translate around a circle with yaw held: steering turns once per lap")


def circle_spin(laps=2, period=10.0):
    return Reference(Circle(R=0.4, laps=laps, period=period, yaw_ratio=-1.0), "circle_spin",
                     flip=False, notes="circle while the body yaws -360 deg per lap")


def straight(dist=1.4, dur=6.0):
    return Reference(Piecewise((0, 0, 0), [(dur, (dist, 0, 0))]), "straight", flip=True,
                     notes="straight line; used with the bump floor")


SCENARIOS = {"square": square, "star": star, "spin": spin, "circle": circle,
             "circle_spin": circle_spin, "straight": straight}
