"""Dimensions, masses and closed-form kinematics of the 4-wheel nutating-caster vehicle.

Units: m, kg, s, rad.  Module frame: origin at the gimbal pivot P, z up, the outer
gimbal axis along +x.  Crank angle phi = 0 puts the crank pin (motor rear) on +x.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import asin, atan2, cos, radians, sin, sqrt


@dataclass
class ModuleGeometry:
    tilt_deg: float = 45.0          # motor axis tilt from vertical
    wheel_R: float = 0.055          # tire centre-line (O-ring) radius
    tire_r: float = 0.005           # O-ring cross-section radius (10 mm cord)
    wheel_offset: float = 0.100     # pivot -> wheel plane along -axis
    rear_length: float = 0.120      # pivot -> crank pin (spherical bearing centre)
    # Structure (see BOM): all values are clearances-driven, not optimised.
    motor_radius: float = 0.0185    # 37 mm gear motor
    motor_a0: float = 0.015         # motor body starts here along +axis
    motor_a1: float = 0.095         # motor body ends here
    tube_radius: float = 0.011      # stationary tube around the 8 mm output shaft
    tube_a0: float = -0.088         # tube end near the wheel
    stub_radius: float = 0.005      # hollow 10 mm stub through the igubal bearing
    stub_a1: float = 0.140          # cable exits here
    ring_half_x: float = 0.042      # outer gimbal ring half length along x
    ring_half_y: float = 0.030      # outer gimbal ring half length along y
    ring_bar: float = 0.004         # half thickness of ring bars (8 mm)
    bearing_x: float = 0.060        # gimbal-x pillow block centre
    post_x: float = 0.135           # vertical posts of the module frame
    arm_z: float = 0.0              # height of the horizontal arms (pivot level)
    crank_arm_dz: float = 0.035     # crank arm runs this much above the pin height
    plate_dz: float = 0.070         # module top plate bottom above the pin height
    cable_radius: float = 0.0025    # 5 mm motor+encoder cable
    cable_bend_R: float = 0.030     # routed bend radius (6 x d)
    cable_free_len: float = 0.30    # free vertical length above the plate (spreads torsion)
    cable_route: str = "over"       # legacy 'pin' layout only: 'over' or 'diag'
    # Cable layout. 'gimbal' (default): motor -> hollow inner trunnion (+y) -> U loop across
    # the inner axis -> channel in the ring -> hollow outer trunnion (+x) -> U loop across the
    # outer axis -> frame.  'pin' (previous): through the hollow crank-pin stub and shaft.
    cable_layout: str = "hook"      # 'hook' (v4), 'gimbal' (v2 hollow), 'clamp' (v3 trial), 'pin' (v1), 'axis' (v5 candidate)
    # v4 'hook': no hollow parts.  Same two U loops as v2, but the cable reaches the loop
    # start on each axis outside its bearing along an external bracket: a hook on the motor
    # housing over the ring bar (inner axis), a bridge on the ring under the pillow block
    # (outer axis).  Trunnions are solid.
    # v3 'clamp' (trial, not adopted - see REPORT.md): solid trunnions.  Each free span starts at a clamp close to its hinge axis
    # with the cable leaving PERPENDICULAR to the axis (so the +-45 deg swing bends the cable
    # instead of twisting it) and forms an arch (R = arch_R) to the next body:
    #   A: clamp on the housing beside the +y trunnion, up along the motor, arch over the ring
    #      bar to a plate on the ring.
    #   B: clamp on the ring bar beside the +x trunnion, arch beside the pillow block (in the
    #      horizontal plane) to a clamp on the frame arm.
    arch_R: float = 0.030           # span A loop radius
    loop_R_B3: float = 0.025        # span B loop radius (room between motor and wheel)
    ring_half_y_clamp: float = 0.034  # wider ring: 11.5 mm slot between motor and ring bar
    pillow_hz: float = 0.008        # pillow block half height (16 mm block)
    loop_R: float = 0.045           # U-loop bend radius as built (9 x d)
    loop_leg: float = 0.010         # straight legs of the inner (A) loop (short: less closing)
    loop_leg_B: float = 0.020       # straight legs of the outer (B) loop
    loop_pitch: float = 0.012       # axial offset between the two legs (> cable diameter)
    arm_y: float = -0.0325          # frame arms/posts moved off the x axis (v4; -0.0225 in v2)
    trunnion_y_end: float = 0.052   # start of loop A on the inner axis (hook end; 0.048 in v2)
    trunnion_x_end: float = 0.110   # start of loop B on the outer axis (bridge end / v2 trunnion end)
    # v5 'axis' (candidate, single free span, no brackets, no loops): the cable leaves a
    # clamp on the INNER side of the motor body (facing the crank axis) at `exit_a` along
    # the axis, runs under the crank arm to the crank axis, enters the hollow hub/crank
    # shaft from below and ends at a clamp on the frame above the shaft.  The pin stub and
    # the trunnions stay solid; only the crank shaft (with the hub) is hollow.
    exit_a: float = 0.060           # housing clamp position along the motor axis
    exit_clear: float = 0.004       # clamp stand-off from the motor surface
    top_free_len: float = 0.10      # free length from the shaft top to the frame clamp
    # 'hang' (candidate, simplest wiring): one slack cable from the stub end straight to a
    # clamp under the top plate, off the crank axis.  Nothing hollow, no brackets.
    hang_clamp: tuple = (0.0, 0.150, 0.145)   # frame clamp (mount frame, z above the pivot)
    hang_len: float = 0.27                    # free cable length
    exit_angle_deg: float = 0.0     # 'side_y': cable leaves at this angle from +y towards the motor axis (90 = along the axis)
    hang_exit: str = "stub"                   # 'stub' (stub end), 'side' (motor face that is up/inner at phi=0), 'side_y' (+y face, cable along +y), 'side_y_ax' (+y face, cable along the motor axis)

    # masses
    m_ring: float = 0.030
    m_housing: float = 0.26
    m_wheel: float = 0.10
    m_crank: float = 0.08
    m_frame: float = 0.45           # frame, pillow blocks, steer motor, belt, top bearing

    @property
    def theta(self) -> float:
        return radians(self.tilt_deg)

    @property
    def s(self) -> float:
        return sin(self.theta)

    @property
    def c(self) -> float:
        return cos(self.theta)

    @property
    def pivot_height(self) -> float:
        """Floor -> pivot so that the lowest tire point touches z = 0."""
        return self.wheel_offset*self.c + self.wheel_R*self.s + self.tire_r

    @property
    def rho(self) -> float:
        """Signed horizontal offset of the contact point along e (axis projection)."""
        return self.wheel_R*self.c - self.wheel_offset*self.s

    @property
    def r_eff(self) -> float:
        """|a x (C-P)|: contact speed per unit wheel spin about the axis."""
        return self.wheel_R + self.tire_r*self.s

    @property
    def crank_radius(self) -> float:
        return self.rear_length*self.s

    @property
    def crank_height(self) -> float:
        return self.rear_length*self.c

    @property
    def wheel_top_z(self) -> float:
        """Highest tire point relative to the pivot (must stay below the frame arms)."""
        return -self.wheel_offset*self.c + self.wheel_R*self.s + self.tire_r


@dataclass
class VehicleGeometry:
    module: ModuleGeometry = field(default_factory=ModuleGeometry)
    half_length: float = 0.21       # module pivots at (+-half_length, +-half_width)
    half_width: float = 0.17
    plate_half: tuple = (0.30, 0.25, 0.0025)
    m_chassis: float = 2.4          # plate (lightened) + battery + electronics
    rocker: bool = False            # optional front axle beam pivoting about body x
    rocker_limit_deg: float = 6.0

    def module_positions(self):
        L, W = self.half_length, self.half_width
        # FL, FR, RL, RR (body x forward, y left)
        return [(L, W), (L, -W), (-L, W), (-L, -W)]


# --------------------------------------------------------------------------- kinematics
def gimbal_state(g: ModuleGeometry, phi: float):
    """Ideal passive gimbal angles and their derivatives d/dphi."""
    s, c = g.s, g.c
    gx = -atan2(s*sin(phi), c)
    gy = asin(s*cos(phi))
    dgx = -s*c*cos(phi)/(c*c + s*s*sin(phi)**2)
    dgy = -s*sin(phi)/sqrt(max(1e-12, 1 - s*s*cos(phi)**2))
    return gx, gy, dgx, dgy


def contact_coefficients(g: ModuleGeometry, phi: float):
    """Material velocity of the tire at the contact, chassis fixed:

        v = (J_phi * dphi + J_psi * dpsi) * t,   t = (-sin phi, cos phi)

    psi is the wheel angle relative to the motor housing (the drive joint).
    The Cardan gimbal makes the housing's spin about its own axis phase dependent,
    so J_phi oscillates around rho - r_eff.
    """
    s, c = g.s, g.c
    _, _, dgx, _ = gimbal_state(g, phi)
    sigma = dgx*s*cos(phi)                  # housing spin about its axis per unit dphi
    cz = -g.pivot_height
    j_phi = s*(s*g.rho + c*cz) + sigma*g.r_eff
    return j_phi, g.r_eff


def summary(g: ModuleGeometry) -> dict:
    return {
        "tilt_deg": g.tilt_deg, "pivot_height_m": g.pivot_height, "rho_m": g.rho,
        "r_eff_m": g.r_eff, "crank_radius_m": g.crank_radius,
        "crank_height_m": g.crank_height, "wheel_top_rel_pivot_m": g.wheel_top_z,
    }


if __name__ == "__main__":
    import json
    g = ModuleGeometry()
    print(json.dumps(summary(g), indent=2))
    for deg in (0, 45, 90, 135, 180):
        print(deg, contact_coefficients(g, radians(deg)))
