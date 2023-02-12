import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import sys

np.seterr(all='raise')
# Finite Volume Method for Fluid Simulation
# Primitive Variables: Density (Mass), Velocity, Pressure
# Conservative Variables: Mass Density, Momentum Density, Energy Density

def get_conservative(rho: np.mat, vx: np.mat, vy: np.mat, p: np.mat, gamma: float, vol: float) -> np.mat:
    """
    Convert from the primitive variables to conservative variables of the system.
    :param rho: np.mat of cell densities
    :param vx: np.mat of cell x-velocities
    :param vy: np.mat of cell y-velocities
    :param p: np.mat of cell pressures
    :param gamma: float value of ideal gas gamma (Adiabatic Index)
    :param vol: float value of cell volume
    :return: Four np.mat of cell: masses, x-momenta, y-momenta, energies
    """
    mass = rho * vol
    momx = vx * mass
    momy = vy * mass
    energy = (p / (gamma - 1) + 0.5 * rho * (vx ** 2 + vy ** 2)) * vol  # Calorically perfect gas approximation

    return mass, momx, momy, energy


def get_primitive(mass: np.mat, momx: np.mat, momy: np.mat, energy: np.mat, gamma: float, vol: np.mat) -> np.mat:
    """
    Convert from the conservative variables to the primitive variables of the system.
    :param mass: np.mat of cell masss
    :param momx: np.mat of cell x-momentums
    :param momy: np.mat of cell y-momentums
    :param energy: np.mat of cell energies
    :param gamma: float value of ideal gas gamma)
    :param vol: float value of cell volumes
    :return: Four np.mat of cell: densities, x-velocities, y-velocities, pressures
    """

    rho = mass / vol
    vx = momx / rho / vol
    vy = momy / rho / vol
    p = (energy / vol - 0.5 * rho * (vx ** 2 + vy ** 2)) * (gamma - 1)

    return rho, vx, vy, p


def get_timestep(c_max: float, cell_width: float, vx: np.mat, vy: np.mat) -> float:
    """
    Using the Courant-Friedrichs-Lewy CFL condition for convergence of hyperbolic PDE's
    as a necessary condition for convergence, we can find a maximum timestep for our simulation
    :param c_max: A constant that is less than 1
    :param cell_width: float of the cell width
    :param vx: np.mat of cell x-velocities
    :param vy: np.mat of cell y-velocities
    :return: float, A timestep value for the simulation
    """
    return c_max * np.min(cell_width / (vx ** 2 + vy ** 2))

def get_timestep_2(c_max: float, cell_width: float, vx: np.mat, vy: np.mat, gamma: float, p: np.mat, rho: np.mat) -> float:
    """
    Using the Courant-Friedrichs-Lewy CFL condition for convergence of hyperbolic PDE's
    as a necessary condition for convergence, we can find a maximum timestep for our simulation
    :param c_max: A constant that is less than 1
    :param cell_width: float of the cell width
    :param vx: np.mat of cell x-velocities
    :param vy: np.mat of cell y-velocities
    :return: float, A timestep value for the simulation
    """

    return c_max * np.min(cell_width / (np.sqrt(gamma * p / rho) + np.sqrt(vx ** 2 + vy ** 2)))



def get_gradient(field: np.mat, step: float) -> np.mat:
    """
    Calculates the gradients of an arbitrary field using linear interpolation of discreet steps.
    :param field: np.mat of the values the field takes
    :param step: float of the size of the step
    :return: Two np.mat of the fields: x-derivatives, y-derivatives
    """

    # === MAY WANT TO ADD A SLOPE LIMITER HERE TO DESCRIBE HIGH RESOLUTION DISCONTINUITIES===

    # Directions for np.roll
    r = -1  # Right
    l = 1  # Left

    f_dx = (np.roll(field, r, axis=0) - np.roll(field, l, axis=0)) / (2 * step)
    f_dy = (np.roll(field, r, axis=1) - np.roll(field, l, axis=1)) / (2 * step)

    return f_dx, f_dy


def extrapolate_cell_center_to_face(f: np.mat, f_dx: np.mat, f_dy: np.mat, dx: float) -> np.mat:
    """
    Uses a first order linear extrapolation to find the values of a field at the edges of a cell,
    given the value of the field at the center of the cell.
    Note yl and yr and the "y left" and "y right" values. This can be thought of as
    bottom and top as well, this is just treating them as their own 1D array.
    :param f: np.mat of an arbitrary field
    :param f_dx: np.mat of the fields x-derivatives
    :param f_dy: np.mat of the fields y-derivatives
    :param dx: float of the width of the cells
    :return:
    """
    # Directions for np.roll
    r = -1  # Right
    l = 1  # Left

    f_xl = f - f_dx * dx / 2
    # Roll f_xl by moving each element one element to the left so that each f_xl
    # would "line up" with each f_xr.
    f_xl = np.roll(f_xl, r, axis=0)
    f_xr = f + f_dx * dx / 2

    f_yl = f - f_dy * dx / 2
    f_yl = np.roll(f_yl, r, axis=1)
    f_yr = f + f_dy * dx / 2

    return f_xl, f_xr, f_yl, f_yr


def get_fluxes(rho_l: np.mat, rho_r: np.mat, vx_l: np.mat, vx_r: np.mat,
             vy_l: np.mat, vy_r: np.mat, p_l: np.mat, p_r: np.mat, gamma: float) -> np.mat:
    """
    A Lax-Friedrichs / Rusanov method to calculate the flux between two states
    Note to self, need to research this in more detail to understand, especially
    stabilizing diffusivity term.
    :param rho_l:
    :param rho_r:
    :param vx_l:
    :param vx_r:
    :param vy_l:
    :param vy_r:
    :param p_l:
    :param p_r:
    :param gamma:
    :return:
    """
    # Calculating Energies of each cell
    energy_l = p_l / (gamma - 1) + 0.5 * rho_l * (vx_l * vx_l + vy_l * vy_l)
    energy_r = p_r / (gamma - 1) + 0.5 * rho_r * (vx_r * vx_r + vy_r * vy_r)

    # Computing the average states
    rho_ave = 0.5 * (rho_l + rho_r)
    momx_ave = 0.5 * (rho_l * vx_l + rho_r * vx_r)
    momy_ave = 0.5 * (rho_l * vy_l + rho_r * vy_r)
    energy_ave = 0.5 * (energy_l + energy_r)

    p_ave = (gamma - 1) * (energy_ave - 0.5 * (momx_ave * momx_ave + momy_ave * momy_ave) / rho_ave)

    # Computing the fluxes - Lax-Friedrichs/Rusanov
    flux_mass = momx_ave
    flux_momx = momx_ave * momx_ave / rho_ave + p_ave
    flux_momy = momx_ave * momy_ave / rho_ave
    flux_energy = (energy_ave + p_ave) * momx_ave / rho_ave

    # Finding the wave speeds
    c_l = np.sqrt(gamma * p_l / rho_l) + np.abs(vx_l)
    c_r = np.sqrt(gamma * p_r / rho_r) + np.abs(vx_r)
    c = np.maximum(c_l, c_r)

    # Stabilizing Diffusive Term
    flux_mass -= c * 0.5 * (rho_l - rho_r)
    flux_momx -= c * 0.5 * (rho_l * vx_l - rho_r * vx_r)
    flux_momy -= c * 0.5 * (rho_l * vy_l - rho_r * vy_r)
    flux_energy -= c * 0.5 * (energy_l - energy_r)

    return flux_mass, flux_momx, flux_momy, flux_energy


def apply_fluxes(f, flux_f_x, flux_f_y, dx, dt) -> np.mat:
    """
    Applying the fluxes to all the conserved variables.
    :param f: np.mat of a conserved variable field
    :param flux_f_x: np.mat of the x-direction fluxes
    :param flux_f_y: np.mat of the y-direction fluxes
    :param dx: float of the cell size
    :param dt: float of the timestep
    :return: np.mat of the updated conserved variable field
    """
    # directions to np.roll()
    r = -1  # right
    l = 1  # left

    # Updating solution
    f += - dt * dx * flux_f_x
    f += dt * dx * np.roll(flux_f_x, l, axis=0)
    f += - dt * dx * flux_f_y
    f += dt * dx * np.roll(flux_f_y, l, axis=1)

    return f


def time_integration(delta_t: float, current_time: float, gamma: float, dx: float, vol: float, courant_factor: float,
                     mass: np.mat, momx: np.mat, momy: np.mat, energy: np.mat, plot_real_time: bool,
                     t_output: float) -> float:

    t = 0
    output_count = 1

    # Checking if there is an image folder, if not, make one.
    if 'sim_images' not in os.listdir():
        os.makedirs('../../sim_images')
    # Clearing all the previous images from past simulation runs
    path = os.path.join(os.curdir + '/sim_images')
    files = os.listdir(path)
    for file in files:
        os.remove(os.path.join(path, file))

    # A flag to define whether it is the first run through
    first_time_flag = True
    while t < delta_t:

        # Little Loading bar, helps keep sane when long simulatiions
        perc = int(t / delta_t * 20)
        sys.stdout.write("[%-20s] %d%%\r" % ('=' * perc, 5 * perc))

        # Getting Primitive Variables
        rho, vx, vy, p = get_primitive(mass, momx, momy, energy, gamma, vol)

        dt = get_timestep_2(courant_factor, dx, vx, vy, gamma, p, rho)

        # Checking plotting condition
        plot_this_iter = False
        if t + dt > output_count * t_output:
            dt = output_count * t_output - t
            plot_this_iter = True


        # Calculating Gradients
        rho_dx, rho_dy = get_gradient(rho, dx)
        vx_dx, vx_dy = get_gradient(vx, dx)
        vy_dx, vy_dy = get_gradient(vy, dx)
        p_dx, p_dy = get_gradient(p, dx)

        # Extrapolating a half time-step in time
        # Extrapolation done by
        rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
        vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1 / rho) * p_dx)
        vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1 / rho) * p_dy)
        p_prime = p - 0.5 * dt * (gamma * p * (vx_dx + vy_dy) + vx * p_dx + vy * p_dy)

        # Extrapolating to face centers
        rho_xl, rho_xr, rho_yl, rho_yr = extrapolate_cell_center_to_face(rho_prime, rho_dx, rho_dy, dx)
        vx_xl, vx_xr, vx_yl, vx_yr = extrapolate_cell_center_to_face(vx_prime, vx_dx, vx_dy, dx)
        vy_xl, vy_xr, vy_yl, vy_yr = extrapolate_cell_center_to_face(vy_prime, vy_dx, vy_dy, dx)
        p_xl, p_xr, p_yl, p_yr = extrapolate_cell_center_to_face(p_prime, p_dx, p_dy, dx)

        # Computing fluxes
        flux_mass_x, flux_momx_x, flux_momy_x, flux_energy_x = get_fluxes(rho_xl, rho_xr, vx_xl, vx_xr,
                                                                          vy_xl, vy_xr, p_xl, p_xr, gamma)
        flux_mass_y, flux_momy_y, flux_momx_y, flux_energy_y = get_fluxes(rho_yl, rho_yr, vy_yl, vy_yr,
                                                                          vx_yl, vx_yr, p_yl, p_yr, gamma)

        # Updating the solution
        mass = apply_fluxes(mass, flux_mass_x, flux_mass_y, dx, dt)
        momx = apply_fluxes(momx, flux_momx_x, flux_momx_y, dx, dt)
        momy = apply_fluxes(momy, flux_momy_x, flux_momy_y, dx, dt)
        energy = apply_fluxes(energy, flux_energy_x, flux_energy_y, dx, dt)

        # update-time
        t += dt

        # Plotting in the real time
        if (plot_real_time and plot_this_iter) or (t >= delta_t):
            plt.cla()
            ax = plt.gca()

            if first_time_flag:
                # This is all to design the grid for the vector field
                im = ax.imshow(rho.T)
                left, right, bottom, top = im.get_extent()
                size = im.get_size()
                temp_x = np.linspace(left, right, size[0])
                temp_y = np.linspace(top, bottom, size[1])
                X, Y = np.meshgrid(temp_x, temp_y)
                del temp_x, temp_y
                vect_int = 2 # Vector Plotting Intervals

            ax.quiver(X[::vect_int, ::vect_int], Y[::vect_int, ::vect_int],
                      vx[::vect_int, ::vect_int], vy[::vect_int, ::vect_int], headwidth=0.9, linewidth=5)
            # plt.clim(0.8, 2.2)
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            fig = plt.gcf()
            plt.pause(0.1)
            fig.savefig(f'sim_images/finite_volume{output_count}.png')

            output_count += 1

    sys.stdout.write("[%-20s] %d%%\r" % ('=' * 20, 100))
    sys.stdout.write('Simulation Complete!')
    # Saving the figure as a gif
    with imageio.get_writer('../../Fluid_Sim.gif', mode='I') as writer:
        for plot_num in range(1, output_count):
            image = imageio.imread(f'sim_images/finite_volume{plot_num}.png')
            writer.append_data(image)

    plt.show()  # Only Call Want to Plot During Sim


    return t + current_time


def main():
    """
    Finite Volume Simulation
    :return: 0 if runs succesfully
    """

    # Simulation Parameters
    n_cells = 128  # Cell resolution
    box_size = 1
    gamma = 5 / 3  # 5/3 for an ideal gas
    courant_factor = 0.4
    t = 0
    t_end = 2
    t_output = 0.01  # animation drawing frequency
    use_slope_limiting = False  # The slope limiting in numerical derivatives for higher resolution discontinuities
    plot_real_time = True  # Choose if simulation animates real time or saves a gif

    # Designing Mesh
    dx = box_size / n_cells
    vol = dx * dx
    xlin = np.linspace(0.5 * dx, box_size - 0.5 * dx, n_cells)
    Y, X = np.meshgrid(xlin, xlin)

    # Initial Conditions
    w0 = 0.1
    sigma = 0.05 / np.sqrt(2.)
    rho = 1. + (np.abs(Y - 0.5) < 0.25)
    vx = -0.5 + (np.abs(Y - 0.5) < 0.25)
    vy = w0 * np.sin(4 * np.pi * X) * (
                np.exp(-(Y - 0.25) ** 2 / (2 * sigma ** 2)) + np.exp(-(Y - 0.75) ** 2 / (2 * sigma ** 2)))
    p = 2.5 * np.ones(X.shape)
    vy[:] = 1
    vx[:64] = 1
    vx[64:] = -1
    # vx = (np.random.rand(*vx.shape) - 0.25)
    # vy = (np.random.rand(*vy.shape) - 0.25)

    # Gettting the conserved variables
    mass, momx, momy, energy = get_conservative(rho, vx, vy, p, gamma, vol)

    # Designing the figure for the simulation
    fig = plt.figure(figsize=(4, 4), dpi=240)

    # Running the simulation

    time_integration(t_end, t, gamma, dx, vol, courant_factor, mass, momx, momy, energy, plot_real_time, t_output)

    return 0


if __name__ == '__main__':
    main()
