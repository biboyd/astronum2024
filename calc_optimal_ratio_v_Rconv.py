import numpy as np
import pandas as pd
import pynucastro as pyna
import matplotlib.pyplot as plt
import unyt
from scipy.optimize import minimize


def load_data(infile):
    df = pd.read_csv(infile, skiprows=14, sep='\\s+', header=None)
    df.columns = ["radius", "density", "temperature", "pressure", "neutron",
                  "hydrogen-1", "helium-4", "carbon-12", "oxygen-16",
                  "neon-20", "neon-23", "sodium-23", "magnesium-23"]

    radius_arr = df.radius.to_numpy() / 1e5  # km
    density_arr = df.density.to_numpy()
    temperature_arr = df.temperature.to_numpy()

    return radius_arr, density_arr, temperature_arr


def calc_mass_arr(rad_arr, rho_arr):
    dr = rad_arr[1] - rad_arr[0]
    rad_in_arr = 1e5 * (rad_arr - dr/2.)  # cm
    rad_out_arr = 1e5 * (rad_arr + dr/2.)  # cm

    # calc Mconv
    mass = 4./3. * np.pi * rho_arr * (rad_out_arr**3 - rad_in_arr**3)
    return mass


def calc_weak_rates(rho_arr, T_arr):

    # grab rate for library
    tl = pyna.TabularLibrary()
    ecap_rate = tl.get_rate_by_name('na23(e,)ne23')
    bdecay_rate = tl.get_rate_by_name('ne23(,e)na23')

    # arrays to save rate and nuloss calculations
    ecap_rate_arr = np.empty(len(rho_arr))
    bdecay_rate_arr = np.empty(len(rho_arr))

    ecap_nu_loss_arr = np.empty(len(rho_arr))
    bdecay_nu_loss_arr = np.empty(len(rho_arr))

    # loop through and calc rate at each radial bin
    for i, (rho, T) in enumerate(zip(rho_arr, T_arr)):

        # assuming even Ye
        Ye = 0.5

        # calc rates and nu loss
        try:
            ecap_rate_arr[i] = ecap_rate.eval(T, rhoY=rho*Ye)
            bdecay_rate_arr[i] = bdecay_rate.eval(T, rhoY=rho*Ye)

            ecap_nu_loss_arr[i] = ecap_rate.get_nu_loss(T, rhoY=rho*Ye)
            bdecay_nu_loss_arr[i] = bdecay_rate.get_nu_loss(T, rhoY=rho*Ye)

        # set zero for low density/temp when we're outside the tabular range
        except ValueError:
            ecap_rate_arr[i] = 0.
            bdecay_rate_arr[i] = 0.

            ecap_nu_loss_arr[i] = 0.
            bdecay_nu_loss_arr[i] = 0.

    return ecap_rate_arr, bdecay_rate_arr, ecap_nu_loss_arr, bdecay_nu_loss_arr


def calc_c12_burn():
    # gonna just fix this value for rn
    # Based on the value of the End of the Sim
    # in units g/s
    return 7.17668016e+25


def calc_const_variables(rad_arr, rho_arr, T_arr, Rconv):
    # constants needed by the calculation
    mass = calc_mass_arr(rad_arr, rho_arr)

    mask_arr = np.where(rad_arr[rad_arr < Rconv])
    Mconv = np.sum(mass[mask_arr])

    # calc Nuc Rate Masses
    ecap_arr, bdecay_arr, ecap_nu, bdecay_nu = calc_weak_rates(rho_arr, T_arr)
    Mecap_arr = mass * ecap_arr
    Mbeta_arr = mass * bdecay_arr

    delMburn = calc_c12_burn()
    delMecap = np.sum(Mecap_arr[mask_arr])
    delMbeta = np.sum(Mbeta_arr[mask_arr])

    return (Mconv, delMburn, delMbeta, delMecap)


def calc_new_ratio(in_ratio, total_Urca, inputs,
                   Mconv_dot=0.0*unyt.Msun/unyt.hr, outside_X=None):

    # calc composition and constants
    Xna23 = total_Urca / (1 + in_ratio)
    Xne23 = total_Urca * in_ratio / (1 + in_ratio)
    Mconv, delMburn, delMbeta, delMecap = inputs

    if outside_X is None:
        outside_X = total_Urca

    # calc mass change for urca stuff
    delMconv = Mconv_dot.in_cgs()*outside_X
    change_in_na23 = delMconv.value + delMburn + Xne23*delMbeta - Xna23*delMecap 
    change_in_ne23 = Xna23*delMecap - Xne23*delMbeta

    # calc the real final ratio though.
    final_ratio = change_in_ne23/change_in_na23
    return final_ratio


def test_ratio(in_ratio, total_Urca, inputs, outside_X=None):
    new_ratio = calc_new_ratio(in_ratio, total_Urca, inputs,
                               outside_X=outside_X)
    return np.abs(in_ratio - new_ratio)


def find_optimal_ratio(Rconv, const_variables, urca_total, guess=1.,
                       min_ratio=0.01, max_ratio=1e5):

    sol = minimize(test_ratio, guess, method='Powell',
                   bounds=[(min_ratio, max_ratio)],
                   args=(urca_total, const_variables),
                   tol=1e-12)

    if sol.success:
        return sol.x[0]
    else:
        print(sol)
        return -100


def plot_results(Rconv, Mconv, ratio, 
                 ylabel="$X({}^{23} \\mathrm{Ne}) / X({}^{23} \\mathrm{Na})$"):
    fig, ax = plt.subplots(1, 1)

    # do main plot w/ Rconv
    ax.plot(Rconv, ratio)

    ax.set_xlabel("$R_{\\mathrm{conv}}$($\\mathrm{km}$)", fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')

    bot, top = ax.get_ylim()
    #ax.vlines(520, bot, top, linestyles='--', colors='k')
    ax.set_ylim(bot, top)
    ax.set_xlim(Rconv[0], Rconv[-1])

    # setup Mconv labeling
    axM = ax.twiny()

    axM.plot(Mconv, ratio)
    axM.cla()
    axM.set_xlim(Mconv[0], Mconv[-1])
    axM.set_title("$M_{\\mathrm{conv}}$ ($\\mathrm{M}_{\\odot}$)", fontsize='large')
    return fig, ax


if __name__ == "__main__":
    infile = "init_models/WD_urca_nuc_T5.6e8_rho4.5e9_Mconv1.0.hse.hse.8192"

    rad_arr, rho_arr, T_arr = load_data(infile)
    mass_arr = calc_mass_arr(rad_arr, rho_arr) * unyt.g
    mass_arr = mass_arr.in_units('Msun').value
    R_in = 420.  # km
    R_out = 720.  # km
    dR = 10.  # km

    Rconv_arr = np.arange(R_in, R_out+dR, dR, dtype=np.float64)
    out_ratio_arr = np.empty_like(Rconv_arr)
    Mconv_arr = np.array([np.sum(mass_arr[np.where(rad_arr < r)]) for r in Rconv_arr])

    urca_tot = 8e-4

    for i, Rconv in enumerate(Rconv_arr):
        # calc necessary constants
        const_inputs = calc_const_variables(rad_arr, rho_arr, T_arr, Rconv)

        # this is quick mix w/o source
        init_guess = 40. #min(40., const_inputs[3]/const_inputs[2])

        # run the minimizing
        curr_ratio = find_optimal_ratio(Rconv, const_inputs, urca_tot,
                                        guess=init_guess)

        # est_ratio = calc_new_ratio(curr_ratio, urca_tot, const_inputs)

        print(f"R={Rconv:0.1f} M={Mconv_arr[i]:0.3f}, guess={init_guess:0.1f} result is: {curr_ratio:0.1f}")

        out_ratio_arr[i] = curr_ratio

    fig, ax = plot_results(Rconv_arr, Mconv_arr, out_ratio_arr)
    ax.plot(519, 9, 'x', color='k', label='3D simulation')
    ax.legend()
    fig.savefig("figures/ratio_vs_rconv.png")

    # save the arrays
    np.save("rconv_arr.npy", Rconv_arr)
    np.save("Mconv_arr.npy", Mconv_arr)
    np.save("optimal_ratio_arr.npy", out_ratio_arr)
