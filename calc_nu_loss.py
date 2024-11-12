import numpy as np
from calc_optimal_ratio_v_Rconv import (calc_weak_rates, load_data,
                                        calc_mass_arr, plot_results)
import unyt
import matplotlib.pyplot as plt
from matplotlib import colormaps
cmap = colormaps['Blues']


def calc_const_nu_loss(Xne23, Xna23, ecap_nu_loss_arr, bdecay_nu_loss_arr, mass_arr):
    # take in array of masses and nu_losses raw rates
    # take mass fractions and calculate the actual nulosses per mass bin
    # calculate energy lost

    # calc losses
    ecap_losses = ecap_nu_loss_arr * Xna23
    bdecay_losses = bdecay_nu_loss_arr * Xne23

    # calc mass times constants to account for molarity etc.
    normed_mass = mass_arr * unyt.avogadros_number_mks.value / 23.

    return normed_mass * (ecap_losses + bdecay_losses)


def plot_nu_losses(rad_arr, Rconv_arr, nu_loss_arr, ratio_arr, sample=4):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    min_color = 0.2
    cmap_range = (1/(1-min_color)) * (np.max(Rconv_arr) - np.min(Rconv_arr))

    for i, Rconv in enumerate(Rconv_arr):
        curr_color = cmap(min_color + (Rconv - np.min(Rconv_arr))/cmap_range)
        print(min_color + (Rconv - np.min(Rconv_arr))/cmap_range)
        if i % sample:
            pass

        else:
            rconv_str = "$R_\\mathrm{conv}$"
            ax.plot(rad_arr, nu_loss_arr[i, :]/(rad_arr[1] - rad_arr[0])/1e41,
                    color=curr_color,
                    label=f"{rconv_str}: {Rconv:0.0f} km. ratio={ratio_arr[i]:0.1f}")

    # plot real sim data
    left, right = ax.set_xlim(0., np.max(Rconv_arr))

    sim_nuloss = np.load("nuloss.npy")
    sim_rad = np.load("radius.npy")/1e5
    ax.plot(sim_rad, sim_nuloss/(sim_rad[1] - sim_rad[0])/1e41, 'k--', label='3D Sim. $\\mathrm{ratio} \\sim 9$')

    ax.set_xlim(left, right)

    # set plot params
    ax.legend(ncols=2, loc='upper left', bbox_to_anchor=(0., 1.1), framealpha=1)
    ax.set_xlabel("Stellar Radius (km)", fontsize='large' )
    ax.set_ylabel('$\\mathrm{\\dot{E}}_{\\nu_e}(r)$ per radial bin ($10^{41}$ erg/s/km)', fontsize='large')

    return fig, ax


if __name__ == "__main__":
    # load in data
    infile = "init_models/WD_urca_nuc_T5.6e8_rho4.5e9_Mconv1.0.hse.hse.8192"

    # load in data and calc appropriate arrays
    rad_arr, rho_arr, T_arr = load_data(infile)
    mass_arr = calc_mass_arr(rad_arr, rho_arr) * unyt.g
    ecap_rate, bdecay_rate, ecap_nu_loss, bdecay_nu_loss = calc_weak_rates(rho_arr, T_arr)

    ratio_arr = np.load("optimal_ratio_arr.npy")
    Rconv_arr = np.load("rconv_arr.npy")
    sum_nu_arr = np.empty_like(Rconv_arr)

    # save unneeded space as negative
    full_nu_loss_arr = np.zeros((len(Rconv_arr), len(rad_arr)))

    urca_tot = 8e-4

    for i, (Rconv, ratio) in enumerate(zip(Rconv_arr, ratio_arr)):
        # limit to conv zone
        mask_arr = np.where(rad_arr < Rconv)

        conv_mass = mass_arr[mask_arr].value
        conv_ecap_nu_loss = ecap_nu_loss[mask_arr]
        conv_bdecay_nu_loss = bdecay_nu_loss[mask_arr]

        # calc urca quantities
        Xne23 = urca_tot * ratio / (1 + ratio)
        Xna23 = urca_tot / (1 + ratio)

        # calc the nu loss and save total and full
        curr_nu_loss_arr = calc_const_nu_loss(Xne23, Xna23, conv_ecap_nu_loss,
                                              conv_bdecay_nu_loss, conv_mass)
        sum_nu_arr[i] = np.sum(curr_nu_loss_arr)

        end_idx = len(curr_nu_loss_arr)
        full_nu_loss_arr[i, :end_idx] = curr_nu_loss_arr

    # calc Mconv for plotting
    mass_arr = mass_arr.in_units('Msun').value
    Mconv_arr = np.array(
        [np.sum(mass_arr[np.where(rad_arr < r)]) for r in Rconv_arr])

    fig, ax = plot_results(Rconv_arr, Mconv_arr,
                           sum_nu_arr,
                           ylabel='$\\dot{E}_{\\nu_e}$(erg/s)')
    ax.plot(519, 4.18e42, 'x', color='k', label='3D simulation')
    ax.hlines(3.32e43, *ax.get_xlim(), colors='tab:orange', label='3D Simulation $\\dot{E}_\\mathrm{nuc}$',zorder=-1)
    ax.legend()

    fig.savefig("figures/nuloss_vs_rconv.png")

    figall, _ = plot_nu_losses(rad_arr, Rconv_arr, full_nu_loss_arr, ratio_arr,
                               sample=6)
    figall.savefig("figures/full_nu_losses.png")
    # save nu loss array data
    np.save("sum_nu_loss.npy", sum_nu_arr)
    np.save("full_nu_loss_v_rad.npy", full_nu_loss_arr)
