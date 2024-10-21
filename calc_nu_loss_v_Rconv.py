import numpy as np
import pandas as pd
import pynucastro as pyna
import matplotlib.pylot as plt
import unyt


def load_data(infile):
    df = pd.read_csv(infile, skiprows=14, sep='\s+', header=None)
    df.columns = ["density", "temperature", "pressure", "neutron", 
                  "hydrogen-1", "helium-4", "carbon-12", "oxygen-16", 
                  "neon-20", "neon-23", "sodium-23", "magnesium-23"]
    density_arr = df.density.to_numpy()
    temperature_arr = df.temperature.to_numpy()
    return density_arr, temperature_arr


def calc_weak_rates(df):

    # grab rate for library
    tl  = pyna.TabularLibray()
    ecap_rate = tl.get_rate_by_name('na23(e,)ne23')
    bdecay_rate = tl.get_rate_by_name('ne23(,e)na23')

    # arrays to save rate and nuloss calculations
    ecap_rate_arr = np.empty(len(df))
    bdecay_rate_arr = np.empty(len(df))

    ecap_nu_loss_arr = np.empty(len(df))
    bdecay_nu_loss_arr = np.empty(len(df))

    # loop through and calc rate at each radial bin
    for i in range(len(df)):
        rho, T = df[['density', 'temperature']].iloc[i]
        Ye = 0.5

        ecap_rate_arr[i] = ecap_rate.eval(T, rhoY=rho*Ye)
        bdecay_rate_arr[i] = bdecay_rate.eval(T, rhoY=rho*Ye)

        ecap_nu_loss_arr[i] = ecap_rate.get_nu_loss(T, rhoY=rho*Ye)
        bdecay_nu_loss_arr[i] = bdecay_rate.get_nu_loss(T, rhoY=rho*Ye)
    return ecap_rate_arr, bdecay_rate_arr, ecap_nu_loss_arr, bdecay_nu_loss_arr


# constants needed by the calculation
def calc_const_variables(rho_arr, temp_arr, Rconv):
    #sph_conv = ds.sphere(ds.domain_center, (Rconv, 'km'))
    #sph_nu   = ds_nu.sphere(ds_nu.domain_center, (Rconv, 'km'))

    ## Mass and mass changes
    #Mconv    = sph_conv.sum('mass')
    #delMburn = sph_conv.sum("c12_to_na23_mass")
    #delMecap = sph_nu.sum("mass ecap")
    #delMbeta = sph_nu.sum("mass beta")

    return (Mconv, delMburn, delMbeta, delMecap)


def calc_new_ratio(in_ratio, total_Urca, inputs,
                   Mconv_dot=0.05*unyt.Msun/unyt.hr,
                   outside_X=None, dt=unyt.hr):

    # calc composition and constants
    Xna23 = total_Urca / (1 + in_ratio)
    Xne23 = total_Urca * in_ratio / (1 + in_ratio)
    Mconv, delMburn, delMbeta, delMecap = inputs

    if outside_X is None:
        outside_X = total_Urca

    # calc mass change for urca stuff
    change_in_na23 = Mconv_dot*outside_X + delMburn + Xne23*delMbeta - Xna23*delMecap 
    change_in_ne23 = Xna23*delMecap - Xne23*delMbeta

    # calc the real final ratio though.
    final_ratio = change_in_ne23/change_in_na23
    return final_ratio


def test_ratio(in_ratio, total_Urca, inputs, outside_X=None, dt=unyt.hr):
    return np.abs(in_ratio - calc_new_ratio(in_ratio, total_Urca, inputs, outside_X=outside_X, dt=dt))/in_ratio



if __name__ == "__main__":
    infile = "init_models/WD_urca_nuc_T5.5e8_rho4.5e9_Mconv1.0.hse.hse.4096"

    rho, temp = load_data(infile)
