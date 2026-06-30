# Program: ParMaCh - 1D Model of Solidification of Magma Chambers 
# Module: Monte-Carlo simulation of the Stokes fall - miscellaneous functions.

import logging
import numpy as np
import warnings
import copy
import random
import gc
import types
import time as time
import matplotlib.pyplot as plt 
from math import log, trunc, sqrt
from tqdm import tqdm
from numpy.typing import NDArray
from typing import Dict, Tuple, Optional, Deque
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, root_scalar
from scipy.integrate import quad
from numba.experimental import jitclass
from numba import njit, boolean, int32, int64, float64, prange

# Modules:
from mPar import *
from mGL import calculate_rayleigh_number
from mFunc import *

# Suppress only the NumbaPendingDeprecationWarning:
from numba.core.errors import NumbaPendingDeprecationWarning 

# Get logger from main.py:
logger = logging.getLogger(__name__)

""" ############################################## """
""" #########    UNMIXED STOKES 1.0    ########### """
""" ############################################## """

def n_sed(Hmix: float, lmd: float, z_max: float, z_min: float, bulk_grow: float, A: float, a_grid: np.ndarray, bins: int,
          gWrms: float, atbl: np.ndarray, ntbl: np.ndarray, tblstokes: bool, f_tbl: interp1d
) -> np.ndarray:
    """ Calculates steady-state distribution in the sediment or the Stokesian unmixed fall """
    
    r_flux = np.zeros_like(a_grid)
    pref = lmd / (bulk_grow * Hmix)
    for i, a in enumerate(a_grid):
        if a <= 0:
            continue
        ab_grid = np.linspace(0, min(a, A), bins)
        zf = z_fall(a, ab_grid, gWrms, lmd, bulk_grow)

        # Restrict to the mixed layer:
        mask = (zf >= z_min) & (zf <= z_max)
        if not np.any(mask):
            continue
        ab_valid = ab_grid[mask]
        if ab_valid.size < 2:
            continue
        integral = np.trapz(f_tbl(ab_valid), ab_valid)
        r_flux[i] = pref * (a**2) * integral

    # Rescale so integral matches nucleation rate:
    #if tblstokes: r_flux[:] = 0.0; r_flux[-1] = np.sum(ntbl)
    if np.sum(r_flux) == 0.0: r_flux[-1] = np.sum(ntbl)
    total = np.sum(r_flux)
    if total > 0:
        r_flux *= np.sum(ntbl) / total # spawn_rate = np.sum(ntbl)!

    return r_flux

def z_fall(a, ab, W, lmd, bulk_grow) -> float:
    """ Auxiliary function fro the Stokesian unmixed fall """
    return (lmd / (3.*bulk_grow))*(a**3 - ab**3) - (W / bulk_grow)*(a - ab)

def n_bulk(z_max: float, z_min: float, Hmix: float, a_grid: np.ndarray, A: float, bulk_grow: float, lmd: float,
           gWrms: float, atbl: np.ndarray, ntbl: np.ndarray, bins: int, tblstokes: bool, f_tbl: interp1d
) -> tuple[np.ndarray, np.ndarray]:
    """ Calculates steady-state distribution in the bulk for the Stokesian unmixed fall """

    n_active = np.zeros_like(a_grid)

    #if tblstokes:
    #    for i, a in enumerate(a_grid):
    #        ab_grid = np.linspace(0, min(a, A), bins)
    #        n_active[:] = (1. / (Hmix * bulk_grow)) * Hmix * f_tbl(ab_grid)
    #else: 
    
    for i, a in enumerate(a_grid):      
        abtmp = min(a, A)
        if abtmp < 0.0: continue
        ab_grid = np.linspace(0, min(a, A), bins)
        zf = z_fall(a, ab_grid, gWrms, lmd, bulk_grow)
        L = np.where(zf <= z_min, Hmix, np.where(zf >= z_max, 0.0, z_max - zf))
        integrand = f_tbl(ab_grid) * L
        n_active[i] = (1. / ((z_max - z_min) * bulk_grow)) * np.trapz(integrand, ab_grid)           # TODO: factor 1/Hmix OK? 
        #n_active[i] = (1. / (bulk_grow)) * np.trapz(integrand, ab_grid)

    return n_active

def mSfJIT(
    Hnow: float, Hmix: float, bulk_grow: float, lmd: float, nu: float, aest: float, ntbl: np.ndarray, atbl: np.ndarray, 
    gWrms: float, tblstokes: bool, bins: int, fdense: bool=False, growth_cutoff: float=1.e-15
) -> tuple[CrystalDistro, CrystalDistro]:
    """ Stokesian fall with crystals being well-mixed within a subvolume of the chamber """
    
    if Hmix > Hnow: 
        print("WARNING: the height of the mixed sub-domain is bigger than the chamber itself!")
        Hmix = 0.01 # TODO
        #raise Exception("[Function mSfJIT error]")

    if tblstokes: Hmix = 0.01
    zmin = Hnow - Hmix
    zmax = Hnow
    atblmax = np.max(atbl)

    # TBL distribution interpolation object:
    ftbl = None if len(atbl) == 1 or len(ntbl) == 1 else interp1d(atbl, ntbl, bounds_error=False, fill_value=0.0) 

    try: # TODO: fix so that it doesnt yell some nonsense!
        tbulkmin = (Hnow - Hmix) / ((2. * ModelParameter.gacc * (ModelParameter.rhoc - ModelParameter.rhof) * (bulk_grow + growth_cutoff)**2) \
                    / (27. * nu ** ModelParameter.rhof) )  
        tbulkmax = ((27. * nu * Hnow * ModelParameter.rhof) / (2. * ModelParameter.gacc * (ModelParameter.rhoc - ModelParameter.rhof) \
                    * (bulk_grow + growth_cutoff)**2))**(1./3.)
        
    except RuntimeWarning: # TODO aint an exception?!
        tbulkmin = 0.0
        tbulkmax = 0.0
    
    abulkmin = tbulkmin*bulk_grow + np.max(atbl) # ; print(f"Expected minimum radius: {1.e3*abulkmin:.3e} [mm]")
    abulkmax = tbulkmax*bulk_grow + np.max(atbl) # ; print(f"Expected maximum radius: {1.e3*abulkmax:.3e} [mm]")

    # Denser grid around expected peak:
    if fdense:
        aleft   = np.linspace(0, abulkmin, bins//4, endpoint=False)
        amiddle = np.linspace(abulkmin, abulkmax, bins//2, endpoint=False)
        aright  = np.linspace(abulkmax, aest, bins//4)
        a_grid  = np.concatenate([aleft, amiddle, aright])
    else:
        a_grid = np.linspace(0, aest, bins)

    ndistBLK = n_bulk(zmax, zmin, Hmix, a_grid, atblmax, bulk_grow, lmd, gWrms, atbl, ntbl, bins, tblstokes, ftbl)
    ndistSED = n_sed(Hmix, lmd, zmax, zmin, bulk_grow, atblmax, a_grid, bins, gWrms, atbl, ntbl, tblstokes, ftbl)

    #print(f"A: {np.sum(ndistSED):.3e} ({np.sum(ntbl)}).")

    return (ndistBLK, ndistSED, a_grid)

# Interpolator of the TBL distribution:
#ftbl = interp1d(atbl, ntbl, bounds_error=False, fill_value=0.0) # NOTE: would not work with NUMBA!

""" ############################################## """
""" #########    UNMIXED STOKES 2.0    ########### """
""" ############################################## """

def integrate_P(z1, z2, Pz, zmin, zmax):
    if z2 <= z1: return 0.0
    val, _ = quad(lambda z: Pz(z, zmin, zmax), z1, z2)
    return val

def P_uniform(z, zmin, zmax):
    return 1.0 / (zmax - zmin)

def P_exponential(z, zmin, zmax, alpha=5.0):
    Z = (1.0 - np.exp(-alpha * (zmax - zmin))) / alpha
    return np.exp(-alpha * (z - zmin)) / Z 

def calculate_nbulk(a: np.ndarray, a0: float, lmd: float, bulk_grow: float, hmix: float, zmin: float, zmax: float, gWrms: float, Pz) -> float:
    """ Calculate the steady-state distribution in the bulk for the unmixed Stokesian fall """

    zcrit = ((lmd / (3.0 * bulk_grow)) * (a**3 - a0**3) - (gWrms / bulk_grow) * (a - a0))
    if np.isscalar(a):
        if zcrit <= zmin: return hmix        
        if zcrit >= zmax: return 0.0
        return hmix * integrate_P(zcrit, zmax, Pz, zmin, zmax)
    else:
        out = np.zeros_like(a)
        mask1 = zcrit <= zmin
        mask2 = (zcrit > zmin) & (zcrit < zmax)
        out[mask1] = hmix

        for i in np.where(mask2)[0]:
            out[i] = hmix * integrate_P(zcrit[i], zmax, Pz, zmin, zmax)
        
        return out

# NOTE: dont delete yet
#out[mask2] = zmax - zcrit[mask2]
#mask1 = ( (lmd / (3. * bulk_grow)) * (a**3 - a0**3) <= zmin )
#mask2 = ( (lmd / (3. * bulk_grow)) * (a**3 - a0**3) >= zmin ) & ( (lmd / (3. * bulk_grow)) * (a**3 - a0**3) <= zmax )
#out[mask2] = zmax - (lmd / (3. * bulk_grow)) *  (a[mask2]**3 - a0**3)
    
def calculate_nsed(a: np.ndarray, a0: float, a1: float, a2: float, bulk_grow: float, hmix: float, lmd: float, gWrms: float, zmin: float, zmax: float, Pz):
    """ Calculate the steady-state distribution in the sediment for the unmixed Stokesian fall """

    zcrit = ((lmd / (3.0 * bulk_grow)) * (a**3 - a0**3) - (gWrms / bulk_grow) * (a - a0))
    out = np.zeros_like(a)
    mask = ( (a >= a1) & (a <= a2) )
    
    dzda = (lmd * a[mask]**2 - gWrms) / bulk_grow
    Pvals = np.array([Pz(zc, zmin, zmax) for zc in zcrit[mask]])
    out[mask] = dzda * Pvals * hmix

    return out

# NOTE: dont delete yet
#out[mask] = ((lmd / (bulk_grow * hmix)) * a[mask]**2) - (gWrms / (bulk_grow * hmix))
#out[out < 0.0] = 0.0

def find_cubic_root(a0: float, lmd: float, bulk_grow: float, zmin: float, zmax: float, gWrms: float) -> tuple[float, float]:
    alpha   = lmd / 3.0
    beta    = - gWrms
    gamma_1 = - ( (lmd / 3.0) * a0**3 + bulk_grow * zmin - gWrms * a0)
    gamma_2 = - ( (lmd / 3.0) * a0**3 + bulk_grow * zmax - gWrms * a0) # NOTE: there was a + sign, I removed it and now the volumetric constraint is satisfied!
    
    def f_1(a): return alpha * a**3 + beta * a + gamma_1 
    def f_2(a): return alpha * a**3 + beta * a + gamma_2
    sol_1 = root_scalar(f_1, bracket=[0, 1e6], method="brentq")
    sol_2 = root_scalar(f_2, bracket=[0, 1e6], method="brentq")
    a1 = sol_1.root
    a2 = sol_2.root

    return a1, a2

def mean_residence_time(a0: float | np.ndarray, hmix: float | np.ndarray, G: float, lmd: float, H: float, gWrms: float, \
                        Pz, TINY: float=1.e-5) -> float:
    """ Calculates mean residence time for one unmixed batch of crystals """
    
    # NOTE: only works for uniform PDF function!
    
    alpha = lmd * G**2 / 3.
    beta  = lmd * G * a0                 
    gamma = lmd * a0**2 - gWrms
    z0_grid = np.linspace(0, hmix, 100)
    dz_grid = H - z0_grid

    t_exact = []
    dz_grid[dz_grid < 0.0] = TINY
    for dz in dz_grid:  
        def f(t): return alpha*t**3 + beta*t**2 + gamma*t - dz 
        sol = root_scalar(f, bracket=[0, 1e6], method="brentq")
        t_exact.append(sol.root)

    if Pz is P_uniform:
        t_mean = np.mean(t_exact)
        #print(f"tmean: {t_mean:.3e}.")
    elif Pz is P_exponential:
        P = Pz(z0_grid, 0.0, hmix)
        
        t_mean = np.trapz(t_exact * (1. - P), z0_grid)
        #print(f"tmean: {t_mean:.3e}.")

        # tmean here < tmean above, does it make sense? no, crystals closer to bottom boundary should spend less time there!
    else:
        raise ValueError("[WARNING] - Invalid PDF!")

    return t_mean

def calculate_suspended_crystals(hmixbatch: np.ndarray, Kbatch: np.ndarray, atr: float, G: float, lmd: float, H: float, gWrms: float, Pz):
    """ Calculates the total suspended population within bulk for the partially-mixed case """

    if isinstance(Pz, int):
        match Pz:
            case 1:
                print("here in case 1")
                Pz = P_uniform
            case 2: 
                Pz = P_exponential
            case _:
                raise("[WARNING] - Invalid PDF (calculate_suspended_crystals)!")

    N_batches = []
    a0batch = np.full(len(hmixbatch), atr)
    for a0, hmix, K in zip(a0batch, hmixbatch, Kbatch):
        t_mean = mean_residence_time(a0, hmix, G, lmd, H, gWrms, Pz)
        N = K * t_mean
        N_batches.append(N)
    Nsus = np.sum(N_batches)

    return Nsus 

""" ############################################## """
""" #########    UNMIXED STOKES 3.0    ########### """
""" ############################################## """

@njit
def exponential_pdf(hmix: float, alpha: float) -> float:
    """
    u = np.random.random()
    if alpha == 0.0:
        return u * hmix
    else:
        return (-np.log(1.0 - u * (1.0 - np.exp(-alpha * hmix))) / alpha)
    """
    zmin = 0.0
    zmax = hmix
    u = np.random.random()
    if alpha == 0.0:
        return zmin + u * (zmax - zmin)
    else:
        return zmin - np.log(
            1.0 - u * (1.0 - np.exp(-alpha * (zmax - zmin)))
        ) / alpha

@njit
def partially_mixed_monte_carlo(
    hmix:    np.ndarray,
    Kpop:    np.ndarray,
    apop:    np.ndarray, # TODO: remove?
    dt:      float,
    nu:      float,
    bgrw:    float,
    lmd:     float,
    Hnow:    float,
    astn:    float,
    c:       Parameters,
    gWrms:   float,
    Pz_mode: int,
    steps:   int=2200,
    ngens:   int=1

) -> tuple[np.ndarray, ...]:
    """ Stochastic Monte-Carlo method for the treatment of the partially-mixed regime """
    
    # NOTE: You can not change the step dt here! Kpop is normalized per apriori calculate time step dt!
    # TODO: přepsat na while + checker steady-statu? | TODO: generations?!

    batches = len(hmix)
    suspended_at_given_time = np.zeros(steps)
    sedimented_per_dt = np.zeros(steps)
    crystals = np.zeros((steps, batches, 4), dtype=np.float64) # NOTE: batch:= [K-count, z-coord, a-size, f-pass]

    totsus_radius = np.empty((ngens, steps*batches), dtype=np.float64)
    totsus_count  = np.empty((ngens, steps*batches), dtype=np.float64)
    totsed_radius = np.empty((ngens, steps*batches), dtype=np.float64)
    totsed_count  = np.empty((ngens, steps*batches), dtype=np.float64)

    for ng in range(ngens):
        # Nullify:
        crystals[:] = 0.0
        suspended_at_given_time = np.zeros(steps)
        sedimented_per_dt = np.zeros(steps)

        for i in range(steps): # loop over time steps!
            # Initialize each time step:
            outcount = 0.0; suspended_mass = 0.0; sedimented_mass = 0.0; idx_fall = 0; idx_susp = 0
            sediment_radius  = np.zeros(steps*batches, np.float64)
            sediment_count   = np.zeros(steps*batches, np.float64)
            suspended_radius = np.zeros(steps*batches, np.float64)
            suspended_count  = np.zeros(steps*batches, np.float64)

            for j in range(batches): # loop over batches
                crystals[i, j, 0] = Kpop[j]    
                if Pz_mode == 1: crystals[i, j, 1] = (np.random.rand()**(1.0 / (0.0 + 1.0))) * hmix[j] 
                elif Pz_mode == 2: 
                    # NOTE: position closer towards the bottom boundary!
                    crystals[i, j, 1] = (hmix[j] - exponential_pdf(hmix[j], alpha=5.0)) 
                else: print("[WARNING] - Invalid PDF (Monte-Carlo)!")         
                crystals[i, j, 2] = astn                                   
                crystals[i, j, 3] = 0.0

            for k in range(i+1):
                for l in range(batches):
                    if crystals[k, l, 3] == 0.0:
                        crystals[k, l, 1] += (wStokes(crystals[k, l, 2], nu, c) - gWrms) * dt
                        crystals[k, l, 2] += bgrw * dt 

                        if crystals[k, l, 1] > Hnow: # outflow distribution!
                            outcount += crystals[k, l, 0] 
                            crystals[k, l, 3] = 1.0
                            sedimented_mass += crystals[k, l, 0]
                            sediment_radius[idx_fall] = crystals[k, l, 2]
                            sediment_count[idx_fall]  = crystals[k, l, 0]
                            idx_fall += 1

            # FIXME: this can be improved!
            for k in range(i+1): # calculate D_B(a,t) suspended CSD
                for l in range(batches):
                    if crystals[k, l, 3] == 0.0:
                        suspended_mass += crystals[k, l, 0]
                        suspended_radius[idx_susp] = crystals[k, l, 2] 
                        suspended_count[idx_susp]  = crystals[k, l, 0]
                        idx_susp += 1

            suspended_at_given_time[i] = suspended_mass
            sedimented_per_dt[i] = sedimented_mass / dt

        # Add individual runs:
        totsus_count[ng, :]  = suspended_count   #np.append(totsus_count, suspended_count)
        totsus_radius[ng, :] = suspended_radius  #np.append(totsus_radius, suspended_count)
        totsed_count[ng, :]  = sediment_count    #np.append(totsed_count, sediment_count)
        totsed_radius[ng, :] = sediment_radius   #np.append(totsed_radius, sediment_radius)

    # Transform into 1D arrays:
    totsus_count  = totsus_count.reshape(-1)
    totsus_radius = totsus_radius.reshape(-1)
    totsed_count  = totsed_count.reshape(-1)
    totsed_radius = totsed_radius.reshape(-1)

    return (crystals, suspended_at_given_time, sedimented_per_dt, [totsed_radius, totsed_count], [totsus_radius, totsus_count])

def call_mPmSfJIT(
    hmix:    np.ndarray,
    Kpop:    np.ndarray,
    apop:    np.ndarray, # TODO: remove? 
    bgrw:    float,
    lmd:     float,
    Hnow:    float,
    astn:    float,
    nu:      float,
    dt:      float,
    gWrms:   float,
    c:       Parameters, # TODO: add nbins=RunConstants.nbins
    Pz_mode: int=2,      # 1: uniform | 2: exponential
    sus_plt: bool=True

) -> None:

    # TODO: pro ngens > 1 asi nějaký dělení jinak křičí IDLE podmínka!
    _, nsusp, nsed, dist_sed, dist_bulk = partially_mixed_monte_carlo(hmix, Kpop, apop, dt, nu, bgrw, lmd, Hnow, Diag.astn, ModelParameter, Pz_mode=Pz_mode, gWrms=gWrms)

    # Number of sedimented crystals per unit time:
    print(f"Number of sedimented crystals per unit time:                                 {nsed[-1]:.3e}.")
    print(f"Number of fall-in crystals & injected stone-like particles per unit time:    {np.sum(Kpop)/dt:.3e}.") # NOTE: this checks out! Great!
    print(f"Number of suspended crystals (falling):                                      {nsusp[-1]:.3e}.")

    if sus_plt:
        nsus_analytical = calculate_suspended_crystals(hmix, Kpop, astn, bgrw, lmd, Hnow, gWrms=gWrms, Pz=Pz_mode) / dt
        print(f"nsus_analytical: {nsus_analytical:.3e}.")
        plt.clf()
        plt.plot(np.arange(0, len(nsusp)), nsusp)
        plt.axhline(y=nsus_analytical, c="r")
        plt.show()  

    distBLK = None; distSED = None
    try: 
        sed_rad = dist_sed[0]; sed_rad = sed_rad[sed_rad > 0.0] # NOTE: sediment distribution!
        sed_cnt = dist_sed[1]; sed_cnt = sed_cnt[sed_cnt > 0.0]

        # TODO: tady to asi umím zlepšit, ale 0.0 pak nefunguje pro da, to je jasný!
        abins = np.linspace(0.0, np.max(sed_rad), num=500)
        nbins = np.zeros(len(abins), dtype=np.float64)
        da = abins[1] - abins[0]
        for i, cnt in enumerate(sed_cnt):
            ibin = int((sed_rad[i]) / da)
            nbins[ibin] += cnt

        distSED = CrystalDistro(abins, nbins)

        # Bulk distribution:
        bulk_rad = dist_bulk[0]; bulk_rad = bulk_rad[bulk_rad > 0.0]
        bulk_cnt = dist_bulk[1]; bulk_cnt = bulk_cnt[bulk_cnt > 0.0]

        abins = np.linspace(0.0, np.max(bulk_rad), num=500)
        nbins = np.zeros(len(abins), dtype=np.float64)
        da = abins[1] - abins[0]
        for i, cnt in enumerate(bulk_cnt):
            ibin = int(bulk_rad[i] / da)
            nbins[ibin] += cnt

        distBLK = CrystalDistro(abins, nbins)

        #############################

        # NOTE: We must account for the suspended mixing particles!
        #print(f"Number of suspended crystals (falling):      {nsusp[-1]:.3e}.")
        #nsusm = np.sum(Kpop * (astn - apop)) / bgrw
        #print(f"Number of suspended crystals (idle-mixing):  {nsusm:.3e}.")
        #print(f"Total number of suspended crystals:          {(nsusp[-1]+nsusm):.3e}.")

    except ValueError: 
        print("ValueError!")

    return (distBLK, distSED)

def call_mSfJIT(
    Hnow: float, Hmix: float, bulk_grow: float, lmd: float, nu: float, aest: float, 
    ntbl: np.ndarray, atbl: np.ndarray, gWrms: float, bins: int, tblstokes: bool=False
) -> tuple[CrystalDistro, CrystalDistro]:

    ndistBLK, ndistSED, adistBLK = mSfJIT(Hnow, Hmix, bulk_grow, lmd, nu, aest, ntbl, atbl, gWrms, tblstokes, bins)

    distBLK = CrystalDistro(adistBLK, ndistBLK)
    distSED = CrystalDistro(adistBLK, ndistSED)

    return (distBLK, distSED)

def call_mSfJIT2(
    atbl: np.ndarray, ntbl: np.ndarray, hmixbatch: np.ndarray,
    Hnow: float, bulk_grow: float, atr: float, lmd: float, gWrms: float,
    P_mode=2

) -> tuple[CrystalDistro, CrystalDistro]:
    """ Call mSfJIT2 function compactly """

    match P_mode:
        case 1:
            P_z = P_uniform
        case 2: 
            P_z = P_exponential
        case _:
            raise("[WARNING] - Invalid PDF!")

    Nsus = calculate_suspended_crystals(hmixbatch, ntbl, atr, bulk_grow, lmd, Hnow, gWrms, Pz=P_z)
    Nout = np.sum(ntbl)
    size = len(ntbl)
    nbulk = np.zeros(size, dtype=np.float64)
    nsed  = np.zeros(size, dtype=np.float64)

    # Loop over all families:
    for hmixj in hmixbatch:
        zmin = Hnow - hmixj
        zmax = Hnow
        if gWrms > 0.0:
            a1, a2 = find_cubic_root(atr, lmd, bulk_grow, zmin, zmax, gWrms)
        else:
            a1 = (atr**3 + 3.*bulk_grow*zmin / lmd)**(1./3.)
            a2 = (atr**3 + 3.*bulk_grow*zmax / lmd)**(1./3.)

        # Superposition: add contribution of the j-th family:
        agrid = np.linspace(atr, a2, num=size)
        nbulk[:] += calculate_nbulk(agrid, atr, lmd, bulk_grow, hmixj, zmin, zmax, gWrms, Pz=P_z)
        nsed[:]  += calculate_nsed(agrid, atr, a1, a2, bulk_grow, hmixj, lmd, gWrms, zmin, zmax, Pz=P_z)

    # Normalize nbulk/nsed and rescale by the # of suspended crystals and nucleation rate respectively:
    nbulk /= np.sum(nbulk); nbulk *= Nsus
    nsed  /= np.sum(nsed);  nsed  *= Nout

    return (CrystalDistro(agrid, nbulk), CrystalDistro(agrid, nsed))


def settling_transition(
    _lmd: float, atbl: np.ndarray, ntbl: np.ndarray,  bulk_grow: float, dtaumix: np.ndarray, 
    Hnow: float, Wrms: float, amaxest: float, flux: float, nu: float, step: int, stpsSed: int, printstp: int,
    Tliq: float, Tbulk: float, Teut: float, Troof: float, distTBL: float, dtSED: float, tc: float
    
) -> tuple[CrystalDistro, CrystalDistro]:
    
    # THE  MIXING-LENGTH APPROACH (NOTE: ONLY FOR TESTING):
    wstmean = _lmd * ( np.power(atbl, 2.0) + atbl * bulk_grow * dtaumix + bulk_grow**2 * np.power(dtaumix, 2.0) / 3.0)
    wstmix  = np.maximum(wstmean - ModelParameter.gamma * Wrms, Wrms)
    Hmixnet = np.minimum(dtaumix * wstmix, Hnow)
    Hmixnet[Hmixnet < 0.0] = 1.e-2 * Hnow
    aidlecr = bulk_grow * Hmixnet / wstmix
    diffvel = wstmean - 2. * Wrms
    Shared.a0idle = np.minimum(atbl + aidlecr, Diag.astn) # NOTE: it is an array and its correct in my opinion, size (Ntbl-1)!
    if amaxest >= Diag.atrn:                                                          
        # Shrink velocity | Stokesian fall:
        if np.any(Hmixnet < 0.0):
            raise Warning("[WARNING] - Negative mixing length!")
        
        # Decide which method is appropriate:
        if np.all(Hmixnet >= Hnow): # the whole TBL is well-mixed!                
            atbl_pass = copy.deepcopy(atbl)
            #if Attributes.SCORR: atbl_pass[:] += aidlecr[:]

            Diag.Hmix = np.min(Hmixnet)
            distBLK, distSED = call_mTaCJIT(Hnow, nu, stpsSed, ntbl, atbl_pass, Wrms, tc, bulk_grow,
                                            flux, dtSED, step, printstp, Tliq, Tbulk, Troof, Teut,
                                            distTBL)
            Diag.setmarker = 2 # dashed --!

        else:
            if np.all(Hmixnet < Hnow):  # the entire TBL distribution in partially-mixed regime!
                Diag.Hmix = np.min(Hmixnet) 

                distBLK_Sf, distSED_Sf = call_mPmSfJIT(Hmixnet, ntbl, atbl, bulk_grow, _lmd, Hnow, Diag.astn, nu, Shared.dtSED, ModelParameter.gamma*Wrms, ModelParameter)
                
                #distBLK_Sf, distSED_Sf = call_mSfJIT2(atbl, ntbl, Hmixnet, Hnow, bulk_grow, Diag.astn, _lmd, ModelParameter.gamma*Wrms)
                #distBLK_Sf.scale(np.power(Shared.dtSED, -1.0))

                # Merge the distributions:
                distBLK = distBLK_Sf
                distSED = distSED_Sf
                Diag.setmarker = 3 # dotted ..!

            else: # only a chunk of the TBL distribution will be well-mixed!                    
                idtrans = np.argmin(np.abs(Hmixnet - Hnow)) # find the closest Hmix to Hnow!
                
                # Split atbl into two groups:
                if idtrans > 1: 
                    atblTaC = atbl[idtrans:]; ntblTaC = ntbl[idtrans:]
                    atblSf  = atbl[:idtrans]; ntblSf  = ntbl[:idtrans]
                    hmixbatch = Hmixnet[:idtrans]
                else:
                    atblTaC = atbl[:]; ntblTaC = ntbl[:]
                    atblSf  = np.array([]); ntblSf = np.array([])
                    hmixbatch = Hmixnet

                # Apply the size-growth correction:
                atblTaC_pass = copy.deepcopy(atblTaC)
                atblTaC_pass += aidlecr[:len(atblTaC)] #atbl[:len(atblTaC)]
                print(f"Unified settling law: {len(atblTaC):d} and partially-mixed: {len(atblSf):d}.") # <- TODO DEBUG LOGGER?

                #print(idtrans, np.min(Hmixnet), np.max(Hmixnet))
                Diag.Hmix = min(np.max(hmixbatch), Hnow)
                if len(atblSf) == 0: 
                    distBLK_TaC, distSED_TaC = call_mTaCJIT(Hnow, nu, stpsSed, ntblTaC, atblTaC_pass, Wrms, tc, bulk_grow,
                                                flux, dtSED, step, printstp, Tliq, Tbulk, Troof, Teut,
                                                distTBL)
                    distBLK = distBLK_TaC 
                    distSED = distSED_TaC 
                    Diag.setmarker = 2 # dashed --!

                else: # STOKESIAN FALL THROUGH THE ENTIRE CHAMBER!
                    if Shared.tblstokes: 
                        #distBLK_Sf, distSED_Sf = call_mSfJIT(Hnow, Hmix, bulk_grow, lmd_, nu, amaxest, ntblSf, atblSf, \
                        #                         ModelParameter.gamma*Wrms, RunConstants.nbins, Shared.tblstokes)

                        # TODO: tohle oprav!!
                        distBLK_Sf, distSED_Sf = call_mSfJIT2(atbl, ntbl, Hmixnet, Hnow, bulk_grow, Diag.astn, _lmd, ModelParameter.gamma*Wrms)

                        # Merge the distributions:
                        distBLK = distBLK_Sf
                        distSED = distSED_Sf

                    else: # split!
                        distBLK_TaC, distSED_TaC = call_mTaCJIT(Hnow, nu, stpsSed, ntblTaC, atblTaC_pass, Wrms, tc, bulk_grow, \
                                                                flux, dtSED, step, printstp, Tliq, Tbulk, Troof, Teut, distTBL
                                                                )           
                        distBLK_Sf, distSED_Sf = call_mPmSfJIT(Hmixnet, ntblSf, atblSf, bulk_grow, _lmd, Hnow, Diag.astn, nu, \
                                                               Shared.dtSED, ModelParameter.gamma*Wrms, ModelParameter
                                                            )
                        
                        #distBLK_Sf, distSED_Sf = call_mSfJIT2(atblSf, ntblSf, hmixbatch, Hnow, bulk_grow, Diag.astn, _lmd, ModelParameter.gamma*Wrms)
                        #distBLK_Sf.scale(np.power(Shared.dtSED, -1.0)) # TODO: check scaling here, ok though?

                        # Merge the distributions:
                        distBLK = distBLK_TaC + distBLK_Sf
                        distSED = distSED_TaC + distSED_Sf
                    Diag.setmarker = 3 # dotted ..!

    else: # dust-like regime: 
        atbl_pass = copy.deepcopy(atbl)
        #if Attributes.SCORR: atbl_pass[:] += aidlecr[:]

        distBLK, distSED = call_mTaCJIT(Hnow, nu, stpsSed, ntbl, atbl_pass, Wrms, tc, bulk_grow,
                                        flux, dtSED, step, printstp, Tliq, Tbulk, Troof, Teut,
                                        distTBL)
        Diag.setmarker = 2 # dashed --!

    return (distBLK, distSED)

######################################################################################################
#% end of the module!