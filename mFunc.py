# Program: ParMaCh - 1D Model of Solidification of Magma Chambers 
# Module: Functions, classes, and procedures.

import logging
import numpy as np
import warnings
import gc
import types
import time as time
import matplotlib.pyplot as plt 
from math import trunc, sqrt
from numpy.typing import NDArray
from typing import Dict, Tuple, Optional, Deque
from scipy.optimize import curve_fit, root_scalar
from scipy.integrate import quad
from numba.experimental import jitclass
from numba import njit, boolean, int32, int64, float64, prange

# Modules:
from mPar import *
from mGL import calculate_rayleigh_number
#from mPlot import plot_2D_tbl_distribution_active

# Suppress only the NumbaPendingDeprecationWarning:
from numba.core.errors import NumbaPendingDeprecationWarning 
warnings.simplefilter("ignore", NumbaPendingDeprecationWarning)

# Get logger from main.py:
logger = logging.getLogger(__name__)

""" ############################################## """
""" ######   SIMPLE FUNCTIONS/ONE-LINERS  ######## """
""" ############################################## """

@njit
def visc0_GIORDANO(T: float, Ea: float, v0: float=2.81867e-9, R: float=8.314) -> float: 
    """ Compute the dynamic viscosity after Giordano et al. (2008). """
     
    return v0 * np.exp(Ea/(R*T))  # [Pa.s] DYNAMIC VISCOSITY (!), divide by density to obtain the kinematic viscosity!

@njit
def visc0_EFF(T: float, Ea: float, phiB: float, R: float=1.67) -> float:
    """ Compute the effective (i.e., crystalinity dependence) dynamic viscosity following the Einstein-Roscoe equation. """
    
    eta_m = visc0_GIORDANO(T=T, Ea=Ea) # melt viscosity, temperature dependence!
    
    return eta_m * np.power (1. - R * phiB, -2.5)

@njit 
def lmd_stokes(nu: float, c: Parameters):
    """ Auxiliary prefactor appearing in the formula for the Stokes velocity. """
    if nu is not None:
        return (2. * c.gacc * (c.rhoc - c.rhof)) / (9. * nu * c.rhof)
    else:
        print("Undefined viscosity!")
        
@njit
def temperature_profile(Tbulk: float, Troof: float, htbl: float, z: float | np.ndarray) -> float | np.ndarray:
    """ Calculate the linear temperature profile in the TBL, return temperature at a given depth. """
    return (z*(Tbulk - Troof) / htbl) + Troof

@njit
def linear_alloy(X: float, TLmax: float, Teut: float, Xeut: float):
    # linear: y=Ax+B
    A = (TLmax - Teut) / (1.0 - Xeut)
    B = TLmax - A
    return A*X + B

@njit
def wStokes(a: float, nu: float, c: Parameters, const: float=1.0):
    """ Given crystal radius and fluid properties, calculate its Stokes velocity. """
    return const*((2.*c.gacc*(c.rhoc - c.rhof)) / (9.*c.rhof * nu))*(a**2)       

def hfce(t, A, B):
    """ Fit the time-dependent heat flux. """
    
    return (A / t**(1./2.)) * np.exp(-B / t)

@njit
def hflux_custom(t, f0, A):
    """ Custom empirically prescribed heat flux decay. """

    return f0 * np.power((1. + t/A), -1./2.)
    
""" ############################################## """
""" ######  KINETIC LAWS (GROWTH/NUCLEATION)  #### """
""" ############################################## """

@njit
def pow_nucl(Tc: float | np.ndarray, Tliq: float | np.ndarray, Tref: float, c: Parameters, norm: bool=False, p: int=1): 
    """ Dimensional nucleation power law. """

    N0 = c.N0POW; norma = 1.0
    if norm: N0 = 1.0
    return N0 * np.power( (Tliq - c.epsdel - Tc) / (c.Tliqd0 - c.epsdel - Tref), p) / norma

@njit
def pow_grow(Tc: float, Tliq: float, Tref: float, c: Parameters, norm: bool=False, q: int=1):
    """ Dimensional growth power law. """

    V0 = c.V0POW; norma = 1.0
    if norm: V0 = 1.0
    return V0 * np.power(((Tliq - Tc) / (c.Tliqd0 - Tref)), q) * norma

@njit
def Hort_nucl(T: float | np.ndarray, Tliq: float | np.ndarray, c: Parameters, norm: bool=False, Tratio: bool=False):
    """ Hortian nucleation law (Hort 1997). """

    N0 = c.N0_HG97
    Tg = c.Tg_HG97
    Ti = c.Ti_HG97
    if norm: N0 = 1.0
    if Tratio:
        Trat = T / Tliq
        # NOTE: specifically for the ratio, only auxiliary for the mPlot.py module:
        result = N0 * np.exp((Tg/(1. - Tg))*(1./Ti - 1./Trat - (((1. - Ti)**3)/(1. - 3.*Ti))
            *(1./(Ti*(1. - Ti)**2) - 1./((Trat)*(1. - Trat)**2))))
    else:
        # Just plug in:
        result = N0 * np.exp((Tg/(1. - Tg))*(1./Ti - Tliq/T - (((1. - Ti)**3)/(1. - 3.*Ti))
            *(1./(Ti*(1. - Ti)**2) - 1./((T/Tliq)*(1. - T/Tliq)**2))))

    return np.where(result > 0.0, result, 0.0)
    
@njit
def Hort_grow(T: float | np.ndarray, Tliq: float | np.ndarray, c: Parameters, norm: bool=False):
    """ Hortian growth law (Hort 1997). """

    V0 = c.V0_HG97
    Tg = c.Tg_HG97
    if norm: V0 = 1.0
    result = V0 * (Tg*(Tliq - T)/(T*(1. - Tg)))*np.exp(-(Tliq*(Tg - T/Tliq))/(T*(1. - Tg)))
    return np.where(result > 0.0, result, 0.0)

@njit
def pow_nucl_dimless(u: float, epsdel: float, p: int=1): # u = Tliqd - Troof!
    """ Dimensionless nucleation power law (Jarvis & Woods). """

    if u <= epsdel:
        return 0.
    else:
        return ((u - epsdel) / (1. - epsdel))**p

@njit
def pow_grow_dimless(u: float, q: int=1): # u = Tliqd - Tbulk!
    """ Dimensionless growth power law  (Jarvis & Woods). """
    return u**q

@njit
def ndimless(chi: float, Tbulk: float, Troof: float, Tliqd: float, dpile: float, epsdel: float):
    """ Auxiliary dimensionless nucleation per unit volume of the chamber. """
    return chi**(1./3.) * (Tbulk - Troof)**(-1./3.) * pow_nucl_dimless(u=(Tliqd - Troof), epsdel=epsdel) / (1. - dpile)

def avg_growth(Tbulk: float, Troof: float, Tliqd: float):
    """ Auxiliary function to calculate the mean crystal growth rate across the boundary layer """
    
    avg_temp = (Tbulk + Troof) / 2.
    if Attributes.NG_METHOD == 1: 
        return pow_grow(avg_temp, Tliqd, Shared.Tref, ModelParameter)
    elif Attributes.NG_METHOD == 2: 
        return Hort_grow(avg_temp, Tliqd, ModelParameter)

def bulk_growth(Tbulk: float, Tliqd: float):
    """ Computes the bulk growth within the convecting fluid. """

    if Attributes.NG_METHOD == nMethod.mLin:
        return pow_grow(Tbulk, Tliqd, Shared.Tref, ModelParameter)
    elif Attributes.NG_METHOD == nMethod.mLab:
        return Hort_grow(Tbulk, Tliqd, ModelParameter)

def return_epsdel(Tliqd: float, Ncut: float=1.e-8) -> float:
    """ Find the numerical value (magnitude) of the nucleation lag """

    find_tnuc = lambda hnuc, ncrit : (np.abs(hnuc - ncrit)).argmin()  
    Ttmp = np.linspace(0.9*Tliqd, Tliqd - RunConstants.TINY, num=500)
    hnuc = Hort_nucl(Ttmp, Tliqd, c=ModelParameter, norm=True)
    _idxnM = find_tnuc(hnuc, Ncut) # mathematical nucleation threshold (must be numerically feasiable!)
    epsdel = Tliqd - Ttmp[_idxnM]

    return epsdel

""" ############################################## """
""" ######    CRYSTAL-BASED CLASSES (CSDs)    #### """
""" ############################################## """

class CrystalBatch:                                                         
    """ Crystal class for storing a family of crystals (radius, residence time, population, etc.) """

    def __init__(self, a, K, z, indep):     
        # Instance characteristics:
        self.radius   = a                     # initial radius 
        self.inidep   = indep                 # nucleation depth (measured from the roof)
        self.zcoord   = z                     # z-coordinate of the crystal family
        self.count    = K                     # number of crystals in the batch
        self.rtimetbl = 0.0                   # residence time in the TBL 
        self.rtimeblk = 0.0                   # residence time in the bulk
        self.rtimedst = 0.0                   # residence time in the dust-like regime
        self.hsettle  = 0.0                   # settling front height measured from the chamber roof

        # Flags:
        self.nucled = False                   # Q: batch nucleated?       (yes/no)
        self.tblpas = False                   # Q: batch fell into bulk?  (yes/no)
        self.sedout = False                   # Q: batch settled?         (yes/no)

    def __str__(self):
        print("Crystal batch with %.3f crystals and radius of %.3f %s nucleated at depth %.5f %s." % 
              (self.count, self.radius, Units.sunit, self.inidep, Units.sunit))
        print("Current z-coordinate: %.5f" % self.zcoord)

    def CrystalNucl(self, Tc, Tliq, dz, dt, NG_METHOD): 
        """ Nucleation of the batch. """

        if self.nucled: 
            raise Exception("This batch has nucleated already!")
        else:
            match(NG_METHOD):
                case nMethod.mLin: 
                    self.count = dz * dt * pow_nucl(Tc, Tliq, Shared.Tref, ModelParameter) 
                case nMethod.mLab:
                    self.count = dz * dt * Hort_nucl(Tc, Tliq, ModelParameter)        
            self.nucled = True

    def CrystalGrow(self, Tc, Troof, Tliq, dt, NG_METHOD):
        """ Crystal radius increment. """

        match(NG_METHOD): 
            case nMethod.mLin: 
                self.radius += dt * pow_grow(Tc, Troof, Tliq)
            case nMethod.mLab:                    
                self.radius += dt * Hort_grow(Tc, Tliq, ModelParameter) 

    def CrystalShift(self, dt, Wrms, nu, vsh=False):
        """ Stokesean shift. """

        if vsh:
            self.zcoord += (wStokes(self.radius, nu, ModelParameter) - Wrms) * dt
        else:
            self.zcoord += wStokes(self.radius, nu, ModelParameter) * dt

class CrystalDistro:
    """ Crystal size distribution (CSD/CSF) class. """

    def __init__(self, adist, ndist):
        self.adist = np.array(adist)
        self.ndist = np.array(ndist)

    def norm(self, boost=1.e8):
        """ Normalize the distribution. """
        
        try:
            self.ndist = self.ndist / np.sum(self.ndist) 
        except RuntimeWarning or ZeroDivisionError:
            result = np.sum(self.adist * boost * self.ndist) / np.sum(boost * self.ndist)
            return result
        
    def scale(self, sfac):
        """ Rescale the distribution by a scale factor. """
        self.ndist = self.ndist * sfac
        
    def amean(self, boost=1.e8):
        """ First moment (mean radius). """
        try:
            result = np.sum(self.adist * boost * self.ndist) / np.sum(boost * self.ndist)
        
        except ZeroDivisionError:
            result = np.sum(self.adist * boost * self.ndist) / np.sum(boost * self.ndist)
        
        return result
    
    def amax(self):
        """ Maximum radius. """

        return np.max(self.adist)
    
    def amin(self):
        """ Minimum non-zero radius. """

        try:
            return self.adist[self.ndist > 0].min()
        
        except ValueError:
            return self.adist[0]

    def amoment2(self):
        """ Second moment of the distribution: spread, how much "weight" away from the origin. """

        return np.sum( (self.adist - self.amean() )**2 * self.ndist) / np.sum(self.ndist)    
    
    def amoment3(self):
        """ Third moment of the distribution: skewness. """

        return np.sum(self.adist**2 * self.ndist) / np.sum(self.ndist)
    
    def da(self):
        """ Returns the bin size. """

        return abs(self.adist[1] - self.adist[0])
    
    def der(self):
        """ Calculate the first derivative of the distribution. """

        return CrystalDistro(adist=self.adist[1:], ndist=( np.abs(self.ndist[1:] - self.ndist[:-1]) / (1. * self.da() )))

    def __add__(self, other, bins=RunConstants.nbins):
        """ Define the "+" operator on the CrystalDistro """

        print(np.sum(self.ndist)+np.sum(other.ndist))

        amax = max(np.max(self.adist), np.max(other.adist))
        amin = min(np.min(self.adist), np.min(other.adist))
        a_sum = np.linspace(amin, amax, num=bins)
        n_sum = np.zeros(bins, dtype=float)
        da = a_sum[1] - a_sum[0]

        for i in prange(len(self.ndist)):
            ix = int(self.adist[i] / da)
            n_sum[ix] += self.ndist[i]

        for i in prange(len(other.ndist)):
            ix = int(other.adist[i] / da)
            n_sum[ix] += other.ndist[i]
    
        print(np.sum(n_sum))

        return CrystalDistro(a_sum, n_sum)

class CrystalDistro2D(CrystalDistro):
    """ Auxiliary class for handling 2D TBL distributions """
    
    def __init__(self, adist, ndist, zdist, Tbulk, Troof, htbl):
        super().__init__(adist, ndist)
        self.zdist = zdist
        self.Troof = Troof
        self.Tbulk = Tbulk
        self.htbl  = htbl

    def tmp_profile(self, z):
        return temperature_profile(
            Tbulk=self.Tbulk, Troof=self.Troof, htbl=self.htbl, z=z
        )

    def dz(self):
        return abs(self.zdist[0] - self.zdist[1])

def comp_hist(phi: tuple[float, float], amin: float, amax: float, bins: int, adjust: int=1.0001) -> CrystalDistro:
    """ From phi (tuple of pairs radius "a" vs. count "#"), compute a non-normalized histogram (CSD). """

    assert bins < ModelParameter.Ntbl, "Increase discretization of the nucleation sublayer!" 
    amax *= adjust
    da = (amax - amin) / bins
    nbins = np.zeros(bins, dtype=float)
    for pair in phi:
        if isinstance(pair[0], tuple): # just a possible bug prevention from mPhase.py!
            print(f"Warning! Tuple {pair[0]}!")
            raise TypeError("Check your implementation!")
        
        # Place crystals into the corresponding bin!
        pos = int((pair[0] - amin) / da) 
        if pos == bins: 
            nbins[pos-1] += pair[1]
            print(f"[CompHist WARNING]: {pair[0]:.3e}, {amax:.3e}!")
            raise Exception("One of the radii is larger than amax!")
        nbins[pos] += pair[1]
    abins = np.linspace(amin + da/2., amax - da/2., num=bins, dtype=float) 

    return CrystalDistro(abins, nbins)

""" ############################################## """
""" ######        PETROLOGICAL SECTION        #### """
""" ############################################## """

class BinaryAlloy:
    """ Binary alloy, described analytically """
    def __init__(self, Tmax1: float, Tmax2: float, Teut: float, X0: float, Xeut: float, Dm1: float, Dm2: float) -> None:
        self._Tmax1 = Tmax1
        self._Tmax2 = Tmax2
        self._Teut  = Teut
        self._Xeut  = Xeut
        self._X0    = X0 
        self._Dm1   = Dm1
        self._Dm2   = Dm2
        self._Lheat = None

        if self._X0 > self._Xeut:
            self._Lheat = Dm1 / ModelParameter.M_an # convert to [J/kg]!
        else:
            self._Lheat = Dm2 / ModelParameter.M_di # convert to [J/kg]!

    # Beware, X only denotes composition of one of the principal elements!
    def branch_A(self, X) -> float:
        """ Return the liquidus temperature on the first branch """

        # Molar fraction of anorthite computed from the melt composition (in wt% An):
        XAn = ((X / ModelParameter.M_an)) / ( (X / ModelParameter.M_an) + ((1. - X) / ModelParameter.M_di) ) # unitless!

        return self._Dm1 / ( self._Dm1 / self._Tmax1 - ModelParameter.gasR * np.log(XAn) ) 
    
    def branch_B(self, X) -> float:
        """ Return the liquidus temperature on the second branch """

        # Molar fraction of anorthite:
        XAn = ((X / ModelParameter.M_an)) / ( (X / ModelParameter.M_an) + ((1. - X) / ModelParameter.M_di) ) # unitless!

        return self._Dm2 / ( self._Dm2 / self._Tmax2 - ModelParameter.gasR * np.log(1. - XAn) )


""" ############################################## """
""" #####  AUXILIARY NJIT-POWERED FUNCTIONS  ##### """
""" ############################################## """

@njit
def linspace_numba(start, stop, num):
    """ @njit equivalent of np.linspace """
    result = np.empty(num, dtype=float64)
    stp = (stop - start) / (num - 1)
    for i in range(num): result[i] = start + i*stp
    return result

@njit 
def comp_histJIT(historyArray, amin, amax, bins, xpar, adjust=1.0001):
    """ Use in the mTaC solver: histogram-generating function (crystals -> bins) """
    amax *= adjust
    da = (amax - amin) / bins
    nbins = np.zeros(bins, dtype=float64)
    abins = linspace_numba(amin + da/2., amax - da/2., num=bins)
    N = int( historyArray.shape[1] / xpar) # = ModelParameter.Ntbl-1
    
    # Loop over individual records:
    for i in prange(historyArray.shape[0]):
        for j in range(N):
            tmpn, tmpa = historyArray[i,j], historyArray[i,j+N]
            # Place the crystal in the corresponding bin:
            pos = int(( tmpa - amin) / da) 
            nbins[pos] += tmpn

    return (abins, nbins)

@njit
def crystal_shift(a0: float, z0: float, ng: int, Tref: float, const: object,
                  htbl: float, Tbulk: float, Troof: float, Tliq: float, dtTBL: float,
                  tplume: float, nutbl: bool=False, ff: bool=True) -> tuple: 
    """ Crystal shifting function for the TBL dynamics """
    rtimetbl = 0.0; arad = a0; zcoord = z0
    
    # Full gravitational extraction:
    if ff: 
        while zcoord < htbl:
            T = temperature_profile(Tbulk, Troof, htbl, zcoord)
            if nutbl:
                _nu = visc0_GIORDANO(T, const.EaMod) / const.rhof
            else:
                _nu = visc0_GIORDANO(Tbulk, const.EaMod) / const.rhof  
            if ng == 1: 
                arad += pow_grow(T, Tliq, Tref, const) * dtTBL
            elif ng == 2: 
                arad += Hort_grow(T, Tliq, const) * dtTBL
            zcoord += wStokes(arad, _nu, const) * dtTBL
            rtimetbl += dtTBL

    else:
        while rtimetbl <= tplume:
            T = temperature_profile(Tbulk, Troof, htbl, zcoord)
            if nutbl:
                _nu = visc0_GIORDANO(T, const.EaMod) / const.rhof
            else:
                _nu = visc0_GIORDANO(Tbulk, const.EaMod) / const.rhof  
            if ng == 1: 
                arad += pow_grow(T, Tliq, Tref, const) * dtTBL
            elif ng == 2: 
                arad += Hort_grow(T, Tliq, const) * dtTBL
            zcoord += wStokes(arad, _nu, const) * dtTBL
            rtimetbl += dtTBL

    return (arad, zcoord, rtimetbl)

#######################################################################
#####           %%   TRACKING OF ALL CRYSTALS METHOD   %%         #####
#####                    FUNCTIONS (NJIT POWERED)                 #####
#######################################################################

@njit
def vert_distBLK2D(a, K, volume, H, n_z_bins=50):
    """ Computes the vertical CSD of the crystals within the bulk """
    N  = len(a)
    dz = H / n_z_bins
 
    z_centers = np.empty(n_z_bins)
    for k in range(n_z_bins):
        z_centers[k] = (k + 0.5) * dz
 
    hist = np.zeros((N, n_z_bins))
 
    for i in range(N):
        for k in range(n_z_bins):
            z_lo = k * dz
            z_hi = z_lo + dz
            overlap = max(0.0, min(z_hi, volume[i]) - z_lo)
            if volume[i] == 0.0: continue
            hist[i, k] = K[i] * overlap / volume[i]
 
    return (a, z_centers, hist)

@njit
def vert_distBLK2D_NEW(a, K, Htop, Hbot, H, bins):
    """ Computer the vertical CSD of the suspended crystals within the bulk """
    
    # 2D histogram:
    z2d = np.linspace(0, H, num=bins)
    a2d = np.linspace(0, np.max(a), num=bins)
    n2D = np.zeros( (len(z2d), len(a2d)), dtype=np.float64)

    ia_all = np.searchsorted(a2d, a, side="right") - 1
    for i in range(len(K)): 
        iK = K[i]; ia = ia_all[i]
        
        # Go over 
        bot = Hbot[i]; top = Htop[i]; vol = (bot - top)        

        # Pour crystals into the 2D histogram:
        overlap_bot = np.maximum(z2d[:-1], bot)
        overlap_top = np.minimum(z2d[1:], top)
        overlap     = np.maximum(0.0, overlap_top - overlap_bot)

        if vol == 0.0: continue
        n2D[:, ia] += iK * overlap / vol
    
    print("CHECK:", np.sum(n2D), np.sum(K))

    return (a2d, z2d, n2D)

@njit
def wShrink(a, Wrms, nu, c):
    """ Compute the shrinking velocity. """
    condition = wStokes(a, nu, c) - c.gamma * Wrms
    result = np.where(condition > 0.0, condition, 0.0)
    return result

@njit
def check_steady_state(sedHold, sed, eps):
    diff = np.abs(sedHold - sed) 
    condition = np.all(diff < eps)   
    if condition:
        return True  
    return False

# JIT-powered tracking method (implemeted solely though numpy without OOP)
@njit(boundscheck=NUMBA_BOUNDSCHECK)
def mTaCJIT(
        nu: float, stps: int,
        ntbl: np.ndarray, atbl: np.ndarray, const: object, bulk_growth: float,
        dtSED: float, Hnow: float, atrn: float, Wrms: float, bins: int, 
        tc: float, nblk_prev: Optional[np.ndarray]=None, ablk_prev: Optional[np.ndarray]=None,
        MaN: bool=False, TBLon: bool=True, tccut: bool=False, eps: float=8e-6, stop: int=1500

    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """ Tracking method of bulk processes powered by njit. 
    
            > trackHistory: np.ndarray

                range:      stps x (4*N)
                [:N]        crystal count
                [N:2*N]     crystal radius
                [2*N:3*N]   residence time upon exceeding the transitional radius
                [3*N:4*N]   advancement of the settling front

            Returns: 
            > distBLK:      tuple(np.array, np.array)
            > distSED:      tuple(np.array, np.array)       
            > trackHistory: np.array(stps, 4*N)  
            > ss_time:      float       
            > sedoutArray:  np.array(stps, 4*N) 
    """        

    # Beware: it can either be (Ntbl-1) or (nbins)!
    xpar = 4
    N = ntbl.shape[0]  
    trackHistory = np.zeros((stps, 4*N), dtype=np.float64)
    trackedArray = np.zeros(4*N, dtype=np.float64)
    sedoutArray = np.zeros((stps, 4*N), dtype=float64)
    trackedArray[:N] = ntbl
    trackedArray[N:2*N] = atbl
    sedHold = np.zeros(bins, dtype=np.float64)
    ss_time = stps * dtSED
    cbulksum = 0.0; csedsum  = 0.0

    if not TBLon: # NOTE: TBL dynamics off, crystals extracted with zero radius!
        trackedArray[:N] = 0.0          
        trackedArray[0]  = np.sum(ntbl)

    # Time loop - calculating the correction:
    for i in range(stps):
        if i == stps - 1: print("[WARNING] Function mTaCJIT warning: STEADY STATE NOT REACHED!")
        # Save the snapshot:
        trackHistory[i,:N] = trackedArray[:N]
        trackHistory[i,N:2*N] = trackedArray[N:2*N]   
        trackHistory[i,2*N:3*N] = trackedArray[2*N:3*N] 
        trackHistory[i,3*N:4*N] = trackedArray[3*N:4*N]    
        # Growth of crystals:
        trackedArray[N:2*N] += np.where(0.0 < trackedArray[:N], bulk_growth*dtSED, 0.0)
        # Sedimentation & make sure hsettle does not exceed height of the chamber:  
        trackedArray[:N] = np.where(trackedArray[3*N:4*N] >= Hnow, 0.0, trackedArray[:N])
        trackedArray[:N] *= (1.0 - wStokes(trackedArray[N:2*N], nu, const) * dtSED  \
                            / (Hnow - trackedArray[3*N:4*N])
                            )
        # Numerical reasons - prevents the rare occurence of the exploding dominator:
        trackedArray[:N] = np.where(trackedArray[:N] < 0.0, 0.0, trackedArray[:N])
        
        # Transitional radius exceedance bulk residence time:           
        if MaN: 
            for j in range(N):
                if trackedArray[N+j] >= atrn and trackedArray[j] > 0.0:
                    trackedArray[2*N+j] += dtSED
                    trackedArray[3*N+j] += wShrink(trackedArray[N+j], Wrms, nu, const) * dtSED

        # Crystal sum:        
        cbulksum += np.sum(trackHistory[i,:N])

        # Check for the steady state:
        if i % stop == 0:
            isp = i / stop
            dcrmnt = trackHistory[(isp-1)*stop:isp*stop, :N] * wStokes(trackHistory[(isp-1)*stop:isp*stop, N:2*N], nu, const) \
                * dtSED / (Hnow - trackHistory[(isp-1)*stop:isp*stop, 3*N:4*N])
            dcrmnt = np.where(dcrmnt > trackHistory[(isp-1)*stop:isp*stop, :N], np.minimum(dcrmnt, trackHistory[(isp-1)*stop:isp*stop, :N]), dcrmnt)
            sedoutArray[(isp-1)*stop:isp*stop, :N] = dcrmnt
            sedoutArray[(isp-1)*stop:isp*stop, N:2*N] = trackHistory[(isp-1)*stop:isp*stop, N:2*N]
            csedsum += np.sum(sedoutArray[(isp-1)*stop:isp*stop, :N])
            
            if abs(csedsum - np.sum(ntbl)) / np.sum(ntbl) < eps:
                amaxblk = np.max(trackHistory[:,N:2*N])
                amaxsed = np.max(sedoutArray[:,N:2*N])
                distBLK = comp_histJIT(trackHistory, 0.0, amaxblk, bins, xpar=xpar)
                distSED = comp_histJIT(sedoutArray, 0.0, amaxsed, bins, xpar=xpar)
                ss_time = i * dtSED
                break
          
    return (distBLK, distSED, trackHistory, ss_time, sedoutArray, None, None)

# JIT-powered tracking method (implemeted solely though numpy without OOP)
@njit(boundscheck=NUMBA_BOUNDSCHECK)
def mTaCJIT_UNIFIED(
        nu: float, stps: int,
        ntbl: np.ndarray, atbl: np.ndarray, const: object, bulk_growth: float,
        dtSED: float, Hnow: float, atrn: float, Wrms: float, bins: int, 
        tc: float, nblk_prev: Optional[np.ndarray]=None, ablk_prev: Optional[np.ndarray]=None,
        #track_prev: Optional[np.ndarray]=None,
        MaN: bool=False, TBLon: bool=True, tccut: bool=False, eps: float=1.e-3, stop: int=2000

    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """ Tracking method of bulk processes powered by njit. 
    
            %> trackHistory: np.ndarray

                range:      stps x (4*N)
                [:N]        crystal count
                [N:2*N]     crystal radius
                [2*N:3*N]   residence time upon exceeding the transitional radius
                [3*N:4*N]   TODO
                [4*N:5*N]   cloud upper-boundary
                [5*N:6*N]   cloud bottom-boundary

            Returns: 
                %> distBLK:      tuple(np.array, np.array)
                %> distSED:      tuple(np.array, np.array)       
                %> distBLK2D:    tuple(np.array, np.array, np.array) TODO
                %> trackHistory: np.array(stps, 4*N)  
                %> ss_time:      float       
                %> sedoutArray:  np.array(stps, 4*N) 
    """        

    xpar = 6
    N = ntbl.shape[0]  
    trackHistory = np.zeros((stps, xpar*N), dtype=np.float64)
    trackedArray = np.zeros(xpar*N, dtype=np.float64)
    sedoutArray  = np.zeros((stps, xpar*N), dtype=np.float64)
    trackedArray[:N] = ntbl
    trackedArray[N:2*N] = atbl

    sedHold = np.zeros(bins, dtype=np.float64)
    ss_time = stps * dtSED
    cbulksum = 0.0; csedsum = 0.0

    if not TBLon: # NOTE: TBL dynamics off, crystals extracted with zero radius!
        trackedArray[:N] = 0.0          
        trackedArray[0]  = np.sum(ntbl)
    distBLK = None; distSED = None
    ff = True; lf = True; idle = True; printclouddown = False; printcloudupper = False

    # Time loop - calculating the correction:
    for i in range(stps):
        if i == stps - 1:
            print(" [WARNING] - Function mTaCJIT_UNIFIED warning: Steady-state not reached!")

        # Save the snapshot:
        trackHistory[i,:N]      = trackedArray[:N]
        trackHistory[i,N:2*N]   = trackedArray[N:2*N]   
        trackHistory[i,2*N:3*N] = trackedArray[2*N:3*N] 
        trackHistory[i,3*N:4*N] = trackedArray[3*N:4*N]    
        trackHistory[i,4*N:5*N] = trackedArray[4*N:5*N]
        trackHistory[i,5*N:6*N] = trackedArray[5*N:6*N]

        # Growth of crystals:
        trackedArray[N:2*N] += np.where(0.0 < trackedArray[:N], bulk_growth*dtSED, 0.0)

        # Shift the cloud upper-boundary:
        trackedArray[4*N:5*N] += dtSED * np.maximum(wShrink(trackedArray[N:2*N], Wrms, nu, const), 0.0) 
        trackedArray[4*N:5*N] = np.minimum(trackedArray[4*N:5*N], 0.999*Hnow)

        """
        if np.any(trackedArray[4*N:5*N] > 0.0) and not printcloudupper:
            print("Cloud descending!")
            print(i*dtSED)
            printcloudupper = True
        """
            
        # Shift the cloud bottom-boundary:
        trackedArray[5*N:6*N] += dtSED * (Wrms + wStokes(trackedArray[N:2*N], nu, const))
        #trackedArray[5*N:6*N] += dtSED * np.maximum(Wrms, wStokes(trackedArray[N:2*N], nu, const))
        trackedArray[5*N:6*N] = np.minimum(trackedArray[5*N:6*N], Hnow)

        """
        if np.all(trackedArray[5*N:6*N] >= Hnow) and not printclouddown:
            print("Cloud hit the bottom!")
            print(i*dtSED)
            printclouddown = True
        """
            
        """
        if np.any(trackedArray[5*N:6*N] >= Hnow) and ff:
            print("First family hit the bottom: ", i, "upper boundary: ", np.min(trackedArray[4*N:5*N]))
            ff = False

        if np.all(trackedArray[5*N:6*N] >= Hnow) and lf:
            print("All families hit the bottom: ", i, "upper boundary: ", np.min(trackedArray[4*N:5*N]))
            lf = False
        """
            
        # NOTE: Sedimentation:
        # > it is proportional to the mean concentration at the bottom of the chamber!
        # > it can only occur when Hbot >= Hnow!
        trackedArray[:N] = np.where(
            trackedArray[5*N:6*N] >= Hnow, # where Hbot already hit the bottom!
            trackedArray[:N] * (1.0 - wStokes(trackedArray[N:2*N], nu, const) * dtSED \
                    / np.abs(trackedArray[4*N:5*N] - trackedArray[5*N:6*N])), trackedArray[:N]) #
        
        trackedArray[:N] = np.where(trackedArray[4*N:5*N] >= Hnow, 0.0, trackedArray[:N]) 
        trackedArray[:N] = np.where(trackedArray[:N] < 0.0, 0.0, trackedArray[:N]) 

        if np.all(trackedArray[5*N:6*N] >= Hnow): idle = False

        """
        if i < 100:
            M = 0
            print(
                "Krok:",     i,
                "Populace:", trackedArray[0+M],
                "Poloměr:",  trackedArray[N+1+M],
                "Shrink:",   wShrink(trackedArray[N+1+M], Wrms, nu, const),
                "Horní:",    trackedArray[4*N+1+M],
                "Dolní:",    trackedArray[5*N+1+M]
            )
        """
                    
        # Crystal sum at time t:        
        cbulksum += np.sum(trackHistory[i,:N])

        # Check for the steady state:
        if i > 0 and i % stop == 0:
            isp = int(i / stop)
            dcrmnt = np.where(trackHistory[(isp-1)*stop:isp*stop, 5*N:6*N] >= Hnow,
                        trackHistory[(isp-1)*stop:isp*stop, :N] * wStokes(trackHistory[(isp-1)*stop:isp*stop, N:2*N], nu, const) * dtSED \
                     / np.abs(trackHistory[(isp-1)*stop:isp*stop, 4*N:5*N] - trackHistory[(isp-1)*stop:isp*stop, 5*N:6*N]), 0.0)

            # You cant settle more than you have:
            dcrmnt = np.where(dcrmnt > trackHistory[(isp-1)*stop:isp*stop, :N], np.minimum(dcrmnt, trackHistory[(isp-1)*stop:isp*stop, :N]), dcrmnt) 

            sedoutArray[(isp-1)*stop:isp*stop, :N] = dcrmnt
            sedoutArray[(isp-1)*stop:isp*stop, N:2*N] = trackHistory[(isp-1)*stop:isp*stop, N:2*N]
            csedsum += np.sum(sedoutArray[(isp-1)*stop:isp*stop, :N])

            if abs(csedsum - np.sum(ntbl)) / np.sum(ntbl) < eps:
                amaxblk = np.max(trackHistory[:,N:2*N])
                amaxsed = np.max(sedoutArray[:,N:2*N])
                distBLK = comp_histJIT(trackHistory, 0.0, amaxblk, bins, xpar=xpar)
                distSED = comp_histJIT(sedoutArray, 0.0, amaxsed, bins, xpar=xpar)
                ss_time = i * dtSED
                break    

    if distBLK is None or distSED is None:
        amaxblk = np.max(trackHistory[:,N:2*N])
        amaxsed = np.max(sedoutArray[:,N:2*N])
        distBLK = comp_histJIT(trackHistory, 0.0, amaxblk, bins, xpar=xpar)
        distSED = comp_histJIT(sedoutArray, 0.0, amaxsed, bins, xpar=xpar)
        ss_time = i * dtSED

    # It should trigger pretty much always (idle-mixing!):
    
    if np.any( (trackHistory[:,4*N:5*N] - trackHistory[:,5*N:6*N]) < 0.99*Hnow): 
        #volume = ( trackHistory[:,5*N:6*N] - trackHistory[:,4*N:5*N] )
        #adist2D, zdistBLK2D, ndistBLK2D = vert_distBLK2D(
        #    a=trackHistory[:,N:2*N].flatten(), K=trackHistory[:,:N].flatten(), volume=volume.flatten(), \
        #    H=Hnow, n_z_bins=bins
        #)

        # TODO: does .flatten do what it is supposed to do?
        #adist2D, zdistBLK2D, ndistBLK2D = vert_distBLK2D_NEW(
        #    a=trackHistory[:,N:2*N].flatten(), K=trackHistory[:,:N].flatten(), Htop=trackHistory[:,4*N:5*N].flatten(), \
        #    Hbot=trackHistory[:,5*N:6*N].flatten(), H=Hnow, bins=bins
        #)

        """
        zdistBLK2D, ndistBLK2D = vert_distBLK(
                K=trackHistory[:,:N].flatten(),
                a=trackHistory[:,N:2*N].flatten(),
                Htop=trackHistory[:,4*N:5*N].flatten(),
                Hbot=trackHistory[:,5*N:6*N].flatten()
            )
        """
        #if 1.e2 * abs( (np.sum(trackHistory[:,:N]) - np.sum(ndistBLK2D)) / np.sum(trackHistory[:,:N]) ) > 10.0:
        #    print("[WARNING] - Error greater than 10% in vertically distributed crystals!")
        pass
    
    adist2D, zdistBLK2D, ndistBLK2D = None, None, None
    return (distBLK, distSED, trackHistory, ss_time, sedoutArray, adist2D, zdistBLK2D, ndistBLK2D)


def call_mTaCJIT(
    Hnow: float, nu: float, stpsSed: int, ntbl: np.ndarray, atbl: np.ndarray,
    Wrms: float, tc: float, bulk_grow: float, flux: float, dtSED: float,
    step: int, printstp: int, Tliq: float, Tbulk: float, Troof: float, Teut: float,  # TODO: useless arguments, get rid of them!
    distTBL: CrystalDistro, con_unif: bool=True

) -> tuple[CrystalDistro, CrystalDistro]: # FIXME: asi nefunguje pro bulk_grow = 0?
    """ Auxiliary: mTaCJIT call in &SED_phase """

    N = len(ntbl) # TODO: remove?
    if con_unif:
        pblk, psed, track, ss_time, ssT, adist2D, zdist2D, ndist2D = mTaCJIT_UNIFIED(
                                        nu=nu, stps=stpsSed, ntbl=ntbl, atbl=atbl, const=ModelParameter, 
                                        bulk_growth=bulk_grow, dtSED=Shared.dtSED, Hnow=Hnow, atrn=Diag.atrn,
                                        Wrms=Wrms, bins=RunConstants.nbins, tc=tc, nblk_prev=None, ablk_prev=None,
                                        MaN=Attributes.TBL, TBLon=Attributes.TBL, tccut=False # TODO: MaN = Attributes.TBL or Attributes.JWLIMIT
                                    )
    else:
        pblk, psed, track, ss_time, ssT, _, _ = mTaCJIT( # FIXME: what did I want to fix?
                                        nu=nu, stps=stpsSed, ntbl=ntbl, atbl=atbl, const=ModelParameter, 
                                        bulk_growth=bulk_grow, dtSED=Shared.dtSED, Hnow=Hnow, atrn=Diag.atrn,
                                        Wrms=Wrms, bins=RunConstants.nbins, tc=tc, nblk_prev=None, ablk_prev=None,
                                        MaN=Attributes.TBL, TBLon=Attributes.TBL, tccut=False # TODO: MaN = Attributes.TBL or Attributes.JWLIMIT
                                    )         
    # TODO: make that fucking 2D distribution...

    #fig, ax = plt.subplots()
    #pcm = ax.pcolormesh(
    #    zdist2D, adist2D, ndist2D,            
    #    cmap="plasma", shading="auto"
    #)
    #cbar = fig.colorbar(pcm, ax=ax)
    #plt.show()

    # Auxiliary filtering: 
    pblk = np.array(pblk)   
    psed = np.array(psed)
    pnsed = psed[1] #[psed[1] > tol]
    pnblk = pblk[1] #[pblk[1] > tol]
    pased = psed[0][:len(pnsed)]
    pablk = pblk[0][:len(pnblk)]

    # Build the distribution objects:    
    distBLK   = CrystalDistro(pablk, pnblk)
    distSED   = CrystalDistro(pased, pnsed)

    # print(np.sum(pnblk)) = print(np.sum(n2D))
    
    #print(pablk, zdist2D.shape, ndist2D.shape)
    #plt.pcolormesh(zdist2D, adist2D, ndist2D)
    #plt.show()


    """
    n2D, a2D, z2D = np.histogram2d(
        pablk, zdist2D, 
        bins=[30, 30],
        weights=
    )
    """

    #n2D, a2D, z2D = np.histogram2d(
    #    pablk, zdist2D, # <--- pablk, zdist2D must be longer arrays!
    #    bins=[pablk, zdist2D],
    #    weights=ndist2D
    #)
    
    #distBLK2D = CrystalDistro2D(
    #    adist=a2D, zdist=z2D, ndist=n2D, Tbulk=Tbulk, Troof=Troof, htbl=Hnow
    #)

    #plt.plot(zdist2D, ndist2D)
    #plt.show()
    #print(zdist2D[0], ndist2D[-1])

    # Save the previous tracked array: 
    Shared.track = track # FIXME: Do I need it?

    # Crystals in/out:
    #logger.info(f"INFLUX,  {Diag.cnt0DMC * Shared.dtSED / Shared.dtTBL:.3e}.") 
    #logger.info(f"OUTFLUX, {np.sum(ssT[:,:len(ntbl)]):.3e}.")     
    #print(f"INFLUX, {Diag.cnt0DMC * Shared.dtSED / Shared.dtTBL:.3e}.") 
    #print(f"OUTFLUX, {np.sum(ssT[:,:len(ntbl)]):.3e}.")                  
    Diag.cin  = Diag.cnt0DMC * Shared.dtSED / Shared.dtTBL
    Diag.cout = np.sum(ssT[:,:len(ntbl)])

    # Create crystal distribution data objects:
    #distSED = CrystalDistro(pased, pnsed)
    #distBLK = CrystalDistro(pablk, pnblk)

    #print("#", np.sum(distBLK.ndist))

    # Steady-state time scale:
    Diag.tssblk = ss_time
    ntime = dtSED * np.arange(0, len(track[0,:]))

    # Plot tracking history:
    # track: array
    """
        [:N]        crystal count
        [N:2*N]     crystal radius
        [2*N:3*N]   residence time upon exceeding the transitional radius
        [3*N:4*N]   advancement of the settling front
    """

    # Save each printstp-th distribution to see how it evolves with time:
    try: 
        dn = ModelParameter.outfile + "/" + ModelParameter.giffile + "/dist_sb_" + str(step)
        if step % printstp == 0: # TODO: PRESUN DO SEDPHASE?
            objs = [
                np.array(pased, dtype=float), 
                np.array(pnsed, dtype=float), 
                np.array(pablk, dtype=float), 
                np.array(pnblk, dtype=float),
                np.array([Diag.atrn]),
                np.array([Diag.astn]),
                np.array([np.max(atbl)]),
                np.array([Diag.blkgrow]),
                np.array(ntbl, dtype=float),
                np.array(atbl, dtype=float)
            ]
            objs_padded = []
            ml = max([len(obj) for obj in objs])
            for _, obj in enumerate(objs):
                if len(obj) < ml:
                    obj_padded = np.pad(obj, (0, ml - len(obj))) 
                    objs_padded.append(obj_padded)
                else:
                    objs_padded.append(obj)
            np.savetxt(dn, np.column_stack(objs_padded), delimiter="\t")

    except FileNotFoundError:
        print("FIX THIS BUG.")

    # Save each printstp-th 2D distribution to see how it evolves with time:
    try: 
        ...

    except FileNotFoundError:
        print("FIX THIS BUG.")

    distBLK2D = None
    return (distBLK, distSED, distBLK2D)

#######################################################################
#####             %%    METHOD OF DISTRIBUTIONS    %%             #####
#####                   FUNCTIONS (NJIT POWERED)                  #####
#######################################################################

@njit
def crystal_scatterJIT(distinp: tuple[np.ndarray, np.ndarray], amaxinp: tuple[np.ndarray, np.ndarray], amaxtar: float, mode: int, \
                       binstar: int, binsinp: int, ainc: Optional[float]=None, amin: float=0.0):
    """
        #% - Mode 1: scatters TBL crystals in BLK bins
        #% - Mode 2: scatters BLK crystals in updated BLK bins
        #% - distinp : inflow distribution (tuple)
        #% - amaxinp, amaxtar : maximum radii of the input and target distributions, respectively
    """

    if mode not in (1, 2): raise ValueError("Invalid mode of the crystal_scatterJIT function selected.")
    nbins = binstar
    inctar = np.zeros(binstar, dtype=float64) # increments of target bins!
    datar = (amaxtar - amin) / binstar
    dainp = (amaxinp - amin) / binsinp     
    imax = int(amaxinp / datar) + 1 

    if mode == 1:
        for i in range(0,imax): # loop over non-empty bulk bins
            jmin = int(i*(datar / dainp))
            jmax = int((i+1)*(datar / dainp))
            if (jmax > nbins - 1): jmax = nbins - 1 # the last bin!
            for j in range(jmin,jmax+1):
                # Computes what fraction of the j-th TBL bin falls into the i-th bulk bin
                frac = min(min(1.0, (j+1) - i*(datar / dainp)), min(1.0, (i+1)*(datar / dainp) - j))
                if frac > 0.0: 
                    inctar[i] += frac * distinp[1][j]
    if mode == 2: 
        imax = binstar
        for i in range(0,imax):
            jmin = max(int((i*datar - ainc) / dainp), 0)
            jmax = min(int(((i+1)*datar- ainc) / dainp), nbins-1)
            if (i == nbins - 1): jmax = nbins - 1
            for j in range(jmin,jmax+1):
                # Computes what fraction of the j-th old bulk bin falls into the i-th new bulk bin
                frac1 = min(1.0, (ainc + (j+1)*dainp - i*datar) / dainp )
                frac2 = min(1.0, ((i+1)*datar - j*dainp - ainc) / dainp )
                if (i == nbins - 1): frac2 = 1.0
                frac = min(frac1, frac2)
                if frac > 0.0: 
                    inctar[i] += frac * distinp[1][j]
    return inctar

@njit
def ssbulkJIT(
    nbulk:      np.ndarray,
    nbulkprev:  np.ndarray,
    tol:        float=1.e-10
    ) -> bool:    
    """ Checks the steady state """

    return np.all(np.abs(nbulk - nbulkprev) < tol)

@njit
def mMoDJIT(   
        stps: int, nbins: int, dtSED: float, ntot: float, Hnow: float,
        ainc: float, amaxTBL: float, amaxBLK: float, bulk_grow: float,
        ndistTBL: np.ndarray, ndistSED: np.ndarray,
        adistTBL: np.ndarray,  nsuspd: np.ndarray,
        const: object, nu :float, jwlimit: bool=False, fac: int=500,
        init_aBLK: Optional[np.ndarray]=None, init_nBLK: Optional[np.ndarray]=None,
        nss: bool=False, tCool: Optional[float]=None

    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:

    """
        - 0) method of distributions, we only work with (dim = nbins) arrays
        - 1) crystal_scatter(mode=1): TBL crystals into BLK bins
        - 2) if growth_on: BLK bins updated 
        - 3) crystal_scatter(mode=2): BLK crystals into updated BLK bins
        - 4) advance in time until steady state is reached
    """

    adistBLK = np.zeros(nbins, dtype=np.float64)
    ndistBLK = np.zeros(nbins, dtype=np.float64) 
    adistBLK[:] = adistTBL
    toff = False
    ntime = np.zeros(stps, dtype=np.float64)    

    # Non-steady state regime:
    if nss: stps = int64(trunc(tCool / dtSED))
    if stps == 0: stps = 1

    # Modification for an initial condition (i.e., distribution from the previous time step):
    if init_nBLK is not None and init_nBLK is not None:
    #if np.sum(init_nBLK) > 0.0:
        ndistBLK[:] = init_nBLK
        adistBLK[:] = init_aBLK
        amaxBLK = np.max(init_aBLK)

    # Time loop:
    for i in range(stps):
        ndistBLKprev = ndistBLK.copy()
        ndistSEDold  = ndistSED.copy()
        if not jwlimit: 
            ndistBLK[0] += ntot
        else: 
            ndistBLK[:] += crystal_scatterJIT(distinp=(adistTBL, ndistTBL), 
                                              amaxinp=amaxTBL, 
                                              amaxtar=amaxBLK, 
                                              mode=1, 
                                              binstar=nbins, 
                                              binsinp=nbins,
                                              ainc=ainc
                                            )
        # Crystal growth:
        amaxBLK_ = amaxBLK
        if bulk_grow > 0.0 and (ndistBLK[-1] > np.mean(ndistBLK) / fac): 
            amaxBLK += ainc 
            adistBLK = linspace_numba(amaxBLK / nbins, amaxBLK - amaxBLK / nbins, num=nbins)

        # Sedimentation and rearrange into BLK bins:
        ndistBLKtmp = crystal_scatterJIT(distinp=(adistBLK, ndistBLK), 
                                         amaxinp=amaxBLK_, 
                                         amaxtar=amaxBLK, 
                                         mode=2,
                                         binstar=nbins, 
                                         binsinp=nbins,
                                         ainc=ainc
                                        )           
        ndistBLK[:] = ndistBLKtmp[:]
        ndistBLK[:] *= (1.0 - wStokes(adistBLK[:], nu, const) * dtSED / Hnow)       
        ndistSED[:] += ndistBLK[:] * (wStokes(adistBLK[:], nu, const) * dtSED / Hnow)

        # Track the number of suspended crystals:
        nsuspd[i] = np.sum(ndistBLK)
        ntime[i] = i*dtSED
        if toff: tss = ntime[i]; ntime[i:] = ntime[i]; break;
        if not nss: toff = ssbulkJIT(ndistBLK, ndistBLKprev)

    # Steady-state increment:
    if nss: tss = dtSED * stps
    ndistSED -= ndistSEDold

    return (adistBLK.copy(), ndistBLK.copy(), ndistSED, i, nsuspd, tss, ntime)


def call_mMoDJIT(
    distTBL: CrystalDistro, stpsSed: int, dtSED: float, bulk_grow: float,
    Hnow: float, nu: float, step: int, printstp: int,

    # TODO: evidentně půlku těch parametrů nepotřebuju:) FIXME: REMOVE IT
    Tliq: float, Teut: float, 
    Tbulk: float, Troof: float, flux: float, Wrms: float

) -> Tuple[CrystalDistro, CrystalDistro]:             
    """ Auxiliary: mMoDJIT call in &SED_phase """

    # Crystal arrays (auxiliary initialization):
    distTBL.norm()
    ntot = Diag.cnt0DMC * (Shared.dtSED / Shared.dtTBL)
    adistTBL = (distTBL.adist).copy()
    amaxTBL  = amaxBLK = np.max(adistTBL)
    ndistTBL = [distTBL.ndist[k] * ntot for k in range(len(distTBL.ndist))] 
    ndistSED = np.zeros(RunConstants.nbins, dtype=float)  
    nsus = np.zeros(stpsSed, dtype=float)

    # Crystal radius increment:
    match Attributes.NG_METHOD:
        case(nMethod.mLin):
            ainc = dtSED * bulk_grow #pow_grow(Tbulk, Tliq, Shared.Tref, ModelParameter)
        case(nMethod.mLab):
            ainc = dtSED * bulk_grow #Hort_grow(Tbulk, Tliq, ModelParameter)
    adistBLK = adistTBL

    # Jarvis & Woods (1994) limit, all crystals have zero size!
    if not Attributes.TBL:
        ndistTBL = [0.0 for _ in ndistTBL]
        ndistTBL[0] = ntot

    adistBLK, ndistBLK, ndistSED, istd, nsus, tss, ntime = mMoDJIT(
                            stps=stpsSed, nbins=RunConstants.nbins, dtSED=Shared.dtSED, ntot=ntot,
                            Hnow=Hnow, ainc=ainc, amaxTBL=amaxTBL, amaxBLK=amaxBLK, bulk_grow=bulk_grow,
                            ndistTBL=ndistTBL, ndistSED=ndistSED, adistTBL=adistTBL,
                            nsuspd=nsus, const=ModelParameter, nu=nu, jwlimit=Attributes.TBL,
                            init_aBLK=None, init_nBLK=None,
                            #init_aBLK=Shared.prev_aBLK, init_nBLK=Shared.prev_nBLK, 
                            nss=False, tCool=Diag.tCool
                            ) 
                            # FIXME: toto ještě vyšetřit, proč mám někdy C-čkovský pointer-error, kde numba selhává?

    # Crystal inflow/outflow & steady-state time:
    Diag.cin  = Diag.cnt0DMC * Shared.dtSED / Shared.dtTBL
    Diag.cout = np.sum(ndistSED)
    Shared.prev_aBLK = adistBLK.copy()
    Shared.prev_nBLK = ndistBLK.copy()
    Diag.tssblk = tss

    #data_blk = np.column_stack((adistBLK, ndistBLK))
    #data_sed = np.column_stack((adistBLK, ndistSED))
    #np.savetxt("sed1.dat", data_sed)
    #np.savetxt("blk1.dat", data_blk)
    
    # Save each K-th distribution to see how it evolves with time!
    dn = ModelParameter.outfile + "/" + ModelParameter.giffile + "/dist_sb_" + str(step)
    if step % printstp == 0:
        objs = [
            adistBLK, 
            ndistBLK, 
            ndistSED
        ]
        objs_padded = []
        ml = max([len(obj) for obj in objs]) 
        for _, obj in enumerate(objs):
            if len(obj) < ml:
                obj_padded = np.pad(obj, (0, ml - len(obj))) 
                objs_padded.append(obj_padded)
            else:
                objs_padded.append(obj)
        np.savetxt(dn, np.column_stack(objs_padded), delimiter="\t")

    distBLK = CrystalDistro(adistBLK, ndistBLK)
    distSED = CrystalDistro(adistBLK, ndistSED)

    Shared.nsus = nsus
    Shared.ntime = ntime

    return (distBLK, distSED)

""" ############################################## """
""" #########   ANALYTICAL SOLUTION    ########### """
""" ############################################## """

def crystal_decay(t, N0, a0, H, G, lmd):
    exponent = (-1.0)*(lmd / H) * ((a0**2)*t + a0*G*(t**2) + (G**2 * t**3) / 3.)
    return N0 * np.exp(exponent) - N0 / 1.e10

@njit
def aux_denominator(a, amax, lmd, Wrms, H, G, gamma=0.7, eps=1e-15, TINY=1e-20):
    if lmd * a**2 <= gamma * Wrms: return H
    result = H * G - (a - amax) * (lmd * a**2 - gamma * Wrms)
    if result < 0.0 or result <= eps:
        return TINY #1.e-3 * H
    else:
        return result

@njit
def trapezoidal_integrate(a: np.ndarray, n: np.ndarray) -> float:
    result = 0.0
    for i in range(1, len(a)):
        da = a[i] - a[i-1]
        result += (1./2.) * da * (n[i] + n[i-1])
    return result

@njit
def phi_vals_arr(a_vals: np.ndarray, atbl: np.ndarray, ntbl: np.ndarray, 
                 N: int, amax: float, lmd: float, H: float, G: float, K: float, 
                 Wrms: float, jw: bool, shrink: bool=True
    ) -> np.ndarray:
    n = len(a_vals)
    phi_vals = np.zeros(n)
    for i in range(n):
        a = a_vals[i]
        if not jw: # Jarvis-Woods limit!
            phi_vals[i] = np.exp(-lmd * a**3 / (3.0 * G * H)) 
        else: 
            if not shrink:
                _ntbl = (ntbl * ((np.exp((lmd * np.power(atbl, 3) / (3.0 * G * H))))))[atbl <= a]
                g0tbl = trapezoidal_integrate(atbl[atbl <= a], _ntbl)
                phi_vals[i] = np.exp(-lmd * a**3 / (3.0 * G * H)) * g0tbl
            else:
                if a < amax:
                    _ntbl = (ntbl * ((np.exp((lmd * np.power(atbl, 3) / (3.0 * G * H))))))[atbl <= a]
                    g0tbl = trapezoidal_integrate(atbl[atbl <= a], _ntbl)
                    phi_vals[i] = np.exp(-lmd * a**3 / (3.0 * G * H)) * g0tbl

                else: # Integrate from amax to a using trapezoid rule!
                    s_vals = np.linspace(1.001*amax, a, N)
                    integrand = np.zeros(N)
                    for j in range(N):
                        integrand[j] = (lmd * s_vals[j]**2) / aux_denominator(s_vals[j], amax, lmd, Wrms, H, G)
                        if integrand[j] > 1.e4: integrand[j] = 0.0
                        #print(j, integrand[j])
                    integral = trapezoidal_integrate(s_vals, integrand)

                    # Calculate phi(a_tr) term!
                    _ntbl = (ntbl * ((np.exp((lmd * np.power(atbl, 3) / (3.0 * G * H))))))[atbl <= amax]
                    g0tbl = trapezoidal_integrate(atbl[atbl <= amax], _ntbl)
                    prefactor = np.exp(-lmd * amax**3 / (3.0 * G * H)) * g0tbl

                    phi_vals[i] = prefactor * np.exp(-integral)

    phi_vals[:] *= (K / G)
    return phi_vals

@njit
def phi_sedimentation(phi_vals, a_vals, atbl, ntbl, amax, lmd, Wrms, H, G, K, jw, dtSED, shrink: bool=False, gamma: float=0.7, TINY: float=1.0):
    n = len(a_vals)
    phi_sed = np.zeros(n)
    for i in range(n):         
        if not jw:
            phi_sed[i] = dtSED * phi_vals[i] * lmd * a_vals[i]**2 / H
        else:
            if not shrink:
                phi_sed[i] = dtSED * phi_vals[i] * lmd * a_vals[i]**2 / H
            else:
                if a_vals[i] <= amax:
                    phi_sed[i] = dtSED * phi_vals[i] * lmd * a_vals[i]**2 / H
                else: 
                    pass
                    """
                    print(
                        (a_vals[i] - amax) * (lmd * a_vals[i]**2 - gamma*Wrms) / G, a_vals[i], amax
                    )
                    H_mod = max(H - (a_vals[i] - amax)*(lmd * a_vals[i]**2 \
                            - gamma*Wrms) / G, TINY)
                    phi_sed[i] = dtSED * phi_vals[i] * lmd * a_vals[i]**2 / H_mod
                    """

    return phi_sed

def steady_state_analytical(
    lmd:      float,                    # auxiliary Stokesian variable
    Wrms:     float,                    # r.m.s convective velocity
    amax:     float,                    # transitional radius
    atbl:     np.ndarray,               # TBL radius array
    ntbl:     np.ndarray,               # TBL count array
    G:        float,                    # bulk-growth
    H:        float,                    # chamber height
    K:        float,                    # nucleation rate per unit chamber volume
    dtSED:    float,
    N:        int=200,                  #RunConstants.nbins,   # discretization
    G_tol:    float=1.e-16,             # to avoid zero division in the growth-limited regime
    jw:       bool=False,               # Jarvis & Woods limit (on/off)
    shrink:   bool=False                # Unified settling law (on/off)

) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # But ntbl should also be normalized per unit volume chamber per second, no?
    #ntbl /= (Shared.dtTBL * H)

    # Normalize for the g0 integral purposes:
    _norm = trapezoidal_integrate(atbl, ntbl)
    ntbl /= _norm

    if G < G_tol: # special case, G approximately zero!
        phi_vals = ntbl*np.power(atbl, 2.0)                         # TODO: correct? probably not but who cares, you can always skip the first step!
        phi_vals_sed = ntbl 
        a_vals = atbl                

    else: 
        a0 = np.max(atbl)
        proot = ( -lmd*a0**2 - sqrt(lmd**2 * a0**4 + lmd*a0*G*H*4)) / (-2.*lmd*a0*G)
        tmaxest = ( (3. * H) / (lmd * G**2) )**(1./3.) 
        amaxest = 2. * proot * G 
        a_vals = linspace_numba(0.0, amaxest, N)
        
        # Get rid of invalid values:
        phi_vals = phi_vals_arr(a_vals, atbl, ntbl, N, amax, lmd, H, G, K, Wrms, jw, shrink=shrink)
        phi_vals[np.isnan(phi_vals)] = 0.0
        phi_vals[np.isinf(phi_vals)] = 0.0

        phi_vals_sed = phi_sedimentation(phi_vals, a_vals, atbl, ntbl, amax, lmd, Wrms, H, G, K, jw, dtSED, shrink=shrink)

    return (a_vals, phi_vals, phi_vals_sed)

def call_mAnGZ(
    distTBL: CrystalDistro, bulk_grow: float, Hnow: float, _lmd: float, Wrms: float,
    Tbulk: float, Tliq: float, Troof: float, Teut: float, flux: float, step: int, dtSED: float,
    printstp: int, jw_in: bool, shrink: bool

) -> tuple[CrystalDistro, CrystalDistro]:
    """ Auxiliary: mAnGZ call in &SED_phase """

    # Beware, K=Diag.cnt0DMC / Shared.dtTBL / Hnow has the meaning of nucleation rate per unit chamber now!
    a_vals, phi_bulk, phi_sed = steady_state_analytical( 
        lmd=_lmd, Wrms=Wrms, amax=Diag.atrn, atbl=distTBL.adist,  
            ntbl=distTBL.ndist, G=bulk_grow, H=Hnow, K=(Diag.cnt0DMC / Shared.dtTBL / Hnow), \
            dtSED=dtSED, jw=jw_in, shrink=shrink
        )
        
    # Rescaling here:
    K = Diag.cnt0DMC / Shared.dtTBL / Hnow   # number of crystals nucleated per unit time per unit volume chamber!
    delta_a = a_vals[1] - a_vals[0] 
    #phi_bulk *= delta_a                     # TODO: this is just to get the distribution in the same units as SED_METHOD=1!
    #phi_bulk *= Hnow
    #phi_sed  *= delta_a
    #phi_sed  /= Hnow
    if Attributes.DEBUGRUN: print(f"COMPARISON: {np.sum(phi_sed)*delta_a/Hnow:.3e}, {K:.3e}.") # TODO: super confused right now, phi_bulk lines MUST BE COMMENTED OFF!??

    data_blk = np.column_stack((a_vals, phi_bulk))
    data_sed = np.column_stack((a_vals, phi_sed))

    # Save each K-th distribution to see how it evolves with time!
    dn = ModelParameter.outfile + "/" + ModelParameter.giffile + "/dist_sb_" + str(step)
    if step % printstp == 0:
        objs = [
            a_vals, 
            phi_bulk, 
            phi_sed
        ]
        objs_padded = []
        ml = max([len(obj) for obj in objs])    
        for _, obj in enumerate(objs):
            if len(obj) < ml:
                obj_padded = np.pad(obj, (0, ml - len(obj))) 
                objs_padded.append(obj_padded)
            else:
                objs_padded.append(obj)
        np.savetxt(dn, np.column_stack(objs_padded), delimiter="\t")
    
    distBLK = CrystalDistro(a_vals, phi_bulk)
    distSED = CrystalDistro(a_vals, phi_sed)

    return (distBLK, distSED)

""" ################################################################### """
""" #########    PREDICTOR - CORRECTOR INTEGRATION SCHEME   ########### """
""" ################################################################### """

# NOTE: does not really help much!
# TODO: rhs cant be deque, it will be just a numpy array of size three and add njit flags

def predictor_evol(order: int, rhs_h0: float, rhs_xl0: float, rhs_tb0: float, rhs_tlJW0: float, rhs: Deque[float]) -> tuple[float, float, float]:
    """ Calculate the predictor step based on the order of the integration scheme. """

    if order == 0 or order == 1 or order == -1: # explicit Euler | Adams-Bashforth 1 | implicit Euler
        rhs_h = rhs_h0; rhs_xl = rhs_xl0; rhs_tb = rhs_tb0; rhs_tl = rhs_tlJW0

    elif order == 2: # Adams-Bashforth 2
        rhs_h  = 3. * rhs_h0  / 2. - rhs[-1][0] / 2.
        rhs_xl = 3. * rhs_xl0 / 2. - rhs[-1][1] / 2.
        rhs_tb = 3. * rhs_tb0 / 2. - rhs[-1][2] / 2.

    elif order == 3: # Adams-Bashforth 3
        rhs_h  = 23. * rhs_h0  / 12. - 16. * rhs[-1][0] / 12. + 5. * rhs[-2][0] / 12.
        rhs_xl = 23. * rhs_xl0 / 12. - 16. * rhs[-1][1] / 12. + 5. * rhs[-2][1] / 12.
        rhs_tb = 23. * rhs_tb0 / 12. - 16. * rhs[-1][2] / 12. + 5. * rhs[-2][2] / 12.

    elif order == 4: # Adams-Bashforth 4
        rhs_h  = 55. * rhs_h0  / 24. - 59. * rhs[-1][0] / 24. + 37. * rhs[-2][0] / 24. - 9. * rhs[-3][0] / 24.
        rhs_xl = 55. * rhs_xl0 / 24. - 59. * rhs[-1][1] / 24. + 37. * rhs[-2][1] / 24. - 9. * rhs[-3][1] / 24.
        rhs_tb = 55. * rhs_tb0 / 24. - 59. * rhs[-1][2] / 24. + 37. * rhs[-2][2] / 24. - 9. * rhs[-3][2] / 24.

    return rhs_h, rhs_xl, rhs_tb, rhs_tl

def corrector_evol(order: int, rhs_hp: float, rhs_xlp: float, rhs_tbp: float, \
                   rhs_h0: float, rhs_xl0: float, rhs_tb0: float, rhs: Deque[float]) -> tuple[float, float, float]:
    """ Calculate the corrector step based on the order of the integration scheme """

    if order == 1:  # Adams-Moulton 1
        rhs_hcorr  = (rhs_h0 + rhs_hp)   / 2.0
        rhs_xlcorr = (rhs_xl0 + rhs_xlp) / 2.0
        rhs_tbcorr = (rhs_tb0 + rhs_tbp) / 2.0

    elif order == 2: # Adams-Moulton 2
        rhs_hcorr  = (5. * rhs_hp  + 8. * rhs_h0  - 1. * rhs[-1][0]) / 12.
        rhs_xlcorr = (5. * rhs_xlp + 8. * rhs_xl0 - 1. * rhs[-1][1]) / 12.
        rhs_tbcorr = (5. * rhs_tbp + 8. * rhs_tb0 - 1. * rhs[-1][2]) / 12.

    elif order == 3: # Adams-Moulton 3
        rhs_hcorr  = (9. * rhs_hp  + 19. * rhs_h0  - 5. * rhs[-1][0] + 1. * rhs[-2][0]) / 24.
        rhs_xlcorr = (9. * rhs_xlp + 19. * rhs_xl0 - 5. * rhs[-1][1] + 1. * rhs[-2][1]) / 24.
        rhs_tbcorr = (9. * rhs_tbp + 19. * rhs_tb0 - 5. * rhs[-1][2] + 1. * rhs[-2][2]) / 24.

    elif order == 4: # Adams-Moulton 4
        rhs_hcorr  = (251. * rhs_hp  + 646. * rhs_h0  - 264. * rhs[-1][0] + 106. * rhs[-2][0] \
            - 19. * rhs[-3][0]) / 720.
        rhs_xlcorr = (251. * rhs_xlp + 646. * rhs_xl0 - 264. * rhs[-1][1] + 106. * rhs[-2][1] \
            - 19. * rhs[-3][1]) / 720.
        rhs_tbcorr = (251. * rhs_tbp + 646. * rhs_tb0 - 264. * rhs[-1][2] + 106. * rhs[-2][2] \
            - 19. * rhs[-3][2]) / 720.

    return rhs_hcorr, rhs_xlcorr, rhs_tbcorr

""" ########################################################### """
""" #########  TIME EVOLUTION AUXILIARY FUNCTIONS   ########### """
""" ########################################################### """

def adapt_timestepPI(
    dt: float, arel0: float, arel1: float, dtmax: float=1.e6,
    kp: float=0.7, ki: float=0.4, sft: float=0.85, tol: float=0.1,
    fmin: float=0.3, fmax: float=2.0, dtmin: float=120.
) -> float:
    """ PI-style adaptive time step controller """

    # Callibration factor:
    factor = sft * (tol / arel1)**(kp) * (arel0 / arel1)**(ki)
    
    # Clumping:
    factor = max(fmin, min(factor, fmax))
    reject: bool = arel1 > tol
    dtnew = max(dtmin, dt*factor) if reject else min(dtmax, dt*factor)

    return dtnew

def ODMC_errorcheck(
    _i: int, mode: int, Tbulk: float, Tbulko: float, Tliqd: float,
    Troof: float, Teut: float, Hnow: float, nu: float, phiB: float,
    crates: int, debug_steps: int, calibration_steps: bool
) -> tuple[bool, bool, bool]:

    # Default values: 
    err = False; eqlix_write = False; eutix_write = False

    if Tbulk > Tbulko: 
        if not calibration_steps: print(" [WARNING] WARMING UP! (numerical instability)")
        logger.critical("Warming up! Numerical instability.")
        err = True

    # Numerical instability: magma becomes superheated
    if Tliqd < Tbulk:
        if not calibration_steps: print(" [WARNING] SUPERHEATED! (terminating)")
        logger.critical("SUPERHEATED")
        err = True

    # Single step, debug run.
    if Attributes.DEBUGRUN and Shared.onsetc is not None:
        print()
        if not calibration_steps: print(f" [DEBUGRUN] - The first step of the crystallization proceeded (step {_i:d}), terminating.")
        err = True

    # Reached eutectic:
    if Tliqd <= Teut and Shared.idxeut is None:
        Shared.idxeut = _i
        logger.warning(
            f"[EUTECTIC] - Reached eutectic the temperature!")
        print()
        if not calibration_steps: print(f" [EUTECTIC] - Reached the eutectic temperature!")
        eutix_write = True
        err = True
        
    # Equilibrium (no crystallisation):
    if Tbulk <= Troof:
        print()
        if not calibration_steps: print(f" [EQUILIBRIUM] - the bulk temperature caught up with the roof temperature!")
        Shared.idxend = _i
        eqlix_write = True
        err = True

    # Constant roof temperature benchmark:
    if Tliqd - Troof - ModelParameter.epsdel <= 1.e-3 and mode == 1:
        print(f"[TERMINATED] - Crystallization ceased!")
        err = True

    # Reached 100 steps:
    if crates == 100 and Attributes.DEBUG100:
        print(f"\nFirst {crates:d} steps done. Terminating the simulation!")
        err = True

    # Reached 2000 steps:
    if crates == 2000 and Attributes.DEBUG200:
        print(f" \nFirst {crates:d} steps done. Terminating the simulation!")
        err = True

    if crates == debug_steps and Attributes.DEBUG_X:
        print(f" \nFirst {debug_steps:d} steps done. Terminating the simulation!")
        err = True

    # Crystalinity sanity check:
    if phiB > 1.0: 
        if not calibration_steps: print(" [WARNING] - Crystalinity above 1.0, volumetrically overlapping crystals!")
        err = True

    _Ra = calculate_rayleigh_number(Hnow=Hnow, Tbulk=Tbulk, Troof=Troof, nu=nu, c=ModelParameter)
    if _Ra <= ModelParameter.Racrit:
        print("[WARNING] - Rayleigh number below the critical value, convection ceased.")
        err = True

    return (err, eqlix_write, eutix_write)

""" ########################################################## """
""" #########  NON-ESSENTIAL AUXILIARY FUNCTIONS   ########### """
""" ########################################################## """

def benchmark_solution(htbl: float, Hnow: float, Tliqd: float, Troof: float, Tbulk: float, lmd: float, c: Parameters,
                       distSED: CrystalDistro, distBLK: CrystalDistro, pshow: bool=False) -> None:

    def dsed(a, A, B): return A * a**2 * np.exp(-B * a**3)
    def dblk(a, A, B): return A * np.exp(-B * a**3)

    if Attributes.NG_METHOD == 1:
        _N = (htbl / Hnow) * pow_nucl(Tc=Troof, Tliq=Tliqd,Tref=Shared.Tref, c=ModelParameter)
        _V = pow_grow(Tc=Tbulk, Tliq=Tliqd,Tref=Shared.Tref, c=ModelParameter)

    if Attributes.NG_METHOD == 2:
        _N = Diag.cnt0DMC / Hnow / Shared.dtTBL
        _V = Hort_grow(T=Tbulk, Tliq=Tliqd, c=ModelParameter)

    try:
        print()
        print("-"*40)
        adistsed = distSED.adist[1:]
        ndistsed = distSED.ndist[1:]
        initial_guessesSED = [1e3, 1e5]
        poptsed, _ = curve_fit(dsed, adistsed, ndistsed, p0=initial_guessesSED)
        pref_sed = ((lmd ) / (Hnow * _V))
        print(f"SEDIMENT:")
        print(f"A predicted {poptsed[0]:.2e} vs. A theory {pref_sed:.2e}.")
        print(f"B predicted {poptsed[1]:.2e} vs. B theory {(lmd / (3.* Hnow * _V)):.2e}.")

        adistblk = distBLK.adist[1:-1]
        ndistblk = distBLK.ndist[1:-1]
        initial_guessesBLK = [1e5, 1e5]
        poptblk, _ = curve_fit(dblk, adistblk, ndistblk, p0=initial_guessesBLK)
        print(f"BULK:")
        print(f"A predicted = {poptblk[0]:.2e} vs. A analytical = {(_N/_V):.2e}")
        print(f"B predicted = {poptblk[1]:.2e} vs. B theory = {(lmd / (3. * Hnow * _V)):.2e}.")
        print("-"*40)

        fig, ax = plt.subplots(nrows=1, ncols=2)
        xblk = np.linspace(0, np.max(adistblk))
        xsed = np.linspace(0, np.max(adistsed))
        ax[0].scatter(adistblk, ndistblk, s=2, label="Computed BLK")
        ax[0].plot(xblk, dblk(xblk, *poptblk), "r:", label="Fit BLK")
        ax[0].legend()
        ax[1].scatter(adistsed, ndistsed, s=2, label="Computed SED")
        ax[1].plot(xsed, dsed(xsed, *poptsed), "r:", label="Fit SED")
        ax[1].legend()
        if pshow: plt.show()
        print()

    except ZeroDivisionError: pass 
    return

def snapshot_functions():
    """ Debugging: memory tracking """
    return {id(obj): obj for obj in gc.get_objects() if isinstance(obj, types.FunctionType)}

def dynamic_print(params: Dict[str, int]) -> None: 
    max_value_width = 0
    for param, (value, unit) in params.items():
        if isinstance(value, (float, int)):
            value_str = f"{value:.2e}" if abs(value) < 1.e-2 or abs(value) > 1.e3 else f"{value:.2e}"
        else:
            value_str = str(value)        
        max_value_width = max(max_value_width, len(value_str))

    for param, (value, unit) in params.items():
        if isinstance(value, (float, int)):
            value_str = f"{value:.2e}" if abs(value) < 1.e-2 or abs(value) > 1.e3 else f"{value:.2e}"
        else:
            value_str = str(value)
        print(f"{param:22} {value_str:>{max_value_width}}  {unit}")

######################################################################################################
#% end of the module!