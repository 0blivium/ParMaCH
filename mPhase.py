# Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 3)
# Module: Subroutines computing steady-state TBL, SED, and BLK distributions

# Standard libraries:
import os
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import time as time
import random
import copy
import warnings
from math import sqrt, pi, gamma, exp, log
from tqdm import tqdm
from numba import njit
from typing import Optional, Type
from collections import deque
from numpy.typing import NDArray
from scipy.integrate import simpson, quad
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar, curve_fit

# Memory tracking:
import gc
import psutil
import objgraph 

# Modules:
from mPar import *
from mGL import *
from m1DHE import *
from mFunc import CrystalDistro, CrystalBatch, CrystalDistro2D, BinaryAlloy
from mFunc import Hort_grow, Hort_nucl, pow_nucl, pow_grow, temperature_profile, hfce, bulk_growth
from mFunc import comp_hist, wStokes, visc0_GIORDANO, predictor_evol, corrector_evol, avg_growth, crystal_shift
from mFunc import lmd_stokes, wShrink, hflux_custom, crystal_decay, return_epsdel
from mFunc import adapt_timestepPI, ODMC_errorcheck, benchmark_solution
from mFunc import call_mTaCJIT, call_mMoDJIT, call_mAnGZ
from mPlot import plot_pnucage, plot_distribution, plot_tbl_distribution, plot_1D
from mPlot import plot_tbl_distribution_active, plot_2D_tbl_distribution_active
from mMisch import settling_transition

# Treat RuntimeWarnings as errors:
warnings.filterwarnings("error", category=RuntimeWarning)

# Get logger from main.py:
logger = logging.getLogger(__name__)

def TBLphase(
    step:   int,                # step of the chamber thermal evolution
    Tbulk:  float,              # bulk temperature
    Troof:  float,              # roof temperature
    Tliq:   float,              # liquidus tempeature
    htbl:   float,              # thickness of the TBL
    hnbl:   float,              # thickness of the nucleation sublayer
    nu:     float,              # viscosity
    nucage: bool=False,         # flag | plot nucleation age across the TBL
    ptbl1D: bool=False,         # flag | plot single 1D TBL outflow distribution
    plmswp: bool=False          # flag | gravitational extraction vs. plume sweep

) -> tuple[CrystalDistro, CrystalDistro, CrystalDistro2D]:
    
    """ 
    ###% NUCLEATION & GROWTH IN THE COLD BOUNDARY LAYER %###

        > calculating phiTBL distribution falling in the convecting bulk (two methods implemented)

        Calculates and saves as Shared.variable:
         - phiTBL (type: np.ndarray[CrystalBatch] of size ngens * Ntbl)
         - phiTBL (type: tuple of pairs (radius, count))
         - numpy array phiTBL stores $max(ngens * Ntbl, Ntbl) CrystalBatch objects, $phiTBLhist creates a tuple of pairs (radius, count):
         - the function returns non-normalized histogram of the TBL crystal size distribution

        Returns:
        - distTBL (CrystalDistro object)
    """

    logger.info(f"%%%%%%%  STEP {step:d}  %%%%%%%")
    logger.info("-" * 40)

    # Save the max/min growth/nucleation rate in the TBL:
    Tnucl = Tliq - ModelParameter.epsdel
    try:
        if Attributes.NG_METHOD == 1:
            Diag.tblgrwmin = pow_grow(Tbulk, Tliq, Shared.Tref, ModelParameter)
            Diag.tblnucmin = pow_nucl(Tnucl, Tliq, Shared.Tref, ModelParameter)
            Diag.tblgrwmax = pow_grow(Troof, Tliq, Shared.Tref, ModelParameter)
            Diag.tblnucmax = pow_nucl(Troof, Tliq, Shared.Tref, ModelParameter)
        if Attributes.NG_METHOD == 2:
            Diag.tblgrwmin = Hort_grow(Tbulk, Tliq, ModelParameter)
            Diag.tblnucmin = Hort_nucl(Tnucl, Tliq, ModelParameter)
            Diag.tblgrwmax = Hort_grow(Troof, Tliq, ModelParameter)
            Diag.tblnucmax = Hort_nucl(Troof, Tliq, ModelParameter)

    except ZeroDivisionError:

        # Careful handling here:
        if Attributes.NG_METHOD == 1:
            Diag.tblgrwmin = pow_grow(Tbulk, Tliq, Shared.Tref, ModelParameter)
            Diag.tblnucmin = 0.0
            Diag.tblgrwmax = pow_grow(Troof, Tliq, Shared.Tref, ModelParameter)
            Diag.tblnucmax = 0.0
        if Attributes.NG_METHOD == 2:
            Diag.tblgrwmin = Hort_grow(Tbulk, Tliq, ModelParameter)
            Diag.tblnucmin = 0.0
            Diag.tblgrwmax = Hort_grow(Troof, Tliq, ModelParameter)
            Diag.tblnucmax = 0.0

    """ Spatial dicretization of the nucleation sublayer """
    zFaces = np.linspace(0.0, hnbl, ModelParameter.Ntbl)
    delta_z = hnbl / ModelParameter.Ntbl
    zCenters = zFaces[:ModelParameter.Ntbl-1] + delta_z / 2.
    TCenters = temperature_profile(Tbulk, Troof, htbl, zCenters)

    """ Auxiliary initialization and time step estimate """
    phiTBLhist, particle_age, ini_depth, SbScystals = [[] for _ in range(4)]
    amin = 0.0; cntsum = 0.0; radsum = 0.0; nCrystals = 0.0; stpsTBL = 500; ngens = 1
    distTBLActive = None; distTBL = None; phiHistActive2D = None; distTBL2D = None; nuc_rand = True

    factor = ((27. * nu * htbl * ModelParameter.rhof) / (2. * ModelParameter.gacc * (ModelParameter.rhoc - ModelParameter.rhof)))**(1./3.)
    tres_est = factor / (avg_growth(Tbulk, Troof, Tliq))**(2./3.)
    dtTBL = tres_est / 500.
    Shared.dtTBL = dtTBL # update the time step in the shared class!

    # Plume detachment time scale (Turcotte & Schubert 2012):
    factor = ((nu * ModelParameter.kappa * RunConstants.Racrit) / (ModelParameter.alpha*ModelParameter.gacc*(Tbulk - Troof)))**(2./3.)
    tplume = factor * (1. / (5.36 * ModelParameter.kappa)) # FIXME
    Diag.tplume = tplume
    plmswp = Attributes.RTIS
    logger.info(f"Time step in the TBL:         {dtTBL:.3e}{Units.tunit}.")
    logger.info(f"Residence time in the TBL:    {tres_est:.3e}{Units.tunit}.")
    logger.info(f"Cold plume release timescale: {tplume:.3e}{Units.tunit}.")    

    """ Computational chunk """
    tbl_start = time.process_time()
    if plmswp: Attributes.TBL_METHOD = 2
    match Attributes.TBL_METHOD:
        # a) step-by-step method:
        case cMethod.mSbS:
            if step == 0 or Attributes.DEBUGRUN: print("[WARNING] - cMethod.mSbS employed.")
            tol = 1.e-6; nold = 0; printstp = 1; nuc_rand = False
            #assert(ModelParameter.Ntbl >= 300)
            CrystalFamily = np.array([CrystalBatch(0., 0., 0., 0.) for _ in np.arange(0, stpsTBL) for _ in np.arange(0, ModelParameter.Ntbl)])
            CrystalFamily = CrystalFamily.reshape(stpsTBL, ModelParameter.Ntbl)

            # Time loop:
            for i in range(stpsTBL):
                phiTBL = np.empty((1, 0), dtype=object)
                if i * dtTBL > tplume and plmswp: 
                    istop = i
                    logger.info(f"Crystals swept by a plume at time: {(i*dtTBL):.3e}.")
                    if Attributes.DEBUGRUN: print(f"Plume timescale exceeded (step {i:d}), the 2D-CSD was swept into the convecting bulk!")
                    break

                # Nucleation of newbies:
                for j in range(ModelParameter.Ntbl-1):
                    tmpz = random.uniform(zCenters[j] + delta_z / 2., zCenters[j] - delta_z / 2.) if nuc_rand else zCenters[j]
                    tmpT = temperature_profile(Tbulk, Troof, htbl, tmpz) if nuc_rand else TCenters[j]
                    CrystalFamily[i,j].CrystalNucl(tmpT, Tliq, delta_z, dtTBL, NG_METHOD=Attributes.NG_METHOD)
                    CrystalFamily[i,j].zcoord = tmpz

                    # Average number of crystals nucleated per time step (count only the first generation):
                    if i == 0: Diag.cnt0DMC += CrystalFamily[i,j].count

                # Shift & growth of oldies:
                for k in np.arange(0,i):
                    for l in range(ModelParameter.Ntbl-1):
                        if (not CrystalFamily[k,l].tblpas):
                            Tc = temperature_profile(Tbulk, Troof, htbl, CrystalFamily[k, l].zcoord)
                            if Attributes.NUTBL:
                                _nu = visc0_GIORDANO(Tc, ModelParameter.EaMod) / ModelParameter.rhof
                            else:
                                _nu = nu
                            CrystalFamily[k,l].CrystalGrow(Tc, Troof, Tliq, dtTBL, NG_METHOD=Attributes.NG_METHOD)
                            CrystalFamily[k,l].CrystalShift(dtTBL, Wrms=0.0,  nu=_nu)
                            if CrystalFamily[k,l].zcoord > htbl:
                                phiTBL = np.append(phiTBL, CrystalFamily[k, l])
                                CrystalFamily[k,l].tblpas = True

                if np.any(phiTBL) and i % printstp == 0:
                    nCrystals = np.sum([item.count for item in phiTBL])
                    diff = abs(nCrystals - nold)
                    print(f"Step (time): {i*dtTBL}{Units.tunit} | Crystal count: {nCrystals:.4f} / {Diag.cnt0DMC:.4f} (convg. {diff:.6e}).")
                    if abs(nCrystals - nold) < tol:
                        logger.info(f"Stable state reached at time: {(i*dtTBL):.3e}.")
                        istop = i
                        break
                    nold = nCrystals
                SbScystals.append(nCrystals)

            # Calculate the active CSD within the TBL:
            CrystalFamilyActive = CrystalFamily[:istop,:].reshape(-1)                    
            active_mask = np.array([not batch.tblpas for batch in CrystalFamilyActive])    
            CrystalFamilyActive = CrystalFamilyActive[active_mask]
            phiHistActive   = [(batch.radius, batch.count) for batch in CrystalFamilyActive]

            phiHistActive2D = [(batch.radius, batch.count, batch.zcoord) for batch in CrystalFamilyActive]
            amin = min(CrystalFamilyActive, key=lambda batch: batch.radius).radius
            amax = max(CrystalFamilyActive, key=lambda batch: batch.radius).radius
            distTBLActive = comp_hist(phi=phiHistActive, amin=amin, amax=amax, bins=RunConstants.nbins)

            if step % Attributes.printstep == 0:
                # Plot CSD within the TBL (volumetrically):
                plot_tbl_distribution_active( 
                    step=step, distTBLActive=distTBLActive, htbl=htbl
                )

                # Plot CSD in its full 2D form (count-size-zcoord):
                abins = np.linspace(0.0, amax, num=RunConstants.nbins) # convert to [mm]!
                zbins = np.linspace(0.0, htbl, num=RunConstants.nbins)
                radius, count, zcoord = np.array(list(zip(*phiHistActive2D)))

                Nij, aedges, zedges = np.histogram2d(
                    radius, zcoord,
                    bins=[abins, zbins],
                    weights=count
                )

                # Crystalinity in the TBL:
                phiB_htbl = (4. / 3.) * pi * np.sum( (distTBLActive.ndist) * np.power(distTBLActive.adist, 3.)) / htbl 

                plot_2D_tbl_distribution_active(
                    step=step, aedges=1.e3*aedges, zedges=1.e3*zedges, N=Nij, phiB_htbl=phiB_htbl
                )

                # Assemble the 2D distribution:
                distTBL2D = CrystalDistro2D(
                    adist=aedges, zdist=zedges, ndist=Nij, Tbulk=Tbulk, Troof=Troof, htbl=htbl
                )

        # b) generation method (master version):
        case cMethod.mGen:
            phiTBL = np.empty((1,0), dtype=object)
            CrystalFamily = np.empty((ModelParameter.Ntbl-1, ngens), dtype=object)
            fcounter = 0
            printstp = int(ModelParameter.Ntbl / 3.)
            assert ngens > 0, "Number of generations has to be greater than zero!"
            tqdmgens = tqdm(range(ngens), desc="Generations") if Attributes.DEBUG else range(ngens)
            for ng in tqdmgens: # gen!
                for i in range(ModelParameter.Ntbl-1): # sublayer!
                    # Nucleation at the centre vs. randomly (uniform probability density) within a layer:
                    tmpz = random.uniform(zCenters[i] + delta_z / 2., zCenters[i] - delta_z / 2.) if nuc_rand else zCenters[i]
                    tmpT = temperature_profile(Tbulk, Troof, htbl, tmpz) if nuc_rand else TCenters[i]
                    CrystalFamily[i,ng] = CrystalBatch(a=0.0, K=0.0, z=tmpz, indep=tmpz)
                    CrystalFamily[i,ng].CrystalNucl(tmpT, Tliq, delta_z, dtTBL, Attributes.NG_METHOD)
                    Diag.cnt0DMC += CrystalFamily[i, ng].count

                    # Crystal shifting (@njit powered) - keep shifting each batch until it falls out of TBL:
                    atmp, zctmp, rtbltmp = crystal_shift(a0=CrystalFamily[i,ng].radius,
                                                         z0=CrystalFamily[i,ng].zcoord,
                                                         ng=Attributes.NG_METHOD,
                                                         Tref=Shared.Tref,
                                                         const=ModelParameter,
                                                         htbl=htbl,
                                                         Tbulk=Tbulk,
                                                         Troof=Troof,
                                                         Tliq=Tliq,
                                                         dtTBL=Shared.dtTBL,
                                                         tplume=tplume,
                                                         nutbl=Attributes.NUTBL,
                                                         ff=True
                                                        )
                    CrystalFamily[i,ng].tblpas = True
                    CrystalFamily[i,ng].radius = atmp
                    CrystalFamily[i,ng].zcoord = zctmp
                    CrystalFamily[i,ng].rtimetbl = rtbltmp
                    particle_age.append(CrystalFamily[i, ng].rtimetbl)
                    ini_depth.append(CrystalFamily[i, ng].inidep)
                    phiTBL = np.append(phiTBL, CrystalFamily[i, ng])
                    fcounter += 1

            # Average number of crystals nucleated per time step:
            Diag.cnt0DMC /= ngens

        case _:
            logger.exception("Invalid method for computing TBL distribution selected.")
            raise Exception("Invalid TBL method.")

    if len(phiTBL) > 1: 
        Shared.phiTBL = phiTBL
        Diag.tbres = max(phiTBL, key=lambda batch: batch.rtimetbl).rtimetbl
        amin = min(phiTBL, key=lambda batch: batch.radius).radius
        amax = max(phiTBL, key=lambda batch: batch.radius).radius
        if Attributes.DEBUG:
            logger.info(f"Maximum/minimum radius falling into the bulk from the TBL: {(amax*RunConstants.mtomm):.2f}\
                                / {(amin*RunConstants.mtomm):.2f} [mm]")
        phiTBLhist = [(batch.radius, batch.count) for batch in phiTBL]
        distTBL = comp_hist(phi=phiTBLhist, amin=0.0,amax=amax, bins=RunConstants.nbins)
        tbl_end = time.process_time()
        logger.info(f"Elapsed CPU time (TBL phase): {(tbl_end - tbl_start):.4f}{Units.tunit}.")
        logger.info(f"Shifting: average number of crystals nucleated per time step (discretized sum): {Diag.cnt0DMC:.3e}")
    
        # Correction to the average number of nucleated crystals:
        match Attributes.NG_METHOD:
            case nMethod.mLin:
                pow_nucls, _ = quad(lambda z: pow_nucl(temperature_profile(Tbulk, Troof, htbl, z),
                                                    Tliq, Shared.Tref, ModelParameter),
                                            a=0.0,
                                            b=hnbl
                                    )
                logger.info(f"Integrated power-law nucleation rate: {(pow_nucls * Shared.dtTBL):.3e}")
                Diag.cnt0DMC = pow_nucls * Shared.dtTBL
            case nMethod.mLab:
                hort_nucls, _ = quad(lambda z: Hort_nucl(temperature_profile(Tbulk, Troof, htbl, z),
                                                    Tliq, ModelParameter),
                                            a=0.0,
                                            b=hnbl
                                    )
                logger.info(f"Integrated Hortian nucleation rate {(hort_nucls * Shared.dtTBL):.3e}.")
                Diag.cnt0DMC = hort_nucls * Shared.dtTBL

        # Apply the correction:
        distTBL.norm()
        distTBL_pass = copy.deepcopy(distTBL)
        distTBL.ndist *= Diag.cnt0DMC

        # Average the ensamble of generations (auxiliary generation averaging for the TaC alogorithm):
        if ngens > 1 and Attributes.TBL_METHOD == 1:
            phiTBLavg = np.array([CrystalBatch(0.0, 0.0, htbl, None) for _ in range(ModelParameter.Ntbl-1)])
            for i in range(len(phiTBLavg)):
                for j in range(ngens):
                    cntsum += phiTBL[j*(ModelParameter.Ntbl-1)+i].count
                    radsum += phiTBL[j*(ModelParameter.Ntbl-1)+i].radius
                phiTBLavg[i].count = cntsum / ngens
                phiTBLavg[i].radius = radsum / ngens
                cntsum, radsum = 0.0, 0.0

            # Update the shared variable:
            Shared.phiTBL = phiTBLavg

        # Plot the particle age vs. nucleation depth | plot 1D TBL outflow distribution:
        if nucage: plot_pnucage(step=step, particle_age=particle_age, ini_depth=ini_depth)
        if ptbl1D: plot_tbl_distribution(step=step, htbl=htbl, hnbl=hnbl, distTBL=distTBL_pass)

        # Save the distribution if necessary:
        if Attributes.DEBUGRUN:
            pass

    else:
        if Attributes.DEBUG: print("No crystals were gravitationally extracted!")

        Shared.phiTBL = CrystalFamilyActive
        distTBL = copy.deepcopy(distTBLActive) # replace the flux with the swept suspended distribution!

    return (distTBL, distTBLActive, distTBL2D)

def SEDphase(step:       int,                     # step of the thermal evolution
             phiTBL:     NDArray[np.object_],     # crystal batch array from TBL (CrystalBatch objects)
             distTBL:    CrystalDistro,           # discretized histogram from TBL
             Tbulk:      float,                   # temperature of bulk
             Troof:      float,                   # temperature of roof
             Tliq:       float,                   # temperature of liquidus
             Hnow:       float,                   # current height of the chamber
             htbl:       float,                   # thickness of the TBL
             Wrms:       float,                   # volume averaged r.m.s. convective velocity
             nu:         float,                   # kinematic viscosity in the bulk
             tc:         float,                   # cooling time
             flux:       float,                   # current heat flux
             Teut:       float,                   # eutectic temperature
             MC:         bool=False,              # switch Monte-Carlo approach vs. semi-analytical solution (cubic roots)
             plot_track: bool=False               # plot the track.pdf figure to see what is happening inside mTaCJIT()

        ) -> tuple[CrystalDistro, CrystalDistro]:
    
    """
    ###% CRYSTAL DYNAMICS IN THE BULK & SETTLING AND SEDIMENTATION %###

        Crystal dynamics function:
            > MODE is None:
                - enforcing a selected settling method (1,3,4)
            > MODE is not None:
                - determines which settling regime should apply and invokes it

        Returns:
        - distSED, distBLK (CSDs in sediment/bulk, CrystalDistro data structures)
    """

    # Auxiliary initialization and time step estimate:
    printstp = Attributes.printstep; stpsSed = 100000; tblon = True; grow_cutoff = 1.e-15; distBLK_idle = None 
    if Attributes.DEBUGRUN: printstp = 1

    # Residence time scale:
    _lmd = lmd_stokes(nu=nu, c=ModelParameter)
    try:
        Diag.tresjw = (Hnow / (_lmd*((bulk_grow + grow_cutoff)**2)))**(1./3.)

    except RuntimeWarning:
        tresjwOLD = Diag.tresjw
        Diag.tresjw = tresjwOLD

    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # TODO: redo this section!
    # Estimate tail and time step:
    stpssafe = 2000; tdecay = []
    try: 
        def aux(t): return crystal_decay(t, N0=Shared.phiTBL[0].count, a0=phiTBL[0].radius, H=Hnow, G=bulk_grow, lmd=_lmd)
        tcut = root_scalar(aux, bracket=[0, 1e30], method="brentq")
        tdecay = tcut.root

    except TypeError or ValueError or RuntimeWarning:
        tdecay = Shared.tdecayold
        tdecay = Hnow / wStokes(Diag.amaxtbl, nu, ModelParameter)

    dtSED = (tdecay / stpssafe / 10.) # * 100.
    Shared.dtSED = dtSED
    Shared.tdecayold = tdecay
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    # Normalize the phiTBL distribution, and calculate the distribution that falls in per time step Shared.dtSED!
    # NOTE: %> Remember, index 0 of phiTBL = max. radius, index -1 of phiTBL = min. radius!
    # NOTE: %> Remember, the radius array is in a descending order! Length is (Ntbl-1)!
    # NOTE: %> Remember, ntbl is defined with respect to time step dt_sed now!
    if phiTBL is not None:
        phitblsum = np.sum([x.count for x in phiTBL])
        [setattr(phiTBL[k], "count", Diag.cnt0DMC * (phiTBL[k].count / phitblsum) * (Shared.dtSED / Shared.dtTBL)) for k in range(len(phiTBL))]
        [setattr(phiTBL[k], "inidep", 0.0) for k in range(len(phiTBL))]
        atbl = np.array([x.radius for x in phiTBL])  
        ntbl = np.array([x.count for x in phiTBL])
        Shared.atbl = atbl; Shared.ntbl = ntbl
    else:
        atbl = distTBL.adist 
        ntbl = (distTBL.ndist / np.sum(distTBL.ndist)) * Diag.cnt0DMC * (Shared.dtSED / Shared.dtTBL)
        Shared.atbl = atbl; Shared.ntbl = ntbl

    if np.min(atbl) > Diag.astn: # the whole TBL distribution undergoes a full Stokesian fall across the entire chamber!
        print("[WARNING] - FULL STONE-LIKE REGIME!") # TODO: dej tam ten set() ať to nekřičí pořád :)
        Shared.tblstokes = True

    # Rebinding to be safe:
    distTBL_pass = copy.deepcopy(distTBL)

    # Check if we are likely to exceed the transitional radius | estimate the maximum radius upon extraction:
    tmaxest = ((3. * Hnow) / (_lmd * (bulk_grow + grow_cutoff)**2)) * 10.0
    amaxest = tmaxest**(0.33) * bulk_grow * 1.0  # <-- FIXME: tohle občas háže error!

    # Special case, step #1 in model class A!
    if bulk_grow == 0.0: 
        distBLK, distSED = call_mMoDJIT(distTBL_pass, stpsSed, dtSED, bulk_grow, Hnow, nu, step,
                                printstp, Tliq, Teut, Tbulk, Troof, flux, Wrms
                               ) 
        distSED_pass = copy.deepcopy(distSED)
        distBLK_pass = copy.deepcopy(distBLK)
        if step % printstp == 0:
            plot_distribution(i=step, distSED=distSED_pass, distBLK=distBLK_pass, distTBL=distTBL_pass, distBLK_idle=distBLK_idle,
                            Hnow=Hnow, Tliq=Tliq, Tbulk=Tbulk, Troof=Troof, Tnucl=(Tliq - ModelParameter.epsdel),
                            amean=distSED.amean(), tsim=Shared.ntime, pop=Shared.nsus, track=Shared.track, Teut=Teut, flux=flux, Wrms=Wrms)
        return (distBLK, distSED)

    # TODO: Přepiš distTBL na pass-on parameter, ne jako referenci!
    distTBL_pass = copy.deepcopy(distTBL)

    # MIXING TIME (USE ONLY FOR THE mTaCJIT solver):
    dtaudust  = (Diag.atrn - atbl)      / bulk_grow
    dtaustone = (Diag.astn - Diag.atrn) / bulk_grow
    dtaumix   = (Diag.astn - atbl)      / bulk_grow

    # A 90-settling scale:
    if abs(Hnow*log(0.9)/bulk_grow) < np.max(dtaumix): 
        print(f"{abs(Hnow*log(0.9)/bulk_grow):.3} {np.max(dtaumix):.3e}")
        print("[WARNING] - Mixing vs. Setttling times are competing!")

    if Attributes.MODE is not None: 
        if Shared.onsetc is None: print(f"The flag --MODE is deprecated, use --SED_METHOD instead!")


        # TODO: just formally!
        distBLK, distSED = settling_transition(
            ...
        )

        dn = ModelParameter.outfile + "/" + ModelParameter.giffile + "/dist_sb_" + str(step)
        if step % printstp == 0:                                   
            objs = [
                distBLK.adist,
                distBLK.ndist,
                distSED.ndist,
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

        return (distBLK, distSED)

    else:
        # Specific choice of the settling method:
        atbl_pass = copy.deepcopy(atbl)

        match Attributes.SED_METHOD:
            ###% METHOD OF DISTRIBUTIONS %###
            case sMethod.mDis:
                if Shared.onsetc is None: print("The growth-mixing correction is not accounted in this approach!")
                distBLK, distSED = call_mMoDJIT(distTBL_pass, stpsSed, dtSED, bulk_grow, Hnow, nu, step,
                                                printstp, Tliq, Teut, Tbulk, Troof, flux, Wrms)
                Diag.setmarker = 1

            ###% METHOD OF CRYSTAL TRACING %###
            case sMethod.mUTaC: # FIXME: crystal tracing doesnt react to the --TBL flag!!!! (2nd March: it does though?)
                if not Attributes.tblaoff: atbl_pass[:] = 0.0 
                distBLK, distSED, distBLK2D = call_mTaCJIT(Hnow, nu, stpsSed, ntbl, atbl_pass, Wrms, tc, bulk_grow,
                                                flux, dtSED, step, printstp, Tliq, Tbulk, Troof, Teut,
                                                distTBL
                                            )
                
                """
                plt.plot(distBLK2D.adist, distBLK.ndist)
                plt.show()
                plt.clf()
                plt.plot(distBLK2D.zdist, distBLK.ndist)
                plt.show()
                plt.clf()
                """

                # Plotting the vertically distributed crystals in the bulk:
                #plot_2D_tbl_distribution_active(
                #    step=step, aedges=1.e3*distBLK2D.adist[1:], zedges=distBLK2D.zdist, N=distBLK2D.ndist[1:], _name="blk"
                #)

                # Assign the settling flag (empirical division rules in Patočka et al. 2022):
                if np.max(distBLK.adist) >= Diag.atrn and np.max(distBLK.adist) < Diag.astn:
                    Diag.setmarker = 2 # transitional regime
                elif np.max(distBLK.adist) >= Diag.astn:
                    Diag.setmarker = 3 # stone-like regime
                else:
                    Diag.setmarker = 1 # dust-like regime

            ###% SEMI-ANALYTICAL SOLUTION %###
            case sMethod.AnGZ: 
                if not Attributes.TBL or not Attributes.tblaoff: tblon = False # NOTE: tblon is True by default!
                distBLK, distSED = call_mAnGZ(distTBL, bulk_grow, Hnow, _lmd, Wrms, Tbulk, Tliq,
                                              Troof, Teut, flux, step, Shared.dtSED, printstp, tblon, shrink=False)
                Diag.setmarker = 1

            case _:
                raise Exception("Invalid method for computing the sedimentation/bulk phase selected!")

    # Rebinding to be safe:
    distSED_pass = copy.deepcopy(distSED)
    distBLK_pass = copy.deepcopy(distBLK)
        
    if step % printstp == 0:
        plot_distribution(i=step, distSED=distSED_pass, distBLK=distBLK_pass, distTBL=distTBL_pass, distBLK_idle=distBLK_idle,
                          Hnow=Hnow, Tliq=Tliq, Tbulk=Tbulk, Troof=Troof, Tnucl=(Tliq - ModelParameter.epsdel),
                          amean=distSED.amean(), tsim=Shared.ntime, pop=Shared.nsus, track=Shared.track, Teut=Teut, flux=flux, Wrms=Wrms)

    if step % printstp == 0 and Attributes.MODE is not None and Diag.setmarker == 2 and plot_track: # FIXME: tohle lépe ošetřit, zatím OK
        _N = ModelParameter.Ntbl - 1
        _Nt = Shared.track.shape[0]
        _vhTrack = wShrink(Shared.track[:, _N:2*_N], Wrms, nu, ModelParameter)
        plot_track(
            step=step, tstps=_Nt, Hnow=Hnow, aTrack=Shared.track[:, _N:2*_N], vhTrack=_vhTrack, dhTrack=Shared.track[:, 3*_N:4*_N]
        ) 

    return (distBLK, distSED)


def calculate_distributions(i: int, Ra: float, Re: float, Wrms: float, Hnow: float, Tbulk: float, Troof: float,
                            Tliqd: float, Tnucl: float, tc: float, _nu: float, flux: float, Teut: float, SingleRun: SingleRunAttributes=None
                        ) -> tuple[float, float, float, float, float, float, float]:
    
    """ Calculate convection profile, spatial discretization TBL/NBL, and final TBL/SED distributions """

    #% 1/2 DISTRIBUTIONS:
    """ Spatial dicretization of the TBL/NBL """
    
    # Thickness of the difussion layer/nucleation sublayer:
    _lmd = lmd_stokes(_nu, ModelParameter)
    Nu = calculate_nusselt_number(Ra, _nu / ModelParameter.kappa, Shared.regime)
    
    # NOTE: The predicted thicknesses can be substantially different (up to 1 order of magnitude difference)!
    htbl = 6.4 * np.power(Ra, -1./3.) * Hnow if not Attributes.TBL else calculate_tbl_thickness(Hnow, Nu) 

    if Attributes.SRUN and SingleRun is not None: htbl = SingleRun.htbl
    hnbl = min(htbl*(Tnucl - Troof) / (Tbulk - Troof), htbl)

    if hnbl < 0.0: raise ValueError("Thickness of the nucleation sublayer can not be negative!")

    """ Print state at step #:_i """
    logger.info(f"GOVERNING PARAMETERS (step: {i:d})")
    logger.info(f"Rayleigh number:                           {Ra:.3e}.")
    logger.info(f"Reynolds number:                           {Re:.3e}.")
    logger.info(f"Convective velocity:                       {Wrms:.3e}{Units.vunit}.")
    logger.info(f"Current height of the chamber:             {Hnow:.3e}{Units.sunit}.")
    logger.info(f"Thickness of the diffusion layer:          {htbl:.3e}{Units.sunit}.")
    logger.info(f"Thickness of the nucletion sublayer:       {hnbl:.3e}{Units.sunit} ({1.e2*hnbl/htbl:.2f}% of boundary layer).")
    logger.info(f"Discretization of the nucleation sublayer: {(htbl / ModelParameter.Ntbl):.3e}{Units.sunit}.\n")

    # Growth rate in the bulk:
    global bulk_grow
    bulk_grow = bulk_growth(Tbulk, Tliqd) if Tbulk < Tliqd else 0.0 
    Diag.blkgrow = bulk_grow

    # Calculate the TBL outflow/suspended distribution:
    # NOTE: TBL crystal flux per time step delta t_b!
    distTBL, distTBLActive, distTBL2D = TBLphase(i, Tbulk, Troof, Tliqd, htbl, hnbl, _nu, ptbl1D=False) 
    distTBL_benchmark = copy.deepcopy(distTBL)
    if bulk_grow == 0.0 and not Attributes.tblaoff: 
        hRate = 0.0; pRate = 0.0; amean = 0.0; ameanblk = 0.0; phi0 = 0.0
        return (htbl, hnbl, hRate, pRate, amean, ameanblk, phi0)

    if Attributes.DEBUGRUN:
        print()
        print()
        print(f"Surface densities:")
        print(f"Number of crystals nucleated/extracted/falling from the distTBL (dtb):   {(np.sum(distTBL.ndist)):.3e}.")
        print(f"Number of crystals nucleated per unit time:                              {(Diag.cnt0DMC / Shared.dtTBL):.3e}.")
        print(f"Nucleation rate per unit volume of the chamber:                          {(Diag.cnt0DMC / Shared.dtTBL / Hnow):.3e}.")

        if distTBLActive is not None: # TODO: do it properly
            phiB_htbl = (4./3.) * pi * np.sum( (distTBLActive.ndist / htbl) * np.power(distTBLActive.adist, 3.) )
            print(f"Number of crystals suspended in TBL:                         {np.sum(distTBLActive.ndist):.3e}.")
            print(f"Number of crystals suspended in TBL (per unit volume TBL):   {np.sum(distTBLActive.ndist)/htbl:.3e}.")
            print(f"Volume occupied within the TBL:                              {phiB_htbl:.3e}.")
            print(f"Crystalinity within TBL: {phiB_htbl:.3e}.")

    # NOTE: this is used to copy-paste arrays from the terminal window
    #print(np.array2string(distTBL.adist, separator=', '))   
    #print(np.array2string(distTBL.ndist, separator=', '))

    # Calculate the maximum TBL radius:
    Diag.amaxtbl = distTBL.amax()
    Diag.amintbl = distTBL.amin()
    if not Attributes.TBL or not Attributes.tblaoff:
        Diag.amaxtbl = 0.0

    # Jarvis-Woods 1994 limit = replace the # of crystals by that nucleated at a constant undercooling of (Tliqd - Troof):
    Diag.cnt0DMC = Diag.cnt0DMC
    if Attributes.NG_METHOD == nMethod.mLin:
        Diag.cntHBJW = htbl * Shared.dtTBL * pow_nucl(Tc=Troof, Tliq=Tliqd, Tref=Shared.Tref, c=ModelParameter)
        Diag.cntHNJW = hnbl * Shared.dtTBL * pow_nucl(Tc=Troof, Tliq=Tliqd, Tref=Shared.Tref, c=ModelParameter) 
    elif Attributes.NG_METHOD == nMethod.mLab:
        Diag.cntHBJW = htbl * Shared.dtTBL * Hort_nucl(T=Troof, Tliq=Tliqd, c=ModelParameter) 
        Diag.cntHNJW = hnbl * Shared.dtTBL * Hort_nucl(T=Troof, Tliq=Tliqd, c=ModelParameter) 
    if not Attributes.TBL: Diag.cnt0DMC = Diag.cntHBJW

    # Bulk crystal dynamics:
    distTBL_to_SEDphase = copy.deepcopy(distTBL)
    distBLK, distSED, = SEDphase(i, Shared.phiTBL, distTBL_to_SEDphase, Tbulk, Troof, Tliqd, Hnow, htbl,
                                    Wrms, _nu, tc, flux, Teut)  # CSDs are per unit area of the reservoir!
    distBLK_idle = None

    if Attributes.DEBUGRUN: 
        print(f"Fall-in per unit time:                  {' ' * 33}{(np.sum(distTBL.ndist)/Shared.dtTBL):.3e}.")
        print(f"Suspended crystals:                     {' ' * 33}{np.sum(distBLK.ndist):.3e}.")
        print(f"Suspended crystals (per unit volume):   {' ' * 33}{(np.sum(distBLK.ndist)/Hnow):.3e}.")
        print(f"Outflow per unit time:                  {' ' * 33}{np.sum(distSED.ndist)/Shared.dtSED:.3e}.")

    # Sanity check/warning - in case we nucleate but it is a negligible amount:
    if np.sum(distBLK.ndist) == 0.0: 
        hRate = 0.0; pRate = 0.0; amean = 0.0; phi0 = 0.0; ameanblk = 0.0;
        return (htbl, hnbl, hRate, pRate, amean, ameanblk, phi0)

    # Mean crystal radius:
    amean = distSED.amean(); ameanblk = distBLK.amean()

    #% 2/2 Production/Sedimentation rate
    dablk = distBLK.da(); dased = distSED.da()
    if Attributes.MODE is not None:
        match Diag.setmarker:
            case 1: # NOTE: dust-like regime using sMethod.AnGZ, already rescaled in &SEDphase!
                pass
            case 2: # CSD per unit chamber volume!
                distBLK.scale(np.power(dablk * Hnow, -1.))
            case 3: # CSD per unit chamber volume!
                distBLK.scale(np.power(dablk * Hnow, -1.)) # NOTE: POZOR, DENSER GRID CAUSES PROBLEMS?
    else:
        if Attributes.SED_METHOD == 1 or Attributes.SED_METHOD == 3:  # sMethod.AnGZ already rescaled in &SEDphase!
            distBLK.scale(np.power(dablk * Hnow, -1.))                # CSD per unit chamber volume!

    # Distributions to benchmark:
    distSED_benchmark = copy.deepcopy(distSED)
    distBLK_benchmark = copy.deepcopy(distBLK)

    #np.savetxt("ablk.dat", distBLK.adist)    
    #np.savetxt("nblk.dat", distBLK.ndist)
    #np.savetxt("ased.dat", distSED.adist)
    #np.savetxt("nsed.dat", distSED.ndist)
    #print(np.array2string(distSED.adist, separator=', '))   
    #print(np.array2string(distSED.ndist, separator=', '))
    #print(np.array2string(distBLK.adist, separator=', '))   
    #print(np.array2string(distBLK.ndist, separator=', '))

    ###############################################################################################################
    # Production rate within the TBL: 
    if Attributes.DEBUGRUN and distTBL2D is not None:        
        if Attributes.NG_METHOD == nMethod.mLin: 
            print("[WARNING] - This debugging procedure is implemented for the Hortian laws!")

        # Solid production within the TBL:
        dz = distTBL2D.dz(); da = distTBL2D.da()
        pRate_TBL = 4. * pi * da * dz * np.sum(
            distTBL2D.ndist * np.power(distTBL2D.adist, 2.0)[1:, None] \
                * Hort_grow(distTBL2D.tmp_profile(distTBL2D.zdist[None, 1:]),
                    Tliqd, ModelParameter)
                ) # TODO: tohle by si chtělo ověřit, že to volumetricky fakt sedí, ale asi jo :D
        print(f"Solid production within the TBL: {pRate_TBL:.3e}.")

        # Compare with the outflow volume:
        pRate_TBL_outflow = (1. / Shared.dtTBL) * (4. / 3.) * pi * np.sum(distTBL.ndist * np.power(distTBL.adist, 3.0))
        print(f"Outflow volume:                  {pRate_TBL_outflow:.3e}.")
    ###############################################################################################################

    # Production rate:
    pRate_idle = 0.0
    pRate = 4. * pi * bulk_grow * simpson(np.power(distBLK.adist, 2.) * distBLK.ndist, distBLK.adist) # solid fraction contribution from the bulk!
    pRate0 = pRate
    distTBL.scale(np.power(Hnow * Shared.dtTBL, -1.))                                                 # convert the TBL flux to units per unit volume of the chamber!

    # NOTE: POZOR, distTBL distribuce je už v konvertovaných "JW" jednotkách!
    # NOTE: Production of solid
    #   %> solid flow from the TBL
    #   %> solid production of the idle crystals
    #   %> solid production of the well-mixed / sinking crystals 
    
    pRate_TBL = (4. / 3.) * pi * np.sum(distTBL.ndist * np.power(distTBL.adist, 3.0))   # solid production flux from the TBL!
    if not Attributes.tblaoff: pRate_TBL = 0.0
    pRate += pRate_TBL

    
    """
    # Idle crystals:
    if bulk_grow > 0.0 and Shared.phiTBL is not None and Attributes.SCORR and not Shared.tblstokes:
        idle_crystals_abins = np.linspace(0.0, np.max(Shared.a0idle), num=len(Shared.phiTBL))
        idle_crystals_nbins = np.zeros_like(idle_crystals_abins)
        for i, (a_i, n_i) in enumerate(zip(distTBL.adist, distTBL.ndist)):
            mask = (idle_crystals_abins >= a_i) & (idle_crystals_abins <= Shared.a0idle[i])
            idle_crystals_nbins[mask] += (n_i / bulk_grow)

        # Idle krystaly podruhé:
        #idle_crystals_nbins[:] = idle_crystals_nbins / np.sum(idle_crystals_nbins)
        #idle_crystals_nbins[:] = np.sum(distTBL.ndist[:] * (Diag.astn - distTBL.adist[:])) / bulk_grow
        # FIXME: pro dust-like ale Diag.astn nedává smysl, ne? měl by to být a0 + acorr, že?
        #ogc = np.sum(distTBL.ndist * (Shared.a0corr - distTBL.adist)) / bulk_grow 
        
        #print(Shared.a0idle, Shared.atbl)
        ogc = np.sum(Shared.ntbl * (Shared.a0idle - Shared.atbl)) / bulk_grow / Shared.dtSED / Hnow
        distBLK_idle = CrystalDistro(idle_crystals_abins, idle_crystals_nbins)
        #print(distBLK_idle.ndist)
        distBLK_idle.norm() 
        distBLK_idle.ndist[:] *= ogc

        da_idle = distBLK_idle.da()
        distBLK_idle.scale(np.power(da_idle, -1.))
        pRate_idle = 4. * pi * bulk_grow * simpson(np.power(distBLK_idle.adist, 2.) * distBLK_idle.ndist, distBLK_idle.adist)
        pRate += pRate_idle # NOTE: idk what to do here
    else: 
        pRate_idle = 0.0
    """
    pRate_idle = 0.0


    # Sedimentation rate after Jarvis & Woods (M&N only gives the correct answer):
    hRateJW = _lmd * (4. / 3.) * pi * simpson(np.power(distBLK.adist, 5.) * distBLK.ndist, distBLK.adist)

    # Sedimentation rate per unit time calculated from the sediment crystal distribution flux:
    distSED.norm()
    hRate = (4. / 3.) * pi * np.sum((distSED.ndist) * np.power(distSED.adist, 3.)) * (Diag.cnt0DMC / Shared.dtTBL)
    if Attributes.MODE is None and Attributes.SED_METHOD == 4 and i > 0: hRate = hRateJW # s sAnGZ:
         
    # Calculate the crystalinity in the reservoir:
    phi0 = (4. / 3.) * pi * simpson(np.power(distBLK.adist, 3.) * distBLK.ndist, distBLK.adist)
    phi0_TBL = (4. / 3.) * pi * simpson(np.power(distTBL.adist, 3.) * distTBL.ndist, distTBL.adist)
    if not Attributes.tblaoff: phi0_TBL = 0.0
    if distBLK_idle is not None: phi0_idle = (4. / 3.) * pi * simpson(np.power(distBLK_idle.adist, 3.) * distBLK_idle.adist, distBLK_idle.adist)
    else: phi0_idle = 0.0
    phi0 += (phi0_TBL + phi0_idle)

    if Attributes.DEBUGRUN:
        print()
        print(f"SOLID PRODUCTION")
        print(f"------------------------------------------------------------------------------------")
        print(f"Bulk (active):                              {pRate0:.3e}.")
        print(f"Bulk (idle):                                {pRate_idle:.3e}.")
        print(f"TBL inflow:                                 {pRate_TBL:.3e}.")
        print(f"Total production:                           {(pRate0 + pRate_idle + pRate_TBL):.3e}")
        print(f"Theoretical chidot I should measure is:     {(hRate / Hnow):.3e}.")
        print(f"------------------------------------------------------------------------------------")
        print()
        print(f"SEDIMENTATION RATE")
        print(f"------------------------------------------------------------------------------------")
        print(f"The sedimentation rate:                     {hRate:.3e}.")
        print(f"The ratio:                                  {(hRate/pRate):.3e}.")
        print(f"Current size of the chamber:                {Hnow:.3e}.")
        print(f"------------------------------------------------------------------------------------")
        print()
        print(f"CRYSTALINITY (INDIVIDUAL CONTRIBUTIONS)")
        print(f"------------------------------------------------------------------------------------")
        print(f"Bulk (active):                              {1.e2*phi0:.3e}%.")
        print(f"Bulk (idle):                                {1.e2*phi0_idle:.3e}%.")
        print(f"TBL (suspended):                            {1.e2*phi0_TBL:.3e}%.")
        print(f"Total crystalinity in the chamber:          {1.e2*(phi0 + phi0_idle + phi0_TBL):.3e}%.")
        print(f"------------------------------------------------------------------------------------")
        print()

    # Share the current steady-state across the functions:
    Shared.distBLK   = distBLK
    Shared.distSED   = distSED
    Shared.distTBL   = distTBL 
    Shared.distTBL2D = distTBL2D
    Shared.prateTBL  = pRate_TBL
    Shared.prateBLK  = pRate0
    Shared.prate     = pRate

    # Check the volumetric constraint:
    _ratio = 1.e2 * abs((hRate / pRate) - Hnow) / Hnow
    if _ratio > 10.0:
        print(f"[WARNING] - The volumetric constraint has been exceeded by more than 10% ({_ratio:.2f}%)!")
        logger.warning("[WARNING] - The volumetric constraint has been exceeded by more than 10% ({_ratio:.2f}%)!")

    # Fitting the analytical solution for the crystal size function (first step):
    if Attributes.DEBUGRUN: # or Attributes.DEBUG_X and i == 1:
        benchmark_solution(
            htbl=htbl, Hnow=Hnow, Tliqd=Tliqd, Troof=Troof, Tbulk=Tbulk, lmd=_lmd, c=ModelParameter,
            distSED=distSED_benchmark, distBLK=distBLK_benchmark, pshow=1
        )


    # HERE: SAVE CSD IN "CSD" UNITS?
    #######################################################################################
    #"""
    dn = ModelParameter.outfile + "/BENCH" + str(i)
    if i % Attributes.printstep == 0:
        a0 = Diag.amaxtbl
        __lmd = lmd_stokes(_nu, ModelParameter)
        alpha = __lmd * Diag.blkgrow**2 / 3.
        beta  = Wrms + __lmd * a0**2
        gama  = __lmd*a0*Diag.blkgrow
        delta = - Hnow

        def f1(t): return alpha * t**3 + beta * t + delta + gama * t**2
        tbot = root_scalar(f1, bracket=[0, 1e6], method="brentq").root
        ttop = (Diag.atrn - a0) / Diag.blkgrow
        tratio = tbot / ttop
        tblgrowmean = (Hort_grow(Tbulk, Tliqd, ModelParameter) + Hort_grow(Troof, Tliqd, ModelParameter)) / 2.0
        objs = [
            np.array(distBLK_benchmark.adist, dtype=float), 
            np.array(distBLK_benchmark.ndist, dtype=float), 
            np.array(distSED_benchmark.adist, dtype=float), 
            np.array(distSED_benchmark.ndist, dtype=float),
            np.array([Diag.atrn]),
            np.array([Diag.astn]),
            np.array([np.max(distTBL.adist)]),
            np.array([Diag.blkgrow]),
            np.array([Hnow]),
            np.array([Shared.dtTBL]),
            np.array([Shared.dtSED]),
            np.array([Tbulk]),
            np.array([Tliqd]),
            np.array([Troof]),
            np.array([tblgrowmean]),
            np.array([tratio]),
            np.array([tbot]),
            np.array([ttop]),
            np.array(distTBL_benchmark.adist, dtype=float),
            np.array(distTBL_benchmark.ndist, dtype=float)
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
    #"""
    #######################################################################################

    return (htbl, hnbl, hRate, pRate, amean, ameanblk, phi0)

def input_troof(Nu: float, XL_p: float, Hnow_p: float, Tbulk_p: float, alloy: BinaryAlloy, const: Parameters) \
        -> tuple[float, float, float, float]:
    """ Constant roof temperature: special case study & benchmarking """

    # Update viscosity:
    if Attributes.nu:
        nu_p = visc0_GIORDANO(Tbulk_p, ModelParameter.EaMod) / const.rhof
    else:
        nu_p = Init.nu

    # Update the heat flux:
    Troof_p = ModelParameter.Troof0
    deltaT = Tbulk_p - Troof_p

    Ra_p = calculate_rayleigh_number(Hnow_p, Tbulk_p, Troof_p, nu_p, ModelParameter)
    Nu = calculate_nusselt_number(Ra_p, nu_p / const.rhof, Shared.regime)
    flux_p = (Nu * deltaT * const.rhof * const.heatcp * const.kappa) / Hnow_p

    # Update the liquidus temperature and nucleation threshold:f
    if XL_p > alloy._Xeut:
        Tliqd_p = alloy.branch_A(XL_p)
    else:
        Tliqd_p = alloy.branch_B(XL_p)
    # if not Attributes.bAnDi:
    #    Tliqd_p = linear_alloy(XL_p, alloy._Tmax1, alloy._Teut, alloy._Xeut)

    Tnucl_p = Tliqd_p - ModelParameter.epsdel 

    return (nu_p, flux_p, Tliqd_p, Tnucl_p, Troof_p)


def input_hflux(tCool: float, XL_p: float, Hnow_p: float, Tbulk_p: float, TliqdJW_p: float, alloy: BinaryAlloy, flux: float, const: Parameters) \
        -> tuple[float, float, float, float, float]:
    """ Varying heat flux: update the derived quantities """

    # Update viscosity:
    if Attributes.nu:
        nu_p = visc0_GIORDANO(Tbulk_p, ModelParameter.EaMod) / const.rhof
    else:
        nu_p = Init.nu

    # Update the heat flux (interpolate the heat flux time series from 1D evolution | fit):
    if flux is None:
        if Attributes.HE and not Attributes.HE_CONST:
            if Attributes.HE_CSTM:
                # Custom ad-hoc specified flux decay:
                flux_p = hflux_custom(t=tCool, f0=Init.flux, A=Shared.fdecay)
            else:
                if Shared.tliqhit + tCool >= Shared.htime[5000]:  # NOTE: empirical index...REMOVE
                    flux_p = abs(hfce(Shared.tliqhit + tCool,
                                Shared.A, Shared.B))  # [W/m2]!  # FIXME: 
                else:
                    flux_p = float(flux_interpolator(Shared.tliqhit + tCool))
        else:
            if Attributes.HE_CSTM:
                flux_p = hflux_custom(t=tCool, f0=Init.flux, A=Shared.fdecay)
            else:
                flux_p = Init.flux
    else:
        flux_p = flux

    # Update the liquidus temperature and nucleation threshold:
    if XL_p > alloy._Xeut:
        Tliqd_p = alloy.branch_A(XL_p)
    else:
        Tliqd_p = alloy.branch_B(XL_p)
    if not Attributes.bAnDi:
        Tliqd_p = TliqdJW_p

    # Update the nucleation threshold: 
    ModelParameter.epsdel = return_epsdel(Tliqd_p)
    Tnucl_p = Tliqd_p - ModelParameter.epsdel
    #print(f"New lag: {ModelParameter.epsdel:.2e}.")    

    # Update the roof temperature:
    deltaT = calculate_deltaT(flux_p, nu_p, Hnow_p, Shared.regime, ModelParameter)
    Troof_p = Tbulk_p - deltaT

    return (nu_p, flux_p, Tliqd_p, Tnucl_p, Troof_p)


def convection_state(nu: float, Hnow: float, Tbulk: float, Troof: float, const: Parameters) \
        -> tuple[float, float, float, float, float]:
    """ Calculate the current diagnostic numbers of the thermal convection """

    Pr = calculate_prandtl_number(nu, const.kappa)
    Ra = calculate_rayleigh_number(Hnow, Tbulk, Troof, nu, const)
    Re = calculate_reynolds_number(Ra, Pr, Shared.regime)
    Nu = calculate_nusselt_number(Ra, Pr, Shared.regime)
    Wrms = nu * Re / Hnow

    return (Pr, Ra, Re, Nu, Wrms)


def calculate_rhs(flux: float, Hnow: float, XL: float, Tliqd: float, hrate: float, prate: float, phiB: float, lheat: float, const: Parameters, kd: float = 0.0) \
        -> tuple[float, float, float]:
    """ Evaluate the right-hand sides of the ODE system (0D case) """

    rhs_h = -hrate
    rhs_xl = -prate * (1. - XL) #* np.power(1. - phiB, - 1.0)  # NOTE: this was wrong????? 
    rhs_tb = -flux / (Hnow * const.rhof * const.heatcp) + lheat * prate / const.heatcp
    rhs_tlJW = -prate * (1. - kd) * (const.TmaxAn - Tliqd)

    return (rhs_h, rhs_xl, rhs_tb, rhs_tlJW)

def calculate_rhs1D(flux: float, Hnow: float, XL: float, Tliqd: float, hrate: float, prate: float, phiB: float, lheat: float, const: Parameters, kd: float = 0.0) \
        -> tuple[float, float, float]:
    """ Evaluate the right-hand sides of the ODE system (1D case) """

    rhs_h = -hrate
    rhs_xl = -prate * (1. - XL)

    return (rhs_h, rhs_xl)


def solve_odes(Hnow, XL, Tbulk, Tliqd, rhs_h, rhs_xl, rhs_tb, rhs_tlJW) -> tuple[float, float, float, float]:
    """ Apply Euler method as predictor """

    Hnow_p    = Hnow + Diag.dtCool * rhs_h
    XL_p      = XL + Diag.dtCool * rhs_xl
    Tbulk_p   = Tbulk + Diag.dtCool * rhs_tb
    TliqdJW_p = Tliqd + Diag.dtCool * rhs_tlJW

    return (Hnow_p, XL_p, Tbulk_p, TliqdJW_p)

def solve_odes1D(Hnow, XL, Tbulk, Tliqd, rhs_h, rhs_xl):
    """ Apply Euler method as predictor """

    Hnow_p = Hnow + Diag.dtCool * rhs_h
    XL_p = XL + Diag.dtCool * rhs_xl

    return (Hnow_p, XL_p)


def calculate_rates(_i: int, tCool: float, Ra: float, Re: float, Wrms: float, Hnow: float, Tbulk: float, Troof: float,
                    Tliqd: float, Tnucl: float, nu: float, const: Parameters, flux: float, Teut: float, 
                    SingleRun: SingleRunAttributes=None
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
    
    """ Calculate the sedimentation/production rates """

    # Calculate the transitional radius:
    Diag.atrn = sqrt((const.gamma * 9. * Wrms * const.rhof * nu) /
                     (2. * const.gacc * (const.rhoc - const.rhof))
                     )
    Diag.astn = sqrt((2.0 * 9. * Wrms * const.rhof * nu) /
                     (2. * const.gacc * (const.rhoc - const.rhof))
                     ) # NOTE: Factor 2.0 from Patočka et al. (2022), stone-like particle transition!

    if Troof <= Tnucl and _i > 0: # NOTE: added _i > 0 for class models A!
        htbl, hnbl, hrate, prate, amean, ameanblk, phiB \
            = calculate_distributions(_i, Ra, Re, Wrms, Hnow, Tbulk, Troof,
                                      Tliqd, Tnucl, Diag.tCool, nu, flux, Teut, SingleRun=SingleRun)

        # FIXME: i had to comment out the Attributes.SRUN part..
        if Shared.onsetc is None: # and not Attributes.SRUN:  # Has crystallization been initiated?
            Shared.onsetc = _i
            logger.warning(f"NO CRYSTALLIZATION! System has been cooling down for time steps: 0-{_i:d} \
                            (~{tCool/RunConstants.ytosec:.3e} years)!")
    else:
        if Shared.onsetc is not None:
            print("[WARNING] - ROOF ABOVE NUCLEATION THRESHOLD!")
            Shared.error = True
        
        Nu = calculate_nusselt_number(Ra, nu / ModelParameter.kappa, Shared.regime)
        htbl = 6.4 * np.power(Ra, -1./3.) * Hnow if not Attributes.TBL else calculate_tbl_thickness(Hnow, Nu)
        hnbl, hrate, prate, amean, ameanblk, phiB = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return (htbl, hnbl, hrate, prate, amean, ameanblk, phiB)

def evol_chamber_step1D(_i: int, tCool: float, Hnow: float, XL: float, Tbulk: float, Troof: float, Tliqd: float, Tnucl: float, nu: float,
        flux: float, lheat: float, alloy: BinaryAlloy) -> tuple:
    """ Evaluates one time step of the magma chamber time evolution (1D ParMaCH) """

    Pr, Ra, Re, Nu, Wrms = convection_state(
        nu, Hnow, Tbulk, Troof, ModelParameter)
    
    htbl, hnbl, hrate, prate, amean, ameanblk, phiB = calculate_rates(
        _i, tCool, Ra, Re, Wrms, Hnow, Tbulk, Troof, Tliqd, Tnucl, nu, ModelParameter, flux, alloy._Teut
    )

    # Released latent heat: TODO: VERTICAL SOURCE! WE NEED distBLK2D and distTBL2D computed!
    lheatTBL = Shared.prateTBL * ModelParameter.rhof * alloy._Lheat if Shared.prateTBL is not None else 0.0 # [W/m^3] volumetric source!
    lheatBLK = Shared.prateBLK * ModelParameter.rhof * alloy._Lheat if Shared.prateBLK is not None else 0.0 # [W/m^3] volumetric source!

    # Solve ODEs:
    rhs_h, rhs_xl = calculate_rhs1D(flux, Hnow, XL, Tliqd, hrate, prate, phiB, lheat, ModelParameter)
    Hnow_p, XL_p  = solve_odes1D(Hnow, XL, Tbulk, Tliqd, rhs_h, rhs_xl) 

    return (Hnow_p, XL_p, lheatTBL, lheatBLK, Ra, Re, flux, Nu, nu, Wrms,
            htbl, hnbl, hrate, prate, amean, ameanblk, phiB
        )

def evol_chamber_step(
        _i: int, tCool: float, Hnow: float, XL: float, Tbulk: float, Troof: float, Tliqd: float, Tnucl: float, nu: float,
        flux: float, lheat: float, alloy: BinaryAlloy, mode: int, order: int, rhs_reset: bool, max_iter: int = 8) -> tuple:
    """ Evaluates one time step of the magma chamber time evolution (0D ParMaCH) """

    if not hasattr(evol_chamber_step, "rhs") or rhs_reset:
        evol_chamber_step.rhs = deque(maxlen=3)
    
    ##% I) CONVECTION STATE & PREDICTOR PHASE %##
    Pr, Ra, Re, Nu, Wrms = convection_state(
        nu, Hnow, Tbulk, Troof, ModelParameter)
    htbl, hnbl, hrate, prate, amean, ameanblk, phiB = calculate_rates(
        _i, tCool, Ra, Re, Wrms, Hnow, Tbulk, Troof, Tliqd, Tnucl, nu, ModelParameter, flux, alloy._Teut
    )

    rhs_h0, rhs_xl0, rhs_tb0, rhs_tlJW0 = calculate_rhs(
        flux, Hnow, XL, Tliqd, hrate, prate, phiB, lheat, ModelParameter)

    # Predictor & solve ODEs:
    if calculate_rates.counter <= 2:
        rhs_h = rhs_h0
        rhs_xl = rhs_xl0
        rhs_tb = rhs_tb0
        rhs_tlJW = rhs_tlJW0

    else:
        rhs_h, rhs_xl, rhs_tb, rhs_tlJW = predictor_evol(
            order=order,
            rhs_h0=rhs_h0,
            rhs_xl0=rhs_xl0,
            rhs_tb0=rhs_tb0,
            rhs_tlJW0=rhs_tlJW0,
            rhs=evol_chamber_step.rhs
        )
    Hnow_p, XL_p, Tbulk_p, TliqdJW_p = solve_odes(
        Hnow, XL, Tbulk, Tliqd, rhs_h, rhs_xl, rhs_tb, rhs_tlJW)

    #print("odes:", Hnow_p, XL_p, Tbulk_p, TliqdJW_p)

    # Update other state variables:
    match mode:
        case 1:  # Constant roof temperature:
            (nu_p, flux_p, Tliqd_p, Tnucl_p, Troof_p) = input_troof(
                Nu=Nu,
                XL_p=XL_p,
                Hnow_p=Hnow_p,
                Tbulk_p=Tbulk_p,
                alloy=alloy,
                const=ModelParameter
            )

        case 2:  # Varying heat flux:
            (nu_p, flux_p, Tliqd_p, Tnucl_p, Troof_p) = input_hflux(
                tCool=tCool + Diag.dtCool,
                XL_p=XL_p,
                Hnow_p=Hnow_p,
                Tbulk_p=Tbulk_p,
                TliqdJW_p=TliqdJW_p,
                alloy=alloy,
                flux=None,
                const=ModelParameter
            )

    # IMPLICIT EULER (via iterations!)
    if order == -1:
        if Shared.onsetc is not None:
            calculate_rates.counter += 1

            # Initial guess:
            Hnow_next, XL_next, Tbulk_next = Hnow_p, XL_p, Tbulk_p
            nu_next, flux_next, Tliqd_next, Tnucl_next, Troof_next = nu_p, flux_p, Tliqd_p, Tnucl_p, Troof_p

            # Iterations:
            for i in range(max_iter):
                Pr_next, Ra_next, Re_next, Nu_next, Wrms_next = convection_state(
                    nu_next, Hnow_next, Tbulk_next, Troof_next, ModelParameter)
                _, _, hrate_next, prate_next, _, _, phiB_next = calculate_rates(
                    _i, tCool + Diag.dtCool, Ra_next, Re_next, Wrms_next, Hnow_next, Tbulk_next, Troof_next, Tliqd_next, Tnucl_next, nu_next, ModelParameter,
                    flux, alloy._Teut
                )

                # Resubstitute and solve the system again:
                rhs_hnew, rhs_xlnew, rhs_tbnew = calculate_rhs(
                    flux_next, Hnow_next, XL_next, hrate_next, prate_next, phiB_next, lheat, ModelParameter)
                Hnow_new, XL_new, Tbulk_new = solve_odes(
                    Hnow, XL, Tbulk, rhs_hnew, rhs_xlnew, rhs_tbnew)

                match mode:
                    case 1:
                        nu_new, flux_new, Tliqd_new, Tnucl_new, Troof_new = input_troof(
                            Nu_next, XL_new, Hnow_new, Tbulk_new, alloy, ModelParameter)
                    case 2:
                        nu_new, flux_new, Tliqd_new, Tnucl_new, Troof_new = input_hflux(
                            tCool, XL_new, Hnow_new, Tbulk_new, alloy, None, ModelParameter)

                if i < max_iter - 1:
                    Hnow_next = Hnow_new
                    XL_next = XL_new
                    Tbulk_next = Tbulk_new
                    nu_next = nu_new
                    flux_next = flux_new
                    Tliqd_next = Tliqd_new
                    Tnucl_next = Tnucl_new
                    Troof_next = Troof_new

        else:
            Hnow_new, Tbulk_new, XL_new, nu_new, flux_new = Hnow_p, Tbulk_p, XL_p, nu_p, flux_p
            Tliqd_new, Tnucl_new, Troof_new = Tliqd_p, Tnucl_p, Troof_p

        return (Hnow_new, Tbulk_new, XL_new, nu_new, flux_new, Tliqd_new, Tnucl_new, Troof_new,
                Ra, Re, flux, Nu, nu, Wrms,
                htbl, hnbl, hrate, prate, amean, ameanblk, phiB
                )

    # EXPLICIT EULER!
    #print(XL, XL_p, Tliqd_p)
    if order == 0:
        if Shared.onsetc is not None:
            calculate_rates.counter += 1
        # Determine the Grossmann-Lohse convection regime:
        #Shared.regime = regimeGL(Ra, Pr, mt=Attributes.MT)
        return (Hnow_p, Tbulk_p, XL_p, nu_p, flux_p, Tliqd_p, Tnucl_p, Troof_p,
                Ra, Re, flux, Nu, nu, Wrms,
                htbl, hnbl, hrate, prate, amean, ameanblk, phiB
                )

    # NOTE: the following chunk executed only for order = [1,2,3,4]
    # MULTISTEP CORRECTOR-PREDICTOR APPROACH!
    if calculate_rates.counter <= 2:  # initiate first three steps with Euler!
        evol_chamber_step.rhs.append([
            rhs_h0, rhs_xl0, rhs_tb0
        ])
        if Shared.onsetc is not None:
            calculate_rates.counter += 1
        return (Hnow_p, Tbulk_p, XL_p, nu_p, flux_p, Tliqd_p, Tnucl_p, Troof_p,
                Ra, Re, flux, Nu, nu, Wrms,
                htbl, hnbl, hrate, prate, amean, ameanblk, phiB
                )

    # % II) CORRECTOR PHASE ##%
    Hnow_p0 = Hnow_p
    Tbulk_p0 = Tbulk_p
    XL_p0 = XL_p
    if Attributes.PCITER:
        for i in range(max_iter):
            Pr_p, Ra_p, Re_p, Nu_p, Wrms_p = convection_state(
                nu_p, Hnow_p, Tbulk_p, Troof_p, ModelParameter)
            _, _, hrate_p, prate_p, _, _, phiB_p = calculate_rates(
                _i, tCool + Diag.dtCool, Ra_p, Re_p, Wrms_p, Hnow_p, Tbulk_p, Troof_p, Tliqd_p, Tnucl_p, nu_p, ModelParameter,
                flux, alloy._Teut
            )
            rhs_hp, rhs_xlp, rhs_tbp = calculate_rhs(
                flux_p, Hnow_p, XL_p, hrate_p, prate_p, phiB_p, lheat, ModelParameter)

            rhs_hcorr, rhs_xlcorr, rhs_tbcorr = corrector_evol(
                order=order,
                rhs_hp=rhs_hp,
                rhs_xlp=rhs_xlp,
                rhs_tbp=rhs_tbp,
                rhs_h0=rhs_h0,
                rhs_xl0=rhs_xl0,
                rhs_tb0=rhs_tb0,
                rhs=evol_chamber_step.rhs
            )

            # Final corrected variables:
            Hnow_f, XL_f, Tbulk_f = solve_odes(
                Hnow, XL, Tbulk, rhs_hcorr, rhs_xlcorr, rhs_tbcorr)
            match mode:
                case 1:
                    nu_f, flux_f, Tliqd_f, Tnucl_f, Troof_f = input_troof(
                        Nu_p, XL_f, Hnow_f, Tbulk_f, alloy, ModelParameter)
                case 2:
                    nu_f, flux_f, Tliqd_f, Tnucl_f, Troof_f = input_hflux(
                        tCool, XL_f, Hnow_f, Tbulk_f, alloy, None, ModelParameter)

            if i < max_iter - 1:
                Hnow_p = Hnow_f
                XL_p = XL_f
                Tbulk_p = Tbulk_f
                nu_p = nu_f
                flux_p = flux_f
                Tliqd_p = Tliqd_f
                Tnucl_p = Tnucl_f
                Troof_p = Troof_f

    # Update the RHS buffer:
    evol_chamber_step.rhs.append([rhs_h0, rhs_xl0, rhs_tb0])

    # Determine the Grossmann-Lohse convection regime:
    Shared.regime = regimeGL(Ra, Pr, mt=Attributes.MT)

    calculate_rates.counter += 1
    return (Hnow_f, Tbulk_f, XL_f, nu_f, flux_f, Tliqd_f, Tnucl_f, Troof_f,
            Ra, Re, flux, Nu, nu, Wrms, htbl, hnbl, hrate, prate, amean, ameanblk, phiB
            )

# Single run solver of the ParMaCH model:
def srun_solver(SingleRun: SingleRunAttributes) -> None:
    _i = 1; tCool = 0.0; Teut = 0.0

    # Single run requires the computation of distTBL2D:
    #Attributes.TBL_METHOD = 2 # FIXME: <--- tohle nebude třeba?
    
    # If the physics_check() passed through, you automatically have crystallisation guaranteed:
    Shared.onsetc = _i

    # Save the initial values of viscosity, heat flux, and liquidus temperature:
    Tbulk = SingleRun.Tbulk
    Troof = SingleRun.Troof
    Tliqd = SingleRun.Tliqd
    Tnucl = SingleRun.Tnucl
    Hnow  = SingleRun.Hnow
    nu    = SingleRun.nu
    flux  = SingleRun.flux
    Ra    = SingleRun.Ra
    Re    = SingleRun.Re 
    Wrms  = SingleRun.Wrms

    # Evade numba error:
    Shared.Tref = Tliqd

    # Parameterization of convection:
    #Pr, Ra, Re, Nu, Wrms = convection_state(nu, Hnow, Tbulk, Troof, ModelParameter)
    
    htbl, hnbl, hrate, prate, amean, ameanblk, phiB = calculate_rates(
        _i, tCool, Ra, Re, Wrms, Hnow, Tbulk, Troof, Tliqd, Tnucl, nu, ModelParameter, flux, Teut
    )

    # prate je kompletní, chci ty separátní 
    #print(f"prateTBL: {Shared.prateTBL:.3e}")
    #print(f"prateBLK: {Shared.prateBLK:.3e}")

    # OK, now I need to compute latent heat from these guys:
    lheatTBL = Shared.prateTBL * ModelParameter.rhof * htbl * ModelParameter.Lheat
    lheatBLK = Shared.prateBLK * ModelParameter.rhof * (Hnow - htbl) * ModelParameter.Lheat

    #print("LATENT HEAT:")
    #print(f"latent heat (TBL): {lheatTBL:.3e}")
    #print(f"latent heat (BLK): {lheatBLK:.3e}")



    # COMPUTE THE LATENT HEAT HERE:
    #plt.plot(Shared.distBLK.adist, Shared.distBLK.ndist)
    #plt.plot(Shared.distSED.adist, Shared.distSED.ndist)
    #plt.show()


    print(" [WARNING] - srun_solver not finished!")
    exit()

    """
        Pro zadané parametry single běhu:
        i)   vstup: single run, z hlediska kompozice je to jedno, prostě spočtu pro vybranou viskozitu a liquidus teplotu
        ii)  přepiš veličiny pomocí SR, pak zavolej evol_chamber_step
             
             jak ale budeš počítat latentní teplo? to musíš udělat v &

        iii) výstup, steady-state snapshot? 
        
    """

    # WHAT DO I WANNA RETURN? I want to return the latent heat sources, but first simply as two numbers!

    return 


# The main ODMC-ParMaCh solver (empirically/estimate decaying heat flux):
def ODMC_solver(
    evol_file: h5py.File,           # h5 outfile file
    alloy:     Type[BinaryAlloy],   # binary alloy
    mode:      int,                 # varying heat flux | constant roof temperature
    order:     int,                 # order of the integration scheme
    steps:     int,                 # DEBUG number of steps
    calibrate: bool=False,          # initial timestep-tuning, e.g., 100 empty steps
    mem_track: bool=False           # memory tracing (on/off)

) -> None:
    
    """
        ###! THERMAL EVOLUTION OF THE CHAMBER - 0D-ENERGY BALANCE !###

            %> tracks the thermal evolution of the chamber, supplied heat flux
            %> settling of a single mineral phase
            %> the simulation is halted upon exceeding the eutectic temperature
    """

    # Heat flux interpolation object:
    if Attributes.HE:
        global flux_interpolator
        flux_interpolator = interp1d(Shared.htime, Shared.froof, kind="cubic")

    # Jarvis & Woods limit, non-modified enthalpy!
    if not Attributes.bAnDi:
        alloy._Lheat = ModelParameter.Lheat
        alloy._Teut = ModelParameter.Teutec

    # Save the initial values of viscosity, heat flux, and liquidus temperature:
    Tbulk = Init.Tbulk; Tliqd = Init.Tliqd; Tnucl = Init.Tnucl; Troof = Init.Troof
    Hnow = Init.Hnow; nu = Init.nu; XL = Init.XL; flux = Init.flux
    uctbl0 = Init.Tliqd - Init.Troof - ModelParameter.epsdel

    try:
        if calibrate:
            # Calculate the first {xsteps} steps to test whether the initial time step is appropriate:
            Diag.tCool = Hnow * ModelParameter.rhof * ModelParameter.heatcp * (Tbulk - Troof) / flux
            Shared.fdecay = 0.8 * Diag.tCool; Shared.dtCool0 = Diag.dtCool; crash_flag = True; xsteps = 10
            if not hasattr(calculate_rates, "counter"): calculate_rates.counter = 0

            while crash_flag:
                _ical = 0; tCool_cal = 0.0
                for _ in range(xsteps): 
                    Tbulk_old = Tbulk
                    (Hnow, Tbulk, XL, nu_i, flux_i, Tliqd, Tnucl, Troof, Ra, Re, flux,
                    Nu, nu, Wrms, htbl, hnbl, hrate, prate, amean, ameanblk, phiB)    \
                        = evol_chamber_step(_ical, tCool_cal, Hnow, XL, Tbulk, Troof, Tliqd, Tnucl, nu, flux, 
                                            alloy._Lheat, alloy, mode, order=order, rhs_reset=True)
                    _ical += 1; tCool_cal += Diag.dtCool

                    termination_error, _, _ = ODMC_errorcheck(
                        _i=_ical, mode=mode, Tbulk=Tbulk, Tbulko=Tbulk_old, Tliqd=Tliqd, Troof=Troof, Teut=alloy._Teut, Hnow=Hnow,  nu=nu, phiB=phiB, 
                        crates=calculate_rates.counter, debug_steps=None, calibration_steps=False
                    )

                    if termination_error: break

                if termination_error:
                    _dtold = Diag.dtCool
                    Diag.dtCool /= 5.0 
                    print(f" [WARNING] The time step was modified ({_dtold:.2e}{Units.tunit} to {Diag.dtCool:.2e}{Units.tunit})!")

                else:
                    print(f" [WARNING] The time step remained the same ({Diag.dtCool:.2e}{Units.tunit}), calibration procedure terminated!")
                    crash_flag = False
                    
                # Restore initial values:
                Tbulk = Init.Tbulk; Tliqd = Init.Tliqd; Tnucl = Init.Tnucl; Troof = Init.Troof
                Hnow = Init.Hnow; nu = Init.nu; XL = Init.XL; flux = Init.flux
                uctbl0 = Init.Tliqd - Init.Troof - ModelParameter.epsdel

                # Restore:
                Shared.onsetc = None

                if Diag.dtCool <= 1e0:
                    print(" [WARNING] Ridiculously small time step, check manually!")
                    logger.critical(" [WARNING] Ridiculously small time step, check manually!")
                    exit()

        # Restore the shared logical flags:
        # FIXME: Shared.onsetc = None

        # Main ODMC simulation: 
        _i = 0; tCool = 0.0; arel1 = 0.0; _onsetc = False; eqlix_write = None; eutix_write = None
        if not hasattr(calculate_rates, "counter") or calculate_rates.counter > 0: calculate_rates.counter = 0
        if mem_track: objgraph.show_growth(limit=10)

        # SOLIDIFICATION LOOP UP TO THE EUTECTIC TEMPERATURE:
        while Tliqd >= alloy._Teut:
            if mode == 1:
                print(f"\r Time evolution at {abs(1.e2 - 1.e2*(Tliqd - Troof - ModelParameter.epsdel) / (uctbl0)):.3f} [%] ({_i:d} steps through). ",
                      end="", flush=True)   
            if mode == 2:
                print(f"\r Time evolution at {abs(1.e2 - 1.e2*(Tliqd - alloy._Teut) / (ModelParameter.Tliqd0 - alloy._Teut)):.3f} [%] ({_i:d} steps through). ",
                      end="", flush=True)
                
            if mode == 3: # FIXME: remove?
                print(" [WARNING] - SINGLE STEADY-STATE COMPUTATION.")        

            # Calculate the characteristic cooling time scale:
            Diag.tCool = Hnow * ModelParameter.rhof * ModelParameter.heatcp * (Tbulk - Troof) / flux
            if _i == 0: Shared.fdecay = 0.8 * Diag.tCool # it was 0.1 for an experiment before
            if _i == 0: Shared.dtCool0 = Diag.dtCool

            #if _i == 0: print(f"ČASOVÝ KROK {Diag.dtCool:3e}.")

            # Nullification of references:
            Diag.cnt0DMC = 0.0; Shared.nsus = None; Shared.track = None

            # Save previous state of the chamber:
            Tbulk_old = Tbulk
            Tliqd_old = Tliqd
            Troof_old = Troof
            Hnow_old = Hnow
            Grate_old = Hort_grow(Tbulk_old, Tliqd_old, ModelParameter)
            Nrate_old = Hort_nucl(Troof_old, Tliqd_old, ModelParameter)
            nu_old = nu

            # Evaluate one step (predictor-corrector method):
            (Hnow, Tbulk, XL, nu_i, flux_i, Tliqd, Tnucl, Troof, Ra, Re, flux,
             Nu, nu, Wrms, htbl, hnbl, hrate, prate, amean, ameanblk, phiB) \
                = evol_chamber_step(_i, tCool, Hnow, XL, Tbulk, Troof, Tliqd, Tnucl, nu, flux, alloy._Lheat, alloy, mode,
                                    order=order, rhs_reset=True)
            flux = flux_i
            nu = nu_i

            if Shared.onsetc is not None:
                print(flux, nu, Tbulk, Troof)

            """
            try:
                print("min grow tbl:", Diag.tblgrwmin)
                print("max grow tbl:", Diag.tblgrwmax)
                print("bulk growth:", Diag.blkgrow)
                print("avg. tbl rate", (Diag.tblgrwmin + Diag.tblgrwmax)/2.0)
                print((Diag.tblgrwmin+Diag.tblgrwmax)/2./Diag.blkgrow)
                print("NUSSELT:", Nu)
                print("PAPER FACTOR:", (1/Nu)**(1./3.) * ((Diag.tblgrwmin+Diag.tblgrwmax)/2./Diag.blkgrow)**(1./3.))

            except RuntimeWarning or ZeroDivisionError: pass
            """

            #if Shared.onsetc is not None:
            #    __lmd = lmd_stokes(nu, ModelParameter)
            #    alpha = __lmd * Diag.blkgrow**2 / 3.
            #    beta  = Wrms #(Wrms+__lmd*Diag.amaxtbl**2)
            #    gama  = 0 #__lmd*Diag.amaxtbl*Diag.blkgrow
            #    delta = -Hnow

            #    def f1(t): return alpha * t**3 + beta * t + delta + gama * t**2
            #    tbot = root_scalar(f1, bracket=[0, 1e6], method="brentq").root
            #    #ttop = (Diag.atrn - Diag.amaxtbl) / Diag.blkgrow
            #    ttop = Diag.atrn / Diag.blkgrow
            #    print(f"ČASY: TBOT={tbot:.4e} | TTOP={ttop:.4e} ")
            #    print(f"PAPER: {tbot/ttop :.2f}")
            #    print(f"IDLE-MIXING CONTRIBUTION: {1e3*Diag.blkgrow*tbot} [mm]")

                #trshld = ((Wrms**3/2)/Hnow) * ( (9*ModelParameter.gamma*ModelParameter.rhof*nu) / (2*ModelParameter.gacc*50.0))**(1./2)
                #print(Diag.blkgrow, trshld)

            """
            if _i > 0: 
                print(Diag.blkgrow)
                print("PAPER FACTOR:", (1./Nu)**(1./3.) * ((Diag.tblgrwmin+Diag.tblgrwmax)/2./Diag.blkgrow)**(1./3.))
                print((Diag.tblgrwmin+Diag.tblgrwmax)/2./Diag.blkgrow)
                print("Tbulk:", Tbulk)
                print("Troof:", Troof)
                print("Tliqd:", Tliqd)
            """
                
            if np.isnan([Hnow, Tbulk, XL, nu_i, flux_i, Tliqd, Tnucl, Troof, Ra, Re, flux,
             Nu, nu, Wrms, htbl, hnbl, hrate, prate, amean, ameanblk, phiB, flux, nu]).any(): 
                print(" [WARNING] - One of the evolutionary variables is NaN!")
                Shared.error = True

            # Calculate the relative gradients of individual variables:
            try:
                Grate = Hort_grow(Tbulk, Tliqd, ModelParameter)
                Nrate = Hort_nucl(Troof, Tliqd, ModelParameter)
                dTbdt = abs(Tbulk_old - Tbulk) / Diag.dtCool
                dTldt = abs(Tliqd_old - Tliqd) / Diag.dtCool
                dTrdt = abs(Troof_old - Troof) / Diag.dtCool
                dGrdt = abs(Grate_old - Grate) / Diag.dtCool
                dNrdt = abs(Nrate_old - Nrate) / Diag.dtCool
                dnudt = abs(nu_old - nu)       / Diag.dtCool
            
            except ZeroDivisionError: pass

            try:
                dGrdh = (Grate - Grate_old) / (Hnow - Hnow_old)
                dnudh = (nu - nu_old) / (Hnow - Hnow_old)
                daddh = ( dnudh/nu + (-1.)/Hnow + dGrdh/Diag.blkgrow )
            
            except RuntimeWarning:
                pass # TODO

            """
            try: 
                arel0 = arel1
                arel1 = abs(amean_old - amean) / abs(amean_old)
                if 1.e2*arel1 > 0.1 and _i > 100: 
                    if Attributes.DEBUG: print(f"Relative difference in the mean radius {1.e2*arel1:.3e} [%].")
                    dtold = Diag.dtCool
                    Diag.dtCool = adapt_timestepPI(
                        dt=Diag.dtCool, arel0=arel0, arel1=arel1, dtmax=Shared.dtCool0
                    )
                    if Attributes.DEBUG: print(f"Time step modified to {Diag.dtCool:.2f}")
                    if Attributes.DEBUG: print(f"Time step modified from {dtold:.2f} [s] to {Diag.dtCool:.2f} [s].")

            except NameError or RuntimeWarning: pass
            """
            if Shared.onsetc: amean_old = amean

            # Possible warnings:
            if flux < 0.1: print(" [WARNING] - You are below the theoretical limit of 0.1 W/m2 by Martin 1997.")

            # Save the state of the simulation: # TODO: MAYBE SAVE ONLY EVERY K-TH step?
            evol_state = [tCool, Tbulk, Troof, abs(Tbulk - Troof), Tliqd, Tnucl, (Init.Hnow - Hnow), amean, Ra, Re, flux,
                          Nu, phiB, nu, Wrms, XL, ameanblk, 0.0, 0.0, 0.0 
                        ]

            diag_state = [Diag.cnt0DMC, Diag.cntHBJW, Diag.cntHNJW, Diag.amaxtbl, Diag.cin, Diag.cout,
                          Diag.tCool, Diag.dtCool, Diag.tbres, Diag.tresjw, Diag.tssblk, Diag.atrn,
                          Diag.blkgrow, Diag.tblnucmin, Diag.tblnucmax, Diag.tblgrwmin, Diag.tblgrwmax,
                          dTbdt, dTldt, dTrdt, dGrdt, dNrdt, prate, hrate, htbl, hnbl, Diag.tplume, Diag.setmarker
                        ]

            evol_file["time_evolution/vars"].resize(
                evol_file["time_evolution/vars"].shape[0] + 1, axis=0)
            evol_file["diagnostics/vars"].resize(
                evol_file["diagnostics/vars"].shape[0] + 1, axis=0)
            evol_file["time_evolution/vars"][-1] = evol_state
            evol_file["diagnostics/vars"][-1] = diag_state
            if isinstance(Shared.onsetc, int) and _onsetc is False:
                evol_file["indices"].create_dataset(
                    "onsetc", data=Shared.onsetc)
                _onsetc = True

            # Update the physical time of the simulation:
            _i += 1; tCool += Diag.dtCool

            # Memory management (numba precaution!):      
            if _i == 1: gc.collect()
            if mem_track and _i % 20 == 0:
                objgraph.show_growth(limit=10)
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss
                collected = gc.collect()
                mem_after = process.memory_info().rss
                print(f"Unreachable objects collected: {collected}")
                print(f"Memory before gc.collect(): {mem_before / 1024**2:.2f} MB.")
                print(f"Memory after  gc.collect(): {mem_after  / 1024**2:.2f} MB.")
                print(f"Memory freed: {(mem_before - mem_after) / 1024**2:.2f} MB.")

            termination_error, eqlix_write, eutix_write = ODMC_errorcheck(
                _i=_i, mode=mode, Tbulk=Tbulk, Tbulko=Tbulk_old, Tliqd=Tliqd, Troof=Troof, Teut=alloy._Teut, phiB=phiB, 
                Hnow=Hnow, nu=nu, crates=calculate_rates.counter, debug_steps=steps, calibration_steps=False
            )

            #termination_error = False

            if Shared.error: Shared.idxend = _i
            if eutix_write: evol_file["indices"].create_dataset("idxeut", data=Shared.idxeut)
            if eqlix_write: evol_file["indices"].create_dataset("idxend", data=Shared.idxend)
            if termination_error: break
            if Shared.error: break
            plt.close("all") # sanity check!

    except KeyboardInterrupt:
        Shared.idxend = _i
        print()
        print("[WARNING] - KEYBOARD INTERRUPTION! Exiting 1DMC solver early!")

    finally:
        evol_file.close()

# The main solver 1DMC-ParMaCh solver:
def IDMC_solver(
    evol_file:     h5py.File,           # h5 outfile file
    alloy:         Type[BinaryAlloy],   # binary alloy
    mem_track:     bool=False,          # memory tracking
    save_flag1D:   bool=True,           # save the state of the 1D equation  
    latent_source: bool=True            # latent heat source       

) -> None:
    # NOTE: no Calibration flag needed since the time step size is determined from the 1D heat equation (or maybe I will add it later?)

    """
        ###! THERMAL EVOLUTION OF THE CHAMBER - 1D-ENERGY BALANCE !###
            %> tracks the thermal evolution of the chamber, supplied heat flux
            %> settling of a single mineral phase
            %> the simulation is halted upon exceeding the eutectic temperature
    """

    # Save the initial values of viscosity, heat flux, and liquidus temperature:
    Tbulk = Init.Tbulk; Tliqd = Init.Tliqd; Tnucl = Init.Tnucl; Troof = Init.Troof
    Hnow = Init.Hnow; nu = Init.nu; XL = Init.XL; flux = Init.flux; suc = Init.suc

    # Save the initial 1D setup:
    z = Init.z; k = Init.k; T = Init.T; rhocp = Init.rhocp

    try:
        # Main 1DMC simulation: 
        _i = 0; tCool = 0.0; arel1 = 0.0; _onsetc = False; eqlix_write = None; eutix_write = None
        if not hasattr(calculate_rates, "counter") or calculate_rates.counter > 0: calculate_rates.counter = 0
        if mem_track: objgraph.show_growth(limit=10)

        Lj = 0.0; Ljp1 = 0.0
        Tbj = 0.0; Tbjp1 = 0.0

        # SOLIDIFICATION LOOP UP TO THE EUTECTIC TEMPERATURE:
        while Tliqd >= alloy._Teut:
            print(f"\r Time evolution at {abs(1.e2 - 1.e2*(Tliqd - alloy._Teut) / (ModelParameter.Tliqd0 - alloy._Teut)):.2e} [%] ({_i:d} steps through). ",
                      end="", flush=True)            

            Ljp1 = Lj; Tbj = Tbjp1

            # Save previous state of the chamber:
            Tbulk_old = Tbulk
            Tliqd_old = Tliqd
            Troof_old = Troof
            Hnow_old = Hnow
            Grate_old = Hort_grow(Tbulk_old, Tliqd_old, ModelParameter)
            Nrate_old = Hort_nucl(Troof_old, Tliqd_old, ModelParameter)
            nu_old = nu

            # Nullification of references:
            Diag.cnt0DMC = 0.0; Shared.nsus = None; Shared.track = None

            # Phase §1 - calculate the steady-state snapshot:
            (Hnow, XL, LheatTBL, LheatBLK, Ra, Re, flux, Nu, nu, Wrms,
            htbl, hnbl, hrate, prate, amean, ameanblk, phiB) \
                = evol_chamber_step1D(_i=_i, tCool=tCool, Hnow=Hnow, XL=XL, Tbulk=Tbulk, Troof=Troof,
                                        Tliqd=Tliqd, Tnucl=Tnucl, nu=nu, flux=flux, lheat=alloy._Lheat,
                                        alloy=alloy
                                    )

            # Update the index of the TBL:
            #Constants1D._zchtbl0 = Constants1D._zchtop0 + htbl # NOTE: it should be already determined in the last loop, lets comment it out!

            # Interpolate the latent heat onto a grid (generate a np.ndarray object):
            # NOTE: for now uniform sources
            # TODO: implement vertically-dependent source!

            # Prescribe the latent heat source onto the computational grid:
            if latent_source: Lheat = define_source(z=z, lheatTBL=LheatTBL, lheatBLK=LheatBLK, const=Constants1D)
            else: Lheat = np.zeros_like(z, dtype=np.float64)

            # Phase §2 - solve a 1D heat equation time step:
            (T, Tbulk, flux, flux_bot, _lhs, _rhs) = step1D_with_latent_heat(step=_i, T=T, L=Lheat, k=k, rhocp=rhocp, z=z, \
                                                       dt=Diag.dtCool, Tbulk_old=Tbulk_old, const=Constants1D
                                                    )

            # Phase §3 - update the derived quantities:
            # NOTE: nu, Tliqd, Tnucl, Troof now correspond to the compute T, Tbulk and flux!
            (nu, _, Tliqd, Tnucl, Troof) = input_hflux(tCool=tCool, XL_p=XL, Hnow_p=Hnow, 
                                                        Tbulk_p=Tbulk, TliqdJW_p=None, alloy=alloy, 
                                                            flux=flux, const=ModelParameter
                                                    )

            # Phase §4 - update the 1D heat equation grid:
            dh = Shared.prate * Hnow * Diag.dtCool if Shared.prate is not None else 0.0
            (z, k, rhocp) = update_grid1D(T=T,
                                          z_old=z,
                                          k_old=k,
                                          rhocp_old=rhocp,
                                          dh=dh,
                                          htbl=htbl,
                                          const=Constants1D
                                        )

            Lj = Lheat; Tbj = Tbulk
            #try:
            #    dLdt = abs(Ljp1 - Lj) / abs(Tbjp1 - Tbj)
            #except RuntimeWarning: 
            #    dLdt = 0.0
            #print(f"Do we need iterations? {np.max(dLdt*(Diag.dtCool/(rhocp)))}.")


            #if _i % Attributes.printstep == 0: print(f"{Tliqd:.7e} | {Tbulk:.7e} | {Troof:.7e} | {flux:.7e}")
            
            # Calculate the relative gradients of individual variables:
            try:
                Grate = Hort_grow(Tbulk, Tliqd, ModelParameter)
                Nrate = Hort_nucl(Troof, Tliqd, ModelParameter)
                dTbdt = abs(Tbulk_old - Tbulk) / Diag.dtCool
                dTldt = abs(Tliqd_old - Tliqd) / Diag.dtCool
                dTrdt = abs(Troof_old - Troof) / Diag.dtCool
                dGrdt = abs(Grate_old - Grate) / Diag.dtCool
                dNrdt = abs(Nrate_old - Nrate) / Diag.dtCool
                dnudt = abs(nu_old - nu)       / Diag.dtCool
            
            except ZeroDivisionError: pass

            # Save the state of the simulation:
            evol_state = [tCool, Tbulk, Troof, abs(Tbulk - Troof), Tliqd, Tnucl, (Init.Hnow - Hnow), amean, Ra, Re, flux,
                          Nu, phiB, nu, Wrms, XL, ameanblk, flux_bot, _lhs, _rhs
                          ]

            diag_state = [Diag.cnt0DMC, Diag.cntHBJW, Diag.cntHNJW, Diag.amaxtbl, Diag.cin, Diag.cout,
                          Diag.tCool, Diag.dtCool, Diag.tbres, Diag.tresjw, Diag.tssblk, Diag.atrn,
                          Diag.blkgrow, Diag.tblnucmin, Diag.tblnucmax, Diag.tblgrwmin, Diag.tblgrwmax,
                          dTbdt, dTldt, dTrdt, dGrdt, dNrdt, prate, hrate, htbl, hnbl, Diag.tplume, Diag.setmarker
                          ]

            evol_file["time_evolution/vars"].resize(
                evol_file["time_evolution/vars"].shape[0] + 1, axis=0)
            evol_file["diagnostics/vars"].resize(
                evol_file["diagnostics/vars"].shape[0] + 1, axis=0)
            evol_file["time_evolution/vars"][-1] = evol_state
            evol_file["diagnostics/vars"][-1] = diag_state
            if isinstance(Shared.onsetc, int) and _onsetc is False:
                evol_file["indices"].create_dataset(
                    "onsetc", data=Shared.onsetc)
                _onsetc = True

            termination_error, eqlix_write, eutix_write = ODMC_errorcheck(
                _i=_i, mode=None, Tbulk=Tbulk, Tbulko=Tbulk_old, Tliqd=Tliqd, Troof=Troof, Teut=alloy._Teut, phiB=phiB, 
                Hnow=Hnow, nu=nu, crates=calculate_rates.counter, debug_steps=None, calibration_steps=False
            )

            # Save the current solution:
            if _i % Attributes.printstep == 0 and save_flag1D: 
                plot_1D(T0=None, T=T, z=z, step_hit=None, Tliqd=ModelParameter.Tliqd0, 
                        tcool=tCool, step=_i, const=Constants1D)
                save_solution(T=T, z=z, k=k, rhocp=rhocp, step=_i)

            _i += 1; tCool += Diag.dtCool
            if Shared.error: Shared.idxend = _i
            if eutix_write: evol_file["indices"].create_dataset("idxeut", data=Shared.idxeut)
            if eqlix_write: evol_file["indices"].create_dataset("idxend", data=Shared.idxend)
            if termination_error: break
            if Shared.error: break
            plt.close("all") # sanity check!

            #if _i == 500: break

    except KeyboardInterrupt:
        Shared.idxend = _i
        print()
        print("[WARNING] - KEYBOARD INTERRUPTION! Exiting 1DMC solver early!")

    finally:
        evol_file.close()

#####################################################################################################################
# % end of the module!