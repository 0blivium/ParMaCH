# Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 3)
# Module: Plotting and visualiation (post-processing)
# Compilation: python3 mPost.py --tar_direct=... --evol_file=... --diag_file=... --share_file=...

import re
import os
import h5py
import pickle
import imageio
import glob
import json
import argparse
import numpy as np
import warnings
import shutil
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from numba import njit
from itertools import repeat
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter

from math import floor, log, pi
from mPar import RunConstants
from mFunc import BinaryAlloy

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(prog="ParMaCh - 1D parametrized model of a magma chamber",
                                 description="Post-processing module",
                                 epilog="Author: MSci. Vít Beran, MFF UK, Prague"
                                 )

parser.add_argument("--tar_direct", default=None,         type=str, help="1. Target directory with the files.")
parser.add_argument("--evol_file",  default=None,         type=str, help="2. File with time evolution of the chamber.")
parser.add_argument("--diag_file",  default="diag.dat",   type=str, help="3. File with the diagnostic values.")
parser.add_argument("--share_file", default="shared.dat", type=str, help="4. File with global constants.")
parser.add_argument("--prm_file",   default="param.dat",  type=str, help="5. File with parameters of the model.")
parser.add_argument("--args_json",  default="args.json",  type=str, help="6. Args 0DMC model ran with.")
parser.add_argument("--model_json", default="model.json", type=str, help="7. Model parameters 0DMC model.")

def load_args_from_file(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

def main(args: argparse.Namespace, texfont : bool=False, size="beamer", savefig : bool=True):

    """ ############################################## """
    """ ######   AUXILIARY IO/ATTRIBUTES SETUP  ###### """
    """ ############################################## """

    os.chdir(args.tar_direct)
    plt.rcParams.update({"font.size": 13})
    plt.rcParams["axes.formatter.useoffset"] = False

    if any(arg is None for arg in vars(args).values()):
        print("One of the files was not specified.")
        return

    if texfont: 
        rcParams["text.usetex"] = True
        rcParams["font.family"] = 'serif'
        rcParams["font.serif"] = ['Computer Modern Roman']

    sed_mode = re.search(f"S_(\d)", args.tar_direct) # NOTE: target directory must have the original name!
    sed_mode = int(sed_mode.group(1))
    match sed_mode:
        case 1:
            sed_method = "dst"
        case 2:
            sed_method = "angz"
        case 3:
            sed_method = "dst"

    class SharedVariables: 
        """ Variables that need to be accessible from anywhere. """

        def __init__(self):
            self.idxend  = None                            
            self.onsetc  = None
            self.auxpad  = None   
            self.Tref    = None                              
            self.dTref   = None                                     
            self.Tliqd0  = None
            self.Teutec  = None
            self.mode    = None
            self.N0_HG97 = None                          
            self.V0_HG97 = None                          
            self.Ti_HG97 = None                           
            self.Tg_HG97 = None                           
             
    Shared = SharedVariables()


    # TODO: import pls
    def Hort_nucl(T, Tliq, N0, Tg, Ti, norm=False, Tratio=False):
        # Hortian nucleation law (Hort 1997). 
        if norm: N0 = 1.0
        if Tratio:
            Trat = T/Tliq
            # Specifically for the ratio, only auxiliary for the mPlot.py module:
            return N0 * np.exp((Tg/(1. - Tg))*(1./Ti - 1/Trat - (((1. - Ti)**3)/(1. - 3.*Ti))
                *(1./(Ti*(1. - Ti)**2) - 1./((Trat)*(1. - Trat)**2))))
        else:
            # Just plug in:
            return N0 * np.exp((Tg/(1. - Tg))*(1./Ti - Tliq/T - (((1. - Ti)**3)/(1. - 3.*Ti))
                *(1./(Ti*(1. - Ti)**2) - 1./((T/Tliq)*(1. - T/Tliq)**2))))

    def Hort_grow(T, Tliq, V0, Tg, norm=False):
        # Hortian growth law (Hort 1997). 
        if norm: V0 = 1.0
        return V0 * (Tg*(Tliq - T)/(T*(1. - Tg)))*np.exp(-(Tliq*(Tg - T/Tliq))/(T*(1. - Tg)))

    def pow_nucl(Tc, Troof, Tliq, Tref, N0, epsdel, norm=False, p=1):   # TODO: REMOVE TROOF
        # Dimensional nucleation power law. 
        if norm: N0 = 1.0
        return N0 * np.power( (Tliq - epsdel - Tc) / (Shared.Tliqd0 - epsdel - Tref), p)

    def pow_grow(Tc, Troof, Tliq, Tref, V0, norm=False, q=1):
        # Dimensional growth power law. 
        if norm: V0 = 1.0
        return V0 * np.power(((Tliq - Tc) / (Shared.Tliqd0 - Tref)), q)  
       
    # -----------------------------------------------------------------------------
    # LOAD DATA:
    #     i) args.evol_file: 
    #    ii) model_args: json file with the ParMaCh {args} namespace
    #   iii) model_const: json file with the class ModelParameters
    # -----------------------------------------------------------------------------

    # MODEL ARGUMENTS AND CONSTANTS/PARAMETERS:
    model_args  = load_args_from_file(args.args_json) # dictionaries!
    model_const = load_args_from_file(args.model_json)  
    with h5py.File(args.evol_file, "r") as evol_file:
        time_evolution_data = evol_file["time_evolution/vars"][:] # (stps, ?) NumPy array
        diagnostics_data    = evol_file["diagnostics/vars"][:]    # (stps, ?) NumPy array
        Shared.idxend       = evol_file["indices/idxend"][()]
        Shared.onsetc       = evol_file["indices/onsetc"][()]

    # TIME EVOLUTION:
    t         = time_evolution_data[:, 0]
    Tbulk     = time_evolution_data[:, 1]
    Troof     = time_evolution_data[:, 2]
    deltaT    = time_evolution_data[:, 3]
    Tliqd     = time_evolution_data[:, 4]
    Tnucl     = time_evolution_data[:, 5]
    hpile     = time_evolution_data[:, 6]
    amean     = time_evolution_data[:, 7]
    Ra        = time_evolution_data[:, 8]
    Re        = time_evolution_data[:, 9]
    hflux     = time_evolution_data[:, 10]
    Nu        = time_evolution_data[:, 11]
    phib      = time_evolution_data[:, 12]
    nu        = time_evolution_data[:, 13]
    Wrms      = time_evolution_data[:, 14]
    XL        = time_evolution_data[:, 15]
    ameanblk  = time_evolution_data[:, 16]
    fluxbot   = time_evolution_data[:, 17] # NOTE: added on 29May

    # DIAGNOSTIC QUANTITIES:
    cnt0DMC   = diagnostics_data[:, 0]
    cntHBJW   = diagnostics_data[:, 1]
    cntHNJW   = diagnostics_data[:, 2]
    atblmax   = diagnostics_data[:, 3]
    cin       = diagnostics_data[:, 4]
    cout      = diagnostics_data[:, 5]
    tCool     = diagnostics_data[:, 6]
    dtCool    = diagnostics_data[:, 7]
    tbres     = diagnostics_data[:, 8]
    tresjw    = diagnostics_data[:, 9]
    tssblk    = diagnostics_data[:, 10]
    atrn      = diagnostics_data[:, 11]
    blkgrow   = diagnostics_data[:, 12]
    tblnucmin = diagnostics_data[:, 13]
    tblnucmax = diagnostics_data[:, 14]
    tblgrwmin = diagnostics_data[:, 15]
    tblgrwmax = diagnostics_data[:, 16]
    dTbdt     = diagnostics_data[:, 17]
    dTldt     = diagnostics_data[:, 18]
    dTrdt     = diagnostics_data[:, 19]
    dGrdt     = diagnostics_data[:, 20]
    dNrdt     = diagnostics_data[:, 21]
    prate     = diagnostics_data[:, 22]
    hrate     = diagnostics_data[:, 23]
    htbl      = diagnostics_data[:, 24]
    hnbl      = diagnostics_data[:, 25]
    tplume    = diagnostics_data[:, 26]
    setmarker = diagnostics_data[:, 27]

    # SHARED VARIABLES:
    Shared.auxpad  = -1
    Shared.Tref    = model_args["Tref"]
    Shared.mode    = model_const["mode"]          
    Shared.Tliqd0  = model_const["Tliqd0"]        
    Shared.Teutec  = model_const["Teutec"]        
    Shared.N0_HG97 = model_const["N0_HG97"]       
    Shared.V0_HG97 = model_const["V0_HG97"]       
    Shared.Ti_HG97 = model_const["Ti_HG97"]       
    Shared.Tg_HG97 = model_const["Tg_HG97"]       

    print()
    print(f"[REPORT] - mPost.py:")
    #print(f"Index idxend: {Shared.idxend:d}.")
    print(f"Index crystallization onset: {Shared.onsetc:d}")
    #print(f"Index of padded arrays: {Shared.auxpad:d}.")

    #plt.plot()


    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    """%%%%%%%%%%%%%%   a) INDIVIDUAL PLOTS     %%%%%%%%%%%"""
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    # Figure size ideal for beamer presentation
    if   size == "beamer": fsize = (14,10) 
    elif size == "paper": fsize = (14,10)
    else: fsize = None

    fig, axT = plt.subplots(figsize=fsize)  
    Tb, = axT.plot(t[:Shared.idxend]/RunConstants.ytosec, Tbulk[:Shared.idxend], c="green",  label="Tbulk")
    Tr, = axT.plot(t[:Shared.idxend]/RunConstants.ytosec, Troof[:Shared.idxend], c="purple", label="Troof")
    Tl, = axT.plot(t[:Shared.idxend]/RunConstants.ytosec, Tliqd[:Shared.idxend], c="red", linestyle="--", label="Tliqd")
    Tn, = axT.plot(t[:Shared.idxend]/RunConstants.ytosec, Tnucl[:Shared.idxend], c="orange", label="Tnucl")
    axT.set_ylabel("Temperature [K]")
    axT.set_xlabel("Time [yr]")
    axT.legend()
    fig.tight_layout()
    if savefig: fig.savefig("plot/individual/temp.pdf", format="pdf", bbox_inches="tight")

    fig, axd = plt.subplots(figsize=fsize)  
    axd.plot(t[1:Shared.idxend]/RunConstants.ytosec, hpile[1:Shared.idxend], linestyle="--", c="k", label=None) 
    axd.set_ylabel("Sediment height $h$ [m]")
    axd.set_xlabel("Time [yr]")
    fig.tight_layout()
    if savefig: fig.savefig("plot/individual/hpile.pdf", format="pdf", bbox_inches="tight")

    fig, axa = plt.subplots(figsize=fsize)
    amean_colors = {1: "black", 2: "blue", 3: "red"}
    marker_names = {1: "Dust-like", 2: "Transitional regime", 3: "Unmixed Stokesian fall"}
    for m, c in amean_colors.items():
        mask = (setmarker == m) & (np.arange(len(setmarker)) >= Shared.onsetc)
        axa.plot(amean[mask]*1.e3, hpile[mask], color=c, linestyle="--", label=f"{marker_names[m]}")
    axa.legend()
    axa.set_xlabel("Mean radius $a_d$ [mm]")
    axa.set_ylabel("Sediment height $h$ [m]")
    fig.tight_layout()
    if savefig: fig.savefig("plot/individual/amean.pdf", format="pdf", bbox_inches="tight")

    fig, axc = plt.subplots(figsize=fsize)
    # Crystal content in the bulk.
    axc.plot(t[:Shared.idxend]/RunConstants.ytosec, phib[:Shared.idxend], c="k", label="$\Phi$")
    axc.set_ylabel("Bulk crystal content [-]")
    axc.set_xlabel("Time [yr]")
    axc.legend()
    if savefig: fig.savefig("plot/individual/crystal_volume_fraction.pdf", format="pdf", bbox_inches="tight")

    fig, axx = plt.subplots(figsize=fsize)
    axx.plot(t[:Shared.idxend]/RunConstants.ytosec, 1.e2*XL[:Shared.idxend], c="k", label="$X_L$")
    axx.set_ylabel("Melt composition [wt% An]")
    axx.set_xlabel("Time [yr]")
    axx.legend()
    if savefig: fig.savefig("plot/individual/xl.pdf", format="pdf", bbox_inches="tight")

    fig, arn = plt.subplots(figsize=fsize)
    arn.plot(t[:Shared.idxend]/RunConstants.ytosec, Troof[:Shared.idxend]/Tliqd[:Shared.idxend], c="k", label="$T_{roof}/T_{liqd}$")
    arn.set_ylabel("Temperature ratio [-]")
    arn.set_xlabel("Time [yr]")
    arn.legend()
    if savefig: fig.savefig("plot/individual/troof_tliqd_ratio.pdf", format="pdf", bbox_inches="tight")

    fig, axder = plt.subplots(figsize=fsize)
    dTb, = axder.plot(t[:Shared.idxend]/RunConstants.ytosec, dTbdt[:Shared.idxend], "g:", label="$\dot{T}_B$")
    dTr, = axder.plot(t[:Shared.idxend]/RunConstants.ytosec, dTrdt[:Shared.idxend], "m:", label="$\dot{T}_R$")
    dTl, = axder.plot(t[:Shared.idxend]/RunConstants.ytosec, dTldt[:Shared.idxend], "r:", label="$\dot{T}_L$")
    axder.set_xlabel("Time [yr]")
    axder.set_ylabel("$Rates of change$]")

    ratesAxis = axder.twinx()
    dN, = ratesAxis.plot(t[:Shared.idxend]/RunConstants.ytosec, dNrdt[:Shared.idxend], "m--", label="$\dot{N}$")
    dG, = ratesAxis.plot(t[:Shared.idxend]/RunConstants.ytosec, dGrdt[:Shared.idxend], "r--", label="$\dot{G}$")

    funcs = [dTb, dTr, dTl, dN, dG] 
    labels = [func.get_label() for func in funcs]
    axder.legend(funcs, labels, loc="best")
    if savefig: fig.savefig("plot/individual/derivace.pdf", bbox_inches="tight", format="pdf")

    fig, axfl = plt.subplots(figsize=fsize)
    # Crystal content in the bulk.
    axfl.plot(t[:Shared.idxend]/RunConstants.ytosec, hflux[:Shared.idxend], c="k", label="$\mathcal{F}(t)$")
    axfl.axhline(y=hflux[-1], c="r", linestyle="--", label=f"Flux at the end: {hflux[-1]:.2e} [W/m2].")
    axfl.set_ylabel("Heat flux density [W/m2]")
    axfl.set_xlabel("Time [yr]")
    axfl.legend()
    if savefig: fig.savefig("plot/individual/flux.pdf", format="pdf", bbox_inches="tight")

    fig, axcdhd = plt.subplots(figsize=fsize)
    # Crystal content in the bulk.
    axcdhd.plot(t[:Shared.idxend]/RunConstants.ytosec, model_args["H0"] - hpile[:Shared.idxend], c="r", label="$H(t)$")
    axcdhd.plot(t[Shared.onsetc:Shared.idxend]/RunConstants.ytosec, hrate[Shared.onsetc:Shared.idxend]/prate[Shared.onsetc:Shared.idxend], c="k", label="Ratio $\dot{h}/\dot{\chi}$")
    #axcdhd.axhline(y=hflux[-1], c="r", linestyle="--", label=f"Flux at the end: {hflux[-1]:.2e} [W/m2].")
    axcdhd.set_ylabel("Solid fraction constraint")
    axcdhd.set_xlabel("Time [yr]")
    axcdhd.legend()
    if savefig: fig.savefig("plot/individual/solid_fraction_constraint.pdf", format="pdf", bbox_inches="tight")

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    """%%%%%%%%%%%%%%    a2) NG REGIMES (in t)  %%%%%%%%%%%"""
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    # axu - plots ubulk, uroof un(t), ug(t)
    # agn - twinx, plots nucleation rate N(t)
    # agg - twinx, plots growth rate G(t) 
    # agh - twinx, plots h(t)

    # Calculate nucleation and growth rates:
    match model_args["NG_METHOD"]:
        case 1: 
            _grate = pow_grow(Tbulk, Troof, Tliqd, Shared.Tref, model_const["V0POW"])
            _nrate = pow_nucl(Troof, Troof, Tliqd, Shared.Tref, model_const["N0POW"], model_const["epsdel"])
            V0ampl = model_const["V0POW"]
            N0ampl = model_const["N0POW"]

        case 2: 
            _grate = Hort_grow(Tbulk, Tliqd, model_const["V0_HG97"], model_const["Tg_HG97"])
            _nrate = Hort_nucl(Troof, Tliqd, model_const["N0_HG97"], model_const["Tg_HG97"], model_const["Ti_HG97"])
            V0ampl = model_const["V0_HG97"]
            N0ampl = model_const["N0_HG97"]

    # Replace all negative values with a zero!
    _nrate = np.where(_nrate < 0.0, 0.0, _nrate)
    fig, axu = plt.subplots(figsize=(20,14))
    #axu.set_title("TEMPORAL EVOLUTION OF THE CHAMBER", fontsize=30)

    # Bulk and roof undercoolings:
    uBulk = Tliqd - Tbulk
    uRoof = Tnucl - Troof
    _deltaT = Tbulk - Troof
    ub, = axu.plot(t[:Shared.idxend]/RunConstants.ytosec, uBulk[:Shared.idxend], c="green", linestyle=":", label="Bulk undercooling (growth)")
    ur, = axu.plot(t[:Shared.idxend]/RunConstants.ytosec, uRoof[:Shared.idxend], c="purple", linestyle=":", label="Roof undercooling (nucleation)")
    ud, = axu.plot(t[:Shared.idxend]/RunConstants.ytosec, _deltaT[:Shared.idxend], c="orange", linestyle=":", label="$\Delta T(t)$")
    uczero = axu.axhline(y=0.0, c="k", linestyle="-", label="Zero undercooling")
    #onconv = axu.axvline(x=t[Shared.onsetc]/RunConstants.ytosec, c="k", linestyle="--", label="Onset of crystallization")
    #t_hatch = t[:Shared.idxend]/RunConstants.ytosec; ur_hatch = uRoof[:Shared.idxend]
    nc = axu.axvspan(t[0]/RunConstants.ytosec, t[Shared.onsetc]/RunConstants.ytosec, alpha=0.15, facecolor="grey", edgecolor="black", \
                hatch="X", label="No crystallization")


    #axs[0,1].fill_between(x, y, where=(x >= x0) & (x <= x1), color='red', alpha=0.5, hatch='X')
    #axu.set_yscale("log")
    axu.set_ylabel("Undercooling [K]", fontsize=20)
    axu.set_xlabel("Time [yr]", fontsize=20)

    axadt = axu.twinx()
    idxamean = np.argmax(amean != 0)

    adt, = axadt.plot(t[Shared.onsetc:Shared.idxend]/RunConstants.ytosec, amean[Shared.onsetc:Shared.idxend]*1.e3, c="b", linestyle="--", label="Crystal size grading (SED)")
    adtblk, = axadt.plot(t[Shared.onsetc:Shared.idxend]/RunConstants.ytosec, ameanblk[Shared.onsetc:Shared.idxend]*1.e3, c="b", linestyle=":", label="Crystal size grading (BLK)")
    axadt.set_ylabel("Mean radius [mm]", fontsize=20)
    axu.legend(loc="best", fontsize=16)
    axu.margins(x=0.01)    

    agn = axu.twinx()
    agn.spines["right"].set_position(("axes", 1.11))
    agn.spines["right"].set_visible(True)
    agn.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    agn.yaxis.get_offset_text().set_x(1.11)
    _agnn, = agn.plot(t[:Shared.idxend]/RunConstants.ytosec, _nrate[:Shared.idxend], "m", label="Nucleation rate $\mathcal{N}(t)$")
    agn.set_ylabel("Normalized nucleation rate $\mathcal{N}(t)$", fontsize=20, c="m")

    agg = axu.twinx()
    agg.spines["right"].set_position(("axes", 1.30))
    agg.spines["right"].set_visible(True)
    agg.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    agg.yaxis.get_offset_text().set_x(1.30)
    _agng, = agg.plot(t[:Shared.idxend]/RunConstants.ytosec, _grate[:Shared.idxend], "r", label="Growth rate $\mathcal{G}(t)$")
    agg.set_ylabel("Normalized growth rate $\mathcal{G}(t)$", fontsize=20, c="r")

    agh = axu.twinx()
    agh.spines["right"].set_position(("axes", -0.15))
    agh.spines["right"].set_visible(True)
    agh.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    agh.yaxis.get_offset_text().set_x(-0.15)
    _aghh, = agh.plot(t[:Shared.idxend]/RunConstants.ytosec, hrate[:Shared.idxend], "c--", label="Settling rate $\dot{h}(t)$")
    _agrp, = agh.plot(t[:Shared.idxend]/RunConstants.ytosec, prate[:Shared.idxend], "c:", label="Production rate $\dot{\chi}(t)$")
    agh.set_yscale("log")
    agh.set_ylabel("Settling/Production rate $\dot{h}$", fontsize=20, c="c")

    agnu = axu.twinx()
    agnu.spines["right"].set_position(("axes", -0.30))
    agnu.spines["right"].set_visible(True)
    agnu.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    agnu.yaxis.get_offset_text().set_x(-0.30)

    _agnun, = agnu.plot(t[:Shared.idxend]/RunConstants.ytosec, nu[:Shared.idxend], "y--", label="Viscosity $\\nu(t)$")
    agnu.set_ylabel("Kinematic viscosity $\\nu(t)$", fontsize=20, c="y")

    funcs = [ub, ur, ud, nc, uczero, adt, adtblk, _agnn, _agng, _aghh, _agrp, _agnun] 
    labels = [func.get_label() for func in funcs]
    axu.legend(funcs, labels, loc="best", fontsize=16)
    if savefig: fig.savefig("plot/summary/regimes.pdf", bbox_inches="tight", format="pdf")

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    """%%%%%%%%%%%%%%     b) SUMMARY PLOT       %%%%%%%%%%%"""
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    """% DIMENSIONAL RESULTS %"""
    mode = "Input heat flux"
    plt.clf()
    figs, axs = plt.subplots(nrows=3, ncols=2, figsize=(20,16))
    figs.subplots_adjust(hspace=0.40, top=0.85)
    title = f"ParMaCh - THERMAL EVOLUTION OF THE CHAMBER \n" + f"Mode: {mode}"
    figs.suptitle(title, fontsize=16)
    figs.tight_layout(pad=9.0)
    #figs.text(0.5, 0.9, f"Jarvis-Woods number $A={Shared.Apar:.2f}$, Stefan number $S={Shared.Spar:.2f}$", 
    #        ha='center', fontsize=14)
        
    # Temperature variables & heat flux:
    Tb, = axs[0,0].plot(t[:Shared.idxend]/RunConstants.ytosec, Tbulk[:Shared.idxend], c="green",  label="$T_B(t)$")
    Tr, = axs[0,0].plot(t[:Shared.idxend]/RunConstants.ytosec, Troof[:Shared.idxend], c="purple", label="$T_R(t)$")
    Tl, = axs[0,0].plot(t[:Shared.idxend]/RunConstants.ytosec, Tliqd[:Shared.idxend], c="red", linestyle="--", label="$T_L(t)$")
    Tn, = axs[0,0].plot(t[:Shared.idxend]/RunConstants.ytosec, Tnucl[:Shared.idxend], c="orange", label="$T_N(t)$")
    #nR, = axs[0,0].plot(t[:Shared.idxend], TnuclR[:Shared.idxend], c="orange", linestyle="--", label="TnuclR")
    axs[0,0].get_yaxis().set_major_formatter(ScalarFormatter(useOffset=False))
    axs[0,0].set_ylabel("Temperature [K]")
    axs[0,0].set_xlabel("Time [yr]")

    FluxAxis = axs[0,0].twinx()
    Flux, =  FluxAxis.plot(t[:Shared.idxend]/RunConstants.ytosec, hflux[:Shared.idxend], c="black", linestyle="--", label="Heat flux")
    FluxAxis.set_ylabel("Heat flux $\mathcal{F}$ [W/m$^2$]")
    #FluxAxis.set_yscale("log")
    funcs = [Tb, Tr, Tl, Tn, Flux] 
    labels = [func.get_label() for func in funcs]
    axs[0,0].legend(funcs, labels, loc="best")
    axs[0,0].margins(x=0.01)

    # Rayleigh and Reynolds number:
    #Ra, = axs[0,1].plot(t[:-1]/RunConstants.ytosec, Ra[:-1], "k--", label="Ra(t)") # TODO
    Rapl, = axs[0,1].plot(t/RunConstants.ytosec, Ra, "k--", label="Ra(t)")
    axs[0,1].set_ylabel("Rayleigh number $Ra$")
    #axs[0,1].set_yscale("log")
    ReAxis = axs[0,1].twinx()
    Repl, = ReAxis.plot(t[:-1]/RunConstants.ytosec, Re[:-1], c="purple", label="Re(t)")
    ReAxis.set_ylabel("Reynolds number $Re$")
    #ReAxis.set_yscale("log")
    labels = [num.get_label() for num in [Rapl, Repl]]
    axs[0,1].legend([Rapl, Repl], labels, loc="best")
    axs[0,1].set_xlabel("Time [yr]")
    
    # Sediment pile:
    axs[1,0].plot(t[Shared.onsetc:Shared.idxend]/RunConstants.ytosec, hpile[Shared.onsetc:Shared.idxend], linestyle="--", c="c") 
    axs[1,0].set_xlabel("Time [yr]")
    axs[1,0].set_ylabel("Sediment height $h$ [m]")

    # Grading: evolution of the mean radius with depth:
    # TODO
    adh, = axs[1,1].plot(amean[Shared.onsetc:Shared.idxend]*1.e3, hpile[Shared.onsetc:Shared.idxend], c="k", label="Mean radius $a_d(h)$ on height (sediment)")
    amt = axs[1,1].twinx()
    adt, = amt.plot(amean[Shared.onsetc:Shared.idxend]*1.e3, t[Shared.onsetc:Shared.idxend]/RunConstants.ytosec, \
                     c="purple", label="Mean radius $a_d(t)$ on time (sediment)")
    amt.set_ylabel("Time [yr]")
    funcs = [adh, adt]
    labels = [func.get_label() for func in funcs]
    axs[1,1].legend(funcs, labels, loc="best")
    axs[1,1].set_xlabel("Mean radius $a_d$ [mm]")
    axs[1,1].set_ylabel("Sediment height $h$ [m]")

    # Kinematic viscosity and Wrms:
    anu, = axs[2,0].plot(t[Shared.onsetc:Shared.idxend]/RunConstants.ytosec, nu[Shared.onsetc:Shared.idxend], c="y", label="Viscosity $\\nu(t)$")
    axs[2,0].set_ylabel("Kinematic viscosity [m$^2$/s]")
    axWrms = axs[2,0].twinx()
    aur, = axWrms.plot(t[Shared.onsetc:Shared.idxend]/RunConstants.ytosec, Wrms[Shared.onsetc:Shared.idxend]*RunConstants.ytosec/1.e3, \
                        c="r", label="Convective flow $W_{rms}$(t)")
    funcs = [anu, aur]
    labels = [func.get_label() for func in funcs]
    axs[2,0].legend(funcs, labels, loc="center left")
    axs[2,0].set_xlabel("Time [yr]")
    axWrms.set_ylabel("Convective r.m.s velocity [km/yr]")

    # Transition radius vs maximum radius:          # TODO: POZOR METODA SED3 TOHLE TROCHU FUCKUPUJE, BUDE DÁVÁT NĚJAKOU RANDOM VYSOKOU HODNOTU!
    #amx, = axs[2,1].plot(t[Shared.onsetc:Shared.idxend]/RunConstants.ytosec, 1.e3*amax[Shared.onsetc:Shared.idxend], c="k", label="Maximum radius $a_{max}$ (bulk/sediment)")
    amn, = axs[2,1].plot(t[Shared.onsetc:Shared.idxend]/RunConstants.ytosec, 1.e3*ameanblk[Shared.onsetc:Shared.idxend], c="purple", label="Mean radius $a_{d}$ (bulk)")
    atr, = axs[2,1].plot(t[Shared.onsetc:Shared.idxend]/RunConstants.ytosec, 1.e3*atrn[Shared.onsetc:Shared.idxend], c="r", label="Transitional radius $a_{tr}$")
    axs[2,1].set_xlabel("Time [yr]")
    axs[2,1].set_ylabel("Radius [mm]")
    axs[2,1].legend()
    if savefig: figs.savefig("plot/summary/time_evol.pdf", format="pdf", bbox_inches="tight")

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    """%%%%%%%%%%%   THERMAL BOUNDARY LAYER     %%%%%%%%%%%"""
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    fig, axtbl = plt.subplots(nrows=2, ncols=2, figsize=(18,14))
    title = f"ParMaCh - THERMAL BOUNDARY LAYER DIAGNOSTICS \n" + f"Mode: {mode}"
    fig.suptitle(title, fontsize=16)
    
    # 1. Developement of hb(t) and hn(t)
    htbl = np.concatenate((htbl[Shared.auxpad+1:], htbl[:Shared.auxpad+1]))
    hnbl = np.concatenate((hnbl[Shared.auxpad+1:], hnbl[:Shared.auxpad+1]))
    axtbl[0,0].plot(t[:Shared.idxend]/RunConstants.ytosec, htbl[:Shared.idxend]*1.e3, label="Thermal boundary layer")
    axtbl[0,0].plot(t[:Shared.idxend]/RunConstants.ytosec, hnbl[:Shared.idxend]*1.e3, label="Nucleation boundary layer")
    axtbl[0,0].set_xlabel("Time [yr]", fontsize=18)
    axtbl[0,0].set_ylabel("Thermal/Nucleation boundary thickness [mm]", fontsize=18)
    axtbl[0,0].legend(loc="best", fontsize=16)

    # 2. Development of the maximum TBL radius:
    atblmax = np.concatenate((atblmax[Shared.auxpad+1:], atblmax[:Shared.auxpad+1]))
    axtbl[0,1].plot(t[:Shared.idxend]/RunConstants.ytosec, atblmax[:Shared.idxend]*1.e3, label="TBL extracted maximum radius [mm]")
    axtbl[0,1].set_xlabel("Time [yr]", fontsize=18)
    axtbl[0,1].set_ylabel("Max. TBL crystal radius [mm]", fontsize=18)
    axtbl[0,1].legend(loc="best", fontsize=16)

    # 3. Count of nucleated crystals
    cnt0DMC = np.concatenate((cnt0DMC[Shared.auxpad+1:], cnt0DMC[:Shared.auxpad+1]))
    cntHBJW = np.concatenate((cntHBJW[Shared.auxpad+1:], cntHBJW[:Shared.auxpad+1]))
    cntHNJW = np.concatenate((cntHNJW[Shared.auxpad+1:], cntHNJW[:Shared.auxpad+1]))
    axtbl[1,0].plot(t[:Shared.idxend]/RunConstants.ytosec, cnt0DMC[:Shared.idxend]*1.e3, label="0DMC model")
    axtbl[1,0].plot(t[:Shared.idxend]/RunConstants.ytosec, cntHBJW[:Shared.idxend]*1.e3, label="J&W ($h_b$)")
    axtbl[1,0].plot(t[:Shared.idxend]/RunConstants.ytosec, cntHNJW[:Shared.idxend]*1.e3, label="J&W ($h_n$)")
    axtbl[1,0].set_xlabel("Time [yr]", fontsize=18)
    axtbl[1,0].set_ylabel("log(#) nucleated per unit time", fontsize=18)
    axtbl[1,0].set_yscale("log")
    axtbl[1,0].legend(loc="best", fontsize=16)

    # 4. Range of the growth/nucleation rate within the TBL
    _gavg = (tblgrwmin + tblgrwmax) / 2.0
    _navg = (tblnucmin + tblnucmax) / 2.0
    grwrng = axtbl[1,1].fill_between(t[:Shared.idxend]/RunConstants.ytosec, tblgrwmin[:Shared.idxend], tblgrwmax[:Shared.idxend], color="lightcoral", \
                    alpha=0.6, label="Growth rate $\mathcal{G}$")
    grwavg, = axtbl[1,1].plot(t[:Shared.idxend]/RunConstants.ytosec, _gavg[:Shared.idxend], c="r", label="TBL (mean growth rate)")
        
    axtbl_nucl = axtbl[1,1].twinx()
    nucavg, = axtbl_nucl.plot(t[:Shared.idxend]/RunConstants.ytosec, _navg[:Shared.idxend], c="m", label="TBL (mean nucleation rate)")
    nucrng = axtbl_nucl.fill_between(t[:Shared.idxend]/RunConstants.ytosec, tblnucmin[:Shared.idxend], tblnucmax[:Shared.idxend], color="plum", \
                    alpha=0.6, label="Growth rate $\mathcal{N}$")
    funcs = [grwrng, grwavg, nucavg, nucrng]
    labels = [func.get_label() for func in funcs]
    axtbl[1,1].legend(funcs, labels, loc="best")

    if savefig: fig.savefig("plot/summary/diag_tbl.pdf", format="pdf", bbox_inches="tight")

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    """%%%%%%%%%%%%%%     c) DIAGNOSTICS       %%%%%%%%%%%"""
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    #Shared.auxpad = np.argmin(TliqHort) - 1
    plt.clf()
    figs, axs = plt.subplots(nrows=3, ncols=2, figsize=(18,14))
    figs.subplots_adjust(hspace=0.40, top=0.85)
    title = f"ParMaCh - DIAGNOSTICS \n" + f"Mode: {mode}"
    figs.suptitle(title, fontsize=16)
    figs.tight_layout(pad=7.0)

    # Resolve padding:
    tssblk = np.concatenate((tssblk[Shared.auxpad+1:], tssblk[:Shared.auxpad+1]))
    tbres  = np.concatenate((tbres[Shared.auxpad+1:], tbres[:Shared.auxpad+1]))
    tresjw = np.concatenate((tresjw[Shared.auxpad+1:], tresjw[:Shared.auxpad+1]))
    tplume = np.concatenate((tplume[Shared.auxpad+1:], tplume[:Shared.auxpad+1]))
    axs[0,0].plot(np.arange(0, len(tbres)) , tbres/RunConstants.ytosec,  label="TBL residence time $t_b$")
    axs[0,0].plot(np.arange(0, len(tssblk)), tssblk/RunConstants.ytosec, label="Bulk steady-state time $t_{ss}^{blk}$")
    axs[0,0].plot(np.arange(0, len(tCool)),  tCool/RunConstants.ytosec,  label="Cooling time scale $t_c$")
    axs[0,0].plot(np.arange(0, len(dtCool)), dtCool/RunConstants.ytosec, label="Time evolution step $\Delta t_c$")
    axs[0,0].plot(np.arange(0, len(tresjw)), tresjw/RunConstants.ytosec, label="Bulk residence time $t_{res}$")
    axs[0,0].plot(np.arange(0, len(tplume)), tplume/RunConstants.ytosec, label="Plume detachment time scale $t_{pl}$")
    axs[0,0].legend(loc="best")
    axs[0,0].set_xlabel("Time steps")
    axs[0,0].set_ylabel("Time scales $\log t$ [yr]")
    axs[0,0].set_yscale("log")

    # Evolution of sediment distribution on time:
    axs[1,0].plot(t[Shared.onsetc:]/RunConstants.ytosec, phib[Shared.onsetc:], c="k")
    axs[1,0].set_xlabel("Time [yr]")
    axs[1,0].set_ylabel("Volume fraction of suspended crystals")

    match model_args["NG_METHOD"]:
        case 1: # Linear/Power laws depending on the undercooling
            pass

            #""" Cross-hatch the region of the linear laws we went through during a single run. """            
           
            """
            tmp = np.linspace(Shared.Tliqd0 - 1e-9, Shared.Tref, num=1500)
            axs[0,1].plot(Shared.Tliqd0 - tmp, pow_grow(tmp, Troof, Shared.Tliqd0, Shared.Tref, model_const["V0POW"], norm=True), "r")
            axs[0,1].plot(Shared.Tliqd0 - model_const["epsdel"] - tmp, pow_nucl(tmp, Troof, Shared.Tliqd0, Shared.Tref, model_const["N0POW"], model_const["epsdel"], norm=True), "purple")

            # Cross-hatch growth:
            x = (tmp / Shared.Tliqd0); y = pow_grow(tmp, Troof, Shared.Tliqd0, Shared.Tref, model_const["V0POW"], norm=True)

            x0, x1 = (Tliqd[0] - Tbulk[0]) / (Shared.Tliqd0 - Shared.Tref), (Tliqd[-1] - Tbulk[-1]) / (Shared.Tliqd0 - Shared.Tref)
            axs[0,1].fill_between(x, y, where=(x <= x0) & (x >= x1), color='red', alpha=0.5, hatch='X', label="Bulk undercooling")
            axs[0,1].fill_between(x, y, where=(x >= x0) & (x <= x1), color='red', alpha=0.5, hatch='X')
            axs[0,1].axvline(x=x0, color="r", linestyle="--")   
            axs[0,1].axvline(x=x1, color="r", linestyle="--")   
            axs[0,1].margins(x=0.01, y=0.01)  
            axs[0,1].set_xlabel("Undercooling $u:= T_L - T$")
            axs[0,1].set_ylabel("Visited (cross-hatched) growth rate")
            axs[0,1].legend(loc="best")

            # Cross-hatch nucleation:
            x = (tmp/Shared.Tliqd0); y = pow_nucl(tmp, Troof, Shared.Tliqd0, Shared.Tref, model_const["N0_HG97"], \
                        norm=True)
            """

        case 2: 
            """ Cross-hatch the region of the Hortian curves we went through during a single run. """
            
            tmp = np.linspace(Shared.Tliqd0 - 1e-9, 700., num=1500)
            axs[0,1].plot(tmp/Shared.Tliqd0, Hort_grow(tmp, Shared.Tliqd0, model_const["V0_HG97"], model_const["Tg_HG97"], \
                        norm=True), color="red")
            axs[0,1].plot(tmp/Shared.Tliqd0, Hort_nucl(tmp, Shared.Tliqd0, model_const["N0_HG97"], model_const["Tg_HG97"], \
                        model_const["Ti_HG97"], norm=True), color="purple")
            
            # Cross-hatch growth:
            x = (tmp/Shared.Tliqd0); y = Hort_grow(tmp, Shared.Tliqd0, model_const["V0_HG97"], model_const["Tg_HG97"], \
                        norm=True)
            
            # Ratio Tbulk and Tliqd throughout the history:
            _rTbTliq = Tbulk[Shared.onsetc:] / Tliqd[Shared.onsetc:]
            _rTbTliqdmax = np.max(_rTbTliq)
            _rTbTliqdmin = np.min(_rTbTliq)

            x0, x1 =  _rTbTliqdmin, _rTbTliqdmax
            axs[0,1].fill_between(x, y, where=(x <= x0) & (x >= x1), color='red', alpha=0.5, hatch='X', label="Bulk undercooling history")
            axs[0,1].fill_between(x, y, where=(x >= x0) & (x <= x1), color='red', alpha=0.5, hatch='X')
            axs[0,1].axvline(x=_rTbTliqdmin, color="r", linestyle="--")   
            axs[0,1].axvline(x=_rTbTliqdmax, color="r", linestyle="--")   
            axs[0,1].margins(x=0.01, y=0.01)  
            axs[0,1].set_xlabel("$T/T_L$")
            axs[0,1].set_ylabel("Visited (cross-hatched) growth rate")
            axs[0,1].legend(loc="best")

            # Cross-hatch nucleation:
            axs[1,1].plot(tmp/Shared.Tliqd0, Hort_grow(tmp, Shared.Tliqd0, model_const["V0_HG97"], model_const["Tg_HG97"], \
                                                        norm=True), color="red") 
            axs[1,1].plot(tmp/Shared.Tliqd0, Hort_nucl(tmp, Shared.Tliqd0, model_const["N0_HG97"], model_const["Tg_HG97"], \
                                                        model_const["Ti_HG97"], norm=True), color="purple") 
            
            x = (tmp/Shared.Tliqd0); y = Hort_nucl(tmp, Shared.Tliqd0, model_const["N0_HG97"], model_const["Tg_HG97"], \
                                                        model_const["Ti_HG97"], norm=True)
            
            # Ratio Troof and Tliqd throughout the history:
            _rTrTliq = Troof[Shared.onsetc:] / Tliqd[Shared.onsetc:]
            _rTrTliqdmax = np.max(_rTrTliq)
            _rTrTliqdmin = np.min(_rTrTliq)

            x0, x1 = _rTrTliqdmin, _rTrTliqdmax
            if model_args["input_hflux"]: 
                axs[1,1].fill_between(x, y, where=(x <= x0) & (x >= x1), color="purple", alpha=0.2, hatch='X', label="TBL undercooling history")
                axs[1,1].fill_between(x, y, where=(x >= x0) & (x <= x1), color="purple", alpha=0.2, hatch='X')
            if model_args["troof_const"]: axs[1,1].fill_between(x, y, where=(x >= x0) & (x <= x1), color="purple", alpha=0.2, hatch='X')
            axs[1,1].axvline(x=_rTrTliqdmin, color="purple", linestyle="--")   
            axs[1,1].axvline(x=_rTrTliqdmax, color="purple", linestyle="--")   
            axs[1,1].margins(x=0.01, y=0.01)  
            axs[1,1].set_xlabel("$T/T_L$")
            axs[1,1].set_ylabel("Visited (cross-hatched) nucleation rate")
            axs[1,1].legend(loc="best")
            
    # Influx/Outflux of crystals:
    axs[2,0].plot(t[Shared.onsetc:]/RunConstants.ytosec, cin[Shared.onsetc:], linestyle=":", c="k", label="Crystal influx")
    axs[2,0].plot(t[Shared.onsetc:]/RunConstants.ytosec, cout[Shared.onsetc:], c="r", label="Crystals outflux")
    axs[2,0].legend(loc="best")
    axs[2,0].set_xlabel("Time [yr]")
    axs[2,0].set_ylabel("Crystal flux [-]")

    # Let us track the production of solid and settling rate:
    pdot, = axs[2,1].plot(t[Shared.onsetc:]/RunConstants.ytosec, prate[Shared.onsetc:], linestyle=":", c="k", label="Solid production $\dot{\chi}$")
    axs[2,1].set_xlabel("Time [yr]")
    axs[2,1].set_ylabel("Production [s$^{-1}$ | m/s]")
    axs[2,1].set_yscale("log")
    hdot, = axs[2,1].plot(t[Shared.onsetc:]/RunConstants.ytosec, hrate[Shared.onsetc:], linestyle=":", c="r", label="Settling rate $\dot{h}$")        
    funcs = [pdot, hdot]
    labels = [func.get_label() for func in funcs]
    axs[2,1].legend(funcs, labels, loc="best")
    axs[2,1].margins(x=0.01)
    if savefig: figs.savefig("plot/summary/diagnostics.pdf", format="pdf", bbox_inches="tight")

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    """%%%%%%%%%%%%%%     d) BINARY ALLOY       %%%%%%%%%%%"""
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    with open("binary_alloy.pkl", "rb") as f: binalloyAnDi = pickle.load(f)
    X = np.linspace(0.0001, 0.9999, num=10000)
    fig, ba = plt.subplots(figsize=fsize)      
    title = f"ParMaCh - BINARY ALLOY \n" + f"Mode: {mode}"
    fig.suptitle(title, fontsize=16)
    ba.plot(X, binalloyAnDi.branch_A(X), label="Liquidus Anorthite (An)", c="k")
    ba.plot(1. - X, binalloyAnDi.branch_B(1. - X), label="Liquidus Diopside (Di)", c="k")
    ba.axhline(y=binalloyAnDi._Teut, linestyle="--", c="b", label="Eutectic temperature $T_E$")
    ba.plot(XL, Tliqd, c="r", label="$T_L(X)$")    
    ba.plot(XL, Tbulk, c="g", label="$T_B(X)$")
    ba.plot(XL, Tnucl, c="y", label="$T_N(X)$")
    ba.plot(XL, Troof, c="purple", label="$T_R(X)$")
    ba.axvline(x=XL[0], c="b", linestyle=":", label="Initial composition $X(0)$")
    ba.axvline(x=binalloyAnDi._Xeut, c="b", linestyle=":", label="Eutectic composition $X_E$")
    ba.margins(x=0.01)
    ba.set_xlabel("Composition $X$ of anortit [%w]")
    ba.set_ylabel("Temperature [K]")
    ba.legend(loc="best")
    plt.text(-0.02, 330, "Diopside", fontsize=14, c="r", verticalalignment="bottom")
    plt.text(0.95, 330, "Anortite", fontsize=14, c="r", verticalalignment="bottom")
    if savefig: plt.savefig("plot/summary/binary_alloy.pdf", format="pdf", bbox_inches="tight")

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    """%%%%%%%%%%%%%% e) DISTRO EVOLUTION       %%%%%%%%%%%"""
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    cnum0 = 15
    cnum_rewrite = True
    runH0 = model_args["H0"]
    figs, axsDE = plt.subplots(nrows=1, ncols=2, figsize=(18,14))
    figs.subplots_adjust(hspace=0., top=0.98)
    title = f"ParMaCh - Crystal size distribution evolution \n" + f"Mode: {mode}"
    figs.suptitle(title, fontsize=16)
    gif_file = "dgf"
    try: 
        if os.path.exists(gif_file):
            os.chdir(gif_file)
            files = os.listdir(os.getcwd())
            numfiles = len(files)
            cnum = numfiles
            if cnum_rewrite: cnum = cnum0
            printstep = int(numfiles / cnum)
            cmap = cm.get_cmap('plasma', cnum)
            norm = colors.Normalize(vmin=0.0, vmax=np.max(hpile))
            handlesSED, labelsSED, handlesBLK, labelsBLK = map(lambda x: list(x), repeat([], 4))
            for counter, f in enumerate(files, start=1):
                if f.endswith("pdf") or (counter % printstep != 0 and printstep > 0):  
                    continue
                _, _, step = f.split("_")
                step = int(step)
                idxcm = cmap(norm(hpile[step]))
                with open(f, "r") as of:
                    match sed_method:
                        case "tac": 
                            try: 
                                (pased, pnsed, pablk, pnblk, _, _, _, _, _, _) = np.loadtxt(of, unpack=True)
                                labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                                labelBLK = f"Bulk {1.e2*hpile[step]/runH0:.1f} [%]"
                                lineSED, = axsDE[0].plot(1.e3*pased, pnsed, c=idxcm, label=labelSED)
                                lineBLK, = axsDE[1].plot(1.e3*pablk, pnblk, c=idxcm, label=labelBLK)

                            except ValueError: 
                                # Older data with only four columns: 
                                (pased, pnsed, pablk, pnblk) = np.loadtxt(of, unpack=True)
                                labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                                labelBLK = f"Bulk {1.e2*hpile[step]/runH0:.1f} [%]"
                                lineSED, = axsDE[0].plot(1.e3*pased, pnsed, c=idxcm, label=labelSED)
                                lineBLK, = axsDE[1].plot(1.e3*pablk, pnblk, c=idxcm, label=labelBLK)
                            
                        case "dst":
                            (adistBLK, ndistBLK, ndistSED) = np.loadtxt(of, unpack=True)
                            labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                            labelBLK = f"Bulk {1.e2*hpile[step]/runH0:.1f} [%]"
                            lineSED, = axsDE[0].plot(1.e3*adistBLK, ndistSED, c=idxcm, label=labelSED)
                            lineBLK, = axsDE[1].plot(1.e3*adistBLK, ndistBLK, c=idxcm, label=labelBLK)
                            
                        case "angz":
                            (adistBLK, ndistBLK, ndistSED) = np.loadtxt(of, unpack=True)
                            labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                            labelBLK = f"Bulk {1.e2*hpile[step]/runH0:.1f} [%]"
                            lineSED, = axsDE[0].plot(1.e3*adistBLK, ndistSED, c=idxcm, label=labelSED)
                            lineBLK, = axsDE[1].plot(1.e3*adistBLK, ndistBLK, c=idxcm, label=labelBLK)
                of.close()
    
            axsDE[0].set_xlabel("Radius $a$ [mm]")
            axsDE[0].set_ylabel("# particles")
            axsDE[1].set_xlabel("Radius $a$ [mm]")
            axsDE[1].set_ylabel("# particles")
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbarsed = plt.colorbar(sm, ax=axsDE[0])
            cbarblk = plt.colorbar(sm, ax=axsDE[1])
            cbarsed.set_label("Stratification height [m]")
            cbarblk.set_label("Stratification height [m]")
            figs.savefig("distro_evolution.pdf", format="pdf", bbox_inches="tight")

    except ZeroDivisionError: print("[WARNING] - CSDs evolution not plotted!")

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    """%%%%%%%%%%%%%% f) CONVECTION   %%%%%%%%%%%"""
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    
    os.chdir("../")   
    fig, axconv = plt.subplots(nrows=2, ncols=2, figsize=(18,14))
    title = f"ParMaCh - CONECTION PARAMTERS | GROSSMANLOHSE SCALING \n" + f"Mode: {mode}"
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(hspace=0.40, top=0.85)
    fig.tight_layout(pad=9.0)
        
    # Grossman-Lohse graph log Pr vs. log Ra:           
    Pr_Iu_IIIu = lambda Ra: 5.7e-33 * (Ra**3.)
    Pr_IIIu_IVu = lambda Ra: 4.8e-8 * (Ra**(2./3.))
    Pr = nu / model_const["kappa"]
    Ra_temp = np.linspace(1.e6, 1e18, num=1000)
    axconv[0,0].scatter(Ra, Pr, s=10, c="r", marker="^", label="Chamber evolution")
    axconv[0,0].plot(Ra_temp, Pr_Iu_IIIu(Ra_temp), label="Transition $I_u$-$III_u$")
    axconv[0,0].plot(Ra_temp, Pr_IIIu_IVu(Ra_temp), label="Transition $III_u$-$IV_u$")
    axconv[0,0].scatter(Ra[0], Pr[0], s=30, c="k", marker="o", label="Initial state")
    axconv[0,0].set_xlabel("$\log$ Rayleigh number")
    axconv[0,0].set_ylabel("$\log$ Prandtl number")
    axconv[0,0].set_yscale("log")
    axconv[0,0].set_xscale("log")
    axconv[0,0].legend()

    # Nusselt number evolution:
    axconv[0,1].plot(t/RunConstants.ytosec, Nu, "k")
    axconv[0,1].set_ylabel("Nusselt number $Nu$")
    axconv[0,1].set_xlabel("Time [yr]")

    # Prandtl, Rayleigh number:
    Pr, = axconv[1,0].plot(t/RunConstants.ytosec, Pr, "r", label="$Pr(t)$")
    axconv[1,0].set_ylabel("Prandtl number $Pr$")
    axconv[1,0].set_xlabel("Time [yr]")
    Raaxis = axconv[1,0].twinx()
    Ra, = Raaxis.plot(t/RunConstants.ytosec, Ra, "m", label="$Ra(t)$")
    Raaxis.set_ylabel("Rayleigh number $Ra$")
    funcs = [Pr, Ra] 
    labels = [func.get_label() for func in funcs]
    axconv[1,0].legend(funcs, labels, loc="best")

    # Reynolds number:
    axconv[1,1].plot(t/RunConstants.ytosec, Re, "b")
    axconv[1,1].set_ylabel("Reynolds number $Re$")
    axconv[1,1].set_xlabel("Time [yr]")

    plt.savefig("plot/summary/convection.pdf", format="pdf", bbox_inches="tight")
    print()

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    """%%%%%%%%%%%%%% g) 1D HEAT EQUATION GIF   %%%%%%%%%%%"""
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    def extract_number(f):
        ext = re.search("1D_(\d+)", f).group(1)
        return int(ext) 

    files = sorted(glob.glob("1Ddata/1D_*.dat"), key=extract_number)
    images = []
    for idf, file in enumerate(files):
        print(f"\rFiles processed (1D evolution): {(idf+1):d} | {len(files):d}.", end="", flush=True)
        data = np.loadtxt(file)
        T, z = data[:, 0], data[:, 1] # T(z,t) | z!        
        z = z[::-1] # invert!

        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=fsize)
        fig.subplots_adjust(hspace=0.3)
        
        # Temperature:
        z1 = 3000.
        z2 = 3000. + hpile[extract_number(file)]
        ax[0].plot(T, z, c="orange", label="Temperature")
        ax[0].fill_between(T, z2, z1, alpha=0.3, hatch="o", label="Sediment")
        ax[0].axhline(y=3000., c="red", linestyle=":", label="Initial chamber")
        ax[0].axhline(y=4000., c="red", linestyle=":")
        ax[0].set_xlabel("T [K]")
        ax[0].set_ylabel("z [m]")
        ax[0].set_title(file)
        ax[0].legend(loc="upper right")

        # Conductivity:
        k = data[:, 2]
        #print(" ", np.sum(k == 2.5e3))
        #print("k:", np.max(k), np.min(k), k)
        ax[1].plot(k, z, c="cyan", label="Thermal conductivity")
        ax[1].axhline(y=3000., c="red", linestyle=":", label="Initial chamber")
        ax[1].axhline(y=4000., c="red", linestyle=":")
        ax[1].set_xlabel("$k$ [W/K/m]")
        #ax[1].set_ylabel("$z$ [m]")
        ax[1].set_xscale("log")
        ax[1].set_yticks([])
        ax[1].legend(loc="upper right") 
        
        # Density & heat capacity:
        rhocp = data[:, 3]
        #print("rhocp:", np.max(rhocp), np.min(rhocp), rhocp)
        ax[2].plot(rhocp, z, c="green", label=r"$\rho c_p$ product")
        ax[2].axhline(y=3000., c="red", linestyle=":", label="Initial chamber")
        ax[2].axhline(y=4000., c="red", linestyle=":")
        ax[2].set_xlabel(r"$\rho c_p$ [J/K/m3]")
        ax[2].set_yticks([])
        #ax[2].set_ylabel("$z$ [m]")
        ax[2].legend(loc="upper right")

        # Evolution of the mean radius in the sediment:
        ax[3].plot(1.e3*amean[Shared.onsetc:extract_number(file)], hpile[Shared.onsetc:extract_number(file)], c="k")
        ax[3].set_xlabel(r"$a_d$ [mm]")
        fig.canvas.draw()

        # Convert to image array:
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
        imageio.mimsave("plot/summary/animation.gif", images, duration=1.0, plugin="pillow")

    print()
    print("Gif created and saved.")
    return 

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args=args)    