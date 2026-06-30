# Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 3)
# Module: Plotting and visualiation (during the simulation)

import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter 
from matplotlib.lines import Line2D
from matplotlib import rcParams

latex = 0
if latex:
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Computer Modern Roman"]

# Modules:
from mFunc import CrystalDistro
from mFunc import Hort_grow, Hort_nucl
from mPar import *

warnings.filterwarnings("ignore", category=UserWarning)

# TODO: finish hint-typing
def plot_distribution(i:             int,                # step after the onset of crystallization
                      distSED:       CrystalDistro,      # sediment distribution
                      distBLK:       CrystalDistro,      # bulk distribution
                      distTBL:       CrystalDistro,      # TBL distribution
                      distBLK_idle:  CrystalDistro,      # TODO
                      Hnow:          float,          
                      Tliq:          float,         
                      Tbulk:         float,
                      Troof:         float,
                      Tnucl:         float,
                      amean:         float,     
                      tsim:          np.ndarray,    
                      pop:           np.ndarray,
                      track:         np.ndarray, 
                      Teut:          float,
                      flux:          float,
                      Wrms:          float,
                      fig_flip:      int=1,
                      norm:          bool=True,
                      savefig:       bool=True,
    ) -> None:
    """ Evaluation of each time step: plot individual distributions """
    
    # Normalize all distributions:
    if norm: 
        for dist in (distSED, distBLK, distTBL): dist.norm()

    figs, axs = plt.subplots(nrows=2, ncols=2, figsize=(18,14))
    #% Crystal distribution SED/BLK:
    axs[0,0].scatter(distSED.adist*RunConstants.mtomm, distSED.ndist, s=10, color="red", marker="^", label="Sediment distribution $D^{SED}$")
    axs[0,0].scatter(distBLK.adist*RunConstants.mtomm, distBLK.ndist, s=10, color="purple", marker="h", label="Bulk distribution $D^{BLK}$")

    bg = f"$\mathcal{{G}}_B$ = {Diag.blkgrow:.3e}"
    axs[0,0].scatter([], [], label=bg)
    axs[0,0].set_xlabel("Radius $a$ [mm]", fontsize=18)    
    axs[0,0].set_ylabel("Crystal density", fontsize=18) 

    # Plot the transitional radius:
    if (amean / Diag.atrn >= 1./3.):
        try: hmix = Diag.Hmix 
        except TypeError: hmix = ((Diag.atrn - Diag.amintbl) / 1.e-13) * Wrms
        if hmix is None:
            atrn_label  = "$a_{tr}$ = " + str(round(Diag.atrn*1e3, 2)) + " [mm]" + "\n" 
            astn_label  = "$a_{stn}$ = " + str(round(Diag.astn*1e3, 2)) + " [mm]" + "\n" 
            amean_label = "$a_{mean}$ = " + str(round(amean*1e3, 2)) + " [mm]" + "\n"
        else:
            atrn_label = "$a_{tr}$ = " + str(round(Diag.atrn*1e3, 2)) + " [mm]" + "\n" \
                        + str(round(hmix, 2)) + " [m]" + " | " + str(round(Hnow, 2)) + " [m]"
            astn_label = "$a_{stn}$ = " + str(round(Diag.astn*1e3, 2)) + " [mm]" + "\n" \
                        + str(round(hmix, 2)) + " [m]" + " | " + str(round(Hnow, 2)) + " [m]"

        #axs[0,0].axvline(x=Diag.atrn*RunConstants.mtomm, c="k", linestyle="--", label=atrn_label)
        #axs[0,0].axvline(x=Diag.astn*RunConstants.mtomm, c="k", linestyle=":", label=astn_label)
        #axs[0,0].axvline(x=amean*RunConstants.mtomm, c="magenta", linestyle="--", label=amean_label)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    axs[0,0].margins(x=0.01, y=0.01)
    axs[0,0].legend(loc="best", fontsize=18) 

    #% Idle distribution:
    lblg = "Growth in bulk at $T_{bulk}$" + "\n $\mathcal{G}=$" + str("%.3e" % Hort_grow(Tbulk, Tliq, ModelParameter)) \
    + "\n $T_{bulk}=$" + str(round(Tbulk, 4)) + Units.Tunit + "\n $T_{liqd}=$" + str(round(Tliq, 4)) + Units.Tunit \
    + "\n $T_{eut} | T_{liqd}=$" + str(round(Teut, 1)) + Units.Tunit + " | " + str(round(Tliq, 1)) + Units.Tunit \
    + "\n $\mathcal{F}=$" + str(round(flux, 3)) + Units.funit
    lrft = "Roof temperature: " + str("%.3f" % Troof) + Units.Tunit
    lnft = "Nucleation treshold: " + str("%.3f" % Tnucl) + Units.Tunit
    match fig_flip:
        case 1:
            lblg = "".join(["Bulk distribution $D^{BLK}_{idle}$\n", lblg])
            if distBLK_idle is not None: axs[0,1].scatter(distBLK_idle.adist*RunConstants.mtomm, distBLK_idle.ndist, s=10, color="purple", marker="h", label=lblg)
            axs[0,1].set_xlabel("Radius $a$ [mm]", fontsize=18)    
            axs[0,1].set_ylabel("Crystal density", fontsize=18) 
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            axs[0,1].margins(x=0.01, y=0.01)
            axs[0,1].legend(loc="best") 

        case 2:
            #% Hort nucleation and growth laws:
            tmp = np.linspace(Tliq - RunConstants.TINY, 700., num=1500)
            axs[0,1].plot(tmp/Tliq, Hort_grow(tmp, Tliq, ModelParameter, norm=True), color="red", label="Growth rate")
            axs[0,1].plot(tmp/Tliq, Hort_nucl(tmp, Tliq, ModelParameter, norm=True), color="purple", label="Nucleation rate")
            axs[0,1].axvline(x=Tbulk/Tliq, color="y", linestyle="--", label=lblg)
            axs[0,1].axvline(x=Troof/Tliq, color="k", linestyle="--",  label=lrft)
            axs[0,1].axvline(x=Tnucl/Tliq, color="b", linestyle=":", label=lnft)
            axs[0,1].margins(x=0.01, y=0.01)  
            axs[0,1].set_xlabel("$T/T_L$", fontsize=18)
            axs[0,1].set_ylabel("Normalized Growth and Nucleation rate", fontsize=18)
            axs[0,1].legend(loc="best")

    #% TBL crystal size distribution:
    axs[1,0].scatter(distTBL.adist*RunConstants.mtomm, distTBL.ndist, s=10, color="red", marker="^", label="Distribution $D^{TBL}_{out}$")
    axs[1,0].set_xlabel("Radius $a$ [mm]", fontsize=18)    
    axs[1,0].set_ylabel("Crystal density", fontsize=18) 
    axs[1,0].legend(loc="best", fontsize=18)

    if pop is not None:
        if isinstance(pop, np.ndarray):
            axs[1,1].plot(tsim[:], pop[:], c="k")
            axs[1,1].set_xlabel("Time [s]", fontsize=18)
            axs[1,1].set_ylabel("\# suspended crystals (bulk)", fontsize=18)

        """
        else: 
            axs[1,1].plot(tsim[:], pop[0][:], c="k")
            #axs[1,1].plot(tsim[:], pop[1][:], c="r")
            axs[1,1].axhline(y=0.0, linestyle="--", c="g")
            axs[1,1].set_xlabel("Time [s]", fontsize=18)
            axs[1,1].set_ylabel("# population (min. initial radius)", fontsize=18)
            nsusAxis = axs[1,1].twinx()
            nsusAxis.set_ylabel("# suspended crystals", fontsize=18)
            
            if len(pop[-1][:]) != 0:
                nsus, = nsusAxis.plot(tsim[:-2], pop[-1][:], c="purple", linestyle="-")
                nsuscum = nsusAxis.axhline(y=pop[-1][-1], c="purple", linestyle="--")
        """
                
    # Tracked crystals:
    if track is not None:    
        bulk_suspended = np.cumsum(track[:,:ModelParameter.Ntbl-1].sum(axis=1))
        axs[1,1].plot(np.arange(len(track[:, 0])), bulk_suspended[:], c="r")
        axs[1,1].set_xlabel("Time steps", fontsize=18)
        axs[1,1].set_ylabel("\# suspended population", fontsize=18)

        """
        zeroed_pop = np.argmin(track, axis=0)
        for j in range(ModelParameter.Ntbl-1):
            axs[1,1].plot(np.arange(len(track[:, 0])), track[:, j])
        #axs[1,1].set_yscale("log")
        #axs[1,1].set_xscale("log")
        axs[1,1].set_xlabel("log(time steps)", fontsize=18)
        axs[1,1].set_ylabel("log(# population)", fontsize=18)
        """
        
    progress = abs(1.e2 - 1.e2*(Tliq - Teut) / (ModelParameter.Tliqd0 - Teut))
    if savefig: figs.savefig(ModelParameter.outfile + "/" + ModelParameter.dstfile + "/dist_snap_" + str(i) + "_" + \
                            str(round(progress, 3)) + ".pdf", \
                            format="pdf", bbox_inches="tight")
        
    plt.close(figs) # prevents the problem with too many figures open in matplotlib!
    return

def plot_single_distribution(
        step:       int,
        distSED:    CrystalDistro,
        distBLK:    CrystalDistro,
        showatr:    bool=True,
        savefig:    bool=True,
        logy:       bool=False

    ) -> None:
    """ Plot a single figure for the distributions """

    #distSED.norm()
    #distBLK.norm()

    # Both distributions:
    fig, ax = plt.subplots(figsize=(13,8))
    if logy: ax.set_yscale("log")
    ax.scatter(distSED.adist*RunConstants.mtomm, distSED.ndist, s=10, color="red", marker="^", label="Sediment distribution $D^{SED}$")
    ax.scatter(distBLK.adist*RunConstants.mtomm, distBLK.ndist, s=10, color="purple", marker="h", label="Bulk distribution $D^{BLK}$")
    ax.set_xlabel("Radius $a$ [mm]", fontsize=22)    
    ax.set_ylabel("Crystal density", fontsize=22) 
    if showatr: ax.axvline(x=Diag.atrn*RunConstants.mtomm, c="k", linestyle="--", label="Transitonal radius $a_{tr}$")
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.margins(x=0.01, y=0.01)
    ax.legend(loc="best", fontsize=22)
    if savefig: fig.savefig(ModelParameter.outfile + "/single_dist_snap" + str(step) + ".pdf", format="pdf")
    # Bulk only:
    fig, ax = plt.subplots(figsize=(13,8))
    if logy: ax.set_yscale("log")
    ax.scatter(distBLK.adist*RunConstants.mtomm, distBLK.ndist, s=10, color="purple", marker="h", label="Bulk distribution $D^{BLK}$")
    ax.set_xlabel("Radius $a$ [mm]", fontsize=22)    
    if not logy: ax.set_ylabel("Crystal density", fontsize=22)
    else: ax.set_ylabel("\# log crystals (normalized)",  fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.margins(x=0.01, y=0.01)
    ax.legend(loc="best", fontsize=22)
    if savefig: fig.savefig(ModelParameter.outfile + "/single_distBLK_snap" + str(step) + ".pdf", format="pdf")
    # Sediment only:
    fig, ax = plt.subplots(figsize=(13,8))
    if logy: ax.set_yscale("log")
    ax.scatter(distSED.adist*RunConstants.mtomm, distSED.ndist, s=10, color="red", marker="^", label="Sediment distribution $D^{SED}$")
    ax.set_xlabel("Radius $a$ [mm]", fontsize=22)    
    if not logy: ax.set_ylabel("Crystal density", fontsize=22) 
    else: ax.set_ylabel("\# log crystals (normalized)",  fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.margins(x=0.01, y=0.01)
    ax.legend(loc="best", fontsize=22)
    if savefig: fig.savefig(ModelParameter.outfile + "/" + ModelParameter.dstfile + "/single_distSED_snap" + str(step) + ".pdf", format="pdf")
    plt.close(fig) # Prevents the problem with too many figures open in matplotlib!
    return

def plot_tbl_distribution(
        step:       int,
        htbl:       float,
        hnbl:       float,
        distTBL:    CrystalDistro,
        savefig:    bool=True

    ) -> None:
    """ Plot a single TBL crystal flux distribution """

    np.savetxt(ModelParameter.outfile + "/fig6_single_tbl.dat", ((distTBL.adist, distTBL.ndist)))

    distTBL.ndist = distTBL.ndist[distTBL.ndist > 0.0]
    distTBL.adist = distTBL.adist[:len(distTBL.ndist)]
    
    distTBL.norm()
    fig, ax = plt.subplots(figsize=(13,8))
    sc = ax.scatter(distTBL.adist*RunConstants.mtomm, distTBL.ndist, s=10, color="red", marker="^", label="Distribution flux $\mathcal{D}_{T}^{out}$")
    ax.set_xlabel("Radius $a$ [mm]", fontsize=22)    
    ax.set_ylabel("Crystal density", fontsize=22) 
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    text_handle = Line2D([], [], linestyle="none",
                        label=(
                            f"$h_b$ = {1e3*htbl:.2f} mm\n"
                            f"$h_n$ = {1e3*hnbl:.2f} mm\n"
                            f"$\mathcal{{G}}_{{\mathrm{{min}}}}$ = {Diag.tblgrwmin:.2e} m/s\n"
                            f"$\mathcal{{G}}_{{\mathrm{{max}}}}$ = {Diag.tblgrwmax:.2e} m/s\n"
                            f"$\mathcal{{N}}_{{\mathrm{{min}}}}$ = {Diag.tblnucmin:.2e} \#/m$^3$/s\n"
                            f"$\mathcal{{N}}_{{\mathrm{{max}}}}$ = {Diag.tblnucmax:.2e} \#/m$^3$/s\n"
                        )
    )
    
    ax.legend(handles=[
        sc, text_handle,
    ], loc="best", fontsize=22)
    ax.margins(x=0.01, y=0.01)
    if savefig: fig.savefig(ModelParameter.outfile + "/" + ModelParameter.dstfile + "/single_tbl_snap" + str(step) + ".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig) # prevents the problem with too many figures open in matplotlib!
    return

def plot_tbl_distribution_active(
        step:          int,
        distTBLActive: CrystalDistro,
        htbl:          float,               # we wanted to plot also the %occupied volume somewhere in the figure
        savefig:       bool=True
        
    ) -> None:
    """ Plot a single TBL distribution of the suspended crystals """

    np.savetxt(ModelParameter.outfile + "/fig6_single_2Dtbl.dat", ((distTBLActive.adist, distTBLActive.ndist)))

    fig, ax = plt.subplots(figsize=(13,8))
    plt.scatter(distTBLActive.adist*RunConstants.mtomm, distTBLActive.ndist/np.sum(distTBLActive.ndist), s=10, color="red", marker="^", label="TBL suspended distribution $D_T$")
    ax.set_xlabel("Radius $a$ [mm]", fontsize=22)    
    ax.set_ylabel("Crystal density", fontsize=22) 
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.legend(loc="best", fontsize=22)
    if savefig: fig.savefig(ModelParameter.outfile + "/" + ModelParameter.dstfile + "/single_tbl_snap_active" + str(step) + ".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    plt.show()
    return

def plot_2D_tbl_distribution_active(
        step:               int,
        aedges:             np.ndarray,
        zedges:             np.ndarray,
        N:                  np.ndarray,
        phiB_htbl:          float=0.0,
        savefig:            bool=True,
        savehist:           bool=False,
        _name:              str="tbl"

    ) -> None:
    """ Plot a 2D TBL distribution of the suspended crystals """
    
    if savehist:
        with open(ModelParameter.outfile + "/hist2d.dat", "wb") as f:
            np.save(f, N)
            np.save(f, aedges)
            np.save(f, zedges)

    N /= np.sum(N)
    fig, ax = plt.subplots(figsize=(13, 8))
    _title = f"Crystalinity within the TBL: $\Phi$ = {1.e2*phiB_htbl:.2f} \%"
    #fig.suptitle(_title, fontsize=22)
    fig.tight_layout(pad=0.4)
    pcm = ax.pcolormesh(
        zedges,
        aedges,
        N,
        shading="auto"
    )

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.set_xlabel("Depth z [mm]", fontsize=22)
    ax.set_ylabel("Radius a [mm]", fontsize=22)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))   # forces scientific notation
    cbar.formatter = formatter
    cbar.update_ticks()
    cbar.set_label("Crystal density", fontsize=22, labelpad=10)
    if savefig: fig.savefig(ModelParameter.outfile + "/" + ModelParameter.dstfile + "/2D_" + str(_name) + "_snap" + str(step) \
                            + ".pdf", format="pdf", bbox_inches="tight")
    return

def plot_track(
        step:    int,
        tstps:   int,
        Hnow:    float,
        aTrack:  np.ndarray,
        vhTrack: np.ndarray,
        dhTrack: np.ndarray,
        savefig: bool=True

    ) -> None:
    """ Plot the tracking history in the bulk region """

    zeroed_pop = np.full(ModelParameter.Ntbl-1, tstps)   #np.argmin(aTrack, axis=0)
    figs, axs = plt.subplots(nrows=2, ncols=2, figsize=(18,14))
    if aTrack is not None:
        for j in range(ModelParameter.Ntbl-1):
            axs[0,0].plot(np.arange(zeroed_pop[j]), aTrack[:zeroed_pop[j],j]*1.e3)   
            axs[0,0].set_ylabel("Radius [mm]")
            axs[0,0].set_xlabel("Time steps [-]")
            axs[0,0].set_yscale("log")
    if vhTrack is not None:
        for j in range(ModelParameter.Ntbl-1):
            axs[0,1].plot(np.arange(zeroed_pop[j]), dhTrack[:zeroed_pop[j],j])
            axs[0,1].set_xlabel("Time steps [-]")
            axs[0,1].set_ylabel("Shrinking velocity [m/s]")
            axs[0,1].set_yscale("log")
    if dhTrack is not None:
        zeroed_pop = np.full(ModelParameter.Ntbl-1, tstps)  #np.argmax(dhTrack, axis=0)        
        for j in range(ModelParameter.Ntbl-1):
            axs[1,0].axhline(y=Hnow) # H0
            axs[1,0].plot(np.arange(zeroed_pop[j]), dhTrack[:zeroed_pop[j],j])
            axs[1,0].set_xlabel("Time steps [-]")
            axs[1,0].set_ylabel("Settling front [m]")
            axs[1,0].set_yscale("log")
    if savefig: figs.savefig(ModelParameter.outfile + "/" + ModelParameter.dstfile + "/track_snap" + str(step) + ".pdf", format="pdf")
    plt.close(figs)
    return

def plot_pnucage(step:         int, 
                 particle_age: np.ndarray, 
                 ini_depth:    float, 
                 savefig:      bool=True
    ) -> None: 
    """ Plot particle age vs. nucleation depth """
    
    figs, ax = plt.subplots(figsize=(18,14))
    ax.scatter(np.array(ini_depth)*RunConstants.mtomm, particle_age, s=10, marker="^", color="purple")
    ax.set_xlabel("Nucleation depth [mm]")
    ax.set_ylabel("Particle age [s]")
    ax.margins(x=0.01, y=0.01)
    if savefig: figs.savefig(ModelParameter.outfile + "/" + ModelParameter.dstfile + "/par_age" + str(step) + ".pdf", format="pdf")
    return

def plot_1D(
        T0:         np.ndarray,
        T:          np.ndarray,
        z:          np.ndarray,
        step_hit:   int,
        Tliqd:      float,
        const:      Constants1DHE,
        tcool:      float=0.0,
        step:       int=None,
        savefig:    bool=True

        ) -> None:
        """ Plot heat flux 1D evolution """

        xpick = 1 # TODO: tohle půjde vylepšit ať velikost těch PDFek není tak velká!
        T = T[::xpick]; z = z[::xpick]
        
        if step_hit is not None: 
            (years, _, qroof, qupp, qlow) = np.loadtxt(ModelParameter.outfile + "/1DHE_flux.dat", unpack=True)
            years = years / RunConstants.ytosec

        fig, axs = plt.subplots(1, 2, figsize=(16,6))
        if T0 is not None:
            T0 = T0[::xpick]
            axs[0].plot(T0, z / 1000., label="Initial temperature")
            axs[0].plot(T, z / 1000., label="Final temperature", linestyle='--')

        else:
            axs[0].plot(T, z / 1000., label=f"Time: {(tcool/RunConstants.ytosec):.2e} yrs | Temperature $T(z,t)$", c="orange", linestyle='--')

        axs[0].set_xlabel("Temperature [K]", fontsize=14)
        axs[0].set_ylabel("Depth [km]", fontsize=14)
        axs[0].plot([Tliqd, Tliqd], [5., 6.], c="m", linestyle="--")
        axs[0].axvline(x=Tliqd, c="m", linestyle="--", label="Initial liquidus")

        #axs[0].axhline(y=(const._zchtop0 - const._dfine)/1000., c="r")
        #axs[0].axhline(y=(const._zchbot0 + const._dfine)/1000., c="r", label="Fine grid") <-- FIXME, remember, im rewriting zchtop0 etc.!

        # Cross-hatch the sediment:
        z1 = z[np.abs(z - const._zchbot0).argmin()]; z2 = z[np.abs(z - const._zchbotINT).argmin()]
        #axs[0].fill_between(T, z, where=(z >= z1) & (z <= z2), color="black", alpha=0.2, hatch="X", label="Sediment")
        axs[0].fill_between(T, z1 / 1.e3, z2 / 1.e3, alpha=0.3, hatch="X", label="Sediment")
        axs[0].invert_yaxis()
        axs[0].legend(fontsize=12)
        axs[0].grid(True)
        axs[0].set_title("Temperature profile", fontsize=14)
        
        if step_hit is not None:
            axs[1].plot(years, np.abs(qroof), label="Heat flux $\mathcal{F}(t)$")
            axs[1].plot(years, np.abs(qupp), label="Heat flux $\mathcal{F}_{+}(t)$")
            axs[1].plot(years, np.abs(qlow), label="Heat flux $\mathcal{F}_{-}(t)$")
            axs[1].axvline(x=years[step_hit], c="r", linestyle=":", label="Liquidus hit")
            axs[1].set_xlabel("Time (years)", fontsize=14)
            axs[1].set_ylabel("Roof heat flux [W/m²]", fontsize=14)
            axs[1].legend(fontsize=12)
            axs[1].grid(True)
            axs[1].set_title("Heat Flux History", fontsize=14)
            textstr = (
                f"Temperatures:\n"
                f"Emplacement temperature: {Constants1D._Temp:.1f} K\n"
                f"Liquidus temperature: {Tliqd:.1f} K\n"
                f"Conductivity ratio: {(const._keff/const._kbg):.1f}\n"
                f"Time to hit liquidus: {years[step_hit]:.1f} years\n"
                f"Initial heat flux: {(qroof[step_hit]):.1f} W/m²\n"
                f"= {(qroof[step_hit])/RunConstants.HFU:.2e} HFU"
                )
            axs[1].text(
                0.43, 0.65, textstr, transform=axs[1].transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
            )

        if savefig: 
            if step is not None:
                fig.savefig(ModelParameter.outfile + "/" + ModelParameter.IDHfile + "/1DHE_" + str(step) + ".pdf", bbox_inches="tight")
            else: 
                fig.savefig(ModelParameter.outfile + "/" + ModelParameter.IDHfile + "/1DHE_INIT.pdf", bbox_inches="tight")
        plt.close(fig)
        return

def plot_1d_grid(z: np.ndarray, interfaces: bool=True, labels: bool=False, savefig: bool=True):
    fig, ax = plt.subplots(figsize=(10, 2))

    ax.plot(z, np.zeros_like(z), "o", ms=4, label="nodes")
    if interfaces and len(z) > 1:
        zi = 0.5 * (z[:-1] + z[1:])
        ax.vlines(zi, -0.15, 0.15,
                  linestyles="dashed",
                  linewidth=1.5,
                  label="interfaces")
    if labels:
        for i, zz in enumerate(z):
            ax.text(zz, 0.08, f"{i}", ha='center')

    # Connect nodes:
    ax.hlines(0, z.min(), z.max(), linewidth=1)

    ax.set_yticks([])
    ax.set_xlabel("z")
    ax.set_title("1D Grid")

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend()
    plt.tight_layout()
    fig.savefig(ModelParameter.outfile + "/grid.pdf", format="pdf", bbox_inches="tight")


def plot_1d_zoom_roof(step: int, T: np.ndarray, z: np.ndarray, const: Constants1DHE, savefig: bool=True) -> None:

    # T... temperature
    # z... vertical coordinate

    fig, ax = plt.subplots(figsize=(10, 2))
    T_select = T[(const._roofidx - 10):(const._roofidx + 10)]
    z_select = z[(const._roofidx - 10):(const._roofidx + 10)]
    z_select = z_select[::-1]
    ax.plot(T_select, z_select, label="Roof zoomed profile.")
    ax.axhline(y=const._zchtop0, c="r", linestyle=":", label="Chamber roof.")
    #ax.axhline(y=const._zchtbl0, c="g", linestyle=":", label="TBL")
    ax.set_xlabel("T [K]")
    ax.set_ylabel("z [m]")
    ax.legend()

    if savefig: fig.savefig(ModelParameter.outfile + "/" + ModelParameter.rfzfile + "/1d_zoom_roof_" + str(step) + ".pdf", format="pdf", bbox_inches="tight")
    return


def plot_1d_zoom_floor(step: int, T: np.ndarray, z: np.ndarray, const: Constants1DHE, savefig: bool=True) -> None:

    # T... temperature
    # z... vertical coordinate

    fig, ax = plt.subplots(figsize=(10, 2))
    T_select = T[(const._botmidx - 10):(const._botmidx + 10)]
    z_select = z[(const._botmidx - 10):(const._botmidx + 10)]
    z_select = z_select[::-1]
    ax.plot(T_select, z_select, label="Floor zoomed profile.")
    ax.axhline(y=const._zchbot0, c="r", linestyle=":", label="Chamber floor.")
    ax.set_xlabel("T [K]")
    ax.set_ylabel("z [m]")
    ax.legend()

    if savefig: fig.savefig(ModelParameter.outfile + "/" + ModelParameter.rfzfile + "/1d_zoom_floor_" + str(step) + ".pdf", format="pdf", bbox_inches="tight")
    return


def plot_1d_interior(step: int, T: np.ndarray,  z: np.ndarray, const: Constants1DHE, savefig: bool=True) -> None:

    # T... temperature
    # z... vertical coordinate

    fig, ax = plt.subplots(figsize=(10, 2))
    T_select = T[(const._roofidx - 10):(const._botmidx + 10)]
    z_select = z[(const._roofidx - 10):(const._botmidx + 10)]
    z_select = z_select[::-1]
    ax.plot(T_select, z_select, label="Floor zoomed profile.")
    ax.axhline(y=const._zchbot0, c="r", linestyle=":", label="Chamber floor.")
    ax.axhline(y=const._zchtop0, c="r", linestyle="--", label="Chamber roof.")
    ax.set_xlabel("T [K]")
    ax.set_ylabel("z [m]")
    ax.legend()

    if savefig: fig.savefig(ModelParameter.outfile + "/" + ModelParameter.rfzfile + "/1d_zoom_interior_" + str(step) + ".pdf", format="pdf", bbox_inches="tight")
    return