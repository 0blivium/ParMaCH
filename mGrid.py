# Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 3)
# Module: Auxiliary plotting script for the systematic search through the parametric space
# Run: python3 mGrid.py --path={} --repeat

import re
import os
import h5py
import json
import string
import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numba import prange
from datetime import datetime 
from itertools import repeat
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LinearLocator, MaxNLocator
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(prog="Landscape figure",
                                 description="ParMaCh - 1D parametrized model of a magma chamber",
                                 epilog="Author: MSci. Vít Beran, MFF UK, Prague"
                            )

parser.add_argument("--path",          default=None,  type=str,             help="Add path to the nodes.")
parser.add_argument("--subtitle",      default=None , type=str,             help="Subtitle of the grid landscape figure.")
parser.add_argument("--flux0",         default=None,  type=float,           help="Initial heat flux.")
parser.add_argument("--H0",            nargs="+",     type=float,           help="Values of initial heights.")
parser.add_argument("--G0",            nargs="+",     type=float,           help="Values of the growth amplitudes.")
parser.add_argument("--X0",            nargs="+",     type=float,           help="Values of the initial compositions.")
parser.add_argument("--repeat",        default=False, action="store_true",  help="Replot the grid figure.")
parser.add_argument("--print_phib",    default=True,  action="store_false", help="Print the maximum crystalinity (across the 0-4 runs displayed).")
parser.add_argument("--print_bulknuc", default=True,  action="store_false", help="Print whether Tbulk dropped below the nucleation lag at any moment.")
parser.add_argument("--print_xldgf",   default=True,  action="store_false", help="Print the XL0 for which the CSD evolution is plotted.")
parser.add_argument("--print_ntag",    default=False, action="store_true",  help="Print the node number.")
parser.add_argument("--print_header",  default=False, action="store_true",  help="Print a header with controlling parameters.")
parser.add_argument("--tblless",       default=False, action="store_true",  help="Plot a TBL-less solution.")
parser.add_argument("--htsol",         default=False, action="store_true",  help="Plot a solution with time-dependent heat flux.")
parser.add_argument("--latex",         default=False, action="store_true",  help="Support of LaTeX in the figure.")

# Auxiliary functions:
def load_args_from_file(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

def extract_hpile_amean_from_each_dir(dirs, args_json, model_json, evol_file):
    if dirs.startswith("."): return
    os.chdir(dirs)

    logger_target = "[WARNING] Ridiculously small time step, check manually!"
    with open("0DMC.log", "r") as f:
        last_line = f.readlines()[-1].strip()
        if logger_target in last_line:
            return ([], [], 0, [], 0, False, True)

    model_args  = load_args_from_file(args_json) 
    model_const = load_args_from_file(model_json)

    with h5py.File(evol_file, 'r') as evol_file:
        time_evolution_data = evol_file["time_evolution/vars"][:] 
        diagnostics_data    = evol_file["diagnostics/vars"][:]    
        idxend              = evol_file["indices/idxend"][()]
        onsetc              = evol_file["indices/onsetc"][()]

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

    # TEST:
    #amean = amean[::1000]
    #hpile = hpile[::1000]

    # DIAGNOSTIC QUANTITIES:
    setmarker = diagnostics_data[:, 27]

    #setmarker = setmarker[::1000]
        
    # RETURN ADDITIONAL QUANTITIES:
    phiBmax = np.max(phib)
    bulknuc_flag = True if (Tbulk < Tnucl).any() else False

    xpad = 1
    return (amean[:-xpad], hpile[:-xpad], onsetc, setmarker[:-xpad], phiBmax, bulknuc_flag, False)

def idxf(comp):
    match comp:
        case "0.65": idx = 3
        case "0.75": idx = 2
        case "0.85": idx = 1
        case "0.95": idx = 0
    return idx

def main(args: argparse.Namespace):
    if args.latex:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = ["Computer Modern Roman"]

    _dtblless = "2026-03-30_13-05-06_RM_1XL_0.75_F2.00e+02H1e+01G1e-08_N1e+03NG_2_S_4_nu_00_O0_GL_1_TBL_AnDi" if args.tblless else None
    _dhtsol =   "2026-06-10_17-47-07_RM_1XL_0.75_F4.00e+01H1e+03G1e-08_N1e+03NG_2_S_2_nu_10_O0_GL_1_TBL_AnDi" if args.htsol else None
    #"2026-03-30_15-25-22_RM_1XL_0.75_F4.00e+01H1e+03G1e-08_N1e+03NG_2_S_4_nu_10_O0_GL_1_TBL_AnDi" if args.htsol else None

    # Create main figure:
    fig = plt.figure(figsize=(26, 14))

    # NOTE: Customize manually:    
    if args.print_header:
        fig.suptitle(
            r"$\mathcal{F}_0 = 200$ W/m$^2$" "\n" 
            r"$\varepsilon_{Hort} = 23 \; \mathrm{K}$" "\n"
            r"$N_0 = 1 \times 10^{3} \; \#/\mathrm{m}^{-3}/s$" "\n"
            r"$Ea = 0.97 \times Ea_{ref} \; \mathrm{J}/\mathrm{mol}$" "\n",
            fontsize=12,
            ha="center"
        )

    ##################################################################
    #####  COMPHREHENSIVE FIGURE OF GRADINGS AND DISTRIBUTIONS   #####
    ##################################################################

    # FIXME: args.H0 args.XL0, and args.G0 not considered here! Fix!
    fig.subplots_adjust(top=0.87)
    outer_grid = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.4)  # 3x3 outer grid!
    ogpath = os.getcwd()
    current_year = str(datetime.now().year)
    handles, labels = [], []
    colorsXL = ["green", "blue", "red", "purple"]
    compositions = ["0.95", "0.85", "0.75", "0.65"] 
    compositions_float = [0.95, 0.85, 0.75, 0.65]
    H0_values = ["1000", "100", "10"] 
    letter_labels = list(string.ascii_lowercase)  
    marker_names = {1: "Dust-like", 2: "Transitional regime", 3: "Unmixed Stokesian fall"}
    linestyle_names = {1: "solid", 2: "dashed", 3: "dotted"}
    cnum_rewrite = True; dstnorm = True; cnum0 = 8; idxCSD = 2
    assert idxCSD <= len(compositions)

    for i in range(9): 
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[i], wspace=0.2) # NOTE: replaced 4 with 3!
        #if i > 1: continue
        #if i != 4: continue

        # Back to "Final Results"!
        print(f"Node {(i+1):d} & Directory {os.getcwd()}")
        row, col = divmod(i, 3)
        ax_dummy = fig.add_subplot(outer_grid[i])
        ax_dummy.axis("off")  # hide dummy axis!
        if row == 0:
            if col == 0:
                ax_dummy.set_title("$G_0 = 1 \\times 10^{-8}$ m/s", fontsize=14, pad=40)
            elif col == 1: 
                ax_dummy.set_title("$G_0 = 1 \\times 10^{-7}$ m/s", fontsize=14, pad=40)
            else:
                ax_dummy.set_title("$G_0 = 1 \\times 10^{-6}$ m/s", fontsize=14, pad=40)

        # Print letter label:
        #ax_dummy.text(
        #    0.0, 1.05,               
        #    f"{letter_labels[i]})",
        #    transform=ax_dummy.transAxes,
        #    fontsize=12,
        #    fontweight="bold",
        #    va="bottom",
        #    ha="left"
        #)

        # --- Vertical H0 labels for nodes 1,4,7 ---
        if i in [0, 3, 6]:  # first column of each row of nodes
            ax_dummy_v = fig.add_subplot(outer_grid[i])
            ax_dummy_v.axis("off")
            bbox = ax_dummy_v.get_position()
            if i == 0: idh = 0
            if i == 3: idh = 1
            if i == 6: idh = 2
            H0 = H0_values[idh]
            fig.text(
                bbox.x0 - 0.04, 
                bbox.y0 + bbox.height / 2 - 0.01,    
                rf"$H_0 = {H0}$ m",
                rotation="vertical",
                va="center",
                ha="center",
                fontsize=14
            )
            
        for j in range(3):
            if args.repeat:
                path = ogpath + "/" + args.path
                os.chdir(path)
            else:   
                os.chdir(args.path)
            
            ax = fig.add_subplot(inner_grid[j])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            #label = rf"{{\bfseries {letter_labels[i]}{j:d})}}
            #label = rf"{{\bfseries {letter_labels[i]}{j:d})}}"
            label =rf"\textbf{{{letter_labels[i]}{j:d})}}"
            ax.text(
                0.0, 1.05,               
                #f"{letter_labels[i]}{j:d})",
                label,
                transform=ax.transAxes,
                fontsize=12,
                #fontweight="bold",
                va="bottom",
                ha="left"
            )

            ax.tick_params(axis="x", labelsize=8)
            idx = None; 
            try:
                if j == 0 or j == 1: # 1st and 2nd panel:
                    print(os.getcwd())
                    os.chdir(f"node{(i+1):d}_{(j+1):d}")
                    cwd = os.getcwd()
                    directories = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d)) and d.startswith(current_year)]
                    if _dtblless in directories: directories.remove(_dtblless)
                    if _dhtsol in directories: directories.remove(_dhtsol)

                    phibtmp = np.array([])
                    bulknuc = None
                    odircnt = 0

                    for itmp, dirs in enumerate(directories): # nodes!
                        comp = re.search(r"XL_([\d.]+)", dirs)
                        comp_f = str(comp.group(1))
                        idx = idxf(comp_f)
                        evol_name = "evol_" + str(dirs) + ".h5"
                        args_name = "args.json"
                        model_name = "model.json"

                        try:
                            amean, hpile, onsetc, setmarker, phibmax, bulknuc_flag, logoff  \
                                = extract_hpile_amean_from_each_dir(dirs=dirs, args_json=args_name, model_json=model_name, evol_file=evol_name)

                        except OSError: continue    
                        odircnt += 1

                        amean = np.array(amean); hpile = np.array(hpile); setmarker = np.array(setmarker)

                        # BEWARE, HARD-CODED:
                        if i == 1 or i == 4 or i == 7: setmarker[:] = 2.0
                        #print("amean:", amean[-5:-1])
                        
                        if len(amean) > 0:
                            try: 
                                idxam = np.where(amean[int(len(amean)/10):] == 0.0)[0][0]
                                amean = amean[:(idxam)]
                                hpile = hpile[:(idxam)]
                                setmarker = setmarker[:(idxam)]

                            except IndexError: pass

                            #if amean[-1] == 0.0:
                            #    amean = amean[:-1]
                            #    hpile = hpile[:-1]
                            #    setmarker = setmarker[:-1]

                        if logoff:
                            os.chdir("..")
                            continue

                        if len(amean) > 0: phibtmp = np.append(phibtmp, phibmax)
                        if bulknuc is None and bulknuc_flag: bulknuc = True
                        
                        # ISOVISCOUS RUNS: 
                        if j == 0:
                            start = onsetc
                            for il in range(onsetc + 1, len(setmarker)):
                                if setmarker[il] != setmarker[il - 1]:  # whenever the marker changes, end current segment!
                                    m = setmarker[il - 1]
                                    ax.plot(1.e3*amean[start:il:], hpile[start:il:],
                                            color=colorsXL[idx],
                                            linestyle=linestyle_names[m])
                                    start = il

                            #if len(setmarker) == 0:
                            #    m = 1
                            #else:
                            
                            m = setmarker[-1]
                            ax.plot(1.e3*amean[start:], hpile[start:],
                                    color=colorsXL[idx],
                                    linestyle=linestyle_names[m])
                            ax.plot([], [], color=colorsXL[idx], label=f"Composition $X_{{L,0}}$ = {1.e2*float(compositions[idx])}")
                            ax.set_title(r"$\nu = \nu_0$")
                            if i == 0 or i == 3 or i == 6:
                                ax.set_ylabel("$h$ [m]")
                            else:
                                ax.set_yticks([])

                        elif j == 1:
                            start = onsetc
                            for il in range(onsetc + 1, len(setmarker)):
                                if setmarker[il] != setmarker[il - 1]: # whenever the marker changes, end current segment!
                                    m = setmarker[il - 1]
                                    ax.plot(1.e3*amean[start:il], hpile[start:il],
                                            color=colorsXL[idx],
                                            linestyle=linestyle_names[m])
                                    start = il

                            #if len(setmarker) == 0:
                            #    m = 1
                            #else:
                                
                            m = setmarker[-1]
                            ax.plot(1.e3*amean[start:], hpile[start:],
                                    color=colorsXL[idx],
                                    linestyle=linestyle_names[m])
                            ax.plot([], [], color=colorsXL[idx], label=f"Composition $X_{{L,0}}$ = {1.e2*float(compositions[idx])}")
                           
                            # xlabel {a_d} is written below Fig. 2:
                            ax.set_xlabel("$a_d$ [mm]")
                            ax.set_title(r"$\nu = \nu(T)$")
                            ax.set_yticks([])

                        if args.print_phib:
                            #if len(phibtmp) == len(directories): # i.e., you went through all simulations, print the highest crystallinity!
                            if odircnt == 4:
                                ax.text(
                                    0.99, 0.99, f"$\Phi_{{max}}$ = {1.e2*np.max(phibtmp):.1f} %",
                                    transform=ax.transAxes,   
                                    ha="right", 
                                    va="top",
                                    fontsize=9
                                )
                        
                        if args.print_bulknuc and bulknuc is not None:
                            if odircnt == 4: # i.e., you went through all simulations, print the highest crystallinity!
                                ax.text(
                                    0.99, 0.95, f"Bulk nucleation!",
                                    transform=ax.transAxes,   
                                    ha="right", 
                                    va="top",
                                    fontsize=9
                                )

                        # Grab handles/labels from first subplot only:
                        if len(handles) == 0 and itmp == 3:
                            handles.append(ax.get_legend_handles_labels()[0])
                            labels.append(ax.get_legend_handles_labels()[1])
                        os.chdir("..")
                                 
                        #""" 
                        if j == 0 and i == 6 and args.tblless and odircnt == 4: # TBL-less solution in Node 7!
                            try:
                                _dirs = _dtblless
                                _evol_name = "evol_" + str(_dirs) + ".h5"
                                _args_name = "args.json"
                                _model_name = "model.json"

                                amean, hpile, onsetc, setmarker, _, _, _ \
                                    = extract_hpile_amean_from_each_dir(dirs=_dirs, args_json=_args_name, model_json=_model_name, evol_file=_evol_name)
                                
                                if len(amean) > 0:
                                    try: 
                                        idxam = np.where(amean[int(len(amean)/10):] == 0.0)[0][0]
                                        amean = amean[:(idxam)]
                                        hpile = hpile[:(idxam)]
                                        setmarker = setmarker[:(idxam)]

                                    except IndexError: pass

                                ax.plot(amean[onsetc:]*1.e3, hpile[onsetc:], color="orange", linestyle="solid")
                                os.chdir("..")

                            except FileNotFoundError:
                                print("Skipping the TBL-less solution.")
                        #"""

                        #"""
                        if j == 1 and i == 0 and args.htsol and odircnt == 4: # Heat flux varying solution in Node 1
                            try:
                                _dirs = _dhtsol
                                _evol_name = "evol_" + str(_dirs) + ".h5"   
                                _args_name = "args.json"
                                _model_name = "model.json"

                                amean, hpile, onsetc, setmarker, _, _, _ \
                                    = extract_hpile_amean_from_each_dir(dirs=_dirs, args_json=_args_name, model_json=_model_name, evol_file=_evol_name)
                                
                                if len(amean) > 0:
                                    try: 
                                        idxam = np.where(amean[int(len(amean)/10):] == 0.0)[0][0]
                                        amean = amean[:(idxam)]
                                        hpile = hpile[:(idxam)]
                                        setmarker = setmarker[:(idxam)]

                                    except IndexError: pass

                                ax.plot(amean[onsetc:]*1.e3, hpile[onsetc:], color="orange", linestyle="solid")
                                os.chdir("..")

                            except FileNotFoundError:
                                print("Skipping the varying heat flux solution.")
                        #"""   

                else:
                    if j == 2: os.chdir(f"node{(i+1):d}_{((j-1)):d}")
                    if j == 3: os.chdir(f"node{(i+1):d}_{((j-1)):d}")
                    cwd = os.getcwd()
                    directories = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d)) and d.startswith(current_year)]
                    if _dtblless in directories: directories.remove(_dtblless)
                    if _dhtsol in directories: directories.remove(_dhtsol)
                    
                    for itmp, dirs in enumerate(directories): # nodes!
                        comp = re.search(r"XL_([\d.]+)", dirs)
                        comp_f = str(comp.group(1))
                        idx = idxf(comp_f)
                        if idx != idxCSD: continue # here: selecting XL=75 wt%

                        evol_name = "evol_" + str(dirs) + ".h5"
                        args_name = "args.json"
                        model_name = "model.json" # you go to the dir!
                        amean, hpile, onsetc, _, _, _, _ = extract_hpile_amean_from_each_dir(dirs=dirs, args_json=args_name, model_json=model_name, evol_file=evol_name)
                        gif_file = "dgf"

                        ####################################### # FIXME: TOHLE SE MI MOC NELÍBÍ :(
                        sed_mode = re.search(f"S_(\d)", dirs)
                        sed_mode = int(sed_mode.group(1))
                        runH0 = re.search(f"H(\d)", dirs)
                        runH0 = float(runH0.group(1))
                        #######################################
                        
                        match sed_mode:
                            case 1:
                                sed_method = "dst"
                            case 3:
                                sed_method = "tac"
                            case 4: 
                                sed_method = "angz"

                        if os.path.exists(gif_file):
                            os.chdir(gif_file) # NOTE: youre in dgf!
                            files = os.listdir(os.getcwd())
                            numfiles = len(files)
                            cnum = numfiles
                            if cnum_rewrite: 
                                cnum = cnum0
                            printstep = int(numfiles / cnum)
                            cmap = cm.get_cmap("plasma", cnum)
                            norm = colors.Normalize(vmin=0.0, vmax=np.max(hpile))
                            handlesSED, labelsSED, handlesBLK, labelsBLK = map(lambda x: list(x), repeat([], 4)) # TODO: remove?
                            if len(files) == 1: 
                                print("Number of files is 1, CSD not plotted!")
                                break
                            for counter, f in enumerate(files, start=1):
                                if f.endswith("pdf") or (counter % printstep != 0 and printstep > 0):  # TODO: oprav, pokud je jenom jeden file... 
                                    continue
                                _, _, step = f.split("_")
                                step = int(step)
                                idxcm = cmap(norm(hpile[step]))
                                with open(f, "r") as of:
                                    if args.print_xldgf:
                                        _XL0 = 1.e2 * compositions_float[idxCSD]
                                        ax.text(
                                            0.99, 0.99, f"$X_{{L,0}}$ = {_XL0:.1f} %",
                                            transform=ax.transAxes,   
                                            ha="right", 
                                            va="top",
                                            fontsize=9
                                        )
                                    #print("SED_METHOD:", sed_method)
                                    if dstnorm:
                                        match sed_method:
                                            case "tac": 
                                                (pased, pnsed, pablk, pnblk, _, _, _, _,) = np.loadtxt(of, unpack=True)
                                                labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                                                lineSED, = ax.plot(1.e3*pased, pnsed/np.sum(pnsed), c=idxcm, label=labelSED)
                                                
                                                """
                                                try: 
                                                    (pased, pnsed, pablk, pnblk, _, _, _, _, _, _) = np.loadtxt(of, unpack=True)
                                                    labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                                                    lineSED, = ax.plot(1.e3*pased, pnsed/np.sum(pnsed), c=idxcm, label=labelSED)

                                                except ValueError:
                                                    (pased, pnsed, pablk, pnblk, _, _, _, _,) = np.loadtxt(of, unpack=True)
                                                    labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                                                    lineSED, = ax.plot(1.e3*pased, pnsed/np.sum(pnsed), c=idxcm, label=labelSED)
                                                """

                                            case "dst":
                                                (adistBLK, ndistBLK, ndistSED) = np.loadtxt(of, unpack=True)
                                                labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                                                lineSED, = ax.plot(1.e3*adistBLK, ndistSED/np.sum(ndistSED), c=idxcm, label=labelSED)
                                                
                                            case "angz":
                                                (adistBLK, ndistBLK, ndistSED) = np.loadtxt(of, unpack=True)
                                                labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                                                lineSED, = ax.plot(1.e3*adistBLK, ndistSED/np.sum(ndistSED), c=idxcm, label=labelSED)
                                    else:
                                        match sed_method:
                                            case "tac": 
                                                (pased, pnsed, pablk, pnblk) = np.loadtxt(of, unpack=True)
                                                labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                                                lineSED, = ax.plot(1.e3*pased, pnsed, c=idxcm, label=labelSED)
                                            case "dst":
                                                (adistBLK, ndistBLK, ndistSED) = np.loadtxt(of, unpack=True)
                                                labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                                                lineSED, = ax.plot(1.e3*adistBLK, ndistSED, c=idxcm, label=labelSED)
                                            case "angz":
                                                (adistBLK, ndistBLK, ndistSED) = np.loadtxt(of, unpack=True)
                                                labelSED = f"Sediment {1.e2*hpile[step]/runH0:.1f} [%]"
                                                lineSED, = ax.plot(1.e3*adistBLK, ndistSED, c=idxcm, label=labelSED)
                                of.close()
                            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                            ax.set_yticks([]) 
                            if j == 2: 
                                cbarsed = plt.colorbar(sm, ax=ax, orientation="vertical") # add one color map to the second graph!
                                #cbarsed.set_label("$h$ [m]", labelpad=10, fontsize=12, loc='center')
                                cbarsed.ax.xaxis.set_label_position("bottom")
                                cbarsed.ax.yaxis.set_ticks_position("right")

                            #cbarsed.set_label("Stratification height [m]")
                        if j == 2: ax.set_title(r"$\mathcal{D}_{S}(h; \nu)$")
                        #if j == 3: ax.set_title(r"$\mathcal{D}_{S}(h; \nu)$")

            except FileNotFoundError: 
                print(f"I skipped node {(i+1):d} and figure {(j+1):d}!")
                ax.set_yticks([]) 
                ax.set_xticks([])
                continue

    if args.repeat:
        path = ogpath + "/" + args.path
        os.chdir(path)
    else:
        os.chdir(args.path)

    big_ax = fig.add_subplot(111, frameon=False)
    big_ax.set_xticks([]); big_ax.set_yticks([])
    big_ax.set_xlabel(""); big_ax.set_ylabel("")

    # Make room at the bottom:
    fig.subplots_adjust(bottom=0.35) 
    try:
        handles = handles[0]
        labels = labels[0]

        def extract_number(label):
            # This looks for a floating-point number at the end of the string
            match = re.search(r'(\d+\.\d+)$', label.strip())
            if match:
                return float(match.group(1))
            else:
                raise ValueError(f"No number found in label '{label}'")

        # Sort the handles:
        combined = list(zip(handles, labels))
        combined.sort(key=lambda hl: extract_number(hl[1]))
        handles_sorted, labels_sorted = zip(*combined)
        handles_sorted = list(handles_sorted)
        labels_sorted  = list(labels_sorted)
        for i, lab in enumerate(labels_sorted):
            labels_sorted[i] = "".join([lab, " wt% An"])

        main_legend = fig.legend(
            handles_sorted, labels_sorted,
            loc="lower center",
            ncol=4,                  
            fontsize=12,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0) 
        )
        fig.add_artist(main_legend)

    except IndexError:
        print(f"Handles has the length of {len(handles):d}!")
        main_legend = None

    custom_lines = [
        Line2D([0], [0], color="black", linestyle="-",  lw=2),  
        Line2D([0], [0], color="black", linestyle="--", lw=2),  
        Line2D([0], [0], color="black", linestyle=":",  lw=2)    
    ]
    custom_labels = ["Dust-like regime", "Transitional regime", "Unmixed stone-like"]

    fig.legend(
        custom_lines, custom_labels,
        loc="lower center",
        ncol=3,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)  # slightly below the first legend
    )

    # Adjust layout to make space for legend:
    plt.subplots_adjust(bottom=0.08)
    plt.savefig("FINAL_GRADING.pdf", format="pdf", dpi=600, bbox_inches="tight")
    print("Figure 1 completed (crystal gradings and distributions).")
    plt.clf()

    """
    ####################################################################
    #####  COMPHREHENSIVE FIGURE OF TEMPERATURE, SETTLING, GROWTH  #####
    ####################################################################
    fig.subplots_adjust(top=0.87)                                       
    outer_grid = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.4)       
    for i in range(9): 
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[i], wspace=0.2)
        # Back to "Final Results"!
        print(f"Node {(i+1):d} & Directory {os.getcwd()}")
        row, col = divmod(i, 3)
        if row == 0:
            ax_dummy = fig.add_subplot(outer_grid[i])
            ax_dummy.axis("off")  # hide dummy axis!
            if col == 0:
                ax_dummy.set_title("$G_0 = 1 \\times 10^{-8}$ m/s", fontsize=14, pad=30)
            elif col == 1: 
                ax_dummy.set_title("$G_0 = 1 \\times 10^{-7}$ m/s", fontsize=14, pad=30)
            else:
                ax_dummy.set_title("$G_0 = 1 \\times 10^{-6}$ m/s [RELEVANT]", fontsize=14, pad=30)
        
        # --- Vertical H0 labels for nodes 1,4,7 ---
        if i in [0, 3, 6]:  # first column of each row of nodes
            ax_dummy_v = fig.add_subplot(outer_grid[i])
            ax_dummy_v.axis("off")
            bbox = ax_dummy_v.get_position()
            if i == 0: idh = 0
            if i == 3: idh = 1
            if i == 6: idh = 2
            H0 = H0_values[idh]
            fig.text(
                bbox.x0 - 0.04,                              # slightly left of node
                bbox.y0 + bbox.height / 2 - 0.01,            # center vertically
                rf"$H_0 = {H0}$ m",
                rotation='vertical',
                va='center',
                ha='center',
                fontsize=14
            )
    # Adjust layout to make space for legend:
    plt.subplots_adjust(bottom=0.08)
    plt.savefig("FINAL_NGTEMP.pdf", bbox_inches="tight")
    print("Figure 2 completed (temperatures, growth, and nucleation).")
    """

    return

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args=args)