# Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 4)
# ParMaCh Generation 1 (Python version)
# Author: MSci. Vít Beran
# Module: Main unit
# Run with: python3 main.py (--flags of your choice)

"""
  "ParMaCh" - Parameterized Model of a Magma Chamber
  - Crystallizing convecting magma reservoir (dimensional formulation)

    Features:
        > Vigorous and turbulent thermal Rayleigh-Benárd convection (Grossman-Lohse 2000)
        > Crystal settling (Martin and Nokes 1989, Patočka et al. 2022)
        > Crystal size distribution function approach (Randolph and Larsen 1971, Marsh 1988)
        > Realistic kinetic law of crystal growth and nucleation (Hort 1997, Couch et al. 2003)
        > 0D energy balance or 1D coupling with host rock
        > Binary eutectic system Anortit-Diopside (Courtial et al. 2000, Gale et al. 2008, Giordano et al. 2008, Krattli and Schmidt 2021)
        > TODO: coupling with MAGEMin software

    Examples: 
        > 0D reference run A: python main.py --SOLVER=0 --input_hflux --hflux=200 --hfluxSI --V0HG97=1.e-6 --XL0=0.75 --ratio=10000 --HE_CSTM --SED_METHOD=3 --nu --Ti97=0.92 --Tg97=0.95
        > 0D reference run B: python main.py --SOLVER=0 --input_hflux --hflux=200 --hfluxSI --V0HG97=1.e-6 --XL0=0.75 --ratio=1000 --HE_CSTM --SED_METHOD=3 --nu --Ti97=0.81 --Tg97=0.93 
        > 1D reference simulation: python main.py --SOLVER=1 --V0HG97=1.e-8 --XL0=0.75 --SED_METHOD=3 --nu --H0=1000 --suc=15.0
    
    Unit tests:
        > # TODO

    For more information, type "python3 main.py --help".
"""

# Standard libraries:
import os   
import h5py
import json
import shutil
import pickle
import sys
import argparse
import glob
import warnings
import subprocess
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from numba import njit
from itertools import chain
from scipy.optimize import curve_fit

# Modules:
from mPar import * 
from mUtils import *
from mFunc import *
from mGL import *
from mPlot import *
from mPhase import ODMC_solver, IDMC_solver, srun_solver
from mJW94 import *
from m1DHE import *   
from mMisch import *

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(prog="1DMC",
                                 description="ParMaCh - 1D parametrized model of a magma chamber\n" \
                                             "Requirements: Python 3.10+, numpy, numba (jit), tqdm, objgraph, h5py",
                                 epilog="Author: MSci. Vít Beran, MFF UK, Prague"
                                )
parser.add_argument("--deltaT",        default=1e2,      type=float,                       help="Temperature contrast (roof vs. bulk)") # FIXME: OBSOLETE
parser.add_argument("--H0",            default=1e3,      type=float,                       help="Initial height of the system.")
parser.add_argument("--hflux",         default=100,      type=float,                       help="Custom input heat flux in HFU units.")
parser.add_argument("--SOLVER",        default=1,        type=int, choices=[0,1,2],        help="ParMaCh solver: O (ODMC), 1 (1DMC), 2 (ODMC, single run).")
parser.add_argument("--TBL_METHOD",    default=1,        type=int, choices=[1,2],          help="1 (generations), 2 (step-by-step)")
parser.add_argument("--NG_METHOD",     default=2,        type=int, choices=[1,2],          help="1 (power-law), 2 (Hortian curves)")
parser.add_argument("--SED_METHOD",    default=3,        type=int, choices=[1,2,3],        help="1 (distributions), 2 (semi-analytical solution), 3 (crystal tracking)")    
parser.add_argument("--stpsCool",      default=1000000,  type=int,                         help="Number of cooling evolution steps.") 
parser.add_argument("--steps1D",       default=5000000,  type=int,                         help="Number of steps in the 1D heat equation.")
parser.add_argument("--XL0",           default=0.99,     type=float,                       help="Initial composition (wt%% An).")
parser.add_argument("--V0HG97",        default=1.e-6,    type=float,                       help="Amplitude of the Hortian growth rate.")
parser.add_argument("--N0HG97",        default=1.e3,     type=float,                       help="Amplitude of the Hortian nucleation rate.")
parser.add_argument("--V0POW",         default=1.e-6,    type=float,                       help="Amplitude of the power-law growth rate.")
parser.add_argument("--N0POW",         default=1.e3,     type=float,                       help="Amplitude of the power-law nucleation rate.")
parser.add_argument("--suc",           default=5.0,      type=float,                       help="Supercooling (emplacement temperature)")
parser.add_argument("--TE",            default=1,        type=int, choices=[1,2,3],        help="Euler vs. AB5 vs. RK4.")
parser.add_argument("--ratio",         default=1000,     type=float,                       help="Initial time step scaling.")
parser.add_argument("--order",         default=0,        type=int, choices=[-1,0,1,2,3,4], help="Order of the integration scheme.")
parser.add_argument("--Tref",          default=1142.,    type=float,                       help="Auxiliary reference temperature for the non-dimensionality.")
parser.add_argument("--kinref",        default=-1,       type=int, choices=[-1,0,1],       help="Reference kinetics (-1 custom, 0 Hort, 1 Couch).")
parser.add_argument("--MODE",          default=None,     type=bool,                        help="Crystal dynamics method (adaptive | uniform).")
parser.add_argument("--Ti97",          default=0.92,     type=float,                       help="Custom nucleation peak.")
parser.add_argument("--Tg97",          default=0.95,     type=float,                       help="Custom growth peak.")
parser.add_argument("--printstep",     default=1000,     type=int,                         help="Save distributions each X steps.")
parser.add_argument("--EA",            default=1.0,      type=float,                       help="Activation energy (multiplication by the reference value).")
parser.add_argument("--dir",           default=None,     type=str,                         help="Rewrite the directory name.")
parser.add_argument("--srun",          default=False,    action="store_true",              help="Single steady-state computation for an arbitrary temperature profile.")
parser.add_argument("--hfluxSI",       default=False,    action="store_true",              help="Custom input heat flux in W/m2.")
parser.add_argument("--ODMC",          default=True,     action="store_false",             help="ODMC solver.") 
parser.add_argument("--IDMC",          default=False,    action="store_true",              help="1DMC solver.")
parser.add_argument("--bAnDi",         default=True,     action="store_false",             help="Turn on/off the anortit-diopside binary.")
parser.add_argument("--DEBUG",         default=False,    action="store_true",              help="Additional information printed in the terminal.")
parser.add_argument("--JWSOLVER",      default=False,    action="store_true",              help="Paper of Jarvis and Woods (1994), implementation.")
parser.add_argument("--JWLIMIT",       default=False,    action="store_true",              help="Jarvis-Woods limit.")
parser.add_argument("--input_hflux",   default=False,    action="store_true",              help="Input heat flux")
parser.add_argument("--troof_const",   default=False,    action="store_true",              help="Special case: keeping Troof constant.")
parser.add_argument("--nu",            default=False,    action="store_true",              help="Temperature dependent viscosity.")
parser.add_argument("--nutbl",         default=False,    action="store_true",              help="Viscosity in the TBL (on/off).")
parser.add_argument("--TBL",           default=True,     action="store_false",             help="TBL dynamics (on/off)")
parser.add_argument("--JW",            default=False,    action="store_true",              help="Complete limit of Jarvis and Woods.")
parser.add_argument("--fr",            default=False,    action="store_true",              help="Full run (= full solidification up to the eutectic temperature).")
parser.add_argument("--DEBUGRUN",      default=False,    action="store_true",              help="Debugger for the very first step of the simulation.")
parser.add_argument("--DEBUG100",      default=False,    action="store_true",              help="First 100 time steps after the onset of crystallization.")
parser.add_argument("--HE",            default=False,    action="store_true",              help="Solve 1D heat equation to get an estimate on time-dependent flux.")
parser.add_argument("--HE_CONST",      default=False,    action="store_true",              help="Solve 1D heat equation from the geological parameters, keep the heat flux constant.")
parser.add_argument("--DEBUG_X",       default=False,    action="store_true",              help="First 100 time steps after the onset of crystallization.")
parser.add_argument("--MT",            default=False,    action="store_true",              help="Rewrite GL with simple Malkus (1954), Racrit heurestic scaling.")
parser.add_argument("--iter",          default=True,     action="store_false",             help="Iterations in the predictor-corrector scheme (on/off).")
parser.add_argument("--tblaoff",       default=True,     action="store_false",             help="Set the TBL radii to 0.0 [mm].")
parser.add_argument("--HE_CSTM",       default=False,    action="store_true",              help="Custom decrease of the heat flux.")
parser.add_argument("--MELTS",         default=None,     action="store_true",              help="Input from MELTS.")
parser.add_argument("--RTIS",          default=False,    action="store_true",              help="Rayleigh-Taylor instability sweeping mechanism (on/off).")
parser.add_argument("--LHEAT",         default=True,     action="store_false",             help="Latent heat in 1D coupling (on/off).")

def save_args_to_file(args, fname, ns=True): 
    with open(fname, "w") as f:
        if ns: json.dump(vars(args), f, indent=4) # better readability!
        else: json.dump(args, f, indent=4)

def main_SNGL(args: argparse.Namespace, SIGN: bool):
    if SIGN: print(chamber)
    print(" Coupling DNS with ParMaCH: ")
    print("="*40)
    print(" Running a 1-snapshot call of ParMaCH...")

    """
        This call of ParMaCh computes a snapshot of the chamber
        with vertically distributed latent heat.
    """

    # 1) Initialize the kinetic parameters:
    #print(" Only the Hortian crystallisation is supported..."); args.NG_METHOD = 2
    kinvals = "-"
    match args.kinref:
        case 0: # mean values from Hort 1997:
            kinvals = "Hort 1997"
            ModelParameter.V0_HG97 = 1.e-6; ModelParameter.N0_HG97 = 1.e3
            ModelParameter.Ti_HG97 = 0.92; ModelParameter.Tg_HG97 = 0.95
        case 1: # rescaled values from Couch 2003:
            kinvals = "Couch 1997"
            ModelParameter.V0_HG97 = 8.e-5; ModelParameter.N0_HG97 = 1.e4 
            ModelParameter.Ti_HG97 = 0.81; ModelParameter.Tg_HG97 = 0.93
        case -1:
            kinvals = "Custom"
            ModelParameter.V0_HG97 = args.V0HG97
            ModelParameter.N0_HG97 = args.N0HG97
            ModelParameter.V0POW   = args.V0POW
            ModelParameter.N0POW   = args.N0POW
            ModelParameter.Ti_HG97 = args.Ti97
            ModelParameter.Tg_HG97 = args.Tg97
    if args.NG_METHOD == 1 and args.kinref != -1: raise Exception("You chose reference kinetics!")

    # Assert mTaC_UNIFIED solver and step-by-step approach in the TBL:
    args.SED_METHOD = 3
    args.TBL_METHOD = 2

    # 2) Attributes Namespace:             
    for key, atrval in vars(args).items():
        if hasattr(Attributes, key):
            setattr(Attributes, key, atrval) 

    # 3) Assemble the outputfile:
    assemble_output_file_name(args=args)
    if not os.path.exists(ModelParameter.outfile): os.mkdir(ModelParameter.outfile)

    # 3) Find the nucleation delay:
    # TODO: this is done elsewhere

    # 4) Assemble the binary alloy | petrological component:
    args.bAnDi = True
    if args.bAnDi:
        if args.XL0 > ModelParameter.Xeut:
            ModelParameter.rhoc = ModelParameter.rhocA
        else:
            ModelParameter.rhoc = ModelParameter.rhocD
    else:
        ModelParameter.rhoc = ModelParameter.rhocA

    # Initialize and save the alloy:
    binalloyAnDi = BinaryAlloy(
                    Tmax1 = ModelParameter.TmaxAn,
                    Tmax2 = ModelParameter.TmaxDi,
                    Teut  = ModelParameter.TeutAD,
                    X0    = args.XL0,
                    Xeut  = ModelParameter.Xeut,
                    Dm1   = ModelParameter.Dm1,
                    Dm2   = ModelParameter.Dm2
                )
    with open(f"{ModelParameter.outfile}/binary_alloy.pkl", "wb") as f:       
        pickle.dump(binalloyAnDi, f)
        f.close()

    #ModelParameter.epsdel = return_epsdel(SingleRun.Tliqd)
    #print(f"Nucleation delay: {ModelParameter.epsdel:.3f}.")

    # 3) Initialisation of the single run:

    # Load the chamber profile from a json file supplied by 2DConLat:
    
    SingleRun = SingleRunAttributes() #deltaT=0.0, epsd=0.0, flux=0.0)
    try:
        print(f" Loading the input parameters from 'parmach_input.json'...")
        print(" ", os.getcwd())
        SingleRun.load_args_from_file("parmach_input.json")

    except FileNotFoundError:
        print(f" The 2DConLat 'parmach_input.json' file with input parameters for the 1-snapshot solver not found!")
        print(f" Terminating ParMaCH!")
        exit()

    # Derive the nucleation lag and the nucleation threshold:
    epsdel = return_epsdel(SingleRun.Tliqd)    
    SingleRun.Tnucl = SingleRun.Tliqd - epsdel
    SingleRun.physics_check()
    SingleRun.parameter_print()

    # 4) Call the 1-snapshot solver:       
    srun_solver(SingleRun=SingleRun, 
                alloy=binalloyAnDi
                )

    print(f" Visualising the latent heat distributions...")
    print(f" Saving the latent heat sources into 'XY.dat'...")
    return

def main_FULL(args: argparse.Namespace, SIGN: bool):
    if SIGN: print(chamber)

    """ Parametrized model of a cooling magma chamber. """
    if args.DEBUG: print(f"Model initiated with the following parameters: {args}.")
    print(" INITIALIZATION OF PARMACH:")
    print("", "-" * 40)
    if args.DEBUGRUN: print("[WARNING] - One debugging time step only!")

    # Choose the ParMaCh application:
    match args.SOLVER:
        case 0:
            args.IDMC = False; args.ODMC = True
            print(" > Employed solver: ODMC (ParMaCH + OD-energy balance).")
        case 1:
            args.input_hflux = True; args.troof_const = False; args.IDMC = True; args.ODMC = False
            args.HE = True
            print(" > Employed solver 1DMC (ParMaCh + 1DHE).")

    """ ############################################## """
    """ ######   AUXILIARY IO/ATTRIBUTES SETUP  ###### """
    """ ############################################## """

    # 0) Foolproof constraints on the input parameters:     
    assert args.hflux >= 0.1                                            # [W/m2] the minimum heat flux through the roof in HFUs from Martin et al. (1997)
    assert args.XL0 != ModelParameter.Xeut                              # we do not start in the eutectic composition!
    assert args.V0HG97 <= 1.e-4,  "Risk of numerical instability!"        
    assert args.V0POW  <= 1.e-4,  "Risk of numerical instability!"       
    assert args.N0HG97  < 1.01e6, "Maximum reported by Hort (1997)"
    assert args.N0POW   < 1.01e6
    assert args.XL0 > 0.0 and args.XL0 < 1.0, "Composition range!"    
    if args.input_hflux == args.troof_const:    
        print("[ERROR] - You may only select one roof region mode!")
        sys.exit(1)

    # Jarvis & Woods 1994 limit:
    if args.JWLIMIT: args.TBL = False; args.NG_METHOD = 1; #args.SED_METHOD = 4

    # 1) Initialize the kinetic parameters:
    kinvals = "-"
    match args.kinref:
        case 0: # mean values from Hort 1997:
            kinvals = "Hort 1997"
            ModelParameter.V0_HG97 = 1.e-6; ModelParameter.N0_HG97 = 1.e3
            ModelParameter.Ti_HG97 = 0.92; ModelParameter.Tg_HG97 = 0.95
        case 1: # rescaled values from Couch 2003:
            kinvals = "Couch 1997"
            ModelParameter.V0_HG97 = 8.e-5; ModelParameter.N0_HG97 = 1.e4 
            ModelParameter.Ti_HG97 = 0.81; ModelParameter.Tg_HG97 = 0.93
        case -1:
            kinvals = "Custom"
            ModelParameter.V0_HG97 = args.V0HG97
            ModelParameter.N0_HG97 = args.N0HG97
            ModelParameter.V0POW   = args.V0POW
            ModelParameter.N0POW   = args.N0POW
            ModelParameter.Ti_HG97 = args.Ti97
            ModelParameter.Tg_HG97 = args.Tg97
    if args.NG_METHOD == 1 and args.kinref != -1: 
        raise Exception("You chose reference kinetics!")

    # 1) Attributes Namespace:             
    for key, atrval in vars(args).items():
        if hasattr(Attributes, key):
            setattr(Attributes, key, atrval)        

    # Initialize the binary alloy (Di-An system for now) vs. JW density:
    if args.bAnDi:
        if args.XL0 > ModelParameter.Xeut:
            ModelParameter.rhoc = ModelParameter.rhocA
        else:
            ModelParameter.rhoc = ModelParameter.rhocD
    else:
        ModelParameter.rhoc = ModelParameter.rhocA

    # Assemble the output file name:
    assemble_output_file_name(args=args)

    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    timestamp = "".join([timestamp, "_"])
    if args.JWSOLVER: timestamp = "JWSOLVER_"
    rename = True
    if args.input_hflux: 
        if args.HE: 
            if args.HE_CONST:
                drv = "_F0"
            else: 
                drv = "_F0C"
        else:
            drv = "_F" + "{:.2e}".format(args.hflux)
    if args.troof_const: drv = "_dT" + "{:.2e}".format(args.deltaT) 
    if args.input_hflux: ModelParameter.outfile = "".join([timestamp, "RM_1", ModelParameter.outfile]) # roof region!
    if args.troof_const: ModelParameter.outfile = "".join([timestamp, "RM_2", ModelParameter.outfile]) 
    _nustatus = "_1" if args.nu else "_0" 
    _nutblstatus = "1" if args.nutbl else "0"
    _glstatus = "_0" if args.MT else "_1"
    _order = str(args.order)
    if rename: 
        if Attributes.NG_METHOD == 1:
            if Attributes.MODE is None:
                ModelParameter.outfile += "XL_" + "{:.2f}".format(args.XL0) + drv + "H" + \
                    "{:.0e}".format(args.H0) + "G" + "{:.0e}".format(ModelParameter.V0POW) + "_N" + "{:.0e}".format(ModelParameter.N0POW) \
                        + "NG_" + str(Attributes.NG_METHOD) + "_S_" + str(Attributes.SED_METHOD) + "_nu" + _nustatus + _nutblstatus \
                        + "_O" + _order + "_GL" + _glstatus
            else: # "0" stands for adaptive settling method!
                ModelParameter.outfile += "XL_" + "{:.2f}".format(args.XL0) + drv + "H" + \
                    "{:.0e}".format(args.H0) + "G" + "{:.0e}".format(ModelParameter.V0POW) + "_N" + "{:.0e}".format(ModelParameter.N0POW) \
                        + "NG_" + str(Attributes.NG_METHOD) + "_S_" + "0" + "_nu" + _nustatus + _nutblstatus \
                        + "_O" + _order + "_GL" + _glstatus
        if Attributes.NG_METHOD == 2:
            if Attributes.MODE is None:
                ModelParameter.outfile += "XL_" + "{:.2f}".format(args.XL0) + drv + "H" + \
                    "{:.0e}".format(args.H0) + "G" + "{:.0e}".format(ModelParameter.V0_HG97) + "_N" + "{:.0e}".format(ModelParameter.N0_HG97) \
                            + "NG_" + str(Attributes.NG_METHOD) + "_S_" + str(Attributes.SED_METHOD) + "_nu" + _nustatus + _nutblstatus  \
                            + "_O" + _order + "_GL" + _glstatus 
            else: # "0" stands for adaptive settling method!
                ModelParameter.outfile += "XL_" + "{:.2f}".format(args.XL0) + drv + "H" + \
                    "{:.0e}".format(args.H0) + "G" + "{:.0e}".format(ModelParameter.V0_HG97) + "_N" + "{:.0e}".format(ModelParameter.N0_HG97) \
                            + "NG_" + str(Attributes.NG_METHOD) + "_S_" + "0" + "_nu" + _nustatus + _nutblstatus  \
                            + "_O" + _order + "_GL" + _glstatus 
    if args.JW: ModelParameter.outfile = "".join([ModelParameter.outfile, "_JW"])
    if args.TBL: ModelParameter.outfile = "".join([ModelParameter.outfile, "_TBL"])
    if args.bAnDi: ModelParameter.outfile = "".join([ModelParameter.outfile, "_AnDi"])
    if args.LHEAT and args.SOLVER == 1: ModelParameter.outfile = "".join([ModelParameter.outfile, "_LHEAT"])
    """
    
    # Rewrite the assembled name of the output file if desired:
    if args.dir is not None: ModelParameter.oufile = args.dir

    # Clear gif distributions:
    gf = os.path.join(ModelParameter.outfile, ModelParameter.giffile)
    df = os.path.join(ModelParameter.outfile, ModelParameter.dstfile)
    zf = os.path.join(ModelParameter.outfile, ModelParameter.rfzfile)
    if os.path.exists(gf):
        shutil.rmtree(gf)
    if os.path.exists(df):
        shutil.rmtree(df)

    if not os.path.exists(ModelParameter.outfile): os.mkdir(ModelParameter.outfile)
    os.makedirs(gf, exist_ok=True)
    os.makedirs(df, exist_ok=True)
    os.makedirs(zf, exist_ok=True)

    # Set up the files:
    dirs = [ModelParameter.outfile + "/plot", ModelParameter.outfile + "/plot/individual", ModelParameter.outfile + "/plot/summary",
            ModelParameter.outfile + f"/{ModelParameter.IDHfile}", ModelParameter.outfile + f"/{ModelParameter.IDHdata}"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
        else:
            shutil.rmtree(dir)
            os.makedirs(dir, exist_ok=True)

    # Clear the previous run (if necessary): 
    out_files = [f"/evol_{ModelParameter.outfile}.h5", "/0DMC.log", "/args.json", "/model.json", "/1DHE_flux.dat", \
                 "/binary_alloy.pkl", "/1DHE_flux.dat"]
    for f_name in out_files:
        tmp_name = ModelParameter.outfile + f_name
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    
    snaps_to_delete = chain(
        glob.glob(ModelParameter.outfile + "/dst/dist_snap*"),
        glob.glob(ModelParameter.outfile + "/dst/track_snap*"),
        glob.glob(ModelParameter.outfile + "/dst/single_*.pdf")
    )
    for f in snaps_to_delete:
        os.remove(f)

    # Set up the logger:
    logging.basicConfig(filename=ModelParameter.outfile + "/0DMC.log",
                        level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
                       )
    logger.info(chamber)

    # Set up the h5 output file:
    HEVOL_VARS = 3; TEVOL_VARS = 20; DIAG_VARS = 28; IDC_VARS = 3
    evol_file_name = ModelParameter.outfile + f"/evol_" + ModelParameter.outfile + ".h5"
    evol_file = h5py.File(evol_file_name, 'w')
    evol_file.create_group("1DHE")
    evol_file.create_group("time_evolution")
    evol_file.create_group("diagnostics")
    evol_file.create_group("indices")

    evol_file["time_evolution"].create_dataset(
        "vars", shape=(0, TEVOL_VARS), maxshape=(None, TEVOL_VARS),
        dtype="float64", chunks=(1, TEVOL_VARS))

    evol_file["diagnostics"].create_dataset(
        "vars", shape=(0, DIAG_VARS), maxshape=(None, DIAG_VARS),
        dtype="float64", chunks=(1, DIAG_VARS))

    """ ############################################## """
    """ ########## START OF THE MAIN UNIT  ########### """
    """ ############################################## """

    # TODO: REPLACE WITH MAGEMIN COUPLING
    if args.MELTS: 
        import petthermotools as ptt
        print(ptt.__version__)
        sys.path.append(r'/home/beran/alphamelts_py-2.3.1/alphamelts-py-2.3.1-linux')

        morb = {
        'SiO2_Liq'  : 50.72,
        'TiO2_Liq'  : 1.47,
        'Al2O3_Liq' : 15.09,
        'FeOt_Liq'  : 8.84,   # total iron as FeO (Fe2O3 + FeO combined)
        'Fe3Fet_Liq': 0.12,   # fraction of total Fe that is Fe3+
        'MnO_Liq'   : 0.17,
        'MgO_Liq'   : 9.11,
        'CaO_Liq'   : 11.44,
        'Na2O_Liq'  : 2.72,
        'K2O_Liq'   : 0.09,
        'P2O5_Liq'  : 0.10,
        'H2O_Liq'   : 0.05,
        }
        print(f"Mass sum: {sum(morb.values()):.2f}.")

        # Run fractional crystallisation:
        results_MELTS = ptt.isobaric_crystallisation(
            bulk          = morb,
            Model         = "MELTSv1.0.2",   # rhyolite-MELTS 1.0.2
            find_liquidus = True,            # finds liquidus T automatically
            T_end_C       = 925.0,           # stop at 925°C
            dt_C          = 5.0,             # 5°C steps
            P_bar         = 1000.0,          # 1 kbar, shallow crustal chamber
            fO2_buffer    = "FMQ",
            fO2_offset    = 0.0,
            Frac_solid    = True,           
            timeout       = 600
        )

        # See what's in the results
        print(results_MELTS.keys())
        #plot_all(results_MELTS, save=True)

    # Initialize all ParMaCh arrays: # INIT CLAS FOR ALL INITIAL CONDITIONS
    stpsCool = args.stpsCool

    # Initialize and save the alloy:
    binalloyAnDi = BinaryAlloy(
                    Tmax1 = ModelParameter.TmaxAn,
                    Tmax2 = ModelParameter.TmaxDi,
                    Teut  = ModelParameter.TeutAD,
                    X0    = args.XL0,
                    Xeut  = ModelParameter.Xeut,
                    Dm1   = ModelParameter.Dm1,
                    Dm2   = ModelParameter.Dm2
                )
    with open(f"{ModelParameter.outfile}/binary_alloy.pkl", "wb") as f:       
        pickle.dump(binalloyAnDi, f)
        f.close()

    # Calculate the initial liquidus temperature:
    ModelParameter.EaMod = Attributes.EA * ModelParameter.Ea
    Init.XL = args.XL0
    if Init.XL > binalloyAnDi._Xeut:
        ModelParameter.Tliqd0 = binalloyAnDi.branch_A(Init.XL) 
    else:
        ModelParameter.Tliqd0 = binalloyAnDi.branch_B(Init.XL)

    """ ############################################## """
    """ ########## START OF THE MAIN UNIT  ########### """
    """ ############################################## """
    """
    #% Solve 1D equation - onset of ParMaCh:
         Calculates:
         %> initial heat flux:               flux[0]
         %> time series (interpolatable):    froof
         %> gives us heat flux on time:      fitted function, A,B coefficients  
    """

    # Magma emplacement temperature (superheated):
    if args.HE:
        z0, T0, k0, rhocp0, dt1D = init_setup1D(H0=args.H0, suc=args.suc, const=Constants1D) 
        Tfinal, _, step_hit, qhit = evolution1D_nosource(T=T0,
                                                         k=k0,
                                                         rhocp=rhocp0,
                                                         z=z0,
                                                         dt=dt1D,
                                                         Tliqd=ModelParameter.Tliqd0,
                                                         max_steps=args.steps1D,
                                                         const=Constants1D
                                                    )
                    
        (htime, _, froof, _, _) = np.loadtxt(ModelParameter.outfile + "/1DHE_flux.dat", delimiter="\t", unpack=True)
        evol_file["1DHE"].create_dataset("flux", data=np.abs(froof))
        evol_file["1DHE"].create_dataset("htime", data=htime)  # TODO: remove .dat and replace fully in plot_1D w/ h5!
        idxmaxf = np.abs(froof).argmax()
        plot_1D(T0=T0, T=Tfinal, z=z0, step_hit=step_hit, Tliqd=ModelParameter.Tliqd0, const=Constants1D)

        Shared.tliqhit = htime[step_hit]
        args.hflux = abs(froof[step_hit]) # W/m2!
        #abs(hfce(htime[step_hit], Shared.A, Shared.B)) # <--- OVERWRITES THE HEAT FLUX HERE!
        print(f" > The default heat flux rewritten to {(args.hflux/RunConstants.HFU):.3e} HFU.")

    else:
        if args.hfluxSI:
            print(f" > You chose custom heat flux {args.hflux:.3e} W/m2.")
        else:
            print(f" > You chose custom heat flux {args.hflux:.3e} HFU.")
            args.hflux *= RunConstants.HFU
            print(f" > Converted to W/m2: {args.hflux:.3e}{Units.funit}")

    # Initial kinematic viscosity and Prandtl number:
    ModelParameter.nu0 = visc0_GIORDANO(T=ModelParameter.Tliqd0, Ea=ModelParameter.EaMod) / ModelParameter.rhof
    ModelParameter.Pr0 = ModelParameter.nu0 / ModelParameter.kappa
    
    # From the initial heat flux and Prandtl number, calculate iteratively the Nusselt number, the GL regime, and corresponding delta T:
    if args.input_hflux:
        print()
        print(" GROSSMANN-LOHSE INITIAL ITERATION:")
        print("", "-" * 40)
        init_GL = start_iterative_Ra_F(flux=args.hflux, Pr=ModelParameter.Pr0, H=args.H0, c=ModelParameter)
        for key, val in init_GL.items():
            if isinstance(val, float):
                print(f" {key}: {val:.3e}.")
            else:
                print(f" {key}: {val}.")
        regime_GL = init_GL["Regime"] # initial regime
        _Nu = init_GL["Nu"]           # initial Nusselt number
        if args.MT: 
            regime_GL = "IVu" # Malkus 1954, independant of the size of the system!
            print(" [WARNING] - scaling IVu adopted throughout the simulation instead (Malkus theory)!")
        Shared.regime = regime_GL
    else:
        # Constant roof temperature: keep the Malkus-like regime (IVu)
        print(" [WARNING] - scaling IVu adopted throughout the simulation instead (Malkus theory)!")
        Attributes.MT = True
        Shared.regime = "IVu"

    # Roof region: 
    rmode = None
    if args.input_hflux:
        """ Calculate the corresponding temperature contrast from the input heat flux. """
        rmode = "Varying flux" if args.HE or args.HE_CSTM else "Constant flux"
        if args.IDMC: rmode = "Varying heat flux (1DHE)"
        ModelParameter.mode = "Input heat flux"
        hflux0 = args.hflux  # initial heat flux, used in main.py
        args.deltaT = calculate_deltaT(hflux0, ModelParameter.nu0, args.H0, regime_GL, ModelParameter)
        print(f" Derived temperature contrast (from heat flux): {args.deltaT:.2e}{Units.Tunit}.") 

    elif args.troof_const:
        """ Special case: constant roof temperature above eutectic (benchmarking). """
        rmode = "Constant Troof"
        ModelParameter.mode = "Constant roof temperature"
        ModelParameter.Troof0 = ModelParameter.Tliqd0 - args.deltaT  
        if ModelParameter.Troof0 < binalloyAnDi._Teut:
            raise ValueError("Roof temperature is under the eutectic!")
        
        _Ra = calculate_rayleigh_number(args.H0, ModelParameter.Tliqd0, ModelParameter.Tliqd0 - args.deltaT, ModelParameter.nu0, ModelParameter)
        _Nu = calculate_nusselt_number(_Ra, ModelParameter.Pr0, Shared.regime)
        hflux0 = (_Nu * args.deltaT * ModelParameter.rhof * ModelParameter.heatcp * ModelParameter.kappa) / args.H0
        print(f"The initial temperature contrast: {(ModelParameter.Tliqd0 - ModelParameter.Troof0):.2e}{Units.Tunit}")
        print(f"The corresponding initial heat flux: {hflux0:.2e}{Units.funit}")
        print(f"The corresponding initial heat flux in HFU: {hflux0/RunConstants.HFU:.2e} [HFU]")
    
    """ ############################################## """
    """ ##########   SOLIDIFICATION LOOP   ########### """
    """ ############################################## """

    """ %%%%%% TIME EVOLUTION %%%%%% """
    
    # Determine the (initial) Hortian lag: 
    if Attributes.NG_METHOD == 2:
        ModelParameter.epsdel = return_epsdel(Tliqd=ModelParameter.Tliqd0)
        
    # Initial conditions:
    Init.Tliqd = ModelParameter.Tliqd0
    Init.Tbulk = ModelParameter.Tliqd0
    Init.Tnucl = ModelParameter.Tliqd0 - ModelParameter.epsdel
    Init.flux  = hflux0
    Init.Hnow  = args.H0
    Init.nu    = ModelParameter.nu0
    Init.XL    = args.XL0
    Init.suc   = args.suc

    # Initial 1D conditions for ParMaCh (AFTER THE INITIAL COOLING PERIOD!):
    try: 
        Init.z     = z0
        Init.k     = k0
        Init.rhocp = rhocp0
        Init.T     = Tfinal

    except UnboundLocalError: pass

    if args.input_hflux: Init.Troof = Init.Tbulk - args.deltaT 
    if args.troof_const: Init.Troof = ModelParameter.Troof0

    # Convection sanity check & check for stagnant lid:
    if calculate_rayleigh_number(args.H0, Init.Tbulk, Init.Troof, Init.nu, ModelParameter) >= RunConstants.Racrit: Shared.convon = True
    else: 
        raise Exception("Rayleigh number below the critical value!")
    nu1 = (visc0_GIORDANO(T=Init.Troof, Ea=ModelParameter.EaMod) / ModelParameter.rhof)
    nu2 = (visc0_GIORDANO(T=Init.Tbulk, Ea=ModelParameter.EaMod) / ModelParameter.rhof)
    print(f" TBL/bulk viscosity contrast (in orders): {np.log10(nu1/nu2):.2e}.")
    if np.log10(nu1/nu2) > 5.0:
        print(f" [WARNING] Viscosity contrast: {np.log10(nu1/nu2):.2e} (orders), \
                severely exceeds the stagnant lid treshold!")
    print()

    # FIXME: the factor of 0.788 is a solution to a transcendental equation, include here!
    lc0 = args.H0**2 / (4. * 0.788 * ModelParameter.kappa)                                                  # maximum solidification time, e.g. Holness et al. (2017)
    tc0 = (args.H0 * ModelParameter.rhof * ModelParameter.heatcp * (Init.Tliqd - Init.Troof)) / hflux0      # initial characteristic cooling time scale
    Diag.dtCool = tc0   

    if Attributes.DEBUG:
        print("-" * 40)
        print(f"#%%%%%%    TIME SCALES ESTIMATES    %%%%%%#")
        print("-" * 40, "\n")
        time_scales = {
            "Initial characteristic cooling scale:":   (tc0,      Units.tunit),
            "Upper solidification time estimate:":     (lc0,      Units.tunit)
        }
        dynamic_print(time_scales)
        print()

    print(" INITIAL CONDITIONS:")
    print("", "-" * 40)
    if Attributes.NG_METHOD == nMethod.mLin: klaws = " linear/power laws"
    if Attributes.NG_METHOD == nMethod.mLab: klaws = " Hortian laws"
    tbldp = " On" if Attributes.TBL else " Off"      # TODO: RESOLVE THIS?!
    gldp  = " On" if not Attributes.MT else " Off"
    nudp  = " On" if Attributes.nu else " Off"

    initial_params = {
        " Roof region:":                (rmode, ""),
        " TBL dynamics:":               (tbldp, ""),
        " GL regimes:":                 (gldp, ""),
        " Varying viscosity:":          (nudp, ""),
        " Kinetic laws:":               (klaws, ""),
        " Kinetics:":                   (kinvals, ""),
        " Prandtl number:":             (ModelParameter.Pr0, ""), 
        " Composition:":                (Init.XL, Units.xunit),
        " Kinematic viscosity:":        (Init.nu, Units.nunit), 
        " Initial height:":             (Init.Hnow, Units.sunit),
        " Liquidus temperature:":       (Init.Tliqd, Units.Tunit), 
        " Roof temperature:":           (Init.Troof, Units.Tunit), 
        " Bulk temperature:":           (Init.Tbulk, Units.Tunit), 
        " Nucleation threshold:":       (Init.Tnucl, Units.Tunit), 
        " Nucleation lag:":             (ModelParameter.epsdel, Units.Tunit),
        " Density of the phase:":       (ModelParameter.rhoc, Units.runit)
    }
    dynamic_print(initial_params)

    """ ############################################## """
    """ ##########   SOLIDIFICATION LOOP   ########### """
    """ ############################################## """
    print()
    print(" SOLIDIFICATION LOOP:")
    print("", "-" * 40)

    # The initial choice of the time step:
    if args.troof_const: 
        mode = 1
        args.Tref = ModelParameter.Troof0
        Diag.dtCool /= args.ratio

    if args.input_hflux: 
        mode = 2
        Diag.dtCool /= args.ratio                

    Shared.Tref = args.Tref    

    # Save model parameters and attributes of the model:
    _model_params = {item: getattr(ModelParameter, item) for item, _ in par_specs}
    _const_params = {item: getattr(RunConstants, item) for item, _ in const_specs}
    save_args_to_file(args, ModelParameter.outfile + "/args.json")
    save_args_to_file(_model_params, ModelParameter.outfile + "/model.json", ns=0)
    save_args_to_file(_const_params, ModelParameter.outfile + "/const.json", ns=0)
    print(" Parameters and arguments saved successfully.")

    """ Solve ODMC model. """
    Shared.idxend = 1000000 # default case!
    evol_file["indices"].create_dataset("idxend", data=Shared.idxend)
    if args.ODMC:   
        match mode:
            case "Input heat flux": print("[0DMC] - Solidification up to the eutectic temperature!") # TODO: only for benchmark!
            case "Constant roof temperature": print("[0DMC] - Solidification up to the roof temperature!")
        
        ODMC_solver(
            evol_file=evol_file, 
            alloy=binalloyAnDi,
            mode=mode,
            order=args.order,
            calibrate=False,  
            steps=stpsCool,
            mem_track=False
            )              

    elif args.IDMC:        
        Diag.dtCool = dt1D # supply the CFL-based time step to the 1DMC solver
        IDMC_solver(
            evol_file=evol_file,
            alloy=binalloyAnDi,
            latent_source=args.LHEAT
        )

    else:
        print(" [WARNING] - 0DMC solver skipped!")

    """ Solve J&W94 model. """
    if args.JWSOLVER:
        Shared.Apar = 1000; Shared.Spar = 1 

        tCoolJW, TliqdJW, TbulkJW, TnuclJW, TroofJW, \
        hPileJW, aMeanJW, fluxJW = JW_solver(
                                   H0=args.H0,
                                   Apar=Shared.Apar,
                                   Spar=Shared.Spar,
                                   epsdel=(ModelParameter.epsdel / (ModelParameter.Tliqd0 - args.Tref)),
                                   of=ModelParameter.outfile,   
                                   stps=1000000,  
                                   mode=mode
                                )    
        # Save the J&W4 output:
        fname = ModelParameter.outfile + "/1DMC_run_JW.dat"   
        comb_cols = np.column_stack((tCoolJW, TbulkJW, TroofJW, TliqdJW, TnuclJW, hPileJW, aMeanJW, fluxJW))
        np.savetxt(fname, comb_cols, delimiter="\t", header="Time, Tbulk, Troof, Tliq, Tnuc, h, ad", comments="#")
        
    else:
        print()
        print(" [WARNING] - J&W solver skipped!")
        logger.warning(" WARNING: J&W solver skipped!")
    return

######################################################################################################

if __name__ == "__main__":  
    args = parser.parse_args([] if "__file__" not in globals() else None) 

    if args.srun: 
        # Execute the main unit for an arbitary input, called from the 2DConLat code:
        main_SNGL(args=args, SIGN=True)

    else: 
        # Execute the main unit and run post-processing: 
        main_FULL(args=args, SIGN=True)   

        print()
        print(" POST-PROCESSING:")
        print(" ", "-" * 40) 
        try:
            print(f"Running ’python3 mPost.py --tar_direct={ModelParameter.outfile} --evol_file=evol_{ModelParameter.outfile}.h5’.")
            subprocess.run(["python3", "mPost.py", f"--tar_direct={ModelParameter.outfile}",
                                                f"--evol_file=evol_{ModelParameter.outfile}.h5"],
                                                check=True)
        except subprocess.CalledProcessError as err:
            print(f"Post-processing terminated with an error!")
            
######################################################################################################
#% end of the module!