# Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 4)
# Module: Utilities

import os
import shutil
import glob
import argparse
import numpy as np
from datetime import datetime
from itertools import chain

from mPar import *

def assemble_output_file_name(args: argparse.Namespace):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    timestamp = "".join([timestamp, "_"])
    if args.JWSOLVER: timestamp = "JWSOLVER_"
    rename = True
    drv = ""
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

    return

######################################################################################################
#% end of the module!