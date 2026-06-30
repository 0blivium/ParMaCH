# Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 3)
# Module: Parameters and constants

import numpy as np
from numba.experimental import jitclass
from numba import float64, float32, int32, int64, int8, types

chamber = """
                              _% ParMaCh Code %_
                            PARameterized model of
                               a MAgma CHamber
    {}
    ################################# TBL ###################################
    #.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#
    #    (                                    *                             #
    #     )       Wrms-->         *                                         #
    #    (                              *       <--Wrms            o        #
    #     )    *               Wrms-->                            (         #
    #    (                                     *            *      )        #
    #     )        *                 *  <--Wrms                   (         #
    #     o                  *                        *            )        #
    #         *                  *       *                        (         #
    #                     <--Wrms       Wrms-->        *           )        #
    # ParMaCh 2026                                                (         #
    #***********************************************************************#
    ##############################  SEDIMENT  ###############################

    {}
    #! Version: PARMACH v4.1
    #! Code: PARMACH - 1D Parameterized Model of a Cooling Magma Chamber
    #! Author: MSci. Vit Beran, Charles University, Prague
    #! Requirements: Python 3.10, h5py, numpy, numba, tqdm, objgraph
    {}
""".format("#"*73, "-"*73, "-"*73)

# Global auxiliary Numba flags (instead of environment value):
NUMBA_BOUNDSCHECK = 0

# Computational model:  TODO: I think that they are njit classes because we need them where exactly?!
@jitclass([
    ("mLin", int8), ("mLab", int8)
])
class nMthd: # 1: pow-law kinetic laws | 2: kinetic laws after Hort (1997)
    def __init__(self):
        self.mLin = 1; self.mLab = 2          

@jitclass([
    ("mDis", int8), ("mTaC", int8), ("mUTaC", int8), ("AnGZ", int8)
])
class sMthd: # 1: method of distributions (MoD) | 2: semi-analytical solution (AnGz) | 3: tracking all crystals (TaC) 
    def __init__(self):
        self.mDis = 1; self.AnGZ = 2; self.mUTaC = 3

@jitclass([
    ("mGen", int8), ("mSbS", int8)
])
class cMthd: # 1: method of generations | 2: step-by-step method
    def __init__(self):
        self.mGen = 1; self.mSbS = 2;

@jitclass([
    ("mNon", int8), ("mShp", int8), ("mVsh", int8)
])
class vStone: # 1: step-by-step method | 2: method of generations
    def __init__(self):
        self.mNon = 1; self.mShp = 2; self.mVsh = 3

const1D_spec = [
    ("_Hdomain", float64), ("_Ttop", float64), ("_Tbot", float64), ("_kappadom", float64),
    ("_kbg", float64), ("_keff", float64), ("_mode", int32), ("_dz1", float64), 
    ("_dz2", float64), ("_zchtop0", float64), ("_zchbot0", float64), ("_Temp", float64),
    ("_rhobg", float64), ("_cheat", float64), ("_dfine", float64), ("_roofidx", int32),
    ("_botmidx", int32), ("_btblidx", int32), ("_theta", float32), ("_zchtbl0", float64),
    ("_zchbotINT", float64), ("_botmidx0", int32)
]
@jitclass(const1D_spec)
class Constants1DHE:
    def __init__(self):
        """ Parameters of the 1D heat equation """
        self._Hdomain   =  7000.   # [m]        height of the domain
        self._Ttop      =  273.    # [K]        top boundary temperature
        self._Tbot      =  500.    # [K]        bottom boundary temperature
        self._kbg       =  2.5     # [W/K/m]    thermal conductivity in the host rock domain
        self._keff      =  2.5e3   # [W/K/m]    modified "effective/convective" thermal conductivity in the convecting fluid 
        self._dz1       =  100.    # [m]        rough discretization 
        self._dz2       =  1.e-3   # [m]        fine discretization - NOTE: mm/cm resolution required!
        self._cheat     =  900.    # [J/kg/K]   heat capacity
        self._rhobg     =  2900.   # [kg/m3]    density of the crust
        self._dfine     =  100.    # [m]        extension of the finely discretized region
        self._mode      =  0       # [-]        bottom boundary condition (0 - constant temperature, 1 - constant heat flux)
        self._zchtop0   =  3000.   # [m]        top of the chamber (petrological parameters specified for the depth of 3 km!)
        self._theta     =  1.0     # [-]        theta-scheme weight 
        self._Temp      =  np.nan  # [K]        temperature of magma after emplacement
        self._zchbot0   =  np.nan  # [m]        bottom of the chamber (+ add args.H0 in §init_setup1D)
        self._zchbotINT =  np.nan  # [m]        initial position of the bottom of the chamber
        self._zchtbl0   =  np.nan  # [m]        the position of the TBL (bottom boundary of it)
        self._roofidx   =  np.nan  # [-]        chamber roof discretization index
        self._botmidx   =  np.nan  # [-]        chamber bottom discretization index
        self._botmidx0  =  np.nan  # [-]        chamber bottom discretization index (initial!)
        self._btblidx   =  np.nan  # [-]        chamber TBL bottom discretization index

par_specs = [
    ("Ntbl", int64), ("rhof", float64), ("rhoc", float64), ("Lheat", float64),
    ("kappa", float64), ("alpha", float64), ("gacc", float64), ("heatcp", float64),
    ("gamma", float64), ("nu0", float64), ("N0POW", float64), ("V0POW", float64),
    ("N0_HG97", float64), ("V0_HG97", float64), ("Ti_HG97", float64), ("Tg_HG97", float64),
    ("epsdel", float64), ("Tsol0", float64), ("Talpha", float64), ("Teutec", float64), ("Tliqd0", float64),
    ("Racrit", float64), ("outfile", types.unicode_type), ("mode", types.unicode_type), ("dstfile", types.unicode_type),
    ("gratio", int64), ("giffile", types.unicode_type), ("Toff", float64), ("epsdelR", float64), ("gasR", float64), 
    ("Xeut", float64), ("Dm1", float64), ("Dm2", float64), ("TmaxAn", float64), ("TmaxDi", float64), ("TeutAD", float64), 
    ("rhocA", float64), ("rhocD", float64), ("M_an", float64), ("M_di", float64), ("Pr0", float64), ("Troof0", float64),
    ("Ea", float64), ("EaMod", float64), ("IDHfile", types.unicode_type), ("IDHdata", types.unicode_type), 
    ("rfzfile", types.unicode_type) 
]
@jitclass(par_specs)
class Parameters:    
    """ Model parameters. 
    #% Data collected from: %#
        # %> crystal kinetics:    from Hort (1997), Couch (2003)
        # %> densities:           from Krattli & Schmidt (2021)
        # %> viscosity:           from Giordano et al. (2008)
        # %> binary diagram:      plagioclase vs. pyroxen (anortit, diopsid) from Courtial et al. 2000O
        
    #% Crystals parameters specified at 1000 bar (3 km depth). 
    """

    def __init__(self):
        # Global parameters of the model:

        self.Ntbl    = 150                            # [-]           number of interfaces in the nucleation layer, number of crystal families
        self.Lheat   = 5.2e5                          # [J/kg]        unmodified entalphy of fusion for anortit (Jarvis & Woods hodnota) 
        self.kappa   = 1.e-6                          # [m^2/s]       thermal diffusivity of the melt
        self.alpha   = 5.e-5                          # [K^-1]        thermal expansivity of the melt
        self.gacc    = 9.81                           # [m/s^2]       gravity acceleration
        self.heatcp  = 1.3e3                          # [J/kg K]      thermal capacity
        self.gamma   = 0.7                            # [-]           mixing coefficient of the unified settling law
        self.epsdel  = 70.0   		                  # [K] 	  	  linear law mathematial nucleation delay
        self.epsdelR = 10                             # [K]           real nucleation delay 
        self.Talpha  = 1473.                          # [K]           point (Ta,Ca) wheres solidus and liquidus intercept
        self.Teutec  = 1203.                          # [K]           eutectic temperature
        self.Toff    = 900.                           # [K]           turn-off temperature 
        self.gasR    = 8.314                          # [J/K/mol]     universal gas constant R           
        self.Racrit  = 660.					          # [-]           critical Rayleigh number        
        self.Tliqd0  = np.nan                         # [K]           liquidus temperature 
        self.Troof0  = np.nan                         # [K]           initial roof temperature
        #self.rhoc    = np.nan                        # [kg/m^3]      density of the solid phase # FIXME
        self.rhoc    = 2700.                          # [kg/m^3]      density of the solid phase
        self.Pr0     = np.nan                         # [-]           initial Prandtl number
        self.nu0     = np.nan				  		  # [m2/s] 		  initial/constant kinematic viscosity 
        self.N0_HG97 = np.nan                         # [#/m3/s]      crystal nucleation amplitude - Hort (1997)
        self.V0_HG97 = np.nan                         # [m/s]         crystal growth amplitude - Hort (1997) 
        self.N0POW   = np.nan                         # [#/m3/s]      crystal nucleation amplitude - power law
        self.V0POW   = np.nan                         # [m/s]         crystal growth amplitude - power law      
        self.Ti_HG97 = 0.92                           # [-]           normalized peak corresponding (nucleation)
        self.Tg_HG97 = 0.95                           # [-]           normalized peak corresponding (growth)         
        self.outfile = ""                             # [-]           output file (log file)
        self.giffile = "dgf"                          # [-]           output file (gif file)
        self.dstfile = "dst"                          # [-]           output file (dist file)
        self.rfzfile = "z1d"                          # [-]           output file (zoom file)
        self.IDHfile = "1D"                           # [-]           output file (1DHE file)
        self.IDHdata = "1Ddata"                       # [-]           output file (1DHE data file)
        self.mode    = ""                             # [-]           roof region mode

        # Binary alloy: virtual diopside & anortite system:
        self.rhof    = 2650.                          # [kg/m^3]      density of the melt/liquid
        self.rhocA   = 2700.                          # [kg/m^3]      density of the solid phase (anortit)
        self.rhocD   = 3250.                          # [kg/m^3]      density of the solid phase (diopsid)
        self.M_an    = 0.27822                        # [kg/mol]      molar mass of anortit
        self.M_di    = 0.21657                        # [kg/mol]      molar mass of diopsid
        self.Dm1     = 5.83e4                         # [J/mol]       modified entalphy of fusion for anorthite # 6.11
        self.Dm2     = 5.88e4                         # [J/mol]       modified entalphy of fusion for diopsid
        self.TmaxAn  = 1458.                          # [K]           maximal liquidus (virtual) temperature for anortit
        self.TmaxDi  = 1302.                          # [K]           maximal liquidus (virtual) temperature for diopsid
        self.TeutAD  = 1203.                          # [K]           eutectic (basaltic) temperature of the An-Di system
        self.Xeut    = 0.413                          # [-]           eutectic composition of the melt
        self.Ea      = 276532.                        # [J/mol]       activation energy
        self.EaMod   = np.nan                         # [J/mol]       modified activation energy

const_specs = [
    ("nbins", int32), ("Racrit", float64), ("HFU", float64), ("ytosec", float64), ("mtomm", float64),
    ("Ti_HG97", float64), ("Tg_HG97", float64), ("Ti_CH03", float64), ("Tg_CH03", float64), ("TINY", float64),
    ("pmaxitr", int32)
]
@jitclass(const_specs)
class Constants: 
    """ Constants that are not expected to change. """ 

    def __init__(self):
        self.nbins   = 100                            # [-]           number of bins for histograms 
        self.Racrit  = 660.                           # [-]           Rayleigh critical number (Schubert & Turcotte, 1991)
        self.HFU     = 41.84e-3                       # [W/m^2]       heat flux unit in Watts!
        self.ytosec  = 31556926.                      # [s]           year to seconds
        self.mtomm   = 1.e3                           # [mm]          meters to milimeters
        self.pmaxitr = 20                             # [-]           the maximum number of Picard iterations
        self.TINY    = 1.e-6                          # [-]           auxiliary numerical constant 
    
class SharedVariables: 
    """ Variables that need to be accessible from anywhere. """

    def __init__(self):
        self.idxend    = None                         # [-]           time index of the end of the simulation
        self.onsetc    = None                         # [-]           time index of the onset of crystallization
        self.convon    = None                         # [-]           convection in the chamber (Ra > Ra_cr)
        self.idxeut    = None                         # [-]           time index of reaching the eutectic temperature
        self.phiTBL    = None                         # [-]           numpy array with CrystalBatch objects falling into the bulk from TBL
        self.track     = None                         # [-]           previous history track array for the mTaC routine
        self.Tref      = None                         # [K]           (benchmarking) the reference temperature for the dimensionless form
        self.Apar      = None                         # [-]           (benchmarking) dimensionless auxiliary parameter A
        self.Spar      = None                         # [-]           (benchmarking) dimensionless auxiliary parameter S
        self.V0ref     = None                         # [-]           (benchmarking) reference crystal growth (w.r.t dT)
        self.N0ref     = None                         # [-]           (benchmarking) reference crystal nucleation (w.r.t. dT)
        self.tliqhit   = None                         # [s]           time of hitting the liquidus temperature
        self.A         = None                         # [-]           1D heat flux fit parameter A
        self.B         = None                         # [-]           1D heat flux fit parameter B
        self.regime    = None                         # [-]           Grossmann-Lohse convection regime
        self.froof     = None                         # [W/m2]        heat flux density at the top of the chamber
        self.htime     = None                         # [s]           heat flux corresponding time marks 
        
        self.prev_aBLK = None                         # [-]           previous step CSD in bulk (radius) # FIXME: tyhle smaž, už je nepotřebujeme
        self.prev_nBLK = None                         # [-]           previous step CSD in bulk (number)
        self.prev_aSED = None                         # [-]           previous step CSD in sediment (radius)
        self.prev_nSED = None                         # [-]           previous step CSD in sediment (number)

        self.distBLK   = None
        self.distSED   = None
        self.distTBL   = None
        self.distTBL2D = None
        self.prate     = None
        self.prateTBL  = None
        self.prateBLK  = None

        self.tdecayold = None                         # [-]           time index of the 1D heat flux fit onset  
        self.onsetfit  = None                         # [-]
        self.error     = None                         # TODO
        self.fall      = None                         # TODO
        self.fdecay    = None                         # TODO
        self.tblstokes = None                         # TODO
        self.dtCool0   = None                         # TODO
        self.a0idle    = None                         # TODO
        self.atbl      = None                         # TODO
        self.ntbl      = None                         # TODO        
        self.nsus      = None                         # TODO  
        self.ntime     = None                         # TODO

class DiagnosticsVariables:
    """ Diagnostic variables that are being tracked """

    def __init__(self):
        self.cnt0DMC   = 0.0                           # [-]           number of crystals nucleated per unit time (ParMaCH)
        self.cntHBJW   = 0.0                           # [-]           number of crystals nucleated per unit time (J&W entire TBL)
        self.cntHNJW   = 0.0                           # [-]           number of crystals nucleated per unit time (J&W nucl. subl.)
        self.amaxtbl   = 0.0                           # [m]           maximum extracted crystal radius from the TBL
        self.amintbl   = 0.0                           # [m]           minimum extracted crystal radius from the TBL (non-zero crystal population)
        self.cin       = 0.0                           # [-]           fall-in crystals
        self.cout      = 0.0                           # [-]           fall-out crystals
        self.tCool     = None                          # [s]           initial convective cooling time scale
        self.dtCool    = None                          # [s]           time step for the cooling and solidification process
        self.a0corr    = None                          # [m]           the initial radius incremented by the correction
        self.atrn      = None                          # [m]           (pointer reference) transitional radius (at given time)
        self.astn      = None                          # [m]           (pointer reference) transitional radius (partially-mixed)
        self.tbres     = 0.0                           # [-]           TBL time scale
        self.tresjw    = 0.0                           # [-]           bulk residence time scale (after JW)     
        self.tssblk    = 0.0                           # [-]           bulk steady state time scale
        self.tplume    = 0.0                           # [s]           cold plume detachment time scale
        self.blkgrow   = 0.0                           # [m/s]         growth in the bulk
        self.tblnucmin = 0.0                           # [/m3/s]       tracked min/max nucleation (TBL)
        self.tblnucmax = 0.0                           # [/m3/s]       tracked min/max nucleation (TBL)
        self.tblgrwmin = 0.0                           # [m/s]         tracked min/max growth (TBL)
        self.tblgrwmax = 0.0                           # [m/s]         tracked min/max growth (TBL)
        self.setmarker = None                          # [-]           settling mode marker
        self.Hmix      = None

class InitVals:
    """ Stored initial values of selected variables. """

    def __init__(self):
        self.nu    = None
        self.flux  = None
        self.Tbulk = None
        self.Troof = None
        self.Tliqd = None
        self.Tnucl = None
        self.XL    = None
        self.Hnow  = None
        self.suc   = None
    
        # Stores the initial state of the 1D heat equation:
        self.z     = None
        self.k     = None
        self.rhocp = None
        self.T     = None
         
class Attributes: # FIXME & TODO: renames, etc.
    """ Model atributes/branching for various scenarios """

    TBL_METHOD    = None  # TBL method 
    NG_METHOD     = None  # nucleation vs. growth method
    SED_METHOD    = None  # Sedimentation method
    SET_METHOD    = None  # ??
    bAnDi         = None  # linear vs. real An-Di system
    DEBUG         = None  # DEBUG?? redundant?
    DEBUGRUN      = None  # Trial debug run
    DEBUG100      = None  # First 100 steps
    DEBUG200      = None  # First 200 steps
    DEBUG_X       = None
    JWLIMIT       = None  
    JWDIMOFF      = None  # dimensionless??
    JWSOLVER      = None  # JW - solver
    JW            = None  # ??
    SM            = None  # ??
    TBL           = None  # TBL limit
    HE            = None
    HE_CONST      = None
    TE            = None  # Time evolution
    nu            = None
    PCITER        = None  # Predictor-corrector iterations 
    NUTBL         = None
    ratio         = None
    MODE          = None
    cMthd         = None
    MT            = None
    tblaoff       = None       
    printstep     = None
    HE_CSTM       = None   
    EA            = 1.0   # activation energy scaling parameter (increase/decrease viscosity)
    SCORR         = None  # idle-mixing correction
    RTIS          = None  # Rayleigh-Taylor instability timescale
    SRUN          = None
    IDMC          = None


class SingleRunAttributes:
    """ Compute one steady-state of the magma reservoir with parameters of your choice """

    # TODO: add loading a json file that would rewrite the initialized variables?

    def __init__(self, deltaT: float, epsd: float):
        self.Tbulk = 1278.
        self.Troof = self.Tbulk - deltaT
        self.Tliqd = 1301.
        self.Tnucl = self.Tliqd - epsd
        self.Hnow  = 1000.
        self.nu    = 1.e-3

        self.physics_check()
        self.nucleation_check() # TODO: do I want it here?

    def physics_check(self):
        if self.Tbulk > self.Tliqd:
           raise ValueError("[SINGLE RUN] - Superheated bulk!") 

    def nucleation_check(self):
        if self.Troof > self.Tnucl:
            print(f"Roof temperature {self.Troof:.2f} vs. nucleation threshold {self.Tnucl:.2f}.")
            raise ValueError("[SINGLE RUN] - No nucleation!")

class AuxUnits:
    """ Auxiliary class for text formatting """    
    
    def __init__(self):
        self.sunit = " [m]"
        self.tunit = " [s]"
        self.vunit = " [m/s]"
        self.Tunit = " [K]"
        self.nunit = " [m2/s]"
        self.funit = " [W/m2]"
        self.runit = " [kg/m3]"
        self.xunit = " [wt% An]"

# Initalization:
ModelParameter = Parameters()
RunConstants = Constants()
nMethod = nMthd()
sMethod = sMthd()
cMethod = cMthd()
mStone  = vStone()
Units   = AuxUnits()
Shared  = SharedVariables()  
Init    = InitVals()
Diag    = DiagnosticsVariables()
Constants1D = Constants1DHE()

# Initialization of the single run:
SingleRun = SingleRunAttributes(deltaT=1.e0, epsd=22.0) # FIXME

######################################################################################################
#% end of the module!