# Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 4)
# Module: 1D Heat Equation

# NOTE: Petrological parameters are specified for the pressure of 1000 bar (~ 3 km depth in the crust)!

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from math import floor
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded

#-------- Ttop=const. -----------
#        \
#         \  HOST ROCK 
#          \
#           *----------*
#                      |
#            LIQ. MELT |
#                      |
#            .*.*.*.*.*|
#           *----------*
#            \
#             \  HOST ROCK
#              \
#               \
#-------- Qbot = const. ---------

# Modules:
from mPar import *
from mGL import *
from mFunc import visc0_GIORDANO
from mPlot import plot_1d_grid, plot_1d_zoom_roof, plot_1d_zoom_floor, plot_1d_interior

""" ############################################## """
""" ######   1D HEAT EQUATION - FUNCTIONS ######## """
""" ############################################## """  

def check_energy0D(
    dt:             float,
    Hnow:           float,
    qtop:           float,
    qbot:           float,
    Tbulk_old:      float,
    Tbulk_new:      float,
    zgrid:          np.ndarray,
    lheat:          np.ndarray,
    rhopcp_melt:    np.ndarray,
    const:          Constants1DHE

) -> None:
    """ Checks 0D energy balance in the intrusion region """

    _LHS = Hnow * rhopcp_melt[const._roofidx+1] * (Tbulk_old - Tbulk_new) / dt
    _lheat = np.trapz(lheat, zgrid)
    _RHS = - (abs(qtop) + abs(qbot)) + _lheat 

    #print(f"RHS: {_RHS:.3e} W/m2 | LHS: {_LHS:.3e} W/m2.")

    return (_LHS, _RHS)

@njit
def compute_heat_flux(T: np.ndarray, z: np.ndarray, k: np.ndarray, const: Constants1DHE) -> np.ndarray:
    """
        Heat flux at all faces [W/m²].
        At the crust/intrusion interfaces (roof and floor), flux is evaluated
        One-sidedly from the melt side (low-gradient, well-resolved).
    """

    # TODO: opravit, tohle možná nepočítá dobře!? I think it does

    nz = len(T)
    q  = np.zeros(nz - 1)
    for i in range(nz - 1):
        if i == const._roofidx - 1 or i == const._roofidx:
            # Roof interface: evaluate from melt side!
            q[i] = -k[const._roofidx] * (T[const._roofidx + 1] - T[const._roofidx]) / (z[const._roofidx + 1] - z[const._roofidx])
        elif i == const._botmidx or i == const._botmidx + 1:
            # Floor interface: evaluate from melt side!
            q[i] = -k[const._botmidx] * (T[const._botmidx] - T[const._botmidx - 1]) / (z[const._botmidx] - z[const._botmidx - 1])
        else:
            dz    = z[i + 1] - z[i]
            dT    = T[i + 1] - T[i]
            k_eff = 2. * k[i] * k[i + 1] / (k[i] + k[i + 1])
            q[i]  = -k_eff * (dT / dz) # heat flux at the face between nodes i and i-1

    #print()
    #print(" Roof heat flux: ", q[const._roofidx])
    #print(" Floor heat flux: ", -q[const._botmidx])
    #print()

    return q

def compute_max_stable_dt(dz_array: np.ndarray, k: np.ndarray, rhocp: np.ndarray, dt: float=1.e9, fac: float=0.9):
    """ CFL criterion """
    dt_list = 0.45 * (np.power(dz_array, 2)) * rhocp / k
    if np.min(dt_list) < dt:
        return fac * np.min(dt_list)    
    else:
        return dt

def find_idxs(z: np.ndarray, const: Constants1DHE) -> tuple[int, int]:
    """ Find the indices of the roof and floor of the magma chamber """
    
    roofidx = np.abs(z - const._zchtop0).argmin() 
    bottidx = np.abs(z - const._zchbot0).argmin()  
    htblidx = np.abs(z - const._zchtbl0).argmin()

    return (roofidx, bottidx, htblidx)

def update_krhocp(z: np.ndarray, const: Constants1DHE) -> tuple[np.ndarray, np.ndarray]:
    """ Update the density/conductivity grid """

    # NOTE: const._zchbot0 should have been already updated in the update_grid1D()
    # NOTE: DEPRECATED, ONLY USED IN THE REFINEMENT SUBROUTINE!

    # Thermal conductivity profile:
    k = np.full(len(z), const._kbg)
    for i in range(len(z)):
        if const._zchtop0 <= z[i] <= const._zchbot0:
            k[i] = const._keff

    # Density & heat capacity:
    rhocp = np.full(len(z), const._rhobg*const._cheat, dtype=np.float64) 
    for i in range(len(z)):
        if const._zchtop0 <= z[i] <= const._zchbot0:
            rhocp[i] = ModelParameter.rhof * ModelParameter.heatcp      

    return (k, rhocp)

def define_source(z: np.ndarray, lheatTBL: float, lheatBLK: float, const: Constants1DHE) -> np.ndarray:
    """ Compute the discretized latent heat source """

    # TODO: eventually, we want interpolation for L(z)!

    zlheat = np.zeros_like(z, dtype=np.float64)
    for i in range(len(z)):
        if const._zchtop0 <= z[i] <= const._zchtbl0:
            zlheat[i] = lheatTBL

        if const._zchtbl0 < z[i] <= const._zchbot0:
            zlheat[i] = lheatBLK

    return zlheat

def build_grid(const: Constants1DHE) -> tuple[np.ndarray, np.ndarray]:
    """ Spatial discretization of the 1D heat equation, build a grid. """

    z1 = np.arange(0.0, const._zchtop0 - const._dfine, const._dz1)
    z2 = np.arange(const._zchtop0 - const._dfine, const._zchbot0 + const._dfine, const._dz2)
    z3 = np.arange(const._zchbot0 + const._dfine, const._Hdomain + const._dz2, const._dz1) # TODO: THIS WAS WRONG? HELLO?
    z = np.concatenate((z1, z2, z3))
    dz_array = np.diff(z)
    dz_array = np.append(dz_array, dz_array[-1])
    
    return (z, dz_array)

def refine_diffusion() -> None:
    """ Solution diffusion after grid refinement (not advancing in time!). """
    return

def modify_dz(dz2_new: float, const: Constants1DHE) -> None:
    """ Modify the discretization step. """

    const._dz2 = dz2_new
    return

def refine_grid(z_old: np.ndarray, T_old: np.ndarray, k_old: np.ndarray, rhocp_old: np.ndarray, const: Constants1DHE, kind="cubic") -> None:
    """ AMR procedure. """

    z_new, dz_arr = build_grid(const) 
    k_new, rhocp_new = update_krhocp(z_new, const)
    if const._theta == 0.0:
        dt1D = compute_max_stable_dt(dz_arr, k_new, rhocp_new)
        print(f"The time step was changed from {Diag.dtCool:.3e} s to {dt1D:.3e} s.") # update the timestep!
        Diag.dtCool = dt1D 

    # Interpolate the temperature:
    interp_T = interp1d(z_old, T_old, kind=kind, fill_value="extrapolate")
    T_new = interp_T(z_new)
    const._roofidx, const._botmidx, const._btblidx = find_idxs(z_new, const)

    return (z_new, T_new)
 
@njit
def explicit_euler(T_old: np.ndarray, lam_m: np.ndarray, lam_p: np.ndarray, s: np.ndarray, dt: float) -> np.ndarray:
    """ Explicit Euler njit-powered. """

    T_new = np.zeros_like(T_old, dtype=np.float64)
    T_new[1:-1] = (T_old[1:-1]
                    + dt * (lam_m * (T_old[:-2] - T_old[1:-1])
                            + lam_p * (T_old[2:] - T_old[1:-1])
                            + s)
                )
    return T_new

def solve_heat_1d_conductive(T: np.ndarray, z: np.ndarray, k: np.ndarray, rhocp: np.ndarray, lheat: np.ndarray,
                             dt: float, stps: int = 1, theta: float = 1.0, q_bottom: float=0.06) -> np.ndarray:
    """
        Finite volume 1D scheme (Tackley & Zadeh, 2010) energy-conserving approach.
            theta = 0.0 : explicit Euler (original behaviour, CFL-limited)
            theta = 0.5 : Crank-Nicolson (2nd order, unconditionally stable)
            theta = 1.0 : fully implicit (1st order, unconditionally stable) 

        Hybrid scheme: fully implicit in temperature and explicit in latent heat
    """

    N = len(z)
    T_new = T.copy()
    for _ in range(stps):
        T_old = T_new.copy()

        # Pre-compute face conductivities (harmonic mean, length N-1):
        k_face = 2. * k[:-1] * k[1:] / (k[:-1] + k[1:])

        # Pre-compute geometry and lambda coefficients (interior nodes): 
        dz_fwd    = z[2:]   - z[1:-1]                             # dz_p, length N-2
        dz_bwd    = z[1:-1] - z[:-2]                              # dz_m, length N-2
        dz_center = (1./2.) * (dz_fwd + dz_bwd)                   # dz_i, length N-2

        lam_m = k_face[:-1] / (dz_bwd * dz_center * rhocp[1:-1])  # lmd_m
        lam_p = k_face[1:]  / (dz_fwd * dz_center * rhocp[1:-1])  # lmd_p
        s     = lheat[1:-1] / rhocp[1:-1]                         # latent heat source

        #print(s, np.sum(s))

        if theta == 0.0: 
            # Explicit Euler:
            T_new = explicit_euler(T_old=T_old, lam_m=lam_m, lam_p=lam_p, s=s, dt=Diag.dtCool)

        else:           
            # Theta method, (semi-)implicit scheme:
            theta_e = 1.0 - theta  

            # Banded storage for scipy.linalg.solve_banded (shape 3 x N)
            ab = np.zeros((3, N))
            ab[0,2:]   = -theta * dt * lam_p                    # super-diagonal
            ab[1,1:-1] =  1.0 + theta * dt * (lam_m + lam_p)    # main diagonal
            ab[2,:-2]  = -theta * dt * lam_m                    # sub-diagonal

            # Dirichlet BC rows (top): 
            ab[1,0]  = 1.0
            #ab[1,-1] = 1.0

            # Neumann condition at the bottom of the domain:
            ab[1, -1]   =  1.0
            ab[2, -2]   = -1.0 # sub-diagonal of last row

            # Explicit part of diffusion + source:
            rhs = T_old.copy()
            rhs[1:-1] += (theta_e * dt * (lam_m * (T_old[:-2] - T_old[1:-1])
                                        + lam_p * (T_old[2:]  - T_old[1:-1]))
                          + dt * s)

            # Dirichlet & Neumann BCs on RHS:
            rhs[0]  = T[0]
            rhs[-1] = q_bottom * (z[-1] - z[-2]) / k[-1] # q_bottom: constant crustal flux: 60 mW/m2!

            T_new = solve_banded((1, 1), ab, rhs)

    return T_new

def evolution1D_nosource(
        T:         np.ndarray, 
        k:         np.ndarray,
        rhocp:     np.ndarray,
        z:         np.ndarray,
        dt:        float,
        Tliqd:     float,
        max_steps: int,
        const:     Constants1DHE,
        libtqdm:   bool=True,
        printstp:  int=1
    
    ) -> tuple[np.ndarray, float, int, float]:
    """ Initial phase of the 1D heat equation (superheated magma after emplacement, no crystallization) """

    with open(ModelParameter.outfile + "/1DHE_flux.dat", "w") as f:
        time = 0.0; liq_hit = None; qhit = None; _Hnow = abs(const._zchtop0 - const._zchbot0)
        roofidx, bottidx, _ = find_idxs(z, const)
        const._roofidx = roofidx  # save the roof index!
        const._botmidx = bottidx  # save the floor index!
        const._botmidx0 = bottidx

        # TODO: možná rovnou tohle udělej v init_setup1D???
        if roofidx is None: raise Exception("Roof index invalid!")
        if bottidx is None: raise Exception("Floor index invalid!")

        zlheat = np.zeros(len(z), dtype=np.float64)
        Tmean = T[roofidx + 1] # initial emplacement temperature

        # INITIAL GUESS:
        initial_run = True
        estNu = 1000.
        const._zchtbl0 = const._zchtop0 + _Hnow / estNu
        _, _, const._btblidx = find_idxs(z, const)
        k[const._roofidx:const._btblidx] = ModelParameter.kappa * ModelParameter.heatcp * ModelParameter.rhof # rewrite conductivity!

        for step in range(max_steps):
            print(f"\r Time evolution: {abs(1.e2 - 1.e2*(Tmean - Tliqd)/(const._Temp - Tliqd)):.2f} [%] (initial cooling). ", end="", flush=True)
            #Told = np.mean(T[(z >= const._zchtbl0) & (z <= const._zchbot0)])
            Told = np.mean(T[(z >= const._zchtop0) & (z <= const._zchbot0)])

            # Initial iterator:
            """
            print()
            if initial_run:
                Nuold = estNu
                for _itr in range(500):
                    T = solve_heat_1d_conductive(T=T, z=z, k=k, rhocp=rhocp, lheat=zlheat, dt=dt, theta=const._theta)
                    Tmean = np.mean(T[(z >= const._zchtbl0) & (z <= const._zchbot0)]) 
                    q_temp = compute_heat_flux(T, z, k, const)
                    qroof = float(q_temp[roofidx])

                    # Assume melt is isoviscous everywhere:
                    Pr = (visc0_GIORDANO(T=Tmean, Ea=ModelParameter.EaMod) / ModelParameter.rhof) / ModelParameter.kappa
                    conv_state = start_iterative_Ra_F(flux=abs(qroof), Pr=Pr, H=_Hnow, c=ModelParameter, Nu_init=estNu)
                    _Ra, estNu, _Re, dT = conv_state["Ra"], conv_state["Nu"], conv_state["Re"], conv_state["dT"]
                    htbl = _Hnow / estNu

                    # Correction:
                    const._zchtbl0 = const._zchtop0 + htbl
                    _, _, const._btblidx = find_idxs(z, const)
                    k[const._roofidx:const._btblidx] = ModelParameter.kappa * ModelParameter.heatcp * ModelParameter.rhof # rewrite conductivity!
                    k[(const._btblidx + 1):const._botmidx0] = const._keff
                    if _itr % 50 == 0:
                        print(f"Iterator step: {_itr:d} | |Nu| = {abs(Nuold - estNu) / estNu:.3e} | {const._btblidx:.5f} | {const._zchtbl0}.")
                    Nuold = estNu

                # Initial iteration:
                initial_run = False
                print("Initial (Nu - Ra - hb - k) non-uniqueness resolved!")

            else:
                T = solve_heat_1d_conductive(T=T, z=z, k=k, rhocp=rhocp, lheat=zlheat, dt=dt, theta=const._theta)
            """
            
            T = solve_heat_1d_conductive(T=T, z=z, k=k, rhocp=rhocp, lheat=zlheat, dt=dt, theta=const._theta)
            time += dt

            # Calculate the mean temperature in the chamber and heat fluxes in the domain:
            #Tmean = np.mean(T[(z >= const._zchtbl0) & (z <= const._zchbot0)]) # NOTE: TESTING
            Tmean = np.mean(T[(z >= const._zchtop0) & (z <= const._zchbot0)]) # NOTE: TESTING
            q_temp = compute_heat_flux(T, z, k, const)

            # Trace heat fluxes in the vicinity of the chamber:
            qroof = float(q_temp[roofidx]); qplus = float(q_temp[roofidx+1]); qminus = float(q_temp[roofidx-1]) 
            qfloor = float(q_temp[bottidx-1])
            zroof = float(z[roofidx])

            print("TOKY:", qroof, qplus, qminus)

            # Parameterization of convection within the chamber:
            Pr = (visc0_GIORDANO(T=Tmean, Ea=ModelParameter.EaMod) / ModelParameter.rhof) / ModelParameter.kappa
            conv_state = start_iterative_Ra_F(flux=abs(qroof), Pr=Pr, H=_Hnow, c=ModelParameter)
            _Ra, _Nu, _Re, dT = conv_state["Ra"], conv_state["Nu"], conv_state["Re"], conv_state["dT"]
            htbl = _Hnow / _Nu

            # Update the position of the thermal boundary layer:
            const._zchtbl0 = const._zchtop0 + htbl
            _, _, const._btblidx = find_idxs(z, const)

            # Tempatures:
            #print(f" TBL region: {dT:.6f} | {T[const._roofidx - 10]:.6f} | {T[const._btblidx]:.6f}.")
            #print(f" BULK region: start - {T[const._btblidx + 1]:.6f} | end - {T[const._botmidx]}.")

            """
            if step % 50 == 0:
                plot_1d_zoom_roof(step=step, T=T, z=z, const=const)
                plot_1d_zoom_floor(step=step, T=T, z=z, const=const)
                plot_1d_interior(step=step, T=T, z=z, const=const)
            """
                
            plt.close("all") # sanity check!

            # Check 0D-energy balance:
            if step % 500 == 0:
                _ , _ = check_energy0D(dt=dt, Hnow=abs(z[const._roofidx] - z[const._botmidx]), qtop=qroof, qbot=qfloor, Tbulk_old=Told,
                            Tbulk_new=Tmean, zgrid=z, lheat=zlheat, rhopcp_melt=rhocp, const=Constants1D
                            )

            if step % printstp == 0: f.write(f"{dt*step:.10e}\t{zroof:.10e}\t{qroof:.10e}\t{qplus:.10e}\t{qminus:.10e}\n") 
            
            # Did we hit liquidus?
            if Tmean <= Tliqd and liq_hit is None:
                liq_hit = True
                time_hit = step * dt
                step_hit = step   
                qhit = q_temp[roofidx]
                break
    f.close()             
    if qhit is None:
        print(" > Liquidus was not hit (more time required), terminating.")
        exit()
    print()
    print(f" > Heat flux upon hitting liquidus: {float(qhit):.3f} [W/m2].")

    # Report time:
    if time_hit is None: raise Exception("More time required to hit liquidus!")
    years = time_hit / 86400. / 365.
    print(f" > Time for magma to cool from {const._Temp:.2f} K to {Tliqd:.2f} K: {time_hit:.2e} s (~{years:.2f} years).")

    return (T, time_hit, step_hit, qhit)

def step1D_with_latent_heat(
    step:      int,             # step of the evolution
    T:         np.ndarray,      # current temperature field
    L:         np.ndarray,      # current vertical latent heat source 
    k:         np.ndarray,      # current thermal conductivity field 
    rhocp:     np.ndarray,      # current density-heat capacity product field
    z:         np.ndarray,      # discretization
    dt:        float,           # time step size
    Tbulk_old: float,           # previous mean temperature
    const:     Constants1DHE,   # 1DHE constants

) -> tuple[np.ndarray, float, float]:

    # Plot the zoomed profiles:
    if step % Attributes.printstep == 0:
        # BEFORE
        plot_1d_zoom_roof(step=step, T=T, z=z, const=const)
        plot_1d_zoom_floor(step=step, T=T, z=z, const=const)
        plot_1d_interior(step=step, T=T, z=z, const=const)

    Tnew = solve_heat_1d_conductive(T=T, z=z, k=k, rhocp=rhocp, lheat=L, dt=dt, theta=const._theta)
    #print(const._zchtop0, const._zchtbl0, const._zchbot0)

    # Determine the new bulk temperature as the mean temperature within the bulk:
    #Tmean = np.mean(Tnew[(z >= const._zchtbl0) & (z <= const._zchbot0)]) # NOTE: Tmean should be computed without the TBL!
    Tmean = np.mean(Tnew[(z >= const._zchtop0) & (z <= const._zchbot0)]) # NOTE: Tmean should be computed without the TBL!

    q_temp = compute_heat_flux(Tnew, z, k, const)
    qroof  = float(q_temp[const._roofidx])   
    qfloor = float(q_temp[const._botmidx-1])

    #print(f"FLUX THROUGH FLOOR: {float(q_temp[const._botmidx]):.2e} | {float(q_temp[const._botmidx-1]):.2e} | {float(q_temp[const._botmidx+1]):.2e} \
    #       \ FLUX THROUGH ROOF: {float(q_temp[const._roofidx]):.2e} | {float(q_temp[const._roofidx-1]):.2e} | {float(q_temp[const._roofidx+1]):.2e} "
    #      )

    # Check 0D-energy balance:  
    #print(z[const._roofidx], z[const._botmidx])
    (_lhs, _rhs) = check_energy0D(dt=dt, Hnow=abs(z[const._roofidx] - z[const._botmidx]), qtop=qroof, qbot=qfloor, Tbulk_old=Tbulk_old,
                   Tbulk_new=Tmean, zgrid=z, lheat=L, rhopcp_melt=rhocp, const=Constants1D
                )
    
    # Plot the zoomed profiles:
    if step % Attributes.printstep == 0:
        # AFTER
        plot_1d_zoom_roof(step=step+1, T=Tnew, z=z, const=const)
        plot_1d_zoom_floor(step=step+1, T=Tnew, z=z, const=const)
        plot_1d_interior(step=step+1, T=Tnew, z=z, const=const)

    return (Tnew, Tmean, abs(qroof), abs(qfloor), _lhs, _rhs)

""" ############################################## """
""" ######   1D HEAT EQUATION - FUNCTIONS ######## """
""" ############################################## """  

def init_setup1D(H0:    float,        # initial height
                 suc:   float,        # initial supercooling
                 const: Constants1DHE # constants 1DHE
) -> tuple:
    """ Initial setup of the computational domain for the 1D heat equation. """
    
    if getattr(init_setup1D, "_has_run", False):
        return

    print(f" [WARNING] - Initializing the 1D heat equation, the args.hflux will be recalculated!")

    # Adjust the size of the convective domain:
    const._zchbot0 = const._zchtop0 + H0
    const._zchbotINT = const._zchtop0 + H0
    if const._zchtop0 > const._Hdomain:
        raise ValueError("[WARNING] - You are outside the computational domain!")

    # Prepare the computational grid:
    z, dz_array = build_grid(const)

    # Initial temperature profile (linear geothermal gradient):
    T0 = const._Ttop + (const._Tbot - const._Ttop) * z / const._Hdomain # TODO: investigate if it is OK or not!

    # Magma emplacement temperature (superheated):
    const._Temp = ModelParameter.Tliqd0 + suc
    for i in range(len(z)):
        if const._zchtop0 <= z[i] <= const._zchbot0:
            T0[i] = const._Temp

    # Thermal conductivity profile:
    k = np.full(len(z), const._kbg) 
    for i in range(len(z)):
        if const._zchtop0 <= z[i] <= const._zchbot0:
            k[i] = const._keff

    # Density & heat capacity:
    rhocp = np.full(len(z), const._rhobg*const._cheat, dtype=np.float64) 
    for i in range(len(z)):
        if const._zchtop0 <= z[i] <= const._zchbot0:
            rhocp[i] = ModelParameter.rhof * ModelParameter.heatcp      

    # Calculate a safe time step:
    if const._theta >= 0.5: dt1D = (H0 / 2.)**2 / (const._keff / ( ModelParameter.rhof * ModelParameter.heatcp )) / 1.e3
    else: dt1D = compute_max_stable_dt(dz_array, k, rhocp)
    print(f" The initial time step: {dt1D:.2e}{Units.tunit}.")

    # Plot the grid:
    #plot_1d_grid(z=z)
    #print(f" Grid plotted and saved.")

    #print("DIMENSIONS:")
    #print("z:", len(z))
    #print("T:", len(T0))
    #print("k:", len(k))
    #print("rho,cp:", len(rhocp))

    init_setup1D._has_run = True
    return (z, T0, k, rhocp, dt1D)

def update_grid1D(T: np.ndarray, z_old: np.ndarray, k_old: np.ndarray, rhocp_old: np.ndarray, dh: float, htbl: float, const: Constants1DHE,
                  INSULATION: int=1) \
-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Add a new sediment layer and rewrite conductivity nodes. """

    if not hasattr(update_grid1D, "state"): update_grid1D.state = 0.0

    # Number of nodes to rewrite!
    rwnds = int(dh // const._dz2)
    if rwnds == 0:
        # First check the remainder and add it to the current state:
        update_grid1D.state += dh
        if update_grid1D.state >= const._dz2:
            rwnds = 1;  update_grid1D.state = (update_grid1D.state - const._dz2)
    else: 
        update_grid1D.state += (dh % const._dz2)
        if (update_grid1D.state > const._dz2): 
            rwnds += 1; update_grid1D.state = (update_grid1D.state - const._dz2)

    # Update density & heat capacity in the sediment: 
    k_old[(const._botmidx - rwnds):(const._botmidx + 1)] = const._kbg 
    
    # TODO: big question! this is energetically dangerous!
    #rhocp_old[(const._botmidx - rwnds):(const._botmidx + 1)] = const._rhobg * const._cheat

    # Update the position of the bottom of the chamber and the TBL:
    #if rwnds > 0: 
    const._zchbot0 -= dh # NOTE: i.e., we actually want to change the nodes!
    const._zchtbl0  = const._zchtop0 + htbl

    # Update the indices of the bottom of the chamber and the TBL:
    _, const._botmidx, const._btblidx = find_idxs(z_old, const)

    match INSULATION:
        case 0:
            pass
        case 1: 
            Tmean = np.mean(T[(z_old >= const._zchtop0) & (z_old <= const._zchbot0)]) 
            #T[(z_old >= const._zchtop0) & (z_old <= const._zchbot0)] = Tmean # NOTE: this is nonsense, you would change the heat fluxes, doesnt make sense
            T[(const._botmidx - rwnds):(const._botmidx + 1)] = Tmean

    z_new = z_old
    k_new = k_old
    rhocp_new = rhocp_old

    return (z_new, k_new, rhocp_new)

def save_solution(T: np.ndarray, z: np.ndarray, k: np.ndarray, rhocp: np.ndarray, step: int) -> None:
    """ Saving the current T(z,t) solution in a data format. """

    path = ModelParameter.outfile + f"/{ModelParameter.IDHdata}/1D_{step}.dat" 
    np.savetxt(path, np.column_stack((T, z, k, rhocp)), fmt="%.8e")
    return

######################################################################################################
#% end of the module!