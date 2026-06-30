# Program: ParMaCh - 1D Model of Solidification of Magma Chambers (version 3)
# Module: Grossmann-Lohse scaling theory of Rayleigh-Benárd convection and associated functions

from numba import njit
from scipy.special import gamma

# Modules:
from mPar import *

@njit
def regimeGL(Ra: float, Pr: float, mt: bool=False) -> str:
    """ Determines GL regime based on the provided Rayeleigh and Prandtl numbers """

    # Recast Malkus theory 1954 (following Jarvis & Woods):
    if mt: regime = "IVu"; return regime

    # Transitional boundaries:
    Pr_Iu_IIIu = 5.7e-33 * Ra**3.
    Pr_IIIu_IVu = 4.8e-8 * Ra**(2./3.)

    # Grossmann-Lohse theory:
    if Pr > Pr_Iu_IIIu:
        regime = "Iu"

    elif Pr > Pr_IIIu_IVu:
        regime = "IIIu"

    else:
        regime = "IVu"

    return regime

@njit
def calculate_rayleigh_number(Hnow : float, Tbulk : float, Troof : float, nu : float, c: Parameters):
    """ Computes the Rayleigh number based on the height of the chamber, temperature contrast, and fluid parameters """
    return ((Hnow**3.) * (Tbulk - Troof) * c.gacc * c.alpha) / (nu * c.kappa)    

@njit
def calculate_reynolds_number(Ra: float, Pr: float, regime: str) -> float:
    """ Calculates the Reynolds number from the GL scaling """

    if   regime == "IVu":
        return 0.16 * Ra**(4./9.) * Pr**(-2./3.)
    
    elif regime == "IIIu":
        return 6.46e-3 * Ra**(4./7.) * Pr**(-6./7.)
    
    else: # regime I_u
        return 0.039 * Ra**(1./2.) * Pr**(-5./6.)

@njit
def calculate_nusselt_number(Ra: float, Pr: float, regime: str) -> float:
    """ Calculates the Nusselt number from the GL scaling """

    if   regime == "IVu":
        return 0.038 * Ra**(1./3.) 
    
    elif regime == "IIIu":
        return 3.43e-3 * Ra**(3./7.) * Pr**(-1./7.)
    
    else: # regime I_u
        return 0.33 * Ra**(1./4.) * Pr**(-1./12.)
    
@njit
def calculate_prandtl_number(nu: float, kappa: float) -> float:
    return nu / kappa

@njit
def calculate_tbl_thickness(H: float, Nu: float) -> float:
    """ Calculates the uppper thermal boundary layer thickness """

    return H / (1.0 * Nu)  # NOTE: the factor one is here to highlight the existence of a single TBL in our system!

@njit
def factorJ0(nu: float, H: float, c: Parameters, regime: str) -> float:
    """ Auxiliary material parameter """

    if   regime == "IVu":
        return 0.038 * ((c.gacc * c.alpha * (c.kappa**2)) / nu)**(1./3.) 
    
    elif regime == "IIIu":
        return 3.43e-3 * (c.gacc * c.alpha)**(3./7.) * c.kappa**(5./7.) * (nu**(-4./7.)) * H**(2./7.)
    
    else: # regime I_u
        return 0.33 * ((c.gacc * c.alpha)**(1./4.) * H**(-1./4.) * nu**(-1./3.) * c.kappa**(5./6.))

@njit
def calculate_deltaT(flux: float, nu: float, H: float, regime: str, c: Parameters) -> float:
    """ Calculates the temperature contrast driving thermal convection """

    if   regime == "IVu":
        return (flux / (c.rhof * c.heatcp * factorJ0(nu, H, c, regime)))**(3./4.)
    
    elif regime == "IIIu":
        return (flux / (c.rhof * c.heatcp * factorJ0(nu, H, c, regime)))**(7./10.)
    
    else: # regime I_u
        return (flux / (c.rhof * c.heatcp * factorJ0(nu, H, c, regime)))**(4./5.)


def calculate_threshold_GBIII(deltaT: float, Hnow: float, nu: float, c: Parameters, bIII: float=6.46e-3):
    """ Calculate the dust-/stone-like deviation threshold. """
    prefIII = (gamma(2./3.) / (gamma(1./3.))) * (bIII**(3./2.) / 3.) * (9. / 2.)**(1./2.)

    return prefIII * (c.alpha * c.gacc * deltaT)**(6./7.) * c.kappa**(3./7.) * Hnow**(1./14.) * nu**(-1./7.) * (c.rhof / c.gacc / (c.rhoc - c.rhof))**(1./2.)


def start_iterative_Ra_F(flux: float, Pr: float, H: float, c: Parameters, Nu_init: float=1.0, tol: float=1.e-6, max_iter: int=500) -> dict:
    """ Given the initial Prandtl number and heat flux, iterative solve for Ra, Nu, Re, and deltaT using Grossmann-Lohse theory """

    # NOTE: start_iterative_Ra_F(...) takes the flux in W/m2!

    Nu = Nu_init # conduction (initial guess) or supplied initial guess in the 1D heat equation!
    Ra = 0.0 # no flow
    k = c.kappa * c.rhof * c.heatcp
    for _ in range(max_iter):
        deltaT = flux * H / (k * Nu)
        Ra = (deltaT * c.gacc * c.alpha * H**3) / (Pr * c.kappa**2)
        regime = regimeGL(Ra, Pr)
        Nu_new = calculate_nusselt_number(Ra, Pr, regime)
        Re_new = calculate_reynolds_number(Ra, Pr, regime)

        # Loop until the implicit equation is solved with desired accuracy:
        if abs(Nu_new - Nu) / (Nu_new) < tol: break
        Nu = Nu_new

    else:
        raise RuntimeError("Grossmann-Lohse iterations did not converge!")

    deltaT_final = (flux * H) / (k * Nu_new)
    return {
        "Ra": Ra,
        "Nu": Nu_new,
        "Re": Re_new,
        "Pr": Pr, 
        "dT": deltaT_final,
        "Regime": regime
    }

######################################################################################################
#% end of the module!