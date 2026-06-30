# Program ParMaCh - 1D Model of Solidification of Magma Chambers
# Module: Benchmark model by Jarvis and Woods (1994), analytical solution for crystal settling

# Standard libraries: 
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.special import gamma

# Modules:
from mPar import * 

# Get logger from main.py:
logger = logging.getLogger(__name__)

def JW_solver(H0: float, Apar: float, Spar: float, epsdel: float, stps: int, of: str, Teutec: float = 0.2,
              Talpha: float = 2.0, mode: int = 1

              ) -> tuple[float, ...]:

    from mFunc import ndimless, pow_grow_dimless

    """ 
        ###! DIMENSIONLESS SOLVER OF THE JARVIS & WOODS MODEL !###
        > TODO
    """

    assert mode in (1, 2) and Teutec > 0.0

    tCool, hPile, amean, Troof, Tliqd, rhs, dTlq, Tliqd, Tbulk, Tnucl, flux, nuvisc = [
        np.zeros(stps+1, dtype=float) for _ in range(12)]
    df, dl = [np.zeros(stps+1, dtype=float) for _ in range(2)]
    dtCool = 1.0  # Diag.dtCool / Shared.tc0
    dtCool /= 10000.

    Shared.idxend = stps

    match mode:
        case 1:  # % Constant roof temperature:

            # Initial conditions:
            Troof[0] = 0.0
            Tliqd[0] = 1.0
            Tbulk[0] = 1.0
            flux[0] = 1.0
            nuvisc[0] = 1.0
            Tnucl[0] = Tliqd[0] - epsdel
            assert Tliqd[0] < Talpha

            for i in range(stps):
                tCool[i+1] = (i+1) * dtCool

                # Growth x nucleation product and crystal production:
                nv = ndimless(chi=nuvisc[i],
                              Tbulk=Tbulk[i],
                              Troof=Troof[i],
                              Tliqd=Tliqd[i],
                              dpile=hPile[i],
                              epsdel=epsdel) * pow_grow_dimless(u=(Tliqd[i] - Tbulk[i]))

                rp = nuvisc[i] * (1. - hPile[i]) * nv

                # Calculate the sediment height increment and mean radius (gamma(4/3) = 0.89):
                hPile[i+1] = hPile[i] + dtCool * Apar * (1 - hPile[i]) * rp
                amean[i+1] = gamma(4./3.) * (3. * H0)**(1./3.) \
                    * (nuvisc[i] * (1. - hPile[i]) * pow_grow_dimless(Tliqd[i] - Tbulk[i]))**(1./3.)

                # Increments/decrements:
                rhs[i] = (-flux[i] / (1. - hPile[i]) + Apar * Spar * rp)
                dTlq[i] = Apar * (Talpha - Tliqd[i]) * rp

                # Update temperature variables:
                Tliqd[i+1] = Tliqd[i] - dtCool * dTlq[i]
                Tbulk[i+1] = Tbulk[i] + dtCool * rhs[i]
                Tnucl[i+1] = Tliqd[i+1] - epsdel
                Troof[i+1] = Troof[0]

                # Update the heatflux and viscosity:
                nuvisc[i+1] = nuvisc[0]
                flux[i+1] = nuvisc[i+1]**(-1./3.) * \
                    (Tbulk[i+1] - Troof[i+1])**(4./3.)

        case 2:  # % Constant/Varying heat flux:

            epsdel = 0.1

            # Initial conditions:
            nuvisc[0] = 1.0
            flux[0] = 0.2
            Tliqd[0] = 1.0
            Tbulk[0] = 1.0
            Troof[0] = Tbulk[0] - (flux[0] / nuvisc[0])**(3./4.)
            Tnucl[0] = Tliqd[0] - epsdel
            assert Tliqd[0] < Talpha

            for i in range(stps):
                tCool[i+1] = (i+1) * dtCool

                # if i % 1000 == 0:
                #    print("d:", abs(Tliqd[i] - Tbulk[i]))

                # Growth x nucleation product and crystal production:
                nv = ndimless(chi=nuvisc[i],
                              Tbulk=Tbulk[i],
                              Troof=Troof[i],
                              Tliqd=Tliqd[i],
                              dpile=hPile[i],
                              epsdel=epsdel) * pow_grow_dimless(u=(Tliqd[i] - Tbulk[i])
                                                                )
                rp = nuvisc[i] * nv * (1. - hPile[i])

                # Calculate the sediment height increment and mean radius (gamma(4/3) = 0.89):
                hPile[i+1] = hPile[i] + dtCool * Apar * (1. - hPile[i]) * rp
                amean[i+1] = gamma(4./3.) * (3. * H0)**(1./3.) \
                    * (nuvisc[i] * (1. - hPile[i]) * pow_grow_dimless(Tliqd[i] - Tbulk[i]))**(1./3.)

                # Update temperature variables:
                Tliqd[i+1] = Tliqd[i] - dtCool * \
                    Apar * (Talpha - Tliqd[i]) * rp
                Tnucl[i+1] = Tliqd[i+1] - epsdel
                Tbulk[i+1] = Tbulk[i] + dtCool * \
                    (-flux[i] / (1. - hPile[i]) + Apar * Spar * rp)
                #print(f"Ratio {(flux[i] / (1. - hPile[i]))/(Apar * Spar * rp)}.")

                df[i] = -flux[i] / (1. - hPile[i])
                dl[i] = Apar * Spar * rp

                # Update the viscosity, the heat flux and the roof temperature:
                # nuvisc[i+1] = nuvisc[0]*exp(-0.0058*tCool[i]) #nuvisc[0]
                nuvisc[i+1] = nuvisc[0]
                #nuvisc[i+1] = 100**(1.-Tbulk[i+1])
                flux[i+1] = flux[0] * (1. + tCool[i+1] / 1.5)**(-3./2.)
                #flux[i+1] = flux[0]
                Troof[i+1] = Tbulk[i+1] - (flux[i+1] / nuvisc[i+1])**(3./4.)

                if Tliqd[i+1] <= Teutec:
                    Shared.idxend = i
                    print("Eutectic temperature reached.")
                    logger.info(f"[Jarvis & Woods Solver]: Reached the eutectic temperature, time index: {i:d}!")
                    break

    fig, axjcm = plt.subplots()
    axjcm.plot(tCool[:Shared.idxend], df[:Shared.idxend], label="flux")
    axjcm.plot(tCool[:Shared.idxend], dl[:Shared.idxend], label="lheat")
    axjcm.legend()
    fig.savefig(of + "/plot/individual/jwrhs.pdf")

    fig, axjcm = plt.subplots()
    axjcm.plot(tCool[:Shared.idxend], Tbulk[:Shared.idxend], label="tb")
    axjcm.plot(tCool[:Shared.idxend], Tliqd[:Shared.idxend], label="tl")
    axjcm.plot(tCool[:Shared.idxend],
               Tliqd[:Shared.idxend] - epsdel, label="tn")
    axjcm.plot(tCool[:Shared.idxend], Troof[:Shared.idxend], label="tr")
    #axjcm.plot(tCool[:Shared.idxend], flux[:Shared.idxend], label="flux")
    axjcm.legend()
    fig.savefig(of + "/plot/individual/jwtemp.pdf")

    fig, axjcm = plt.subplots()
    axjcm.plot(amean[:Shared.idxend], hPile[:Shared.idxend], label="amean")
    axjcm.legend()
    fig.savefig(of + "/plot/individual/jwamean.pdf")

    fig, axjcm = plt.subplots()
    axjcm.plot(tCool[:Shared.idxend], flux[:Shared.idxend], label="flux")
    axjcm.legend()
    fig.savefig(of + "/plot/individual/jwflux.pdf")

    np.savetxt("jwflux.dat", (tCool[:Shared.idxend], flux[:Shared.idxend]))

    return (tCool, Tliqd, Tbulk, Tnucl, Troof, hPile, amean, flux)

#####################################################################################################################
# % end of the module!