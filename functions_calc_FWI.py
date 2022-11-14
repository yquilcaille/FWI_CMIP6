"""
Functions used for the calculation of the Canadian Fire Weather Index, described in Wang et al, 2015 (https://cfs.nrcan.gc.ca/publications?id=36461).

A large part of the equations are based on the implementation in https://github.com/buckinha/pyfwi/blob/master/pyFWI/FWIFunctions.py
Some minor differences were brought by Yann Quilcaille, not affecting the equations:
 - Lawson equations have been removed, they were not necessary for the calculation of the FWI on CMIP6 data.
 - The code is generalized, now handling not only scalar, but also arrays. This is essential to speed up the computation of large datasets.
 - Handling the issues RH < 0 and WIND < 0
 - Handling exceptions with DMC and DC both 0, causing the BUI to become a non-attributed value. It happens for Rain > 2.8 and Temp in [-2.8, -1.1] and some more complex conditions (Dr <= 0, pr <= 0).
 - Handling exception of FFMC > 101 in ISI (due to the approximation of a coefficient in FFMC: 101 * 147.2 / 59.5 ~=250 )
 
The initial equations have also been edited in the following ways by Yann Quilcaille:
 - The DMC uses an estimation of the day length ("DayLength"). Three methods are implemented:
     - In the original equations, it is only for Canada, and depends on the month.
     - In https://github.com/buckinha/pyfwi/blob/master/pyFWI/FWIFunctions.py, it depends on 4 bins of latitudes and the month
     - In https://github.com/NCAR/fire-indices, it depends continuously on latitude and the day of the year using this function: https://www.ncl.ucar.edu/Document/Functions/Crop/daylight_fao56.shtml
 - The DC uses an estimation of the drying factor ("DryingFactor"). Three methods are implemented:
     - In the original equations, values are targeted on Canada.
     - In https://github.com/buckinha/pyfwi/blob/master/pyFWI/FWIFunctions.py, the values for the Southern hemisphere are those for the northern shifted by 6 months.
     - In https://rdrr.io/rforge/cffdrs/src/R/dcCalc.R, the same idea is done, but near the equator, one same value is applied for all months.
 - Correction for overwintering DC is integrated. From one fire season to another, only the DC has to be carried.
     - The original version does not include this correction.
     - A correction is proposed in https://rdrr.io/rforge/cffdrs/man/wDC.html
"""


import math

import numpy as np


class InvalidLatitude(Exception):
    """
    Exception to handle variables not covered by DataDict
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value) + " is not a valid Latitude."


def FFMC(TEMP, RH, WIND, RAIN, FFMCPrev):
    """
    Calculates today's Fine Fuel Moisture Code

    Parameters
    ----------
    TEMP: array
        Temperature of the day at noon (celsius)
    RH: array
        Relative humidity of the day at noon (%)
    WIND: array
        Wind speed of the day at noon  (km/h)
    RAIN: array
        24-hour accumulated rainfall at noon (mm)
    FFMCPrev: array
        Previous day's FFMC (1)

    Returns
    ----------
    FFMC: array
        FFMC on this day (1)
    """

    # preparing initial value mo
    mo = 147.2 * (101.0 - FFMCPrev) / (59.5 + FFMCPrev)

    # case of RAIN > 0.5: modification of mo in this case
    ind_RG05 = np.where(RAIN > 0.5)
    rf = RAIN[ind_RG05] - 0.5
    mr = mo[ind_RG05] + 42.5 * rf * np.exp(-100.0 / (251.0 - mo[ind_RG05])) * (
        1.0 - np.exp(-6.93 / rf)
    )
    # case of m0 > 150: need to add a single term
    ind_moG150 = np.where(mo[ind_RG05] > 150)
    mr[ind_moG150] += (
        0.0015 * pow(mo[ind_RG05][ind_moG150] - 150.0, 2) * pow(rf[ind_moG150], 0.5)
    )
    # limiting mr to 250
    mr[np.where(mr > 250)] = 250
    # updating mo on these cases:
    mo[ind_RG05] = mr

    # variable that will be output
    m = np.nan * np.ones(mo.shape)

    # Preparing threshold
    ed = (
        0.942 * pow(RH, 0.679)
        + 11.0 * np.exp((RH - 100.0) / 10.0)
        + 0.18 * (21.1 - TEMP) * (1.0 - np.exp(-0.115 * RH))
    )

    # Case mo > ed
    ind_moGed = np.where(mo > ed)
    ko = 0.424 * (1.0 - pow(RH[ind_moGed] / 100.0, 1.7)) + 0.0694 * pow(
        WIND[ind_moGed], 0.5
    ) * (1.0 - pow(RH[ind_moGed] / 100.0, 8))
    kd = ko * 0.581 * np.exp(0.0365 * TEMP[ind_moGed])
    m[ind_moGed] = ed[ind_moGed] + (mo - ed)[ind_moGed] * pow(10.0, -kd)

    # Case mo <= ed, with subcases
    ind_moLed = np.where(mo <= ed)
    tmp = m[ind_moLed]  # numpy cant affect values with cases AND subcases
    ew = (
        0.618 * pow(RH[ind_moLed], 0.753)
        + 10.0 * np.exp((RH[ind_moLed] - 100.0) / 10.0)
        + 0.18 * (21.1 - TEMP[ind_moLed]) * (1.0 - np.exp(-0.115 * RH[ind_moLed]))
    )
    #   Subcase m0 < ew
    ind_moLew = np.where(mo[ind_moLed] < ew)
    k1 = 0.424 * (
        1.0 - pow((100.0 - RH[ind_moLed][ind_moLew]) / 100.0, 1.7)
    ) + 0.0694 * pow(WIND[ind_moLed][ind_moLew], 0.5) * (
        1.0 - pow((100.0 - RH[ind_moLed][ind_moLew]) / 100.0, 8)
    )
    kw = k1 * 0.581 * np.exp(0.0365 * TEMP[ind_moLed][ind_moLew])
    tmp[ind_moLew] = ew[ind_moLew] - (ew[ind_moLew] - mo[ind_moLed][ind_moLew]) * pow(
        10.0, -kw
    )
    #   Subcase m0 >= ew
    ind_moGew = np.where(mo[ind_moLed] >= ew)
    tmp[ind_moGew] = mo[ind_moLed][ind_moGew]
    m[ind_moLed] = tmp

    return 59.5 * (250.0 - m) / (147.2 + m)


def DMC(TEMP, RH, RAIN, DMCPrev, LAT, numb_day, MONTH, cfg):
    """
    Calculates today's Duff Moisture Code

    Parameters
    ----------
    TEMP: array
        Temperature of the day at noon (celsius)
    RH: array
        Relative humidity of the day at noon (%)
    RAIN: array
        24-hour accumulated rainfall at noon (mm)
    DMCPrev: array
        Previous day's DMC (1)
    LAT: array
        Latitude in decimal degrees of the location (degree North)
    NUMB_DAY: scalar
        Number of the day in the year
    MONTH: int
        numeral month, from 1 to 12
    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.

    Returns
    ----------
    DMC: array
        DMC on this day (1)
    """

    # Changing DMCPrev
    # case of RAIN > 1.5:
    ind_RG15 = np.where(RAIN > 1.5)
    re = 0.92 * RAIN[ind_RG15] - 1.27
    mo = 20.0 + np.exp(5.6348 - DMCPrev[ind_RG15] / 43.43)
    # Preparing b:
    b = np.nan * np.ones(mo.shape)
    # Case DMCPrev<=33
    ind_DMCPrevL33 = np.where(DMCPrev[ind_RG15] <= 33)
    b[ind_DMCPrevL33] = 100.0 / (0.5 + 0.3 * DMCPrev[ind_RG15][ind_DMCPrevL33])
    # Case 33<DMCPrev<=65
    ind_DMCPrevL65 = np.where((33 < DMCPrev[ind_RG15]) & (DMCPrev[ind_RG15] <= 65))
    b[ind_DMCPrevL65] = 14.0 - 1.3 * np.log(DMCPrev[ind_RG15][ind_DMCPrevL65])
    # Case 65<DMCPrev
    ind_DMCPrevG65 = np.where((65 < DMCPrev[ind_RG15]))
    b[ind_DMCPrevG65] = 6.2 * np.log(DMCPrev[ind_RG15][ind_DMCPrevG65]) - 17.2

    # finishing case of Rain > 1.5, with subcases
    mr = mo + 1000.0 * re / (48.77 + b * re)
    pr = 244.72 - 43.43 * np.log(mr - 20.0)
    #   Subcase of pr > 0
    tmp = DMCPrev[ind_RG15]  # numpy cant affect values with cases AND subcases
    ind_prG0 = np.where(pr > 0)
    tmp[ind_prG0] = pr[ind_prG0]
    ind_prL0 = np.where(pr <= 0)
    tmp[ind_prL0] = 0
    DMCPrev[ind_RG15] = tmp

    # Preparing k
    k = np.zeros(TEMP.shape)  # set to 0 for all except where TEMP > -1.1
    ind_TEMPG11 = np.where(TEMP > -1.1)
    d1 = DayLength(LAT, numb_day, MONTH, cfg)
    k[ind_TEMPG11] = (
        1.894
        * (TEMP[ind_TEMPG11] + 1.1)
        * (100.0 - RH[ind_TEMPG11])
        * d1[ind_TEMPG11]
        * 0.000001
    )

    return DMCPrev + 100.0 * k


def DC(TEMP, RAIN, DCPrev, LAT, MONTH, cfg):
    """
    Calculates today's Drought Code

    Parameters
    ----------
    TEMP: array
        Temperature of the day at noon (celsius)
    RAIN: array
        24-hour accumulated rainfall at noon (mm)
    DCPrev: array
        Previous day's DMC (1)
    LAT: array
        Latitude in decimal degrees of the location (degree North)
    MONTH: int
        numeral month, from 1 to 12
    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.

    Returns
    ----------
    DC: array
        DC on this day (1)
    """

    # Updating DCPrev
    # Case Rain > 2.8, with subcases
    ind_RG28 = np.where(RAIN > 2.8)
    rd = 0.83 * RAIN[ind_RG28] - 1.27
    Qo = 800.0 * np.exp(-DCPrev[ind_RG28] / 400.0)
    Qr = Qo + 3.937 * rd
    Dr = 400.0 * np.log(800.0 / Qr)
    #   Sub-case Dr > 0
    tmp = DCPrev[ind_RG28]  # numpy cant affect values with cases AND subcases
    ind_DrG0 = np.where(Dr > 0)
    tmp[ind_DrG0] = Dr[ind_DrG0]
    ind_DrL0 = np.where(Dr <= 0)
    tmp[ind_DrL0] = 0.0
    DCPrev[ind_RG28] = tmp

    Lf = DryingFactor(LAT, MONTH, cfg)

    # Preparing V
    V = Lf
    # Case TEMP > -2.8
    ind_TGm28 = np.where(TEMP > -2.8)
    V[ind_TGm28] += 0.36 * (TEMP[ind_TGm28] + 2.8)
    # Case V < 0
    V[np.where(V < 0)] = 0

    return DCPrev + 0.5 * V


def ISI(WIND, FFMC):
    """
    Calculates today's Initial Spread Index

    Parameters
    ----------
    WIND: array
        Wind speed of the day at noon  (km/h)
    FFMC: array
        Fine Fuel Moisture Code of the day (1)

    Returns
    ----------
    ISI: array
        ISI on this day (1)
    """

    fWIND = np.exp(0.05039 * WIND)

    m = 147.2 * (101.0 - FFMC) / (59.5 + FFMC)
    # handling cases of FFMC > 101 (due to the approximation of a coefficient in FFMC: 101 * 147.2 / 59.5 ~=250 )
    m[np.where(FFMC > 101)] = 0

    fF = 91.9 * np.exp(-0.1386 * m) * (1.0 + pow(m, 5.31) / 49300000.0)

    return 0.208 * fWIND * fF


def BUI(DMC, DC):
    """
    Calculates today's Buildup Index

    Parameters
    ----------
    DMC: array
        Duff Moisture Code of the day (1)
    DC: array
        Drought Code of the day (1)

    Returns
    ----------
    BUI: array
        BUI on this day (1)
    """

    # Preparing U
    U = np.nan * np.ones(DMC.shape)

    # Case DMC <= 0.4 * DC, with subcases
    ind_dmcL04dc = np.where(DMC <= 0.4 * DC)
    tmp = U[ind_dmcL04dc]
    # handling the cases of DMC and DC being 0
    ind_dmc0dc0 = np.where(
        np.isclose(DMC[ind_dmcL04dc], 0) & np.isclose(DC[ind_dmcL04dc], 0)
    )
    tmp[ind_dmc0dc0] = 0
    ind_dmcN0dcN0 = np.where(
        ~np.isclose(DMC[ind_dmcL04dc], 0) | ~np.isclose(DC[ind_dmcL04dc], 0)
    )
    tmp[ind_dmcN0dcN0] = (
        0.8
        * DMC[ind_dmcL04dc][ind_dmcN0dcN0]
        * DC[ind_dmcL04dc][ind_dmcN0dcN0]
        / (DMC[ind_dmcL04dc][ind_dmcN0dcN0] + 0.4 * DC[ind_dmcL04dc][ind_dmcN0dcN0])
    )
    U[ind_dmcL04dc] = tmp

    # Case DMC > 0.4 * DC, with subcases
    ind_dmcG04dc = np.where(DMC > 0.4 * DC)
    tmp = U[ind_dmcG04dc]
    # handling the cases of DMC and DC being 0
    ind_dmc0dc0 = np.where(
        np.isclose(DMC[ind_dmcG04dc], 0) & np.isclose(DC[ind_dmcG04dc], 0)
    )
    tmp[ind_dmc0dc0] = 0
    ind_dmcN0dcN0 = np.where(
        ~np.isclose(DMC[ind_dmcG04dc], 0) | ~np.isclose(DC[ind_dmcG04dc], 0)
    )
    tmp[ind_dmcN0dcN0] = DMC[ind_dmcG04dc][ind_dmcN0dcN0] - (
        1.0
        - 0.8
        * DC[ind_dmcG04dc][ind_dmcN0dcN0]
        / (DMC[ind_dmcG04dc][ind_dmcN0dcN0] + 0.4 * DC[ind_dmcG04dc][ind_dmcN0dcN0])
    ) * (0.92 + pow(0.0114 * DMC[ind_dmcG04dc][ind_dmcN0dcN0], 1.7))
    U[ind_dmcG04dc] = tmp

    # Taking max of U and 0
    U[np.where(U <= 0)] = 0
    return U


def FWI(ISI, BUI):
    """
    Calculates today's Fire Weather Index

    Parameters
    ----------
    ISI: array
        ISI on this day (1)
    BUI: array
        BUI on this day (1)

    Returns
    ----------
    FWI: array
        FWI on this day (1)
    """

    # Preparing fD
    fD = np.nan * np.ones(BUI.shape)

    # Case BUI <= 80
    ind_buiL80 = np.where(BUI <= 80)
    fD[ind_buiL80] = 0.626 * pow(BUI[ind_buiL80], 0.809) + 2.0
    ind_buiG80 = np.where(BUI > 80)
    fD[ind_buiG80] = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * BUI[ind_buiG80]))

    B = 0.1 * ISI * fD

    # Preparing S (just complying to notations of the original script)
    S = np.copy(B)

    # Case of B > 1
    ind_BG1 = np.where(B > 1)
    S[ind_BG1] = np.exp(2.72 * pow(0.434 * np.log(B[ind_BG1]), 0.647))

    # Simply making sure that they are no negative values
    S[np.where(S < 0)] = 0
    return S


def DryingFactor(Latitude, Month, cfg):
    """
    Calculates the Drying Factor.
    NB: this parameter is called "Day-length adjustment in DC". This name is not used here because of the adjustments on the parameter "Effective day-length" used in DMC.

    Option for the drying factor from cfg.adjust_DryingFactor:
        - 'original': values for the Northern hemisphere applied everywhere (Wagner et al, 1987: https://cfs.nrcan.gc.ca/pubwarehouse/pdfs/19927.pdf)
        - 'NSH': the values for the Southern hemisphere are those for the northern shifted by 6 months (https://github.com/buckinha/pyfwi/blob/master/pyFWI/FWIFunctions.py)
        - 'NSHeq': the same idea is applied, but near the equator, one same value is applied for all months (https://rdrr.io/rforge/cffdrs/src/R/dcCalc.R)

    Parameters
    ----------
    Latitude: array
        Latitude in decimal degrees of the location (degree North)
    Month: int
        numeral month, from 1 to 12
    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.

    Returns
    ----------
    DryingFactor: array
        Drying Factor (1)
    """

    if len(np.array(Month).shape) > 0:
        raise Exception(
            "This code has been rewritten so that only the month must be a scalar... dont know what to do if there are multiple values."
        )

    # LfN are the original values from the Canadian Fire Weather Index
    LfN = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    # LfS are the same values shifted by 6 months
    LfS = [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8]
    # Lfeq is the average
    Lfeq = 1.4

    if cfg.adjust_DryingFactor == "original":
        retVal = LfN[Month - 1] * np.ones(Latitude.shape)

    elif cfg.adjust_DryingFactor == "NSH":
        # setting drying factor at different latitudes
        retVal = np.nan * np.ones(Latitude.shape)
        ind_LatG0 = np.where(Latitude > 0)
        retVal[ind_LatG0] = LfN[Month - 1]
        ind_LatL0 = np.where(Latitude <= 0)
        retVal[ind_LatL0] = LfS[Month - 1]

    elif cfg.adjust_DryingFactor == "NSHeq":
        # setting drying factor at different latitudes
        retVal = np.nan * np.ones(Latitude.shape)
        ind_LatG0 = np.where(Latitude > 20)
        retVal[ind_LatG0] = LfN[Month - 1]

        ind_LatG0 = np.where((Latitude > -20) & (Latitude <= 20))
        retVal[ind_LatG0] = Lfeq

        ind_LatL0 = np.where(Latitude <= -20)
        retVal[ind_LatL0] = LfS[Month - 1]

    else:
        raise Exception("Unknown name for the type of drying factor")

    return retVal


def DayLength(Latitude, numb_day, MONTH, cfg):
    """
    Calculates the effective Day Length
    Option for the effective day length cfg.adjust_DayLength:
        - 'original': values adapted for Canadian latitudes, depends on the month (Wagner et al, 1987: https://cfs.nrcan.gc.ca/pubwarehouse/pdfs/19927.pdf)
        - 'bins': depends on 4 bins of latitudes and the month (https://github.com/buckinha/pyfwi/blob/master/pyFWI/FWIFunctions.py)
        - 'continuous': depends continuously on latitude and the day of the year (https://github.com/NCAR/fire-indices & https://www.ncl.ucar.edu/Document/Functions/Crop/daylight_fao56.shtml)


    Parameters
    ----------
    Latitude: array
        Latitude in decimal degrees of the location (degree North)
    numb_day: scalar
        Number of the day in the year
    Month: int
        numeral month, from 1 to 12
    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.

    Returns
    ----------
    DayLength: array
        Day Length (h)
    """

    if cfg.adjust_DayLength == "original":
        DayLength46N = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
        retVal = DayLength46N[MONTH - 1] * np.ones(Latitude.shape)

    elif cfg.adjust_DayLength == "bins":
        # preparing values of day length
        DayLength46N = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
        DayLength20N = [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8]
        DayLength20S = [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2]
        DayLength40S = [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8]

        # setting day length at different latitudes
        retVal = np.nan * np.ones(Latitude.shape)
        ind_latG33L90 = np.where((33 < Latitude) & (Latitude <= 90))
        retVal[ind_latG33L90] = DayLength46N[MONTH - 1]
        ind_latG0L33 = np.where((0 < Latitude) & (Latitude <= 33))
        retVal[ind_latG0L33] = DayLength20N[MONTH - 1]
        ind_latGm30L0 = np.where((-30 < Latitude) & (Latitude <= 0))
        retVal[ind_latGm30L0] = DayLength20S[MONTH - 1]
        ind_latGm90Lm30 = np.where((-90 <= Latitude) & (Latitude <= -30))
        retVal[ind_latGm90Lm30] = DayLength40S[MONTH - 1]

    elif cfg.adjust_DayLength == "continuous":
        lat = Latitude * np.pi / 180  # degree -> radian
        sun_dec = 0.409 * np.sin(2 * np.pi / 365 * numb_day - 1.39)  # equation 24
        # preparing equation 25, with special cases handled
        val_for_arccos = -np.tan(lat) * np.tan(sun_dec)
        val_for_arccos[np.where(val_for_arccos < -1)] = -1
        val_for_arccos[np.where(val_for_arccos > 1)] = 1
        sunset_hour_angle = np.arccos(val_for_arccos)  # equation 25
        retVal = 24 / np.pi * sunset_hour_angle  # equation 34

    else:
        raise Exception("Unknown name for the type of day length")

    if np.any(np.isnan(retVal)):
        raise InvalidLatitude(Latitude)

    return retVal


def test_FireSeason(TEMP_wDC, SeasonActive, threshold_start=12, threshold_end=5):
    """
    Calculation whether the fire season is active or not.
    Based on Wotton & Flannigan 1993 method, as written in https://rdrr.io/github/jordan-evens-nrcan/cffdrs/src/R/fireSeason.r

    NB: because of the size of the datasets, we choose to use a more explicit loop, instead of using dask xarrays:
        tmp = test.where( test > threshold_start, drop=True )
        d_start = tmp.time.groupby(tmp.time.dt.year).min('time')
        tmp = test.where( test < threshold_end, drop=True )
        d_end = tmp.time.groupby(tmp.time.dt.year).max('time')


    Parameters
    ----------
    TEMP_wDC: array
        Temperature at noon over 2 days before, the current day and the next 2 days (celsius)
    SeasonActive: array
        Boolean providing information on where the Fire Season is active on the former day (bool)
    threshold_start: scalar
        Threshold in temperature at noon (celsius): if the temperature is above for the 2 former days and the current day, the fire season is assumed to start.
    threshold_end: scalar
        Threshold in temperature at noon (celsius): if the temperature is below for the current day and the 2 next days, the fire season is assumed to end.

    Returns
    ----------
    val_SA: array
        Boolean providing information on where the Fire Season is active on the current day (bool)
    """

    # warning, numpy or xarray dont accept to affect values if writing variable[ind1][ind2] = values
    # hence writing in a somewhat convoluted manner
    val_SA = np.copy(SeasonActive)

    # FIRST, DEALING WITH PLACES WHERE FIRE SEASON IS NOT ACTIVE: CHECKING IF STARTS
    ind_SeasonNotActive = np.where(val_SA == False)
    val_SAFalse = val_SA[ind_SeasonNotActive]

    # taking values over the current day and the next 2 days: if all above starting temperature threshold, it means that the fire season starts on the current day
    test_start = np.where(
        np.all(
            (
                TEMP_wDC[2:, ind_SeasonNotActive[0], ind_SeasonNotActive[1]]
                > threshold_start
            ),
            axis=0,
        )
    )

    # Current day and 2 next days are above threshold: starting fire season today
    val_SAFalse[test_start] = True
    val_SA[ind_SeasonNotActive] = val_SAFalse

    # THEN, DEALING WITH PLACES WHERE FIRE SEASON IS ACTIVE: CHECKING IF STOPS
    ind_SeasonActive = np.where(val_SA)
    val_SATrue = val_SA[ind_SeasonActive]

    # taking values over the last 2 days and the current day: if all below ending temperature threshold, it means that the fire season ends on the current day
    test_end = np.where(
        np.all(
            (
                TEMP_wDC[: 2 + 1, ind_SeasonActive[0], ind_SeasonActive[1]]
                < threshold_end
            ),
            axis=0,
        )
    )

    # Last 2 days and current day are below threshold: ending fire season today
    val_SATrue[test_end] = False
    val_SA[ind_SeasonActive] = val_SATrue

    return val_SA


def wDC(DCf, rw, a=0.75, b=0.75):
    """
    Computes the overwintering Drought Code (DC) value. All variables names are laid out in the same manner as Lawson & Armitage (2008) (http://cfs.nrcan.gc.ca/pubwarehouse/pdfs/29152.pdf).

    Parameters
    ----------
    DCf: array
        "Final Fall DC Value": values of DC at the end of the last fire season (1)
    rw: array
        "Winter precipitation": precipitations accumulated over the last non-fire season (mm)
    a: scalar
        user-selected value accounting for carry-over fraction (1). Default here is the median value from Lawson & Armitage (2008).
        1.0  : Daily DCa calculated up to 1 November, continuous snow cover, or freeze-up, whichever comes first
        0.75 : Daily DC calculations stopped before any of the above conditons met or the area is subject to occasional winter chinook conditions, leaving the ground bare and subject to moisture depletion
        0.5  : Forested areas subject to long periods in fall or winter that favor depletion of soil moisture Effectiveness of winter precipitation in recharging moisture reserves in spring

    b: scalar
        user-selected value accounting for wetting efficiency fraction (1). Default here is the median value from Lawson & Armitage (2008).
        0.9  : Poorly drained, boggy sites with deep organic layers
        0.75 : Deep ground frost does not occur until late fall, if at all; moderately drained sites that allow infiltration of most of the melting snowpack
        0.5  : Chinook-prone areas and areas subject to early and deep ground frost; well-drained soils favoring rapid percolation or topography favoring rapid runoff before melting of ground frost

    Returns
    ----------
    DCs: array
        Overwintered Drought Code, i.e. Spring startup DC value (1)
    """
    # Eq. 3 - Final fall moisture equivalent of the DC
    Qf = 800 * np.exp(-DCf / 400)
    # Eq. 2 - Starting spring moisture equivalent of the DC
    Qs = a * Qf + b * (3.94 * rw)
    # Eq. 4 - Spring start-up value for the DC
    DCs = 400 * np.log(800 / Qs)
    # Constrain DC
    DCs[np.where(DCs < 15)] = 15
    return DCs


def DC_with_overwintering(
    TEMP_wDC,
    FormerSeasonActive,
    rw,
    dcf,
    CounterSeasonActive,
    dcPREV,
    TEMP,
    RAIN,
    LAT,
    MONTH,
    cfg,
):
    """
    Computes the overwintered Drought Code (DC) value. Many different cases have to be considered:
    Cases for DC: (Important, the initialization of whether the fire season is active is set to False)
        1. Former day: Fire Season Active
            1.1. Current day: Fire Season Active
                --> LOCAL SUMMER: calculates **DC normally**
            1.2. Current day: Fire Season Inactive
                --> LOCAL END SUMMER / START WINTER: does not calculate DC anymore, and saves value of DC of former day, corresponds to fall value used for next summer.
        2. Former day: Fire Season Inactive
            2.1. Current day: Fire Season Active
                --- LOCAL START SUMMER / END WINTER:
                2.1.1. First Fire Season Active
                    --> Correct dates for fall DC and accumulated precipitations are unknown. Calculates **DC normally**.
                2.1.2. Not the first Fire Season Active
                    --> Correct dates for fall DC and accumulated precipitations are known. Calculates **overwintered DC**.
            2.2. Current day: Fire Season Inactive
                --- LOCAL WINTER:
                2.2.1. No Fire Season Active yet
                    --> Correct dates for fall DC and accumulated precipitations are unknown for the first Fire Season. Must calculate **DC normally**.
                2.2.2. Already had one Fire Season Active
                    --> Correct dates for fall DC and accumulated precipitations are known. Accumulates precipitation.

    FWI can be calculated where walues for DC are correctly known, that is to say:
        - where the Fire Season Active, be it in the first fire season or not
        - where the Fire Season is not active, BUT no fire season happened there before, preventing of correctly overwintering DC.

    Parameters
    ----------
    TEMP_wDC: array
        Temperature at noon over 2 days before, the current day and the next 2 days (celsius)
    FormerSeasonActive: array
        Boolean providing information on where the Fire Season is active on the former day (bool)
    rw: array
        "Winter precipitation": precipitations accumulated over the last non-fire season (mm)
    dcf: array
        "Final Fall DC Value": values of DC at the end of the last fire season (1)
    CounterSeasonActive: array
        Counting the number of elapsed fire seasons.
    dcPREV: array
        Previous day's DMC (1)
    TEMP: array
        Temperature of the day at noon (celsius)
    RAIN: array
        24-hour accumulated rainfall at noon (mm)
    LAT: array
        Latitude in decimal degrees of the location (degree North)
    MONTH: int
        numeral month, from 1 to 12
    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.

    Returns
    ----------
    dictionary to carry on the calculations
        dc: array
            Current day's DMC (1)
        CurrentSeasonActive: array
            Boolean providing information on where the Fire Season is active on the current day (bool)
        rw: array
            "Winter precipitation": updated precipitations accumulated over the last non-fire season (mm)
        dcf: array
            "Final Fall DC Value": updated values of DC at the end of the last fire season (1)
        CounterSeasonActive: array
            Updated counter for the number of elapsed fire seasons.
        ind_calc_FWI: array
            Boolean indicating where the FWI has to be calculated, i.e. when fire season is active or before the very first fire season had started (for initialization purposes)
    """
    # warning, numpy or xarray dont accept to affect values if writing variable[ind1][ind2] = values
    # hence writing in a somewhat convoluted manner

    # Producing new map of where the fire season is active. Need to know where it was already active to know which test to apply where.
    CurrentSeasonActive = test_FireSeason(TEMP_wDC, FormerSeasonActive)

    # Values that will be modified and returned here
    dc = np.nan * np.ones(
        CurrentSeasonActive.shape
    )  # values at this current timestep: either we calculate it or it will be nan because not in the fire season
    # rw: building on former timestep, accumulating precipitations
    # dcf: transmits from former timestep, values at former local autumn

    # LOCAL SUMMER ---> Grid cells where the fire season was active last day and is still active at the current day: normal DC
    ind_TT = np.where(FormerSeasonActive & CurrentSeasonActive)
    dc[ind_TT] = DC(
        TEMP[ind_TT], RAIN[ind_TT], dcPREV[ind_TT], LAT[ind_TT], MONTH, cfg
    )  # (case 1.1)

    # LOCAL END SUMMER / START WINTER ---> Grid cells where the fire season was active last day and has ended this current day: writes DCf of the former active day for when it will resume
    ind_TF = np.where(FormerSeasonActive & (CurrentSeasonActive == False))
    dcf[ind_TF] = dcPREV[ind_TF]  # (case 1.2)

    # LOCAL END WINTER / START SUMMER ---> Grid cells where the fire season was not active last day but starts this current day: calculates wDC, resets rw and DCf and increment CounterSeasonActive
    ind_FT = np.where((FormerSeasonActive == False) & CurrentSeasonActive)
    # WARNING: the *FIRST* time that this is encountered on a gridcell, the real dcf and rw are unknown, in this grid cell. Must sort out the 2 cases.
    dc_tmp = dc[ind_FT]
    # Skipping this very first season, starting at the next one.
    ind_CSA0 = np.where(CounterSeasonActive[ind_FT] == 0)
    dc_tmp[ind_CSA0] = DC(
        TEMP[ind_FT][ind_CSA0],
        RAIN[ind_FT][ind_CSA0],
        dcPREV[ind_FT][ind_CSA0],
        LAT[ind_FT][ind_CSA0],
        MONTH,
        cfg,
    )  # (case 2.1.1)
    # For other seasons, values for dcf and rw are correctly calculated.
    ind_CSAG0 = np.where(CounterSeasonActive[ind_FT] > 0)
    dc_tmp[ind_CSAG0] = wDC(
        dcf[ind_FT][ind_CSAG0], rw[ind_FT][ind_CSAG0], a=0.75, b=0.75
    )  # (case 2.1.2)
    # affecting to dc
    dc[ind_FT] = dc_tmp
    rw[ind_FT] = 0
    dcf[ind_FT] = np.nan
    CounterSeasonActive[ind_FT] = (
        CounterSeasonActive[ind_FT] + 1
    )  # next season can be overwintered

    # LOCAL WINTER ---> Grid cells where the fire season was not active last day and is still inactive this current day: keep accumulating precipitations
    ind_FF = np.where((FormerSeasonActive == False) & (CurrentSeasonActive == False))
    # WARNING: as long as a fire season has not been encoutered on a gridcell, the real dcf and rw are unknown, cannot overwinter. Must sort out the 2 cases.
    dc_tmp = dc[ind_FF]
    # cannot overwinter yet, must calculate DC normally
    ind_CSA0 = np.where(CounterSeasonActive[ind_FF] == 0)
    dc_tmp[ind_CSA0] = DC(
        TEMP[ind_FF][ind_CSA0],
        RAIN[ind_FF][ind_CSA0],
        dcPREV[ind_FF][ind_CSA0],
        LAT[ind_FF][ind_CSA0],
        MONTH,
        cfg,
    )  # (case 2.2.1)
    dc[ind_FF] = dc_tmp
    # can overwinter where had already a first active season
    ind_CSAG0 = np.where(CounterSeasonActive[ind_FF] > 0)
    rw_tmp = rw[ind_FF]
    rw_tmp[ind_CSAG0] = rw_tmp[ind_CSAG0] + RAIN[ind_FF][ind_CSAG0]  # (case 2.2.2)
    rw[ind_FF] = rw_tmp

    # Calculating where the FWI has to be calculated: it is where the CurrentSeasonActive was True,  BUT ALSO where the CounterSeasonActive was 0
    ind_calc_FWI = np.where(
        CurrentSeasonActive
        | ((CurrentSeasonActive == False) & (CounterSeasonActive == 0))
    )

    return {
        "dc": dc,
        "SeasonActive": CurrentSeasonActive,
        "rw": rw,
        "DCf": dcf,
        "CounterSeasonActive": CounterSeasonActive,
        "ind_calc_FWI": ind_calc_FWI,
    }


def calcFWI(vars_calc, cfg):
    """
    Global function for the calculation of today's Fire Weather Index using today's climate variables and former day's variables.

    Parameters
    ----------
    vars_calc: dictionary
        Variables for computation of the FWI.
        TEMP: array
            Temperature of the day at noon (celsius)
        RAIN: array
            24-hour accumulated rainfall at noon (mm)
        RH: array
            Relative humidity of the day at noon (%)
        WIND: array
            Wind speed of the day at noon  (km/h)
        FFMCPrev: array
            Previous day's FFMC (1)
        DCPrev: array
            Previous day's DC (1)
        DMCPrev: array
            Previous day's DMC (1)
        LAT: array
            Latitude in decimal degrees of the location (degree North)
        NUMB_DAY: scalar
            Number of the day in the year
        MONTH: int
            numeral month, from 1 to 12
        TEMP_wDC: array
            Only if overwintering is required: Temperature at noon over 2 days before, the current day and the next 2 days (celsius)
        SeasonActive: array
            Only if overwintering is required: Boolean providing information on where the Fire Season is active on the former day (bool)
        rw: array
            Only if overwintering is required: "Winter precipitation": updated precipitations accumulated over the last non-fire season (mm)
        DCf: array
            Only if overwintering is required: "Final Fall DC Value": updated values of DC at the end of the last fire season (1)
        CounterSeasonActive: array
            Only if overwintering is required: Updated counter for the number of elapsed fire seasons.

    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.

    Returns
    ----------
    vars_calc: dictionnary
        Input "vars_calc" is updated over all its values.
    """
    # directly correcting for spurious values in RAIN, RH and WIND
    vars_calc = corrections_vars(vars_calc)

    # Calculating the drought code, with check for the overwintering:
    if cfg.adjust_overwinterDC == "wDC":
        out = DC_with_overwintering(
            vars_calc["TEMP_wDC"],
            vars_calc["SeasonActive"],
            vars_calc["rw"],
            vars_calc["DCf"],
            vars_calc["CounterSeasonActive"],
            vars_calc["dcPREV"],
            vars_calc["TEMP"],
            vars_calc["RAIN"],
            vars_calc["LAT"],
            vars_calc["MONTH"],
            cfg,
        )
        for var in [
            "dc",
            "SeasonActive",
            "rw",
            "DCf",
            "CounterSeasonActive",
            "ind_calc_FWI",
        ]:
            vars_calc[var] = np.copy(out[var])
    else:
        vars_calc["dc"] = DC(
            vars_calc["TEMP"],
            vars_calc["RAIN"],
            vars_calc["dcPREV"],
            vars_calc["LAT"],
            vars_calc["MONTH"],
            cfg,
        )
        # without overwintering, will always calculates FWI everywhere
        shp_lat, shp_lon = vars_calc["LAT"].shape
        vars_calc["ind_calc_FWI"] = (
            np.repeat(np.arange(shp_lat), shp_lon),
            np.repeat(np.arange(shp_lon)[np.newaxis, :], shp_lat, axis=0).flatten(),
        )

    # Calculating the 2 other codes:
    vars_calc["ffmc"] = FFMC(
        vars_calc["TEMP"],
        vars_calc["RH"],
        vars_calc["WIND"],
        vars_calc["RAIN"],
        vars_calc["ffmcPREV"],
    )
    vars_calc["dmc"] = DMC(
        vars_calc["TEMP"],
        vars_calc["RH"],
        vars_calc["RAIN"],
        vars_calc["dmcPREV"],
        vars_calc["LAT"],
        vars_calc["NUMB_DAY"],
        vars_calc["MONTH"],
        cfg,
    )

    # Calculating the 2 indexes built on them:
    vars_calc["isi"] = ISI(vars_calc["WIND"], vars_calc["ffmc"])
    vars_calc["bui"] = BUI(vars_calc["dmc"], vars_calc["dc"])

    # Calculating the FWI:
    vars_calc["fwi"] = FWI(vars_calc["isi"], vars_calc["bui"])

    # np.where(np.isnan(vars_calc['fwi'][vars_calc['ind_calc_FWI'][0],vars_calc['ind_calc_FWI'][1]]))

    # Next day, values for the previous day will be called. They are today's DC, DMC and FFMC
    for var in ["ffmc", "dc", "dmc"]:
        vars_calc[var + "PREV"] = np.copy(vars_calc[var])

    return vars_calc


def corrections_vars(vars_c):
    """
    Correct variables for calculation of the FWI.

    Parameters
    ----------
    vars_c: dictionary
        Variables to correct.
        RH: array
            Relative humidity of the day at noon (%)
        WIND: array
            Wind speed of the day at noon  (km/h)

    Returns
    ----------
    vars_c: dictionnary
        "vars_c" is corrected over all its values.
    """

    # Preparing Relative Humidity
    vars_c["RH"][np.where(vars_c["RH"] > 100)] = 100
    vars_c["RH"][np.where(vars_c["RH"] < 0)] = 0

    # Preparing Wind
    vars_c["WIND"][np.where(vars_c["WIND"] < 0)] = 0

    # Preparing Rain
    vars_c["RAIN"][np.where(vars_c["RAIN"] < 0)] = 0
    return vars_c


def init_prev_values(data, cfg):
    """
    Initialize values for calculation of the FWI, accordingly to Wagner et al, 1987 (https://cfs.nrcan.gc.ca/pubwarehouse/pdfs/19927.pdf).

    Parameters
    ----------
    data: xarrays
        Data produced by "func_prepare_datasets". Could actually be simply lat & lon.

    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.

    Returns
    ----------
    var_calc: dictionnary
        FFMC: array
            set at 85 (1).
        DC: array
            set at 15 (1).
        DMC: array
            set at 6 (1).
        TEMP_wDC: None
            Only if overwintering is required. Set at None.
        SeasonActive: None
            Only if overwintering is required. Set at None.
        DCf: None
            Only if overwintering is required. Set at None.
        rw: None
            Only if overwintering is required. Set at None.
        CounterSeasonActive: None
            Only if overwintering is required. Set at None.
    """

    # prepare the dictionary that will be returned
    var_calc = {}

    # shape of data
    shp = (data[cfg.list_vars[0]].lat.size, data[cfg.list_vars[0]].lon.size)

    # rough first estimation from: https://cwfis.cfs.nrcan.gc.ca/background/dsm/fwi
    for var in ["ffmcPREV", "dmcPREV", "dcPREV"]:
        var_calc[var] = {"ffmcPREV": 85, "dmcPREV": 6, "dcPREV": 15}[var] * np.ones(shp)

    # adding other variables for overwintering, that will be correctly estimated in functions_support.py, 'prepare_variables_FWI'
    if cfg.adjust_overwinterDC == "wDC":
        for var in ["TEMP_wDC", "SeasonActive", "DCf", "rw", "CounterSeasonActive"]:
            var_calc[var] = None

    return var_calc
