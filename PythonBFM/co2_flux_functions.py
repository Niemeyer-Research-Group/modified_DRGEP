import numpy
from PythonBFM.seasonal_cycling_functions import calculate_density

def calculate_co2_flux(co2_flux_parameters, environmental_parameters, constant_parameters, conc, temper, wind, salt):
    """ calculates the air-sea flux of co2 """
    
    # Species concentrations
    n1p = conc[1]              # Phosphate (mmol P m^-3)
    n5s = conc[5]              # Silicate (mmol Si m^-3)
    o3c = conc[48]             # Dissolved inorganic carbon(mg C m^-3)
    o3h = conc[49]             # Total alkalinity (mmol Eq m^-3)
    
    # Calculate Schmidt number, ratio between the kinematic viscosity and the molecular diffusivity of carbon dioxide
    schmidt_number_o3c = (co2_flux_parameters["c1"] - co2_flux_parameters["c2"]*temper + co2_flux_parameters["c3"]*(temper**2) - co2_flux_parameters["c4"]*(temper**3))
    schmidt_ratio_o3c = co2_flux_parameters["schmidt_o3c"]/schmidt_number_o3c

    # schmidt_ratio is limited to 0 when T > 40 °C 
    if schmidt_ratio_o3c<0.0:
        schmidt_ratio_o3c = 0.0

    # Compute Chemical enhancement the Temperature dependent gas transfer
    bt = 2.5*(0.5246 + 1.6256e-2*temper + 4.9946e-4*(temper**2))

    # Calculate wind dependency + Chemical enhancement including conversion cm/hr => m/s
    ken = (bt + co2_flux_parameters["d"]*(wind**2))*numpy.sqrt(schmidt_ratio_o3c)*constant_parameters["cm2m"]*constant_parameters["hours_per_day"]/constant_parameters["sec_per_day"]

    # K0, solubility of co2 in the water (K Henry) from Weiss 1974; K0 = [co2]/pco2 [mol kg-1 atm-1]
    tk = (temper + constant_parameters["c_to_kelvin"])
    tk_100 = tk/100.0
    k0 = numpy.exp((93.4517/tk_100) - 60.2409 + (23.3585*numpy.log(tk_100)) + salt*(0.023517 - (0.023656*tk_100) + 0.0047036*(tk_100**2)))

    # partial pressure of atmospheric CO2
    pco2_air = co2_flux_parameters["atm_co2_0"]

    # get density [kg m^-3]
    rho = calculate_density(temper,salt,environmental_parameters["del_z"])

    # initialize ph if it doesn't already have a value
    try:
        pH
    except NameError:
        pH = co2_flux_parameters["ph_initial"]

    # calculate all constants needed to convert between various measured carbon species
    tk_inv = 1.0/tk
    tk_log = numpy.log(tk)
    salt_sqrt = numpy.sqrt(salt)
    
    # chlorinity
    scl = salt/1.80655
    
    # ionic strength 
    ionic_strength = 19.924*salt/(1000.0 - 1.005*salt)
    ionic_strength_sqrt = numpy.sqrt(ionic_strength)
    
    # Calculate concentrations for borate, sulfate, and fluoride as a function of chlorinity
    # Uppstrom (1974)
    bt = 0.000232*scl/10.811
    
    # Morris & Riley (1966)
    st = 0.14*scl/96.062
    
    # Riley (1965)
    ft = 0.000067*scl/18.9984
    
    # change units from mmol m^-3 to mol/kg
#    pt = n1p/rho*1e-3
#    sit = n5s/rho*1e-3
    pt = 0.0
    sit = 0.0
    
    # change DIC (o3h) units from mg C m^-3 to mol kg^-1
    ldic = o3c*constant_parameters["omega_c"]*constant_parameters["g_per_mg"]/rho
    
    # convert TA from inits of mmol eq m^-3 to mol eq kg^-1
    alk = o3h/rho*constant_parameters["mol_per_mmol"]
    
    # calculate Acidity constants
    # constants according to Mehrbach et al. (1973) as refitted by Dickson and Millero (1987) 
    # ph scale: seawater  (Millero, 1995, p.664)
    # Standard OCMIP computation. Natural seawater
    # calculate carbonate equilibrium I, k1, [H][hco3]/[H2co3]
    k1 = 10.0**(-1.0*(3670.7*tk_inv - 62.008 + 9.7944*tk_log - 0.0118*salt + 0.000116*(salt**2)))

    # calculate carbonate equilibrium II, k2, [H][co3]/[hco3]
    k2 = 10.0**(-1.0*(1394.7*tk_inv + 4.777 - 0.0184*salt + 0.000118*(salt**2)))

    # calculate k1p, [H][H2PO4]/[H3PO4] (ph scale: total)
    lnK = -4576.752*tk_inv + 115.525 - 18.453* tk_log + (-106.736*tk_inv + 0.69171)*salt_sqrt + (-0.65643*tk_inv - 0.01844)*salt
    k1p = numpy.exp(lnK)

    # calculate k2p, [H][HPO4]/[H2PO4] (ph scale: total)
    lnK = -8814.715*tk_inv + 172.0883 - 27.927* tk_log +(-160.340*tk_inv + 1.3566)*salt_sqrt + (0.37335*tk_inv - 0.05778)*salt
    k2p = numpy.exp(lnK)
    
    # calculate k3p, [H][PO4]/[HPO4] (ph scale: total)
    lnK = -3070.75*tk_inv - 18.126 + (17.27039*tk_inv + 2.81197)*salt_sqrt + (-44.99486*tk_inv - 0.09984)*salt
    k3p = numpy.exp(lnK)
    
    # calculate ksi, [H][SiO(OH)3]/[Si(OH)4]
    ksi = -8904.2*tk_inv + 117.385 - 19.334*tk_log + (-458.79*tk_inv + 3.5913)*ionic_strength_sqrt + (188.74*tk_inv - 1.5998)*ionic_strength + (-12.1652*tk_inv + 0.07871)*(ionic_strength**2) + numpy.log(1.0 - (0.001005*salt))
    ksi = numpy.exp(lnK)
    
    # calculate kw, [H][OH] (ph scale: SWS)
    intercept = 148.9802
    lnK = intercept - 13847.26*tk_inv - 23.6521*tk_log + (118.67*tk_inv - 5.977 + 1.0495*tk_log)*salt_sqrt - 0.01615*salt
    kw = numpy.exp(lnK)
    
    # calculate ks, [H][SO4]/[HSO4]  (ph scale: "free")
    lnK = -4276.1*tk_inv + 141.328 - 23.093*tk_log + (-13856.0*tk_inv + 324.57 - 47.986*tk_log)*ionic_strength_sqrt + (35474.0*tk_inv - 771.54 + 114.723*tk_log)*ionic_strength - 2698.0*tk_inv*(ionic_strength**1.5) + 1776.0*tk_inv*(ionic_strength**2) + numpy.log(1.0 - 0.001005*salt)
    ks = numpy.exp(lnK)
    
    # calculate kf, [H][F]/[HF] (ph scale: "free")
    lnK = 1590.2*tk_inv - 12.641 + 1.525*ionic_strength_sqrt + numpy.log(1.0 - 0.001005*salt)
    kf = numpy.exp(lnK)
    
    # calculate kb, [H][BO2]/[HBO2] (ph scale: total)
    lnK = (-8966.90 - 2890.53*salt_sqrt - 77.942*salt + 1.728*(salt**1.5) - 0.0996*(salt**2))*tk_inv + (148.0248 + 137.1942*salt_sqrt + 1.62142*salt) + (-24.4344 - 25.085*salt_sqrt - 0.2474*salt)*tk_log + 0.053105*salt_sqrt*tk
    kb = numpy.exp(lnK)

    # calculate [H+] total when DIC and TA are known
    small_interval = (pH>1.0 and pH<9.0)
    if small_interval:
        h1 = 10.0**(-(pH + co2_flux_parameters["m2phdelt"]))
        h2 = 10.0**(-(pH - co2_flux_parameters["m2phdelt"]))
        h_plus, error = find_roots_of_f_TA(h1, h2, co2_flux_parameters["m2xacc"], co2_flux_parameters["m2maxit"], k1, k2, k1p, k2p, k3p, ksi, kw, ks, kf, kb, bt, st, ft, pt, sit, ldic, alk)
    if not small_interval or error>0:
        h1 = 10.0**(-11.0)
        h2 = 10.0**(-2.0)
        h_plus, error = find_roots_of_f_TA(h1, h2, co2_flux_parameters["m2xacc"], co2_flux_parameters["m2maxit"], k1, k2, k1p, k2p, k3p, ksi, kw, ks, kf, kb, bt, st, ft, pt, sit, ldic, alk)
    
    # Derive [co2] and Compute other diagnostic variables (hco3, co3 and ph) and pco2, the co2 partial pressure in the water
    h_plus2 = h_plus*h_plus
    co2 = ldic*h_plus2/(h_plus2 + k1*h_plus + k1*k2)
    pco2_sea = co2/k0
    pH = -numpy.log(h_plus)
        
    # convert partial pressure of oceanic CO2 to uatm
    pco2_sea = pco2_sea*constant_parameters["uatm_per_atm"]
    
    # flux co2 in mmol C m^-2 s^-1
    do3cdt_air_sea_flux = ken*(pco2_air - pco2_sea)*k0*rho/1000.0
    
    # convert flux to units of mg C m^-3 s^-1
    do3cdt_air_sea_flux = do3cdt_air_sea_flux/constant_parameters["omega_c"]/environmental_parameters["del_z"]
    
    return do3cdt_air_sea_flux
    
def calculate_Hplus(pH, k1, k2, k1p, k2p, k3p, ksi, kw, ks, kf, kb, bt, st, ft, pt, sit, ldic, alk):
    """ This function expresses total alkalinity (TA) as a function of DIC, 
    hSWS (H+ on sea water scale) and constants. It also calculates the 
    derivative of this function with respect to hSWS.
    
    This function was obtained from ModuleCO2System.F90
    
    Function inputs:
        x:           H+ on sea water scale
    
    Function outputs:
        f_TA:        calculated value for TA
        df_TA_dhSWS: derivative of f_TA with respect to hSWS
    """
    
    t1 = 1.0
    t2 = 2.0
    t3 = 3.0
    
    # derive H+ and other constants
    pH_2 = pH*pH
    pH_3 = pH_2*pH
    k12 = k1*k2
    k12p = k1p*k2p
    k123p = k12p*k3p
    c = t1 + st/ks + ft/kf
    a = pH_3 + k1p*pH_2 + k12p*pH + k123p
    a2 = a*a
    da = t3*pH_2 + t2*k1p*pH + k12p
    b = pH_2 + k1*pH + k12
    b2 = b*b
    db = t2*pH + k1
    
    fn = k1*pH*ldic/b + t2*ldic*k12/b + bt/(t1 + pH/kb)
    df = ((k1*ldic*b) - k1*pH*ldic*db)/b2 - t2*ldic*k12*db/b2 - bt/kb/(t1 + pH/kb)**t2
    
    fn += (kw/pH + pt*k12p*pH/a + t2*pt*k123p/a + sit/(t1 + pH/ksi) - pH/c - 
          st/(t1 + ks/(pH/c)) - ft/(t1 + kf/(pH/c)) - pt*pH_3/a - alk)
    
    df += (-kw/pH_2 + (pt*k12p*(a - pH*da))/a2 - t2*pt*k123p*da/a2 - 
          sit/ksi/(t1 + pH/ksi)**t2 - t1/c - st*(t1 + ks/(pH/c))**(-t2)*(ks*c/pH_2) - 
          ft*(t1 + kf/(pH/c))**(-t2)*(kf*c/pH_2) - pt*pH_2*(t3*a - pH*da)/a2)
    
    return fn, df

def find_roots_of_f_TA(x1, x2, xacc, maxit, k1, k2, k1p, k2p, k3p, ksi, kw, ks, kf, kb, bt, st, ft, pt, sit, ldic, alk):
    """ This function finds the roots of the total alkalinity function
    
    This function was obtained from ModuleCO2System.F90
    
    Function inputs:
        x1
        x2
        xacc:           Accuracy of the iterative scheme for OCMIP
        maxit:          Maximum number of iterations for OCMIP
    
    Function outputs:
        error
    """
    
    error = 0
    fl, df = calculate_Hplus(x1, k1, k2, k1p, k2p, k3p, ksi, kw, ks, kf, kb, bt, st, ft, pt, sit, ldic, alk)
    fh, df = calculate_Hplus(x2, k1, k2, k1p, k2p, k3p, ksi, kw, ks, kf, kb, bt, st, ft, pt, sit, ldic, alk)
    if fl==0:
        drtsafe2 = x1
        error = 1
    elif fh ==0:
        drtsafe2 = x2
        error = 1
    elif fl<0:
        xl = x1
        xh = x2
    else:
        xh = x1
        xl = x2
        swap = fl
        fl = fh
        fh = swap

    drtsafe2 = 0.5*(x1 + x2)
    dxold = numpy.abs(x2 - x1)
    dx = dxold
    f, df = calculate_Hplus(drtsafe2, k1, k2, k1p, k2p, k3p, ksi, kw, ks, kf, kb, bt, st, ft, pt, sit, ldic, alk)
    
    j = 0
    ready = False
    
    while True:
        j+=1
        if ((drtsafe2 - xh)*df - f)*((drtsafe2 - xl)*df - f) >= 0 or numpy.abs(2.0*f) > numpy.abs(dxold*df):
            dxold = dx
            dx = 0.5*(xh - xl)
            drtsafe2 = xl + dx
            ready = (xl == drtsafe2)
        else:
            dxold = dx
            dx = f/df
            temp = drtsafe2
            drtsafe2 = drtsafe2 - dx
            ready = (temp == drtsafe2)
        ready = numpy.abs(dx)<xacc
        if not ready:
            f, df = calculate_Hplus(drtsafe2, k1, k2, k1p, k2p, k3p, ksi, kw, ks, kf, kb, bt, st, ft, pt, sit, ldic, alk)
            if f<0:
                xl = drtsafe2
                fl = f
            else:
                xh = drtsafe2
                fh = f
        if ready and j<maxit:
            break
    if j>maxit:
        error = 2
    
    return drtsafe2, error
