from PythonBFM.other_functions import eTq_vector

def pel_chem_eqns(pel_chem_parameters, environmental_parameters, constant_parameters, temper, conc, flPTN6r):
    """ calculates the non-living equations for DOM, POM, and nutrients """
    
    # State variables
    o2o = conc[0]              # Dissolved oxygen (mg O_2 m^-3)
    n3n = conc[2]              # Nitrate (mmol N m^-3)
    n4n = conc[3]              # Ammonium (mmol N m^-3)
    n6r = conc[6]              # Reduction equivalents (mmol S m^-3)
    r6s = conc[47]             # Particulate organic silicate (mmol Si m^-3)
    
    # Regulating factors
    eo = max(constant_parameters["p_small"], o2o)/(max(constant_parameters["p_small"], o2o)+ pel_chem_parameters["h_o"])
    er = n6r/(n6r + pel_chem_parameters["h_r"])
    
    # Temperature regulating factors
    fTn = eTq_vector(temper, environmental_parameters["basetemp"], environmental_parameters["q10n"])
    fTr6 = eTq_vector(temper, environmental_parameters["basetemp"], environmental_parameters["q10n5"])
    
    # Nitrification in the water  [mmol N m^-3 s^-1]   
    dn4ndt_nit_n3n = max(0.0, pel_chem_parameters["lambda_N4nit"]*n4n*fTn*eo)

    # Denitrification flux [mmol N m^-3 s^-1] from PelChem.F90 line 134
    rPAo = flPTN6r/constant_parameters["omega_r"]
    dn3ndt_denit = max(0.0, pel_chem_parameters["lambda_N3denit"]*fTn*er*rPAo/pel_chem_parameters["m_o"]*n3n)
    
    # Reoxidation of reduction equivalents [mmol S m^-3 s^-1]
    dn6rdt_reox = pel_chem_parameters["lambda_N6reox"]*eo*n6r
    
    # Dissolution of biogenic silicate [mmol Si m^-3 s^-1]
    dr6sdt_rmn_n5s = pel_chem_parameters["lambda_srmn"]*fTr6*r6s
    
    return (dn4ndt_nit_n3n, dn3ndt_denit, dn6rdt_reox, dr6sdt_rmn_n5s)
