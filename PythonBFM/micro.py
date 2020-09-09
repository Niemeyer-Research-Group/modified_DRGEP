from PythonBFM.other_functions import eTq_vector, get_concentration_ratio

def microzoo_eqns(conc, microzoo_parameters, constant_parameters, environmental_parameters, zc, zn, zp, i_c, i_n, i_p, temp):
    """ Calculates the micorzooplankton (Z5 & Z6) terms needed for the zooplankton biological rate equations
        Equations come from the BFM user manual and the fortran code MicroZoo.F90
    """

    # Dissolved oxygen concentration (mg O_2 m^-3)
    o2o = conc[0]
    
    # Concentration ratios
    zn_zc = get_concentration_ratio(zn, zc, constant_parameters["p_small"])
    zp_zc = get_concentration_ratio(zp, zc, constant_parameters["p_small"])

    # Temperature regulating factor
    fTZ = eTq_vector(temp, environmental_parameters["basetemp"], environmental_parameters["q10z"])
    
    # Oxygen dependent regulation factor
    fZO = min(1.0, (o2o/(o2o + microzoo_parameters["z_o2o"])))
    
    #---------------------- Microzooplankton Respiration ----------------------
    # Zooplankton total repiration rate (eqn. 2.4.8, and matches fortran code)
    rrac = i_c*(1.0 - microzoo_parameters["etaZ"] - microzoo_parameters["betaZ"])
    rrsc = microzoo_parameters["bZ"]*fTZ*zc
    dZcdt_rsp_o3c = rrac + rrsc
    
    #------------- Microzooplankton mortality and activity excretion ----------
    # From fortran code MesoZoo.F90 lines 327-331
    rdc = ((1.0 - fZO)*microzoo_parameters["d_ZO"] + microzoo_parameters["d_Z"])*zc
    reac = i_c*(1.0 - microzoo_parameters["etaZ"])*microzoo_parameters["betaZ"]
    rric = reac + rdc
    dZcdt_rel_r1c = rric*constant_parameters["epsilon_c"]
    dZcdt_rel_r6c = rric*(1.0 - constant_parameters["epsilon_c"])    

    #------------------- Microzooplankton nutrient dynamics -------------------
    # Organic Nitrogen dynamics (from fortran code) [mmol N m^-3 s^-1]
    rrin = i_n*microzoo_parameters["betaZ"] + rdc*zn_zc
    dZndt_rel_r1n = rrin*constant_parameters["epsilon_n"]
    dZndt_rel_r6n = rrin - dZndt_rel_r1n

    # Organic Phosphorus dynamics (from fortran code) [mmol P m^-3 s^-1]
    rrip = i_p*microzoo_parameters["betaZ"] + rdc*zp_zc
    dZpdt_rel_r1p = rrip*constant_parameters["epsilon_p"]
    dZpdt_rel_r6p = rrip - dZpdt_rel_r1p

    #--------------- Microzooplankton Dissolved nutrient dynamics -------------     
    # Equations from fortran code (MicroZoo.F90 line 368-371)
    runc = max(0.0, i_c*(1.0 - microzoo_parameters["betaZ"])-rrac)
    runn = max(0.0, i_n*(1.0 - microzoo_parameters["betaZ"]) + rrsc*zn_zc)
    runp = max(0.0, i_p*(1.0 - microzoo_parameters["betaZ"]) + rrsc*zp_zc)
    dZpdt_rel_n1p = max(0.0, runp/(constant_parameters["p_small"] + runc) - microzoo_parameters["p_Zopt"])*runc
    dZndt_rel_n4n = max(0.0, runn/(constant_parameters["p_small"] + runc) - microzoo_parameters["n_Zopt"])*runc
    
    return dZcdt_rel_r1c, dZcdt_rel_r6c, dZcdt_rsp_o3c, dZndt_rel_r1n, dZndt_rel_r6n, dZpdt_rel_r1p, dZpdt_rel_r6p, dZpdt_rel_n1p, dZndt_rel_n4n
    
