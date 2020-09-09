from PythonBFM.other_functions import eTq_vector, get_concentration_ratio

def get_mesozoo_predation_terms(conc, mesozoo3_parameters, mesozoo4_parameters, zoo_availability_parameters, environmental_parameters, constant_parameters, temp):
    """ Calculates the predation terms for mesozooplankton """
    
    # Species concentrations
    p1c = conc[10]             # Diatoms carbon (mg C m^-3)
    p1n = conc[11]             # Diatoms nitrogen (mmol N m^-3)
    p1p = conc[12]             # Diatoms phosphate (mmol P m^-3)
    p1l = conc[13]             # Diatoms chlorophyll (mg Chl-a m^-3)
    p1s = conc[14]             # Diatoms silicate (mmol Si m^-3) 
    p2c = conc[15]             # NanoFlagellates carbon (mg C m^-3)
    p2n = conc[16]             # NanoFlagellates nitrogen (mmol N m^-3)
    p2p = conc[17]             # NanoFlagellates phosphate (mmol P m^-3)
    p2l = conc[18]             # NanoFlagellates chlorophyll (mg Chl-a m^-3)
    p3c = conc[19]             # Picophytoplankton carbon (mg C m^-3)
    p3n = conc[20]             # Picophytoplankton nitrogen (mmol N m^-3)
    p3p = conc[21]             # Picophytoplankton phosphate (mmol P m^-3)
    p3l = conc[22]             # Picophytoplankton chlorophyll (mg Chl-a m^-3)
    p4c = conc[23]             # Large phytoplankton carbon (mg C m^-3)
    p4n = conc[24]             # Large phytoplankton nitrogen (mmol N m^-3)
    p4p = conc[25]             # Large phytoplankton phosphate (mmol P m^-3) 
    p4l = conc[26]             # Large phytoplankton chlorophyll (mg Chl-a m^-3)
    z3c = conc[27]             # Carnivorous mesozooplankton carbon (mg C m^-3)
    z3n = conc[28]             # Carnivorous mesozooplankton nitrogen (mmol N m^-3)
    z3p = conc[29]             # Carnivorous mesozooplankton phosphate (mmol P m^-3)
    z4c = conc[30]             # Omnivorous mesozooplankton carbon (mg C m^-3)
    z4n = conc[31]             # Omnivorous mesozooplankton nitrogen (mmol N m^-3)
    z4p = conc[32]             # Omnivorous mesozooplankton phosphate (mmol P m^-3)
    z5c = conc[33]             # Microzooplankton carbon (mg C m^-3)
    z5n = conc[34]             # Microzooplankton nitrogen (mmol N m^-3)
    z5p = conc[35]             # Microzooplankton phosphate (mmol P m^-3)
    z6c = conc[36]             # Heterotrophic flagellates carbon (mg C m^-3)
    z6n = conc[37]             # Heterotrophic flagellates nitrogen (mmol N m^-3)
    z6p = conc[38]             # Heterotrophic flagellates phosphate (mmol P m^-3)
    
    # concentration ratios
    conc_ratio_n = {
            "p1": get_concentration_ratio(p1n, p1c, constant_parameters["p_small"]),
            "p2": get_concentration_ratio(p2n, p2c, constant_parameters["p_small"]),
            "p3": get_concentration_ratio(p3n, p3c, constant_parameters["p_small"]),
            "p4": get_concentration_ratio(p4n, p4c, constant_parameters["p_small"]),
            "z3": get_concentration_ratio(z3n, z3c, constant_parameters["p_small"]),
            "z4": get_concentration_ratio(z4n, z4c, constant_parameters["p_small"]),
            "z5": get_concentration_ratio(z5n, z5c, constant_parameters["p_small"]),
            "z6": get_concentration_ratio(z6n, z6c, constant_parameters["p_small"])
            }
    conc_ratio_p = {
            "p1": get_concentration_ratio(p1p, p1c, constant_parameters["p_small"]),
            "p2": get_concentration_ratio(p2p, p2c, constant_parameters["p_small"]),
            "p3": get_concentration_ratio(p3p, p3c, constant_parameters["p_small"]),
            "p4": get_concentration_ratio(p4p, p4c, constant_parameters["p_small"]),
            "z3": get_concentration_ratio(z3p, z3c, constant_parameters["p_small"]),
            "z4": get_concentration_ratio(z4p, z4c, constant_parameters["p_small"]),
            "z5": get_concentration_ratio(z5p, z5c, constant_parameters["p_small"]),
            "z6": get_concentration_ratio(z6p, z6c, constant_parameters["p_small"])
            }
    
    # Zooplankton temperature regulating factor
    fTZ = eTq_vector(temp, environmental_parameters["basetemp"], environmental_parameters["q10z"])
    
    # Calculate total potential food given the non-dim prey availability
    # There is no parameter for capture efficiency in mesozooplankton
    # From MesoZoo.F90 lines 247-259
    # Phytoplankton LFG: Food availability of prey Phytoplankton for predator Z3 and Z4
    available_phyto_c3 = (zoo_availability_parameters["del_z3p1"]*p1c) + (zoo_availability_parameters["del_z3p2"]*p2c) + (zoo_availability_parameters["del_z3p3"]*p3c) + (zoo_availability_parameters["del_z3p4"]*p4c)
    available_phyto_c4 = (zoo_availability_parameters["del_z4p1"]*p1c) + (zoo_availability_parameters["del_z4p2"]*p2c) + (zoo_availability_parameters["del_z4p3"]*p3c) + (zoo_availability_parameters["del_z4p4"]*p4c)
    
    # Mesozooplankton LFG
    available_mesozoo_c3 = (zoo_availability_parameters["del_z3z3"]*z3c) + (zoo_availability_parameters["del_z3z4"]*z4c)
    available_mesozoo_c4 = (zoo_availability_parameters["del_z4z3"]*z3c) + (zoo_availability_parameters["del_z4z4"]*z4c)
    
    # Microzooplankton LFG
    available_microzoo_c3 = (zoo_availability_parameters["del_z3z5"]*z5c) + (zoo_availability_parameters["del_z3z6"]*z6c)
    available_microzoo_c4 = (zoo_availability_parameters["del_z4z5"]*z5c) + (zoo_availability_parameters["del_z4z6"]*z6c)
#    sys.exit(available_microzoo_c4)
    
    # Total potential food (from Meso.F90 'rumc')
    f_c3 = available_phyto_c3 + available_mesozoo_c3 + available_microzoo_c3
    f_c4 = available_phyto_c4 + available_mesozoo_c4 + available_microzoo_c4
    
    # Calculate total food uptake rate (from Meso.F90 'rugc')
    total_uptake_rate_z3 = fTZ*mesozoo3_parameters["r_Z0"]*(mesozoo3_parameters["nu_z"]*f_c3/((mesozoo3_parameters["nu_z"]*f_c3) + mesozoo3_parameters["r_Z0"]))*z3c
    total_uptake_rate_z4 = fTZ*mesozoo4_parameters["r_Z0"]*(mesozoo4_parameters["nu_z"]*f_c4/((mesozoo4_parameters["nu_z"]*f_c4) + mesozoo4_parameters["r_Z0"]))*z4c
    
    # Calculate specific uptake rate considering potentially available food (from Meso.F90 'sut')
    specific_uptake_rate_z3 = total_uptake_rate_z3/(constant_parameters["p_small"] + f_c3)
    specific_uptake_rate_z4 = total_uptake_rate_z4/(constant_parameters["p_small"] + f_c4)

    # Total Gross Uptakes from every LFG
    dZ3cdt_prd = {
            "p1": specific_uptake_rate_z3*zoo_availability_parameters["del_z3p1"]*p1c,
            "p2": specific_uptake_rate_z3*zoo_availability_parameters["del_z3p2"]*p2c,
            "p3": specific_uptake_rate_z3*zoo_availability_parameters["del_z3p3"]*p3c,
            "p4": specific_uptake_rate_z3*zoo_availability_parameters["del_z3p4"]*p4c,
            "z3": specific_uptake_rate_z3*zoo_availability_parameters["del_z3z3"]*z3c,
            "z4": specific_uptake_rate_z3*zoo_availability_parameters["del_z3z4"]*z4c,
            "z5": specific_uptake_rate_z3*zoo_availability_parameters["del_z3z5"]*z5c,
            "z6": specific_uptake_rate_z3*zoo_availability_parameters["del_z3z6"]*z6c
            }

    dZ4cdt_prd = {
            "p1": specific_uptake_rate_z4*zoo_availability_parameters["del_z4p1"]*p1c,
            "p2": specific_uptake_rate_z4*zoo_availability_parameters["del_z4p2"]*p2c,
            "p3": specific_uptake_rate_z4*zoo_availability_parameters["del_z4p3"]*p3c,
            "p4": specific_uptake_rate_z4*zoo_availability_parameters["del_z4p4"]*p4c,
            "z3": specific_uptake_rate_z4*zoo_availability_parameters["del_z4z3"]*z3c,
            "z4": specific_uptake_rate_z4*zoo_availability_parameters["del_z4z4"]*z4c,
            "z5": specific_uptake_rate_z4*zoo_availability_parameters["del_z4z5"]*z5c,
            "z6": specific_uptake_rate_z4*zoo_availability_parameters["del_z4z6"]*z6c
            }
    # Total ingestion rate
    ic3 = 0.0
    in3 = 0.0
    ip3 = 0.0
    
    for key in dZ3cdt_prd:
        ic3 += dZ3cdt_prd[key]
        in3 += dZ3cdt_prd[key]*conc_ratio_n[key]
        ip3 += dZ3cdt_prd[key]*conc_ratio_p[key]
    
    ic4 = 0.0
    in4 = 0.0
    ip4 = 0.0
    
    for key in dZ4cdt_prd:
        ic4 += dZ4cdt_prd[key]
        in4 += dZ4cdt_prd[key]*conc_ratio_n[key]
        ip4 += dZ4cdt_prd[key]*conc_ratio_p[key]
  
    return dZ3cdt_prd, dZ4cdt_prd, ic3, in3, ip3, ic4, in4, ip4


def get_microzoo_predation_terms(conc, microzoo5_parameters, microzoo6_parameters, zoo_availability_parameters, environmental_parameters, constant_parameters, temp):
    """ Calculates the predation terms for microzooplankton """
    
    # Species concentrations
    b1c = conc[7]              # Pelagic bacteria carbon (mg C m^-3)
    b1n = conc[8]              # Pelagic bacteria nitrogen (mmol N m^-3)
    b1p = conc[9]              # Pelagic bacteria phosphate (mmol P m^-3)
    p1c = conc[10]             # Diatoms carbon (mg C m^-3)
    p1n = conc[11]             # Diatoms nitrogen (mmol N m^-3)
    p1p = conc[12]             # Diatoms phosphate (mmol P m^-3)
    p1l = conc[13]             # Diatoms chlorophyll (mg Chl-a m^-3)
    p1s = conc[14]             # Diatoms silicate (mmol Si m^-3) 
    p2c = conc[15]             # NanoFlagellates carbon (mg C m^-3)
    p2n = conc[16]             # NanoFlagellates nitrogen (mmol N m^-3)
    p2p = conc[17]             # NanoFlagellates phosphate (mmol P m^-3)
    p2l = conc[18]             # NanoFlagellates chlorophyll (mg Chl-a m^-3)
    p3c = conc[19]             # Picophytoplankton carbon (mg C m^-3)
    p3n = conc[20]             # Picophytoplankton nitrogen (mmol N m^-3)
    p3p = conc[21]             # Picophytoplankton phosphate (mmol P m^-3)
    p3l = conc[22]             # Picophytoplankton chlorophyll (mg Chl-a m^-3)
    p4c = conc[23]             # Large phytoplankton carbon (mg C m^-3)
    p4n = conc[24]             # Large phytoplankton nitrogen (mmol N m^-3)
    p4p = conc[25]             # Large phytoplankton phosphate (mmol P m^-3) 
    p4l = conc[26]             # Large phytoplankton chlorophyll (mg Chl-a m^-3)
    z3c = conc[27]             # Carnivorous mesozooplankton carbon (mg C m^-3)
    z3n = conc[28]             # Carnivorous mesozooplankton nitrogen (mmol N m^-3)
    z3p = conc[29]             # Carnivorous mesozooplankton phosphate (mmol P m^-3)
    z4c = conc[30]             # Omnivorous mesozooplankton carbon (mg C m^-3)
    z4n = conc[31]             # Omnivorous mesozooplankton nitrogen (mmol N m^-3)
    z4p = conc[32]             # Omnivorous mesozooplankton phosphate (mmol P m^-3)
    z5c = conc[33]             # Microzooplankton carbon (mg C m^-3)
    z5n = conc[34]             # Microzooplankton nitrogen (mmol N m^-3)
    z5p = conc[35]             # Microzooplankton phosphate (mmol P m^-3)
    z6c = conc[36]             # Heterotrophic flagellates carbon (mg C m^-3)
    z6n = conc[37]             # Heterotrophic flagellates nitrogen (mmol N m^-3)
    z6p = conc[38]             # Heterotrophic flagellates phosphate (mmol P m^-3)
    
    # concentration ratios
    conc_ratio_n = {
            "b1": get_concentration_ratio(b1n, b1c, constant_parameters["p_small"]),
            "p1": get_concentration_ratio(p1n, p1c, constant_parameters["p_small"]),
            "p2": get_concentration_ratio(p2n, p2c, constant_parameters["p_small"]),
            "p3": get_concentration_ratio(p3n, p3c, constant_parameters["p_small"]),
            "p4": get_concentration_ratio(p4n, p4c, constant_parameters["p_small"]),
            "z5": get_concentration_ratio(z5n, z5c, constant_parameters["p_small"]),
            "z6": get_concentration_ratio(z6n, z6c, constant_parameters["p_small"])
            }
    
    conc_ratio_p = {
            "b1": get_concentration_ratio(b1p, b1c, constant_parameters["p_small"]),
            "p1": get_concentration_ratio(p1p, p1c, constant_parameters["p_small"]),
            "p2": get_concentration_ratio(p2p, p2c, constant_parameters["p_small"]),
            "p3": get_concentration_ratio(p3p, p3c, constant_parameters["p_small"]),
            "p4": get_concentration_ratio(p4p, p4c, constant_parameters["p_small"]),
            "z5": get_concentration_ratio(z5p, z5c, constant_parameters["p_small"]),
            "z6": get_concentration_ratio(z6p, z6c, constant_parameters["p_small"])
            }
    
    # Zooplankton temperature regulating factor
    fTZ = eTq_vector(temp, environmental_parameters["basetemp"], environmental_parameters["q10z"])
    
    # Capture efficiencies 
    capture_efficiencies_z5 = {
            "b1": b1c/(b1c + microzoo5_parameters["mu_z"]),
            "p1": p1c/(p1c + microzoo5_parameters["mu_z"]),
            "p2": p2c/(p2c + microzoo5_parameters["mu_z"]),
            "p3": p3c/(p3c + microzoo5_parameters["mu_z"]),
            "p4": p4c/(p4c + microzoo5_parameters["mu_z"]),
            "z5": z5c/(z5c + microzoo5_parameters["mu_z"]),
            "z6": z6c/(z6c + microzoo5_parameters["mu_z"])
            }
    
    capture_efficiencies_z6 = {
            "b1": b1c/(b1c + microzoo6_parameters["mu_z"]),
            "p1": p1c/(p1c + microzoo6_parameters["mu_z"]),
            "p2": p2c/(p2c + microzoo6_parameters["mu_z"]),
            "p3": p3c/(p3c + microzoo6_parameters["mu_z"]),
            "p4": p4c/(p4c + microzoo6_parameters["mu_z"]),
            "z5": z5c/(z5c + microzoo6_parameters["mu_z"]),
            "z6": z6c/(z6c + microzoo6_parameters["mu_z"])
            }
    
    # Calculate total potential food given the non-dim prey availability and capture efficiency
    # From MicroZoo.F90 lines 209-237
    # Bacteria LFG: Food availability of prey Bacteria for predator Z5 and Z6
    available_bact_c5 = zoo_availability_parameters["del_z5b1"]*b1c*capture_efficiencies_z5["b1"]
    available_bact_n5 = zoo_availability_parameters["del_z5b1"]*b1c*capture_efficiencies_z5["b1"]*conc_ratio_n["b1"]
    available_bact_p5 = zoo_availability_parameters["del_z5b1"]*b1c*capture_efficiencies_z5["b1"]*conc_ratio_p["b1"]
    available_bact_c6 = zoo_availability_parameters["del_z6b1"]*b1c*capture_efficiencies_z6["b1"]
    available_bact_n6 = zoo_availability_parameters["del_z6b1"]*b1c*capture_efficiencies_z6["b1"]*conc_ratio_n["b1"]
    available_bact_p6 = zoo_availability_parameters["del_z6b1"]*b1c*capture_efficiencies_z6["b1"]*conc_ratio_p["b1"]
    
    # Phytoplankton LFG: Food availability of prey Phytoplankton for predator Z5 and Z6
    available_phyto_c5 = ((zoo_availability_parameters["del_z5p1"]*p1c*capture_efficiencies_z5["p1"]) + (zoo_availability_parameters["del_z3p2"]*p2c*capture_efficiencies_z5["p2"]) + 
                          (zoo_availability_parameters["del_z3p3"]*p3c*capture_efficiencies_z5["p3"]) + (zoo_availability_parameters["del_z3p4"]*p4c*capture_efficiencies_z5["p4"]))
    available_phyto_n5 = ((zoo_availability_parameters["del_z5p1"]*p1c*capture_efficiencies_z5["p1"]*conc_ratio_n["p1"]) + (zoo_availability_parameters["del_z3p2"]*p2c*capture_efficiencies_z5["p2"]*conc_ratio_n["p2"]) + 
                          (zoo_availability_parameters["del_z3p3"]*p3c*capture_efficiencies_z5["p3"]*conc_ratio_n["p3"]) + (zoo_availability_parameters["del_z3p4"]*p4c*capture_efficiencies_z5["p4"]*conc_ratio_n["p4"]))
    available_phyto_p5 = ((zoo_availability_parameters["del_z5p1"]*p1c*capture_efficiencies_z5["p1"]*conc_ratio_p["p1"]) + (zoo_availability_parameters["del_z3p2"]*p2c*capture_efficiencies_z5["p2"]*conc_ratio_p["p2"]) + 
                          (zoo_availability_parameters["del_z3p3"]*p3c*capture_efficiencies_z5["p3"]*conc_ratio_p["p3"]) + (zoo_availability_parameters["del_z3p4"]*p4c*capture_efficiencies_z5["p4"]*conc_ratio_p["p4"]))
    available_phyto_c6 = ((zoo_availability_parameters["del_z6p1"]*p1c*capture_efficiencies_z6["p1"]) + (zoo_availability_parameters["del_z6p2"]*p2c*capture_efficiencies_z6["p2"]) + 
                          (zoo_availability_parameters["del_z6p3"]*p3c*capture_efficiencies_z6["p3"]) + (zoo_availability_parameters["del_z6p4"]*p4c*capture_efficiencies_z6["p4"]))
    available_phyto_n6 = ((zoo_availability_parameters["del_z6p1"]*p1c*capture_efficiencies_z6["p1"]*conc_ratio_n["p1"]) + (zoo_availability_parameters["del_z6p2"]*p2c*capture_efficiencies_z6["p2"]*conc_ratio_n["p2"]) + 
                          (zoo_availability_parameters["del_z6p3"]*p3c*capture_efficiencies_z6["p3"]*conc_ratio_n["p3"]) + (zoo_availability_parameters["del_z6p4"]*p4c*capture_efficiencies_z6["p4"]*conc_ratio_n["p4"]))
    available_phyto_p6 = ((zoo_availability_parameters["del_z6p1"]*p1c*capture_efficiencies_z6["p1"]*conc_ratio_p["p1"]) + (zoo_availability_parameters["del_z6p2"]*p2c*capture_efficiencies_z6["p2"]*conc_ratio_p["p2"]) + 
                          (zoo_availability_parameters["del_z6p3"]*p3c*capture_efficiencies_z6["p3"]*conc_ratio_p["p3"]) + (zoo_availability_parameters["del_z6p4"]*p4c*capture_efficiencies_z6["p4"]*conc_ratio_p["p4"]))
    
    # Phytoplankton LFG: Food availability of prey Microzooplankton for predator Z5 and Z6
    available_microzoo_c5 = (zoo_availability_parameters["del_z5z5"]*z5c*capture_efficiencies_z5["z5"]) + (zoo_availability_parameters["del_z5z6"]*z6c*capture_efficiencies_z5["z6"])
    available_microzoo_n5 = (zoo_availability_parameters["del_z5z5"]*z5c*capture_efficiencies_z5["z5"]*conc_ratio_n["z5"]) + (zoo_availability_parameters["del_z5z6"]*z6c*capture_efficiencies_z5["z6"]*conc_ratio_n["z6"])
    available_microzoo_p5 = (zoo_availability_parameters["del_z5z5"]*z5c*capture_efficiencies_z5["z5"]*conc_ratio_p["z5"]) + (zoo_availability_parameters["del_z5z6"]*z6c*capture_efficiencies_z5["z6"]*conc_ratio_p["z6"])
    
    available_microzoo_c6 = (zoo_availability_parameters["del_z6z5"]*z5c*capture_efficiencies_z6["z5"]) + (zoo_availability_parameters["del_z6z6"]*z6c*capture_efficiencies_z6["z6"])
    available_microzoo_n6 = (zoo_availability_parameters["del_z6z5"]*z5c*capture_efficiencies_z6["z5"]**conc_ratio_n["z5"]) + (zoo_availability_parameters["del_z6z6"]*z6c*capture_efficiencies_z6["z6"]**conc_ratio_n["z6"])
    available_microzoo_p6 = (zoo_availability_parameters["del_z6z5"]*z5c*capture_efficiencies_z6["z5"]**conc_ratio_p["z5"]) + (zoo_availability_parameters["del_z6z6"]*z6c*capture_efficiencies_z6["z6"]**conc_ratio_p["z6"])
    
    # Total potential food (from MicroZoo.F90 'rumc', 'rumn', 'rump')
    f_c5 = available_bact_c5 + available_phyto_c5 + available_microzoo_c5
    f_n5 = available_bact_n5 + available_phyto_n5 + available_microzoo_n5
    f_p5 = available_bact_p5 + available_phyto_p5 + available_microzoo_p5
    f_c6 = available_bact_c6 + available_phyto_c6 + available_microzoo_c6
    f_n6 = available_bact_n6 + available_phyto_n6 + available_microzoo_n6
    f_p6 = available_bact_p6 + available_phyto_p6 + available_microzoo_p6
    
    # Calculate total food uptake rate (from MicroZoo.F90 line 243 'rugc')
    total_uptake_rate_z5 = fTZ*microzoo5_parameters["r_Z0"]*(f_c5/(f_c5 + microzoo5_parameters["h_Z_F"]))*z5c
    total_uptake_rate_z6 = fTZ*microzoo6_parameters["r_Z0"]*(f_c6/(f_c6 + microzoo6_parameters["h_Z_F"]))*z6c
    
    # Calculate specific uptake rate considering potentially available food (from MicroZoo.F90 line 244 'sut')
    specific_uptake_rate_z5 = total_uptake_rate_z5/(f_c5 + constant_parameters["p_small"])
    specific_uptake_rate_z6 = total_uptake_rate_z6/(f_c6 + constant_parameters["p_small"])
    
    # Total Gross Uptakes from every LFG
    dZ5cdt_prd = {
            "b1": specific_uptake_rate_z5*zoo_availability_parameters["del_z5b1"]*b1c*capture_efficiencies_z5["b1"],
            "p1": specific_uptake_rate_z5*zoo_availability_parameters["del_z5p1"]*p1c*capture_efficiencies_z5["p1"],
            "p2": specific_uptake_rate_z5*zoo_availability_parameters["del_z5p2"]*p2c*capture_efficiencies_z5["p2"],
            "p3": specific_uptake_rate_z5*zoo_availability_parameters["del_z5p3"]*p3c*capture_efficiencies_z5["p3"],
            "p4": specific_uptake_rate_z5*zoo_availability_parameters["del_z5p4"]*p4c*capture_efficiencies_z5["p4"],
            "z5": specific_uptake_rate_z5*zoo_availability_parameters["del_z5z5"]*z5c*capture_efficiencies_z5["z5"],
            "z6": specific_uptake_rate_z5*zoo_availability_parameters["del_z5z6"]*z6c*capture_efficiencies_z5["z6"]
            }
        
    dZ6cdt_prd = {
            "b1": specific_uptake_rate_z6*zoo_availability_parameters["del_z6b1"]*b1c*capture_efficiencies_z6["b1"],
            "p1": specific_uptake_rate_z6*zoo_availability_parameters["del_z6p1"]*p1c*capture_efficiencies_z6["p1"],
            "p2": specific_uptake_rate_z6*zoo_availability_parameters["del_z6p2"]*p2c*capture_efficiencies_z6["p2"],
            "p3": specific_uptake_rate_z6*zoo_availability_parameters["del_z6p3"]*p3c*capture_efficiencies_z6["p3"],
            "p4": specific_uptake_rate_z6*zoo_availability_parameters["del_z6p4"]*p4c*capture_efficiencies_z6["p4"],
            "z5": specific_uptake_rate_z6*zoo_availability_parameters["del_z6z5"]*z5c*capture_efficiencies_z6["z5"],
            "z6": specific_uptake_rate_z6*zoo_availability_parameters["del_z6z6"]*z6c*capture_efficiencies_z6["z6"]
            }
    
    # Total ingestion rate
    ic5 = 0.0
    in5 = 0.0
    ip5 = 0.0
    
    for key in dZ5cdt_prd:
        ic5 += dZ5cdt_prd[key]
        in5 += dZ5cdt_prd[key]*conc_ratio_n[key]
        ip5 += dZ5cdt_prd[key]*conc_ratio_p[key]
    
    ic6 = 0.0
    in6 = 0.0
    ip6 = 0.0
    
    for key in dZ6cdt_prd:
        ic6 += dZ6cdt_prd[key]
        in6 += dZ6cdt_prd[key]*conc_ratio_n[key]
        ip6 += dZ6cdt_prd[key]*conc_ratio_p[key]
    
    return dZ5cdt_prd, dZ6cdt_prd, ic5, in5, ip5, ic6, in6, ip6
