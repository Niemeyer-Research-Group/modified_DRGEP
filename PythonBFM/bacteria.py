import sys
from PythonBFM.other_functions import eTq_vector, get_concentration_ratio

def bacteria_eqns(conc, bacteria_parameters, constant_parameters, environmental_parameters, temper):
    """ Calculates the terms needed for the bacteria biological rate equations
        Equations come from the BFM user manual
    """
    
    # State variables
    o2o = conc[0]              # Dissolved oxygen (mg O_2 m^-3)
    n1p = conc[1]              # Phosphate (mmol P m^-3)
    n4n = conc[3]              # Ammonium (mmol N m^-3)
    b1c = conc[7]              # Pelagic bacteria carbon (mg C m^-3)
    b1n = conc[8]              # Pelagic bacteria nitrogen (mmol N m^-3)
    b1p = conc[9]              # Pelagic bacteria phosphate (mmol P m^-3)
    r1c = conc[39]             # Labile dissolved organic carbon (mg C m^-3)
    r1n = conc[40]             # Labile dissolved organic nitrogen (mmol N m^-3)
    r1p = conc[41]             # Labile dissolved organic phosphate (mmol P m^-3)
    r2c = conc[42]             # Semi-labile dissolved organic carbon (mg C m^-3)
    r3c = conc[43]             # Semi-refractory Dissolved Organic Carbon (mg C m^-3)
    r6c = conc[44]             # Particulate organic carbon (mg C m^-3)
    r6n = conc[45]             # Particulate organic nitrogen (mmol N m^-3)
    r6p = conc[46]             # Particulate organic phosphate (mmol P m^-3)
    
    # concentration ratios   
    bp_bc = get_concentration_ratio(b1p, b1c, constant_parameters["p_small"])
    bn_bc = get_concentration_ratio(b1n, b1c, constant_parameters["p_small"])
    r1p_r1c = get_concentration_ratio(r1p, r1c, constant_parameters["p_small"])
    r6p_r6c = get_concentration_ratio(r6p, r6c, constant_parameters["p_small"])
    r1n_r1c = get_concentration_ratio(r1n, r1c, constant_parameters["p_small"])
    r6n_r6c = get_concentration_ratio(r6n, r6c, constant_parameters["p_small"])
    
    # Temperature effect on pelagic bacteria
    fTB = eTq_vector(temper, environmental_parameters["basetemp"], environmental_parameters["q10b"])

    # oxygen non-dimensional regulation factor[-]
    # Oxygen environment: bacteria are both aerobic and anaerobic
    f_B_O = max(constant_parameters["p_small"],o2o)**3/(max(constant_parameters["p_small"],o2o)**3 + bacteria_parameters["h_B_O"]**3)
    
    # external nutrient limitation
    f_B_n = n4n/(n4n + bacteria_parameters["h_B_n"])
    f_B_p = n1p/(n1p + bacteria_parameters["h_B_p"])
    
    # Bacteria mortality (lysis) process [mg C m^-3 s^-1]
    dBcdt_lys = (bacteria_parameters["d_0B"]*fTB + bacteria_parameters["d_B_d"]*b1c)*b1c
    dBcdt_lys_r1c = dBcdt_lys*constant_parameters["epsilon_c"]
    dBcdt_lys_r1n = dBcdt_lys*bn_bc*constant_parameters["epsilon_n"]
    dBcdt_lys_r1p = dBcdt_lys*bp_bc*constant_parameters["epsilon_p"]
    dBcdt_lys_r6c = dBcdt_lys*(1.0 - constant_parameters["epsilon_c"])
    dBcdt_lys_r6n = dBcdt_lys*bn_bc*(1.0 - constant_parameters["epsilon_n"])
    dBcdt_lys_r6p = dBcdt_lys*bp_bc*(1.0 - constant_parameters["epsilon_p"])


    # Substrate availability
    if bacteria_parameters["bact_version"]==1 or bacteria_parameters["bact_version"]==2:
        # nutrient limitation (intracellular)
        nut_lim_n = min(1.0, max(0.0, bn_bc/bacteria_parameters["n_B_opt"]))         # Nitrogen
        nut_lim_p = min(1.0, max(0.0, bp_bc/bacteria_parameters["p_B_opt"]))         # Phosphorus
        f_B_n_P = min(nut_lim_n, nut_lim_p)
        
        # Potential uptake by bacteria
        potential_upt = f_B_n_P*fTB*bacteria_parameters["r_0B"]*b1c
        
        # correction of substrate quality depending on nutrient content
        f_r1_n_P = min(1.0, r1p_r1c/bacteria_parameters["p_B_opt"], r1n_r1c/bacteria_parameters["n_B_opt"])
        f_r6_n_P = min(1.0, r6p_r6c/bacteria_parameters["p_B_opt"], r6n_r6c/bacteria_parameters["n_B_opt"])
    else:
        sys.exit('This code does not support this parameterization option, only bact_version=1')
        
    # Calculate the realized substrate uptake rate depending on the type of detritus and quality
    upt_R1c = (bacteria_parameters["v_B_r1"]*f_r1_n_P + bacteria_parameters["v_0B_r1"]*(1.0 - f_r1_n_P))*r1c
    upt_R2c = bacteria_parameters["v_B_r2"]*r2c
    upt_R3c = bacteria_parameters["v_B_r3"]*r3c
    upt_R6c = bacteria_parameters["v_B_r6"]*f_r6_n_P*r6c
    realized_upt = constant_parameters["p_small"] + upt_R1c + upt_R2c + upt_R3c + upt_R6c
    
    # Actual uptake by bacteria
    actual_upt = min(potential_upt, realized_upt)
    
    # Carbon fluxes into bacteria
    dBcdt_upt_r1c = actual_upt*upt_R1c/realized_upt
    dBcdt_upt_r2c = actual_upt*upt_R2c/realized_upt
    dBcdt_upt_r3c = actual_upt*upt_R3c/realized_upt
    dBcdt_upt_r6c = actual_upt*upt_R6c/realized_upt
    
    # Organic Nitrogen and Phosphrous uptake
    dBcdt_upt_r1n = r1n_r1c*dBcdt_upt_r1c
    dBcdt_upt_r6n = r6n_r6c*dBcdt_upt_r6c
    dBcdt_upt_r1p = r1p_r1c*dBcdt_upt_r1c
    dBcdt_upt_r6p = r6p_r6c*dBcdt_upt_r6c
    
    # Bacteria respiration [mc C m^-3 s^-1]
    dBcdt_rsp_o3c = (bacteria_parameters["gamma_B_a"] + bacteria_parameters["gamma_B_O"]*(1.0 - f_B_O))*actual_upt + bacteria_parameters["b_B"]*b1c*fTB

    # Fluxes from bacteria
    if bacteria_parameters["bact_version"]==1:
        
        # There is no Carbon excretion
        dBcdt_rel_r2c = 0.0
        dBcdt_rel_r3c = 0.0
        
        # Dissolved Nitrogen dynamics
        dBndt_upt_rel_n4n = (bn_bc - bacteria_parameters["n_B_opt"])*b1c*bacteria_parameters["v_B_n"]
            
        # Dissolved Phosphorus dynamics
        dBpdt_upt_rel_n1p = (bp_bc - bacteria_parameters["p_B_opt"])*b1c*bacteria_parameters["v_B_p"]

    # BACT2 parameterization
    if bacteria_parameters["bact_version"]==2:
        print('This code does not support this parameterization option, only bact_version=1')
        
    # BACT3 parameterization
    if bacteria_parameters["bact_version"]==3:
        print('This code does not support this parameterization option, only bact_version=1')

    # Term needed for denitrification flux (dn3ndt_denit) (from PelBac.F90 line 352)
    flPTN6r = (1.0 - f_B_O)*dBcdt_rsp_o3c*constant_parameters["omega_c"]*constant_parameters["omega_r"]
    
    return (dBcdt_lys_r1c, dBcdt_lys_r1n, dBcdt_lys_r1p, dBcdt_lys_r6c, dBcdt_lys_r6n, dBcdt_lys_r6p, 
            dBcdt_upt_r1c, dBcdt_upt_r6c, dBpdt_upt_rel_n1p, dBndt_upt_rel_n4n, dBcdt_upt_r2c, dBcdt_upt_r3c, 
            dBcdt_rel_r2c, dBcdt_rel_r3c, dBcdt_rsp_o3c, flPTN6r, f_B_O, f_B_n, f_B_p)
    
