import numpy

def calculate_oxygen_reaeration(oxygen_reaeration_parameters, environmental_parameters, constant_parameters, conc, temper, salt, wind):
    """ calculates the oxygen reaeration between air and water column, 
    as forced by temperature and wind.
    """
    
    # State variables
    o2o = conc[0]              # Dissolved oxygen (mg O_2 m^-3)
    
    # calc absolute temperature divided by 100.0 (input for the next empirical equation)
    abt = (temper + constant_parameters["c_to_kelvin"])/100.0
    
    # calc theoretical oxygen saturation for temp + salinity and convert into proper units of [mmol O_2/m^3]
    #  From WEISS 1970 DEEP SEA RES 17, 721-735
    oxy_sat = numpy.exp(-173.4292 + (249.6339/abt) + (143.3483*numpy.log(abt))-(21.8492*abt) + salt*(-0.033096 + 0.014259*abt - 0.0017*(abt**2)))*44.661

    # Calculate Schmidt number, ratio between the kinematic viscosity and the molecular diffusivity of carbon dioxide
    schmidt_number_o2o = (oxygen_reaeration_parameters["k1"] - oxygen_reaeration_parameters["k2"]*temper + oxygen_reaeration_parameters["k3"]*(temper**2) - oxygen_reaeration_parameters["k4"]*(temper**3))
    schmidt_ratio_o2o = oxygen_reaeration_parameters["schmidt_o2o"]/schmidt_number_o2o
    
    # schmidt_ratio is limited to 0 when T > 40 °C 
    if schmidt_ratio_o2o<0.0:
        schmidt_ratio_o2o = 0.0

    # Calculate wind dependency, including conversion cm/hr => m/s
    wind_dependency = (oxygen_reaeration_parameters["d"]*(wind**2))*numpy.sqrt(schmidt_ratio_o2o)*constant_parameters["cm2m"]*constant_parameters["hours_per_day"]/constant_parameters["sec_per_day"]
    
    # flux o2 [mmol m^-2 s^-1]
    dOdt_wind = wind_dependency*(oxy_sat - o2o)/environmental_parameters["del_z"]
    
    return dOdt_wind