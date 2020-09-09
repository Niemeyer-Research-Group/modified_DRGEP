import numpy
import copy
import sys
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
from PythonBFM.BFM_reduced_rate_eqns import bfm_reduced_rate_eqns

def calc_error_chl_1(species_removed, solution_full_model, t_span, c0):
    """ calculates error in peak chlorophyll concentration during a spring bloom
    This is for the sum of p1l + p2l + p3l + p4l
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of chl-a concentrations (p1l + p2l + p3l + p4l)
    chl_sum_conc_full = solution_full_model.y[13] + solution_full_model.y[18] + solution_full_model.y[22] + solution_full_model.y[26]
    chl_sum_conc_reduced = solution_reduced_model.y[13] + solution_reduced_model.y[18] + solution_reduced_model.y[22] + solution_reduced_model.y[26]

    # Set time range for searching for max value
    # January to March in year 6
    t_min = 86400*365*6
    t_max = t_min + 86400*90
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of chl concentrations
    spring_bloom_full = chl_sum_conc_full[index_t_min_full:index_t_max_full]
    spring_bloom_reduced = chl_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    
    # Find peak chl-a concentration in spring bloom
    spring_bloom_peak_full = max(spring_bloom_full)
    spring_bloom_peak_reduced = max(spring_bloom_reduced)
    
    # Compute the error
    error = 100*numpy.abs(spring_bloom_peak_full - spring_bloom_peak_reduced)/spring_bloom_peak_full
    
    # If peak is zero in reduced model, set error to zero
    if spring_bloom_peak_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_chl_2(species_removed, solution_full_model, t_span, c0):
    """ calculates error in peak chlorophyll concentration during a spring bloom
    This is for the sum of p2l + p3l + p4l
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of chl-a concentrations (p1l + p2l + p3l + p4l)
    chl_sum_conc_full = solution_full_model.y[18] + solution_full_model.y[22] + solution_full_model.y[26]
    chl_sum_conc_reduced =  solution_reduced_model.y[18] + solution_reduced_model.y[22] + solution_reduced_model.y[26]

    # Set time range for searching for max value
    # January to March in year 6
    t_min = 86400*365*6
    t_max = t_min + 86400*90
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of chl concentrations
    spring_bloom_full = chl_sum_conc_full[index_t_min_full:index_t_max_full]
    spring_bloom_reduced = chl_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    
    # Find peak chl-a concentration in spring bloom
    spring_bloom_peak_full = max(spring_bloom_full)
    spring_bloom_peak_reduced = max(spring_bloom_reduced)
    
    # Compute the error
    error = 100*numpy.abs(spring_bloom_peak_full - spring_bloom_peak_reduced)/spring_bloom_peak_full
    
    # If peak is zero in reduced model, set error to zero
    if spring_bloom_peak_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_chl_3(species_removed, solution_full_model, t_span, c0):
    """ calculates error in peak chlorophyll concentration during a spring bloom
    This is for the sum of p3l and p4l
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of chl-a concentrations (p1l + p2l + p3l + p4l)
    chl_sum_conc_full = solution_full_model.y[22] + solution_full_model.y[26]
    chl_sum_conc_reduced = solution_reduced_model.y[22] + solution_reduced_model.y[26]

    # Set time range for searching for max value
    # January to March in year 6
    t_min = 86400*365*6
    t_max = t_min + 86400*90
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of chl concentrations
    spring_bloom_full = chl_sum_conc_full[index_t_min_full:index_t_max_full]
    spring_bloom_reduced = chl_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    
    # Find peak chl-a concentration in spring bloom
    spring_bloom_peak_full = max(spring_bloom_full)
    spring_bloom_peak_reduced = max(spring_bloom_reduced)
    
    # Compute the error
    error = 100*numpy.abs(spring_bloom_peak_full - spring_bloom_peak_reduced)/spring_bloom_peak_full
    
    # If peak is zero in reduced model, set error to zero
    if spring_bloom_peak_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_chl_4(species_removed, solution_full_model, t_span, c0):
    """ calculates error in average chlorophyll concentration during a spring bloom
    This is for the sum of p1l + p2l + p3l + p4l
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of chl-a concentrations (p1l + p2l + p3l + p4l)
    chl_sum_conc_full = solution_full_model.y[13] + solution_full_model.y[18] + solution_full_model.y[22] + solution_full_model.y[26]
    chl_sum_conc_reduced = solution_reduced_model.y[13] + solution_reduced_model.y[18] + solution_reduced_model.y[22] + solution_reduced_model.y[26]

    # Set time range for searching for max value
    # January to March in year 6
    t_min = 86400*365*6
    t_max = t_min + 86400*90
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of chl concentrations
    spring_bloom_full = chl_sum_conc_full[index_t_min_full:index_t_max_full]
    spring_bloom_reduced = chl_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    
    # Find average chl-a concentration in spring bloom
    spring_bloom_avg_full = numpy.mean(spring_bloom_full)
    spring_bloom_avg_reduced = numpy.mean(spring_bloom_reduced)
    
    # Compute the error
    error = 100*numpy.abs(spring_bloom_avg_full - spring_bloom_avg_reduced)/spring_bloom_avg_full
    
    # If peak is zero in reduced model, set error to zero
    if spring_bloom_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_chl_5(species_removed, solution_full_model, t_span, c0):
    """ calculates error in average chlorophyll concentration during a spring bloom
    This is for the sum of p2l + p3l + p4l
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of chl-a concentrations (p1l + p2l + p3l + p4l)
    chl_sum_conc_full = solution_full_model.y[18] + solution_full_model.y[22] + solution_full_model.y[26]
    chl_sum_conc_reduced =  solution_reduced_model.y[18] + solution_reduced_model.y[22] + solution_reduced_model.y[26]

    # Set time range for searching for max value
    # January to March in year 6
    t_min = 86400*365*6
    t_max = t_min + 86400*90
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of chl concentrations
    spring_bloom_full = chl_sum_conc_full[index_t_min_full:index_t_max_full]
    spring_bloom_reduced = chl_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    
    # Find average chl-a concentration in spring bloom
    spring_bloom_avg_full = numpy.mean(spring_bloom_full)
    spring_bloom_avg_reduced = numpy.mean(spring_bloom_reduced)
    
    # Compute the error
    error = 100*numpy.abs(spring_bloom_avg_full - spring_bloom_avg_reduced)/spring_bloom_avg_full
    
    # If average is zero in reduced model, set error to zero
    if spring_bloom_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_chl_6(species_removed, solution_full_model, t_span, c0):
    """ calculates error in average chlorophyll concentration during a spring bloom
    This is for the sum of p3l and p4l
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of chl-a concentrations (p1l + p2l + p3l + p4l)
    chl_sum_conc_full = solution_full_model.y[22] + solution_full_model.y[26]
    chl_sum_conc_reduced = solution_reduced_model.y[22] + solution_reduced_model.y[26]

    # Set time range for searching for max value
    # January to March in year 6
    t_min = 86400*365*6
    t_max = t_min + 86400*90
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of chl concentrations
    spring_bloom_full = chl_sum_conc_full[index_t_min_full:index_t_max_full]
    spring_bloom_reduced = chl_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    
    # Find average chl-a concentration in spring bloom
    spring_bloom_avg_full = numpy.mean(spring_bloom_full)
    spring_bloom_avg_reduced = numpy.mean(spring_bloom_reduced)
    
    # Compute the error
    error = 100*numpy.abs(spring_bloom_avg_full - spring_bloom_avg_reduced)/spring_bloom_avg_full
    
    # If average is zero in reduced model, set error to zero
    if spring_bloom_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_dic_7(species_removed, solution_full_model, t_span, c0):
    """ calculates error in average dissolved inorganic carbon (DIC) concentration
    during the month of january in the sixth year of the ten year simulation
    DIC is 'O3c' and its index is 48
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*6
    t_max = t_min + 86400*31
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of DIC concentration
    dic_full = solution_full_model.y[48][index_t_min_full:index_t_max_full]
    dic_reduced = solution_reduced_model.y[48][index_t_min_reduced:index_t_max_reduced]
    
    # Find average DIC concentration
    dic_avg_full = numpy.mean(dic_full)
    dic_avg_reduced = numpy.mean(dic_reduced)
    
    # Compute the error
    error = 100*numpy.abs(dic_avg_full - dic_avg_reduced)/dic_avg_full
    
    # If avg is zero in reduced model, set error to zero
    if dic_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_dic_8(species_removed, solution_full_model, t_span, c0):
    """ calculates error in annual average dissolved inorganic carbon (DIC) concentration
    during the sixth year of the ten year simulation
    DIC is 'O3c' and its index is 48
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*6
    t_max = 86400*365*7
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of DIC concentration
    dic_full = solution_full_model.y[48][index_t_min_full:index_t_max_full]
    dic_reduced = solution_reduced_model.y[48][index_t_min_reduced:index_t_max_reduced]
    
    # Find peak chl-a concentration
    dic_avg_full = numpy.mean(dic_full)
    dic_avg_reduced = numpy.mean(dic_reduced)
    
    # Compute the error
    error = 100*numpy.abs(dic_avg_full - dic_avg_reduced)/dic_avg_full
    
    # If average is zero in reduced model, set error to zero
    if dic_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_dic_9(species_removed, solution_full_model, t_span, c0):
    """ calculates error in annual average dissolved inorganic carbon (DIC) concentration
    during eighth year of the ten year simulation
    DIC is 'O3c' and its index is 48
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*8
    t_max = 86400*365*9
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of DIC concentration
    dic_full = solution_full_model.y[48][index_t_min_full:index_t_max_full]
    dic_reduced = solution_reduced_model.y[48][index_t_min_reduced:index_t_max_reduced]
    
    # Find average DIC concentration
    dic_avg_full = numpy.mean(dic_full)
    dic_avg_reduced = numpy.mean(dic_reduced)
    
    # Compute the error
    error = 100*numpy.abs(dic_avg_full - dic_avg_reduced)/dic_avg_full
    
    # If average is zero in reduced model, set error to zero
    if dic_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_dic_10(species_removed, solution_full_model, t_span, c0):
    """ calculates error in peak dissolved inorganic carbon (DIC) concentration
    during the eighth year of the ten year simulation
    DIC is 'O3c' and its index is 48
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*8.5
    t_max = 86400*365*9.5
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of DIC concentration
    dic_full = solution_full_model.y[48][index_t_min_full:index_t_max_full]
    dic_reduced = solution_reduced_model.y[48][index_t_min_reduced:index_t_max_reduced]
    
    # Find peak DIC concentration    
    # if statement in case reduction automation removes all spcies
    if len(species_removed) < 50:
        dic_peak_full = max(dic_full)
        dic_peak_reduced = max(dic_reduced)
        
        # Compute the error
        error = 100*numpy.abs(dic_peak_full - dic_peak_reduced)/dic_peak_full
        
    # if all species are removed then error = nan
    else:
        error = numpy.nan
    
    return error, solution_reduced_model


def calc_error_dic_11(species_removed, solution_full_model, t_span, c0):
    """ calculates error in time of peak dissolved inorganic carbon (DIC) concentration
    during the eighth year of the ten year simulation
    DIC is 'O3c' and its index is 48
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*8.5
    t_max = 86400*365*9.5
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of DIC concentration and time
    dic_full = solution_full_model.y[48][index_t_min_full:index_t_max_full]
    dic_reduced = solution_reduced_model.y[48][index_t_min_reduced:index_t_max_reduced]
    time_full = solution_full_model.t[index_t_min_full:index_t_max_full]
    time_reduced = solution_reduced_model.t[index_t_min_reduced:index_t_max_reduced]
    
    # if statement in case reduction automation removes all spcies
    if len(species_removed) < 50:
        # Find the index for the peak DIC concentration
        dic_peak_index_full = numpy.argmax(dic_full)
        dic_peak_index_reduced = numpy.argmax(dic_reduced)
        
        # Find the time of the peak DIC
        dic_peak_time_full = time_full[dic_peak_index_full]
        dic_peak_time_reduced = time_reduced[dic_peak_index_reduced]
        
        # Compute the error
        error = 100*numpy.abs(dic_peak_time_full - dic_peak_time_reduced)/dic_peak_time_full
        
        # If time of peak is zero in reduced model, set error to zero
        if dic_peak_time_reduced == 0.0:
            error = 100
            
    # if all species are removed then error = nan
    else:
        error = numpy.nan
    
    return error, solution_reduced_model


def calc_error_r6n_12(species_removed, solution_full_model, t_span, c0):
    """ calculates error in the peak particulate organic phosphate concentration 
    during the eighth year of the ten year simulation.
    DIC is 'R6n' and its index is 45
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*8
    t_max = 86400*365*9
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of r6n concentration
    r6n_full = solution_full_model.y[45][index_t_min_full:index_t_max_full]
    r6n_reduced = solution_reduced_model.y[45][index_t_min_reduced:index_t_max_reduced]
    
    # Find peak r6n concentration
    r6n_peak_full = max(r6n_full)
    r6n_peak_reduced = max(r6n_reduced)
    
    # Compute the error
    error = 100*numpy.abs(r6n_peak_full - r6n_peak_reduced)/r6n_peak_full
    
    # If peak is zero in reduced model, set error to zero
    if r6n_peak_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_r6n_13(species_removed, solution_full_model, t_span, c0):
    """ calculates error in the average particulate organic phosphate concentration 
    during the eighth year of the ten year simulation.
    DIC is 'R6n' and its index is 45
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*8
    t_max = 86400*365*9
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of r6n concentration
    r6n_full = solution_full_model.y[45][index_t_min_full:index_t_max_full]
    r6n_reduced = solution_reduced_model.y[45][index_t_min_reduced:index_t_max_reduced]
    
    # Find avergae r6n concentration
    r6n_avg_full = numpy.mean(r6n_full)
    r6n_avg_reduced = numpy.mean(r6n_reduced)
    
    # Compute the error
    error = 100*numpy.abs(r6n_avg_full - r6n_avg_reduced)/r6n_avg_full
    
    # If average is zero in reduced model, set error to zero
    if r6n_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_r6n_14(species_removed, solution_full_model, t_span, c0):
    """ calculates error in time of peak particulate organic nitrogen (PON) concentration
    during the eighth year of the ten year simulation
    PON is 'R6n' and its index is 45
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value in full model
    t_min_full = 86400*365*8
    t_max_full = 86400*365*9
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min_full:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max_full:
            index_t_max_full = index
            break
    
    # Slice list of PON concentration and time for full model
    r6n_full = solution_full_model.y[45][index_t_min_full:index_t_max_full]
    time_full = solution_full_model.t[index_t_min_full:index_t_max_full]
    
    # Find the index for the peak PON concentration
    r6n_peak_index_full = numpy.argmax(r6n_full)
    
    # Find the time of the peak PON concentration
    r6n_peak_time_full = time_full[r6n_peak_index_full]
    
    # Set bounds for searching for reduced model peak to be +/- 6 months around r6n_peak_time_full
    t_min_reduced = r6n_peak_time_full - (86400*365/2)
    t_max_reduced = r6n_peak_time_full + (86400*365/2)
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min_reduced:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max_reduced:
            index_t_max_reduced = index
            break
    
    # Slice list of PON concentration and time for reduced model
    r6n_reduced = solution_reduced_model.y[45][index_t_min_reduced:index_t_max_reduced]
    time_reduced = solution_reduced_model.t[index_t_min_reduced:index_t_max_reduced]
    
    # Find the indices for the peak PON concentration
    r6n_peak_indices_reduced = argrelextrema(r6n_reduced, numpy.greater, order=10)
    # if len(species_removed) == 10:
    #     sys.exit(r6n_peak_indices_reduced, type(r6n_peak_indices_reduced), len(r6n_peak_indices_reduced))
    
    
    # Find the time of the peak PON concentration for each index in r6n_peak_indices_reduced
    r6n_peak_times_reduced = numpy.zeros(len(r6n_peak_indices_reduced[0]))
    for i, peak_index in enumerate(r6n_peak_indices_reduced[0]):
        # sys.exit(r6n_peak_indices_reduced)
        r6n_peak_times_reduced[i] = time_reduced[peak_index]
    
    # Compute the error
    error = min(abs(r6n_peak_time_full - r6n_peak_times_reduced)/(365*86400))*100
    
    return error, solution_reduced_model


def calc_error_o2o_15(species_removed, solution_full_model, t_span, c0):
    """ calculates error in average oxygen concentration
    during the month of january in the 8th year of the ten year simulation
    DIC is 'O2o and its index is 0
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*8
    t_max = t_min + 86400*31
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of o2o concentration
    o2o_full = solution_full_model.y[0][index_t_min_full:index_t_max_full]
    o2o_reduced = solution_reduced_model.y[0][index_t_min_reduced:index_t_max_reduced]
    
    # Find average o2o concentration
    o2o_avg_full = numpy.mean(o2o_full)
    o2o_avg_reduced = numpy.mean(o2o_reduced)
    
    # Compute the error
    error = 100*numpy.abs(o2o_avg_full - o2o_avg_reduced)/o2o_avg_full
    
    # If avg is zero in reduced model, set error to zero
    if o2o_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model



def calc_error_o2o_16(species_removed, solution_full_model, t_span, c0):
    """ calculates error in the average oxygen concentration 
    during the eighth year of the ten year simulation.
    DIC is 'O2o' and its index is 0
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*8
    t_max = 86400*365*9
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of o2o concentration
    o2o_full = solution_full_model.y[0][index_t_min_full:index_t_max_full]
    o2o_reduced = solution_reduced_model.y[0][index_t_min_reduced:index_t_max_reduced]
    
    # Find avergae o2o concentration
    o2o_avg_full = numpy.mean(o2o_full)
    o2o_avg_reduced = numpy.mean(o2o_reduced)
    
    # Compute the error
    error = 100*numpy.abs(o2o_avg_full - o2o_avg_reduced)/o2o_avg_full
    
    # If average is zero in reduced model, set error to zero
    if o2o_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_o2o_17(species_removed, solution_full_model, t_span, c0):
    """ calculates error in the peak oxygen concentration 
    during the eighth year of the ten year simulation.
    DIC is 'O2o' and its index is 0
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*8.5
    t_max = 86400*365*9.5
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of o2o concentration
    o2o_full = solution_full_model.y[0][index_t_min_full:index_t_max_full]
    o2o_reduced = solution_reduced_model.y[0][index_t_min_reduced:index_t_max_reduced]
    
    # Find peak o2o concentration
    o2o_peak_full = max(o2o_full)
    o2o_peak_reduced = max(o2o_reduced)
    
    # Compute the error
    error = 100*numpy.abs(o2o_peak_full - o2o_peak_reduced)/o2o_peak_full
    
    # If peak is zero in reduced model, set error to zero
    if o2o_peak_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_o2o_18(species_removed, solution_full_model, t_span, c0):
    """ calculates error in time of peak oxygen concentration
    during the eighth year of the ten year simulation
    oxygen is 'O2o' and its index is 0
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Set time range for searching for max value
    t_min = 86400*365*8.5
    t_max = 86400*365*9.5
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of o2o concentration and time
    o2o_full = solution_full_model.y[0][index_t_min_full:index_t_max_full]
    o2o_reduced = solution_reduced_model.y[0][index_t_min_reduced:index_t_max_reduced]
    time_full = solution_full_model.t[index_t_min_full:index_t_max_full]
    time_reduced = solution_reduced_model.t[index_t_min_reduced:index_t_max_reduced]
    
    # Find the index for the peak o2o concentration
    o2o_peak_index_full = numpy.argmax(o2o_full)
    o2o_peak_index_reduced = numpy.argmax(o2o_reduced)
    
    # Find the time of the peak o2o
    o2o_peak_time_full = time_full[o2o_peak_index_full]
    o2o_peak_time_reduced = time_reduced[o2o_peak_index_reduced]
    
    # Compute the error
    error = 100*numpy.abs(o2o_peak_time_full - o2o_peak_time_reduced)/o2o_peak_time_full
    
    # If time of peak is zero in reduced model, set error to zero
    if o2o_peak_time_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_pc_19(species_removed, solution_full_model, t_span, c0):
    """ calculates error in average phytoplankton carbon concentration during a spring bloom
    This is for the sum of p1c + p2c + p3c + p4c
    This is for the spring bloom (Jan - March) in the 8th year of a 10 year simulation
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of phytoplankton carbon concentrations (p1l + p2l + p3l + p4l)
    pc_sum_conc_full = solution_full_model.y[10] + solution_full_model.y[15] + solution_full_model.y[19] + solution_full_model.y[23]
    pc_sum_conc_reduced = solution_reduced_model.y[10] + solution_reduced_model.y[15] + solution_reduced_model.y[19] + solution_reduced_model.y[23]

    # Set time range for searching for max value
    # January to March in year 8
    t_min = 86400*365*8
    t_max = t_min + 86400*90
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # slice list of phytoplankton carbon concentrations
    spring_bloom_full = pc_sum_conc_full[index_t_min_full:index_t_max_full]
    spring_bloom_reduced = pc_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    
    # Find average phytoplankton carbon concentration in spring bloom
    spring_bloom_avg_full = numpy.mean(spring_bloom_full)
    spring_bloom_avg_reduced = numpy.mean(spring_bloom_reduced)
    
    # Compute the error
    error = 100*numpy.abs(spring_bloom_avg_full - spring_bloom_avg_reduced)/spring_bloom_avg_full
    
    # If average is zero in reduced model, set error to zero
    if spring_bloom_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_pc_20(species_removed, solution_full_model, t_span, c0):
    """ calculates error in peak phytoplankton carbon concentration during a spring bloom
    This is for the sum of p1c + p2c + p3c + p4c
    This is for the spring bloom (Jan - March) in the 8th year of a 10 year simulation
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of phytoplankton carbon concentrations (p1l + p2l + p3l + p4l)
    pc_sum_conc_full = solution_full_model.y[10] + solution_full_model.y[15] + solution_full_model.y[19] + solution_full_model.y[23]
    pc_sum_conc_reduced = solution_reduced_model.y[10] + solution_reduced_model.y[15] + solution_reduced_model.y[19] + solution_reduced_model.y[23]

    # Set time range for searching for max value
    # January to March in year 8
    t_min = 86400*365*8
    t_max = t_min + 86400*90
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # Slice list of phytoplankton carbon concentrations
    spring_bloom_full = pc_sum_conc_full[index_t_min_full:index_t_max_full]
    spring_bloom_reduced = pc_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    
    # Find peak phytoplankton carbon concentration in spring bloom
    spring_bloom_peak_full = max(spring_bloom_full)
    spring_bloom_peak_reduced = max(spring_bloom_reduced)
    
    # Compute the error
    error = 100*numpy.abs(spring_bloom_peak_full - spring_bloom_peak_reduced)/spring_bloom_peak_full
    
    # If peak is zero in reduced model, set error to zero
    if spring_bloom_peak_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_pc_21(species_removed, solution_full_model, t_span, c0):
    """ calculates error in time of peak phytoplankton carbon concentration during a spring bloom
    This is for the sum of p1c + p2c + p3c + p4c
    This is for the spring bloom (Jan - March) in the 8th year of a 10 year simulation
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of phytoplankton carbon concentrations (p1l + p2l + p3l + p4l)
    pc_sum_conc_full = solution_full_model.y[10] + solution_full_model.y[15] + solution_full_model.y[19] + solution_full_model.y[23]
    pc_sum_conc_reduced = solution_reduced_model.y[10] + solution_reduced_model.y[15] + solution_reduced_model.y[19] + solution_reduced_model.y[23]

    # Set time range for searching for max value
    # January to March in year 8
    t_min = 86400*365*8
    t_max = t_min + 86400*90
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # Slice list of phytoplankton carbon concentrations and time data
    spring_bloom_full = pc_sum_conc_full[index_t_min_full:index_t_max_full]
    spring_bloom_reduced = pc_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    time_full = solution_full_model.t[index_t_min_full:index_t_max_full]
    time_reduced = solution_reduced_model.t[index_t_min_reduced:index_t_max_reduced]
    
    # Find the index for the peak phytoplankton carbon concentration
    pc_peak_index_full = numpy.argmax(spring_bloom_full)
    pc_peak_index_reduced = numpy.argmax(spring_bloom_reduced)
    
    # Find the time of the peak phytoplankton carbon
    pc_peak_time_full = time_full[pc_peak_index_full]
    pc_peak_time_reduced = time_reduced[pc_peak_index_reduced]
    
    # Compute the error in the time of peak
    error_time_of_peak = 100*numpy.abs(pc_peak_time_full - pc_peak_time_reduced)/pc_peak_time_full
    
    # Cheack magnitude of peak
    pc_peak_full = spring_bloom_full[pc_peak_index_full]
    pc_peak_reduced = spring_bloom_reduced[pc_peak_index_reduced]
    error_peak = 100*numpy.abs(pc_peak_full - pc_peak_reduced)/pc_peak_full
    
    # Compare error in time of peak and peak magnitude
    # if peak magnitude has > than 80% then output this error
    if error_peak > 80:
        error = error_peak
    else:
        error = error_time_of_peak
    
    return error, solution_reduced_model


def calc_error_bzc_22(species_removed, solution_full_model, t_span, c0):
    """ calculates error in average non-photosynthesizers carbon concentration 
    during year 8 of a 10 year simulation
    This is for the sum of B1c + Z3c + Z4c + Z5c + Z6c
    B1c is index 7, Z3c is index 27, Z4c is index 30, Z5c is index 33, Z6c is index 36
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of non-photosynthesizers carbon concentrations (B1c + Z3c + Z4c + Z5c + Z6c)
    bzc_sum_conc_full = solution_full_model.y[7] + solution_full_model.y[27] + solution_full_model.y[30] + solution_full_model.y[33] + solution_full_model.y[36]
    bzc_sum_conc_reduced = solution_reduced_model.y[7] + solution_reduced_model.y[27] + solution_reduced_model.y[30] + solution_reduced_model.y[33] + solution_reduced_model.y[36]

    # Set time range for searching for max value
    t_min = 86400*365*8
    t_max = 86400*365*9
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # Slice list of non-photosynthesizers carbon concentrations
    bzc_full = bzc_sum_conc_full[index_t_min_full:index_t_max_full]
    bzc_reduced = bzc_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    
    # Find average non-photosynthesizers carbon concentration in spring bloom
    bzc_avg_full = numpy.mean(bzc_full)
    bzc_avg_reduced = numpy.mean(bzc_reduced)
    
    # Compute the error
    error = 100*numpy.abs(bzc_avg_full - bzc_avg_reduced)/bzc_avg_full
    
    # If average is zero in reduced model, set error to zero
    if bzc_avg_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_bzc_23(species_removed, solution_full_model, t_span, c0):
    """ calculates error in peak non-photosynthesizers carbon concentration 
    during year 8 of a 10 year simulation
    This is for the sum of B1c + Z3c + Z4c + Z5c + Z6c
    B1c is index 7, Z3c is index 27, Z4c is index 30, Z5c is index 33, Z6c is index 36
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of non-photosynthesizers carbon concentrations (B1c + Z3c + Z4c + Z5c + Z6c)
    bzc_sum_conc_full = solution_full_model.y[7] + solution_full_model.y[27] + solution_full_model.y[30] + solution_full_model.y[33] + solution_full_model.y[36]
    bzc_sum_conc_reduced = solution_reduced_model.y[7] + solution_reduced_model.y[27] + solution_reduced_model.y[30] + solution_reduced_model.y[33] + solution_reduced_model.y[36]

    # Set time range for searching for max value
    t_min = 86400*365*8
    t_max = 86400*365*9
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # Slice list of non-photosynthesizers carbon concentrations
    bzc_full = bzc_sum_conc_full[index_t_min_full:index_t_max_full]
    bzc_reduced = bzc_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    
    # Find peak non-photosynthesizers carbon concentration in spring bloom
    bzc_peak_full = max(bzc_full)
    bzc_peak_reduced = max(bzc_reduced)
    
    # Compute the error
    error = 100*numpy.abs(bzc_peak_full - bzc_peak_reduced)/bzc_peak_full
    
    # If peak is zero in reduced model, set error to zero
    if bzc_peak_reduced == 0.0:
        error = 100
    
    return error, solution_reduced_model


def calc_error_bzc_24(species_removed, solution_full_model, t_span, c0):
    """ calculates error in time of peak non-[phtotsynthesizers carbon concentration 
    during year 8 of a 10 year simulation
    This is for the sum of B1c + Z3c + Z4c + Z5c + Z6c
    B1c is index 7, Z3c is index 27, Z4c is index 30, Z5c is index 33, Z6c is index 36
    """
    
    # Integrate reduced model
    # Rempove indicated species in concentration list and create a multiplier for rate eqns to "remove species"
    conc_reduced = copy.copy(c0)
    multiplier = numpy.ones(len(c0))
    for index in species_removed.values():
        conc_reduced[index] = 0.0
        multiplier[index] = 0.0
    solution_reduced_model = solve_ivp(lambda time, conc: bfm_reduced_rate_eqns(time, conc, multiplier), t_span, conc_reduced, method='RK23')
    
    # Get sum of non-photosynthesizers carbon concentrations (B1c + Z3c + Z4c + Z5c + Z6c)
    bzc_sum_conc_full = solution_full_model.y[7] + solution_full_model.y[27] + solution_full_model.y[30] + solution_full_model.y[33] + solution_full_model.y[36]
    bzc_sum_conc_reduced = solution_reduced_model.y[7] + solution_reduced_model.y[27] + solution_reduced_model.y[30] + solution_reduced_model.y[33] + solution_reduced_model.y[36]

    # Set time range for searching for max value
    t_min = 86400*365*8
    t_max = 86400*365*9
    
    # Find index associated for t_min and t_max for slicing the list
    for index,time in enumerate(solution_full_model.t):
        if time >= t_min:
            index_t_min_full = index
            break
    for index,time in enumerate(solution_full_model.t):
        if time > t_max:
            index_t_max_full = index
            break
        
    for index,time in enumerate(solution_reduced_model.t):
        if time >= t_min:
            index_t_min_reduced = index
            break
    for index,time in enumerate(solution_reduced_model.t):
        if time > t_max:
            index_t_max_reduced = index
            break
    
    # Slice list of non-photosynthesizers carbon concentrations and time data
    bzc_full = bzc_sum_conc_full[index_t_min_full:index_t_max_full]
    bzc_reduced = bzc_sum_conc_reduced[index_t_min_reduced:index_t_max_reduced]
    time_full = solution_full_model.t[index_t_min_full:index_t_max_full]
    time_reduced = solution_reduced_model.t[index_t_min_reduced:index_t_max_reduced]
    
    # Find the index for the peak bzc concentration
    bzc_peak_index_full = numpy.argmax(bzc_full)
    bzc_peak_index_reduced = numpy.argmax(bzc_reduced)
    
    # Find the time of the peak bzc
    bzc_peak_time_full = time_full[bzc_peak_index_full]
    bzc_peak_time_reduced = time_reduced[bzc_peak_index_reduced]
    
    # Compute the error
    error_time_of_peak = 100*numpy.abs(bzc_peak_time_full - bzc_peak_time_reduced)/bzc_peak_time_full
    
    # Cheack magnitude of peak
    bzc_peak_full = bzc_full[bzc_peak_index_full]
    bzc_peak_reduced = bzc_reduced[bzc_peak_index_reduced]
    error_peak = 100*numpy.abs(bzc_peak_full - bzc_peak_reduced)/bzc_peak_full
    
    # Compare error in time of peak and peak magnitude
    # if peak magnitude has > than 80% then output this error
    if error_peak > 80:
        error = error_peak
    else:
        error = error_time_of_peak
    
    return error, solution_reduced_model

