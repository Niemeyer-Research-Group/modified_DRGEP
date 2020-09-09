import numpy
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import brewer2mpl

from scipy.integrate import solve_ivp
from PythonBFM.BFM50_rate_eqns import bfm50_rate_eqns
from pyMARS_DRGEP_functions import get_importance_coeffs
import error_fcns

def calc_modified_DRGEP_dic(rate_eqn_fcn, conc, t):
    """ Calculates the percent difference between the new rate eqn and the 
    original. The new rate is based on turning one species 'off' """

    c_original = list(conc)
    c_new = list(conc)
    number_of_species = len(conc)

    # calculate original rate values
    dc_dt_og = rate_eqn_fcn(t, conc)

    percent_error_matrix = numpy.zeros([number_of_species, number_of_species])

    for j in range(number_of_species):
        c_new[j] = 0.0
        dc_dt_new = rate_eqn_fcn(t, c_new)
        percent_error = calc_percent_error(dc_dt_new, dc_dt_og)
        percent_error_matrix[:,j] = percent_error
        c_new[j] = c_original[j]

    # Find the maximum value along each row (used for normalization)
    row_max = numpy.amax(percent_error_matrix, axis=1)

    # Normalize percent_error_matrix by row max which is new_dic_matrix
    dic_matrix = numpy.zeros([number_of_species, number_of_species])
    for i in range(number_of_species):
        if row_max[i] == 0:
            dic_matrix[i,:] = 0.0
        else:
            dic_matrix[i,:] = percent_error_matrix[i,:]/row_max[i]

    # Set diagonals to zero to avoid self_directing graph edges
    numpy.fill_diagonal(dic_matrix, 0.0) 

    return dic_matrix


def calc_percent_error(new_array, old_array):
    """ Calculates the percent error between two matricies.
    This is used for the calculating the New Method's direct interaction coeffs
    """
    percent_error = []
    if len(new_array)==len(old_array):
        for i in range(len(new_array)):
            if new_array[i] == old_array[i]:
                percent_error.append(0.0)
            else:
                percent_error.append(abs(new_array[i] - old_array[i])/abs(old_array[i])*100)

    return percent_error


def reduce_modified_DRGEP(error_fcn, species_names, species_safe, threshold, overall_interaction_coeffs, solution_full_model, t_span, c0):
    """ calculates the number of species and error for a given threshold value
    """

    # Find species to remove using cutoff threshold
    species_removed = {}
    for species in overall_interaction_coeffs:
        if overall_interaction_coeffs[species] < threshold and species not in species_safe:
            species_removed[species] = numpy.nan
    # Assign index to dictionary values for the species_removed
    for index, species in enumerate(species_names):
        if species in species_removed:
            species_removed[species] = index

    # Count how many species are remaining
    num_species = len(species_names) - len(species_removed)

    # For the reduced model, call function to get error
    error, solution_reduced_model = error_fcn(species_removed, solution_full_model, t_span, c0)

    return error, num_species, species_removed, solution_reduced_model


def run_modified_DRGEP(overall_interaction_coeffs, error_limit, error_fcn, species_names, species_targets, species_safe, rate_eqn_fcn, t_span, c0):
    """ Iterates through different threshold values to find a reduced model that meets error criteria
    """

    assert species_targets, 'Need to specify at least one target species.'

    # begin reduction iterations
    logging.info('Beginning reduction loop')
    logging.info(45 * '-')
    logging.info('Threshold | Number of species | Max error (%)')

    # make lists to store data
    threshold_data = []
    num_species_data = []
    error_data = []
    species_removed_data = []
    solution_reduced_models = []
    output_data = {}

    # start with detailed (starting) model
    solution_full_model = solve_ivp(rate_eqn_fcn, t_span, c0, method='RK23')

    first = True
    error_current = 0.0
    threshold = 1e-5 #1e-5 #4e-4 #1e-6 #0.001
    threshold_increment = 0.0001 #0.01
    threshold_multiplier = 2
    while error_current <= error_limit:
        error_current, num_species, species_removed, solution_reduced_model = reduce_modified_DRGEP(error_fcn, species_names, species_safe, threshold, overall_interaction_coeffs, solution_full_model,  t_span, c0)

        # reduce threshold if past error limit on first iteration
        if first and error_current > error_limit:
            error_current = 0.0
            threshold /= 10
            threshold_increment /= 10
            if threshold <= 1e-10:
                raise SystemExit(
                    'Threshold value dropped below 1e-6 without producing viable reduced model'
                    )
            logging.info('Threshold value too high, reducing by factor of 10')
            continue

        logging.info(f'{threshold:^9.2e} | {num_species:^17} | {error_current:^.5f}')

        # store data
        threshold_data.append(threshold)
        num_species_data.append(num_species)
        error_data.append(error_current)
        species_removed_data.append(species_removed)
        solution_reduced_models.append(solution_reduced_model)

        threshold += threshold_increment
        # threshold *= threshold_multiplier
        first = False

        # Stop reduction process if num species reaches one
        if num_species == 1:
            break

        # Stop interating if threshold exceeds 1
        if threshold >= 3e-1:
            break

    if error_current > error_limit:
        threshold -= (2 * threshold_increment)
        error_current, num_species, species_removed, solution_reduced_model = reduce_modified_DRGEP(error_fcn, species_names, species_safe, threshold, overall_interaction_coeffs, solution_full_model, t_span, c0)

    logging.info(45 * '-')
    logging.info('New method reduction complete.')

    # Store all output data to a dictionary
    output_data['threshold_data'] = threshold_data
    output_data['num_species_data'] = num_species_data
    output_data['error_data'] = error_data
    output_data['species_removed_data'] = species_removed_data
    output_data['solution_full_model'] = solution_full_model
    output_data['solution_reduced_models'] = solution_reduced_models

    return output_data


if __name__ == '__main__':

    # Names of species in the system
    species_names = ['O2o', 'N1p', 'N3n', 'N4n', 'O4n', 'N5s', 'N6r', 'B1c', 'B1n', 'B1p', 
                     'P1c', 'P1n', 'P1p', 'P1l', 'P1s', 'P2c', 'P2n', 'P2p', 'P2l',
                     'P3c', 'P3n', 'P3p', 'P3l', 'P4c', 'P4n', 'P4p', 'P4l',
                     'Z3c', 'Z3n', 'Z3p', 'Z4c', 'Z4n', 'Z4p', 'Z5c', 'Z5n', 'Z5p',
                     'Z6c', 'Z6n', 'Z6p', 'R1c', 'R1n', 'R1p', 'R2c', 'R3c', 'R6c', 
                     'R6n', 'R6p', 'R6s', 'O3c', 'O3h']

    # Initial concentrations
    c0 = [300.0,                    # O2o
          1.0,                      # N1p
          5.0,                      # N3n
          1.0,                      # N4n
          200.0,                    # O4n
          8.0,                      # N5s
          1.0,                      # N6r
          1.0,                      # B1c
          1.67e-2,                  # B1n
          1.85e-3,                  # B1p
          1.0,                      # P1c
          1.26e-2,                  # P1n
          7.86e-4,                  # P1p
          2.50e-2,                  # P1l
          1.00e-2,                  # P1s
          1.0,                      # P2c
          1.26e-2,                  # P2n
          7.86e-4,                  # P2p
          1.50e-2,                  # P2l
          1.0,                      # P3c
          1.26e-2,                  # P3n
          7.86e-4,                  # P3p
          2.00e-2,                  # P3l
          1.0,                      # P4c
          1.26e-2,                  # P4n
          7.86e-4,                  # P4p
          2.00e-2,                  # P4l
          1.0,                      # Z3c
          1.5e-2,                   # Z3n
          1.67e-3,                  # Z3p
          1.0,                      # Z4c
          1.5e-2,                   # Z4n
          1.67e-3,                  # Z4p
          1.0,                      # Z5c
          1.67e-2,                  # Z5n
          1.85e-3,                  # Z5p
          1.0,                      # Z6c
          1.67e-2,                  # Z6n
          1.85e-3,                  # Z6p
          1.0,                      # R1c
          1.26e-2,                  # R1n
          7.862e-4,                 # R1p
          0.1,                      # R2c
          1.0,                      # R3c
          1.0,                      # R6c
          1.26e-2,                  # R6n
          7.862e-4,                 # R6p
          1.45e-2,                  # R6s
          27060.0,                  # O3c
          2660.0                    # O3h
          ]

    # Information for reduction
    # User needs to update these values
    scenario = 'Oxygen1'
    species_targets = ['O2o']
    species_safe = []
    error_limit = 15
    error_fcn = error_fcns.calc_error_o2o_15
    rate_eqn_fcn = bfm50_rate_eqns
    # Time span for integration
    t_span = [0, 86400*365*10]
    # Time at which the DIC values are obtained
    t = 86400*365

    # Log input data to file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='output.log', level=logging.INFO)
    logging.info('Scenario: {}'.format(scenario))
    logging.info('Error function: {}'.format(error_fcn))
    logging.info('Error limit: {}'.format(error_limit))
    logging.info('Target species: {}'.format(species_targets))
    logging.info('Retained species: {}'.format(species_safe))

    # Get direct interaction coefficients
    dc_dt_og_0 = rate_eqn_fcn(t, c0)
    dic_matrix = calc_modified_DRGEP_dic(rate_eqn_fcn, c0, t)

    # Get overall interaction coefficients
    overall_interaction_coeffs = get_importance_coeffs(species_names, species_targets, [dic_matrix])

    # Make dictionary of target species and their index values
    target_species = {}
    for index, species in enumerate(species_names):
        if species in species_targets:
            target_species[species] = index

    # Run new method
    reduction_data = run_modified_DRGEP(overall_interaction_coeffs, error_limit, error_fcn, species_names, species_targets, species_safe, rate_eqn_fcn, t_span, c0)
