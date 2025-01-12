import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np
import os
import sys

lib_name = {'linux': 'libtwproc.so', 'darwin': 'libtwproc.dylib', 'win32': 'TwProcDll.dll'}
proc_lib = ct.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), lib_name[sys.platform]))


tw_align_chromatogram = proc_lib.TwAlignChromatogram
tw_align_chromatogram.restype = ct.c_double
def TwAlignChromatogram(segment_length, slack, chromatogram_length, target, sample, sample_aligned, flag):
    double_ptr = ndpointer(np.float64) if isinstance(target, np.ndarray) else ct.POINTER(ct.c_double)
    tw_align_chromatogram.argtypes = [
        ct.c_double,
        ct.c_double,
        ct.c_double,
        double_ptr,
        double_ptr,
        double_ptr,
        ct.c_double
    ]

    return tw_align_chromatogram(segment_length, slack, chromatogram_length, target, sample, sample_aligned, flag)


tw_findpeaks = proc_lib.TwFindPeaks
tw_findpeaks.restype = ct.c_double
def TwFindPeaks(
    flag,
    mass_to_charge,
    signal,
    peak_center_from_second_derivative,
    peak_amplitude_from_second_derivative,
    number_of_peaks_found_from_second_derivative,
    peak_center_from_fitting,
    peak_amplitude_from_fitting,
    number_of_peaks_found_from_fitting,
    second_derivative,
    spectrum_size,
    peak_shape_table,
    peak_shape_table_size,
    smooth_factor,
    max_num_peaks_allowed_from_derivative,
    threshold_percent_of_max_second_derivative,
    threshold_min_y,
    threshold_percent_of_max_y,
    fix_peak_width,
    peak_width_factor,
    min_fwhm_ratio_to_separate_two_peaks,
    epsilon_position,
    epsilon_amplitude,
    epsilon_sigma,
    max_num_peaks_allowed_to_force_add,
    threshold_residual_to_y,
    tolerance_min,
    tolerance_max,
    mass_resolving_power):
    double_ptr = ndpointer(np.float64) if isinstance(mass_to_charge, np.ndarray) else ct.POINTER(ct.c_double)
    tw_findpeaks.argtypes = [
        ct.c_double,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        ct.c_double,
        double_ptr,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double
    ]

    return tw_findpeaks(
        flag,
        mass_to_charge,
        signal,
        peak_center_from_second_derivative,
        peak_amplitude_from_second_derivative,
        number_of_peaks_found_from_second_derivative,
        peak_center_from_fitting,
        peak_amplitude_from_fitting,
        number_of_peaks_found_from_fitting,
        second_derivative,
        spectrum_size,
        peak_shape_table,
        peak_shape_table_size,
        smooth_factor,
        max_num_peaks_allowed_from_derivative,
        threshold_percent_of_max_second_derivative,
        threshold_min_y,
        threshold_percent_of_max_y,
        fix_peak_width,
        peak_width_factor,
        min_fwhm_ratio_to_separate_two_peaks,
        epsilon_position,
        epsilon_amplitude,
        epsilon_sigma,
        max_num_peaks_allowed_to_force_add,
        threshold_residual_to_y,
        tolerance_min,
        tolerance_max,
        mass_resolving_power)


tw_generate_formula = proc_lib.TwGenerateFormula
tw_generate_formula.restype = ct.c_double
def TwGenerateFormula(
    seed,
    seed_length,
    max_mz,
    charge,
    max_carbon,
    max_oxygen_to_carbon,
    min_hydrogen,
    min_hydrogen_to_carbon,
    min_double_bond_equivalent_per_carbon,
    max_double_bound_equivalent_per_carbon,
    target_peak_list_length,
    target_peak_list_center):
    double_ptr = ndpointer(np.float64) if isinstance(target_peak_list_center, np.ndarray) else ct.POINTER(ct.c_double)
    tw_generate_formula.argtypes = [
        ct.c_char_p,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        double_ptr,
        double_ptr
    ]

    return tw_generate_formula(
        seed,
        seed_length,
        max_mz,
        charge,
        max_carbon,
        max_oxygen_to_carbon,
        min_hydrogen,
        min_hydrogen_to_carbon,
        min_double_bond_equivalent_per_carbon,
        max_double_bound_equivalent_per_carbon,
        target_peak_list_length,
        target_peak_list_center)


tw_match_peaks = proc_lib.TwMatchPeaks
tw_match_peaks.restype = ct.c_double
def TwMatchPeaks(
    target_peak_list_length,
    target_peak_list_center,
    matched_peak_list_length,
    matched_peak_list_center,
    matched_peak_list_index_in_target_peak_list):
    double_ptr = ndpointer(np.float64) if isinstance(target_peak_list_center, np.ndarray) else ct.POINTER(ct.c_double)
    tw_match_peaks.argtypes = [
        ct.c_double,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr
    ]

    return tw_match_peaks(
        target_peak_list_length,
        target_peak_list_center,
        matched_peak_list_length,
        matched_peak_list_center,
        matched_peak_list_index_in_target_peak_list)


if __name__ == '__main__':
    pass