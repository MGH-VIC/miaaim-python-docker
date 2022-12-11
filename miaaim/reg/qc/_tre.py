# Module for target registration error for landmark pairs
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# import modules
import numpy as np
import pandas as pd
from pathlib import Path
import os

def tre(points_1, points_2):
    """Compute target registration error for landmark pairs based on BIRL
    package and Automatic Non-rigid Histological Image Registration (ANHIR)
    challenge hosted by ASBI 2019 conference.

    Parameters
    ----------
    points_1: str
        Path to csv file containing points

    points_2: str

    Returns
    -------
    ndarray: list of errors of size nb points.

    >> tre(np.random.random((6, 2)),
    ...   np.random.random((9, 2)))
    """

    # get common landmark points (assumes ordered correctly)
    nb_common = min([len(pts) for pts in [points_1, points_2] if pts is not None])
    # if no common points raise error
    if nb_common <= 0:
        raise ValueError('no common landmarks for metric')
    # get points in common
    points_1 = np.asarray(points_1)[:nb_common]
    points_2 = np.asarray(points_2)[:nb_common]
    # compute differences as euclidean distance between points
    diffs = np.sqrt(np.sum(np.power(points_1 - points_2, 2), axis=1))
    return diffs

def tre_statistics(points_fixed, points_warp):
    """Compute summary statistics of target registration error from a set
    of matched landmark points after registration.

    Parameters
    ----------
    points_target: ndarray
        Landmark points in the fixed image.

    points_warp: ndarray
        Warped landmark points from moving to fixed image.

    Returns
    -------
    tuple(ndarray,dict): (np.array<nb_points, 1>, dict)
    """

    # check for overlap points
    if not all(pts is not None and list(pts) for pts in [points_ref, points_est]):
        return [], {'overlap points': 0}

    lnd_sizes = [len(points_ref), len(points_est)]
    if min(lnd_sizes) <= 0:
        raise ValueError('no common landmarks for metric')
    diffs = compute_tre(points_ref, points_est)

    inter_dist = distance.cdist(points_ref[:len(diffs)], points_ref[:len(diffs)])
    # inter_dist[range(len(points_ref)), range(len(points_ref))] = np.inf
    dist = np.mean(inter_dist, axis=0)
    weights = dist / np.sum(dist)

    dict_stat = {
        'Mean': np.mean(diffs),
        'Mean_weighted': np.sum(diffs * weights),
        'STD': np.std(diffs),
        'Median': np.median(diffs),
        'Min': np.min(diffs),
        'Max': np.max(diffs),
        'overlap points': min(lnd_sizes) / float(max(lnd_sizes))
    }
    return diffs, dict_stat

def read_and_compute_tre(path_1, path_2):
    """Read resulting transformed and target points from image registration
    and compute target registration error.

    Parameters
    ----------
    path_1: str
        Path to target points file.

    path_2: str
        Path to warped or source points file.

    Returns
    -------
    ndarray: list of errors of size nb points.

    """
    raise NotImplementedError

def read_and_compute_tre_statistics(path_fixed, path_warped):
    """Read resulting transformed and target points from image registration
    and compute target registration error.

    Parameters
    ----------
    path_fixed: str
        Path to fixed points file.

    path_warped: str
        Path to warped or source points file.

    Returns
    -------
    ndarray: list of errors of size nb points.

    """
    raise NotImplementedError
