# Note that much of this code was borrowed/adapted from Alexandre Perez 
# (https://github.com/alexprz/meta_analysis_notebook).

import os
import multiprocessing
from joblib import Parallel, delayed
import nilearn
import numpy as np
from nilearn import masking, plotting, image
from nipy.labs.statistical_mapping import get_3d_peaks
import nibabel as nib
import scipy
from matplotlib import pyplot as plt
import ntpath
from nimare.dataset import Dataset
import nimare
import copy
from nistats import thresholding
import glob
import csv
import IPython
import pandas as pd
from nilearn.input_data import NiftiMasker

template = nilearn.datasets.load_mni152_template()
Ni, Nj, Nk = template.shape
affine = template.affine
gray_mask = masking.compute_gray_matter_mask(template)

def get_list_of_teams_we_have(teams_in_spreadsheet):
    team_list_file = '../data-narps/orig/team_folder_list.csv'
    with open(team_list_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader: 
            team_list = row

    for n_team in np.arange(len(team_list)):
        team_list[n_team] = team_list[n_team][0:8]
        
    team_list = list(set(teams_in_spreadsheet).intersection(set(team_list)))
    
    team_list.sort()
    
    return team_list

def img_paths_to_matrix_of_imgs(img_paths):
    order = 'C'
    for n, path in enumerate(img_paths):
        img = nilearn.image.load_img(path)
        img_resampled = image.resample_to_img(img, template)
        array = img_resampled.get_fdata()
        flat = array.ravel(order=order)
        flat = np.array(flat)
        flat = np.expand_dims(flat, axis=1)
        if n == 0:
            y = flat
        else:
            y = np.concatenate((y, flat), axis=1)
    y=y.T
    Y = np.matrix(y)
    Y = np.nan_to_num(Y, copy=False, nan=0)
    return Y

def get_path_list_from_team_list(teams, all_paths):
    team_paths = []
    for path in all_paths:
        for team in teams:
            if team in path:
                team_paths.append(path)
    return team_paths

def get_voxel_matrix(img_paths):
    order = 'C'
    for n, path in enumerate(img_paths):
        img = nilearn.image.load_img(path)
        img_resampled = image.resample_to_img(img, template)
        array = img_resampled.get_fdata()
        flat = array.ravel(order=order)
        flat = np.array(flat)
        flat = np.expand_dims(flat, axis=1)
        if n == 0:
            y = flat
        else:
            y = np.concatenate((y, flat), axis=1)
    y=y.T
    Y = np.matrix(y)
    Y = np.nan_to_num(Y, copy=False, nan=0)
    return Y

def get_1s_ttest_design_matrix(Y):
    x = np.ones_like(Y[:,0])
    X = np.matrix(x).T
    return X
    
def glm(Y, X):
    beta = ((X.T * X)**-1) * X.T * Y
    
    n = Y.shape[0]
    residual = np.array(Y - X*beta)
    SSE = np.squeeze(np.sum(np.power(residual,2), axis=0))
    SD = np.sqrt(SSE/(n-1))
    SE = SD/np.sqrt(n)
    
    Z = np.arctanh(beta)
    
    T = beta/SE
    
    SE_map = fun.flat_mat_to_nii(flat_mat=SE, ref_niimg=template, order=order)
    B_map = fun.flat_mat_to_nii(flat_mat=beta, ref_niimg=template, order=order)
    Z_map = fun.flat_mat_to_nii(flat_mat=Z, ref_niimg=template, order=order)
    T_map = fun.flat_mat_to_nii(flat_mat=T, ref_niimg=template, order=order)
    
    return T_map, SE_map, B_map, Z_map
    
    
def mni_mask(img_nii):
    mask = nilearn.datasets.load_mni152_brain_mask()
    masker = NiftiMasker(mask_img=mask)
    masker_fit = masker.fit_transform(img_nii)
    img_nii_masked = masker.inverse_transform(masker_fit)
    return img_nii_masked

def flat_mat_to_nii(flat_mat, ref_niimg, order, dims=(91,109,91), copy_header=False):
    arr = np.reshape(np.array(np.squeeze(flat_mat)), dims, order=order)
    nii = nilearn.image.new_img_like(ref_niimg, arr, copy_header=copy_header)
    nii_masked = mni_mask(nii)
    return nii_masked


def z_to_p(z_arr):
    return 1-scipy.stats.norm.cdf(z_arr)

def p_to_z(p_arr):
    """
    scipy.special.ndtri returns the argument x for which the area under the Gaussian 
    probability density function (integrated from minus infinity to x) is equal to y.
    """
    return scipy.special.ndtri(1-p_arr)

def fisher_meta_analysis(img_paths):
    N_img = len(img_paths)
    
    p_values = np.zeros((N_img, Ni, Nj, Nk))

    for n in range(N_img):
        img = nilearn.image.load_img(img_paths[n])
        img_resampled = image.resample_to_img(img, template)
        arr = img_resampled.get_fdata()
        #np.nan_to_num(arr, copy=False, nan=0)

        p_values[n, :, :, :] = z_to_p(arr)

    T_f = np.nan_to_num(-2*np.sum(np.log(p_values), axis=0))

    # chi2 survival funtion: 
    p_f = scipy.stats.chi2.sf(T_f, 2 * N_img)

    fisher_arr = p_to_z(p_f)
    maxnan = fisher_arr[fisher_arr != np.inf].max()
    minnan = fisher_arr[fisher_arr != -np.inf].min()
    np.nan_to_num(fisher_arr, copy=False, posinf=maxnan, neginf=minnan) # Truncate inf values
    fisher_img = nib.Nifti1Image(np.array(fisher_arr), affine)

    return fisher_img



def nan_to(img_nii, number_for_nan):
    arr = img_nii.get_fdata()
    np.nan_no_num(arr, copy=False, posinf=0)
    

def run_Fishers(ds_dict):
    """Run Fishers on given data."""
    ds = Dataset(ds_dict)
    ma = nimare.meta.ibma.Fishers()
    res = ma.fit(ds)

    return res.get_map('z')

def df_strip(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == np.object:
            df[c] = pd.core.strings.str_strip(df[c])
        df = df.rename(columns={c:c.strip()})
    return df

def get_data_paths_from_orig(filename):
    path_orig = '../data-narps/orig/'
    
    team_list_file = '../data-narps/orig/team_folder_list.csv'
    with open(team_list_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader: 
            team_list = row

    data_list = []
    for team in team_list:
        data_list.append(path_orig + team + '/' + filename)
    
    return data_list
    
    

def get_activations(path, threshold, space='ijk'):
    """
    Retrieve the activation coordinates from an image.

    Args:
        path (string or Nifti1Image): Path to or object of a
            nibabel.Nifti1Image from which to extract coordinates.
        threshold (float): Peaks under this threshold will not be detected.
        space (string): Space of coordinates. Available : 'ijk' and 'pos'.

    Returns:
        (tuple): Size 3 tuple of np.array storing respectively the X, Y and
            Z coordinates

    """
    I, J, K = [], [], []
    try:
        img = nilearn.image.load_img(path)
    except ValueError:  # File path not found
        print(f'File {path} not found. Ignored.')
        return None

    if np.isnan(img.get_fdata()).any():
        print(f'Img {path} contains Nan. Ignored.')
        return [], [], []

    img = nilearn.image.resample_to_img(img, template)

    peaks = get_3d_peaks(img, mask=gray_mask, threshold=threshold)

    if not peaks:
        return I, J, K
        #return None

    for peak in peaks:
        I.append(peak[space][0])
        J.append(peak[space][1])
        K.append(peak[space][2])

    del peaks
    
    I, J, K = np.array(I), np.array(J), np.array(K)
    if space == 'ijk':
        I, J, K = I.astype(int), J.astype(int), K.astype(int)
        
    return I, J, K

def get_data_paths(input_dir, image_type='zscore'):
    image_types = {'zscore':'.nii.gz',
                   'contrast':'_con.nii.gz',
                   'standard_error':'_se.nii.gz',
                    }
    file_ending = image_types[image_type]
    
    img_paths = []
    i = 0

    while True:
        path = os.path.abspath(f'{input_dir}hypo1_unthresh_{i}{file_ending}')
        if not os.path.isfile(path):
            break

        img_paths.append(path)
        i += 1
    return img_paths

def peaks_to_binary_image(I, J, K):
    arr = np.zeros(template.shape)
    arr[I, J, K] = 1
    img = nib.Nifti1Image(arr, affine)
    return img

# The following functions are used only to convert the input images to a NiMARE input format for IBMA analysis. 
# These functions also extract peaks coordinates from full images for CBMA analysis.
# The understanding of these functions is not crucial.
def get_sub_dict(XYZ, path_dict, sample_size):
    """
    Build sub dictionnary of a study using the nimare structure.

    Args:
        XYZ (tuple): Size 3 tuple of list storing the X Y Z coordinates.
        path_dict (dict): Dict which has map name ('t', 'z', 'con', 'se')
            as keys and absolute path to the image as values.
        sample_size (int): Number of subjects.

    Returns:
        (dict): Dictionary storing the coordinates for a
            single study using the Nimare structure.

    """
    d = {
        'contrasts': {
            '0': {
                'metadata': {'sample_sizes': 119}
            }
        }
    }

    if XYZ is not None:
        d['contrasts']['0']['coords'] = {
                    'x': list(XYZ[0]),
                    'y': list(XYZ[1]),
                    'z': list(XYZ[2]),
                    'space': 'MNI'
                    }
        d['contrasts']['0']['sample_sizes'] = sample_size

    if path_dict is not None:
        d['contrasts']['0']['images'] = path_dict

    return d


def extract_from_paths(paths, sample_size, data=['coord', 'path'], level=.05, 
                       height_control='fdr', cluster_threshold=None):
    """
    Extract data (coordinates, paths...) from the data and put it in a
        dictionnary using Nimare structure.

    Args:
        path_dict (dict): Dict which keys are study names and values
            absolute paths (string).
        data (list): Data to extract. 'coord' and 'path' available.
        sample_size (int): Number of subjects in the experiment. 
        threshold (float): value below threshold are ignored. Used for
            peak detection.

    Returns:
        (dict): Dictionnary storing the coordinates using the Nimare
            structure.

    """

    # Computing a new dataset dictionary
    def extract_pool(path):
        """Extract activation for multiprocessing."""
        #print(f'Extracting {path}...')
        
        threshold = thresholding.map_threshold(path, alpha=level, \
                    height_control=height_control, cluster_threshold=cluster_threshold)[1]
        #threshold=1.96
        
        XYZ = None
        if 'coord' in data:
            XYZ = get_activations(path, threshold, space='pos')
            if XYZ is None:
                return
            if len(XYZ[0]) < 1:
                return

        if 'path' in data:
            base, filename = ntpath.split(path)
            file, ext = filename.split('.', 1)

            path_dict = {'z': path}
            for map_type in ['t', 'con', 'se']:
                file_path = f'{base}/{file}_{map_type}.{ext}'
                if os.path.isfile(file_path):
                    path_dict[map_type] = file_path

            return get_sub_dict(XYZ, path_dict, sample_size)

        if XYZ is not None:
            return get_sub_dict(XYZ, None, sample_size)

        return

    n_jobs = multiprocessing.cpu_count()
    res = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(extract_pool)(path) for path in paths)

    # Removing potential None values
    res = list(filter(None, res))
    
    # Merging all dictionaries
    return {k: v for k, v in enumerate(res)}


def run_ALE(ds_dict):
    """Run ALE on given data."""
    ds = Dataset(ds_dict)
    ma = nimare.meta.cbma.ale.ALE()
    res = ma.fit(ds)

    img_ale = res.get_map('ale')
    img_p = res.get_map('p')
    img_z = res.get_map('z')

    return img_ale, img_p, img_z

def run_MFX_GLM(ds_dict):
    """Run MFX_GLM on given data."""
    ds = Dataset(ds_dict)
    ma = nimare.meta.ibma.MFX_GLM()
    res = ma.fit(ds)

    return res.get_map('t')


def fdr_threshold(img_list, img_p, q=0.05):
    """Compute FDR and threshold same-sized images."""
    arr_list = [copy.copy(img.get_fdata()) for img in img_list]
    arr_p = img_p.get_fdata()
    aff = img_p.affine

    fdr = nimare.stats.fdr(arr_p.ravel(), q=q)

    for arr in arr_list:
        arr[arr_p > fdr] = 0

    res_list = [nib.Nifti1Image(arr, aff) for arr in arr_list]

    return res_list
