# Note that much of this code was borrowed/adapted from Alexandre Perez 
# (https://github.com/alexprz/meta_analysis_notebook).

import os
import multiprocessing
from joblib import Parallel, delayed
import nilearn
import numpy as np
from nilearn import masking, plotting
from nipy.labs.statistical_mapping import get_3d_peaks
import nibabel as nib
import scipy
from matplotlib import pyplot as plt
import ntpath
from nimare.dataset import Dataset
import nimare
import copy
from nistats import thresholding

template = nilearn.datasets.load_mni152_template()
Ni, Nj, Nk = template.shape
affine = template.affine
gray_mask = masking.compute_gray_matter_mask(template)

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
        return None

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

def get_data_paths(input_dir):
    img_paths = []
    i = 0

    while True:
        path = os.path.abspath(f'{input_dir}hypo1_unthresh_{i}.nii.gz')

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
