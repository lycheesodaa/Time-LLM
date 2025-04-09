import math

import pandas as pd
import numpy as np
import pywt
from PyEMD import EMD, EEMD, CEEMDAN
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


def pad_to_length(array: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad array to target length with zeros.

    Args:
        array: Input array
        target_length: Desired length

    Returns:
        Padded array
    """
    pad_width = target_length - len(array)
    if pad_width > 0:
        return np.pad(array, (0, pad_width), mode='constant')
    return array


class WaveletDecomposition:
    def __init__(self, wavelet: str = 'db4', level: int = 3):
        """
        Initialize wavelet decomposition.

        Args:
            wavelet: Wavelet type (default: 'db4')
            level: Decomposition level (default: 3)
        """
        self.wavelet = wavelet
        self.level = level
        self.coefficients = None

    def calculate_max_length(self, data_length: int) -> int:
        """
        Calculate maximum length of wavelet coefficients at any level.

        Args:
            data_length: Length of input data

        Returns:
            Maximum length among all coefficient levels
        """
        max_length = data_length
        for i in range(self.level):
            max_length = pywt.dwt_coeff_len(max_length, pywt.Wavelet(self.wavelet), 'symmetric')
        return max_length

    def decompose(self, data: np.ndarray, return_dict: bool = False) -> Dict | np.ndarray:
        """
        Perform wavelet decomposition.

        Args:
            data: 1D numpy array of prices
            return_dict: Flag to indicate if a full numpy array should be returned instead of a dict

        Returns:
            Dictionary containing coefficients and details at each level or a full numpy array with approximation at
            index 0.
        """
        # Perform wavelet decomposition
        self.coefficients = pywt.wavedec(data, self.wavelet, level=self.level)

        if return_dict:
            # Return raw coefficients (different lengths)
            decomp = {
                'approximation': self.coefficients[0],
                'details': {}
            }
            # print("\nRaw coefficient lengths:")
            # print(f"Approximation: {len(self.coefficients[0])}")

            for i in range(1, len(self.coefficients)):
                decomp['details'][f'D{i}'] = self.coefficients[i]
                # print(f"Detail D{i}: {len(self.coefficients[i])}")

            return decomp
        else:
            # Pad each level to original signal length
            components = np.zeros((self.level + 1, data.shape[0]))
            for i in range(len(components)):
                components[i] = pad_to_length(self.coefficients[0], len(components[0]))

            # print("\nPadded component shapes:")
            # print(f"Array shape: {components.shape}")
            # print("All components padded to maximum deconstructed length")

            return components

    def reconstruct(self, coefficients: List = None) -> np.ndarray:
        """
        Reconstruct signal from wavelet coefficients.

        Args:
            coefficients: List of coefficients (if None, uses stored coefficients)

        Returns:
            Reconstructed signal
        """
        if coefficients is None:
            coefficients = self.coefficients
        return pywt.waverec(coefficients, self.wavelet)

    def denoise(self, data: np.ndarray, threshold_method: str = 'soft',
                threshold_level: float = None) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.

        Args:
            data: Input signal
            threshold_method: 'soft' or 'hard'
            threshold_level: Custom threshold level (if None, uses universal threshold)

        Returns:
            Denoised signal
        """
        # Decompose
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)

        # Calculate threshold
        if threshold_level is None:
            threshold_level = np.sqrt(2 * np.log(len(data)))

        # Apply thresholding
        denoised_coeffs = []
        for i in range(len(coeffs)):
            if i == 0:  # Skip approximation coefficients
                denoised_coeffs.append(coeffs[i])
            else:
                if threshold_method == 'soft':
                    thresh_coeffs = pywt.threshold(coeffs[i], threshold_level, mode='soft')
                else:
                    thresh_coeffs = pywt.threshold(coeffs[i], threshold_level, mode='hard')
                denoised_coeffs.append(thresh_coeffs)

        # Reconstruct
        return pywt.waverec(denoised_coeffs, self.wavelet)


class EMDDecomposition:
    def __init__(self, decomp_type='emd', max_imfs=9):
        """Initialize EMD decomposition."""
        self.decomp_type = decomp_type
        if self.decomp_type == 'emd':
            self.emd = EMD(DTYPE=np.float16, spline_kind='akima')
        elif self.decomp_type == 'eemd':
            self.emd = EEMD(DTYPE=np.float16, spline_kind='akima')
        elif self.decomp_type == 'ceemdan':
            self.emd = CEEMDAN(DTYPE=np.float16, spline_kind='akima')
        self.imfs = None
        self.residue = None
        self.max_imfs = max_imfs

    def decompose(self, data: np.ndarray, return_dict: bool = False) -> Dict | np.ndarray:
        """
        Perform EMD decomposition.

        Args:
            data: 1D numpy array of prices
            return_dict: Flag to indicate whether to return dict, or just the IMFs array

        Returns:
            Dictionary containing IMFs and residue or a numpy array containing IMFs only
        """
        # Perform EMD
        if self.decomp_type == 'emd':
            self.imfs = self.emd.emd(data, max_imf=self.max_imfs)
        elif self.decomp_type == 'eemd':
            self.imfs = self.emd.eemd(data, max_imf=self.max_imfs)
        elif self.decomp_type == 'ceemdan':
            self.imfs = self.emd.ceemdan(data, max_imf=self.max_imfs)
        self.residue = data - np.sum(self.imfs, axis=0)

        # Create dictionary to store decomposition
        decomp = {
            'imfs': self.imfs,
            'residue': self.residue
        }

        if not return_dict:
            return decomp['imfs']

        return decomp

    def reconstruct(self, imfs: np.ndarray = None, residue: np.ndarray = None) -> np.ndarray:
        """
        Reconstruct signal from IMFs and residue.

        Args:
            imfs: Array of IMFs (if None, uses stored IMFs)
            residue: Residue (if None, uses stored residue)

        Returns:
            Reconstructed signal
        """
        if imfs is None:
            imfs = self.imfs
        if residue is None:
            residue = self.residue

        return np.sum(imfs, axis=0) + residue

    def denoise(self, data: np.ndarray, n_imfs: int = None) -> np.ndarray:
        """
        Denoise signal by removing high-frequency IMFs.

        Args:
            data: Input signal
            n_imfs: Number of IMFs to keep (if None, uses half of total IMFs)

        Returns:
            Denoised signal
        """
        # Decompose
        imfs = self.emd.emd(data)

        if n_imfs is None:
            n_imfs = len(imfs) // 2

        # Keep only specified number of IMFs (removing high-frequency components)
        denoised_imfs = imfs[:n_imfs]

        # Reconstruct
        return np.sum(denoised_imfs, axis=0)


def plot_decomposition(original: np.ndarray, decomp: Dict, method: str,
                       dates: pd.DatetimeIndex = None):
    """
    Plot original signal and its decomposition.

    Args:
        original: Original signal
        decomp: Decomposition dictionary
        method: 'wavelet' or 'emd'
        dates: DatetimeIndex for x-axis (optional)
    """
    if method == 'wavelet':
        n_plots = len(decomp['details']) + 2  # +2 for original and approximation
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

        # Plot original signal
        axes[0].plot(original)
        axes[0].set_title('Original Signal')

        # Plot approximation
        axes[1].plot(decomp['approximation'])
        axes[1].set_title('Approximation')

        # Plot details
        for i, (name, detail) in enumerate(decomp['details'].items(), 2):
            axes[i].plot(detail)
            axes[i].set_title(f'Detail Level {name}')

    elif method == 'emd':
        n_plots = len(decomp['imfs']) + 2  # +2 for original and residue
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

        # Plot original signal
        if dates is not None:
            axes[0].plot(dates, original)
        else:
            axes[0].plot(original)
        axes[0].set_title('Original Signal')

        # Plot IMFs
        for i, imf in enumerate(decomp['imfs'], 1):
            if dates is not None:
                axes[i].plot(dates, imf)
            else:
                axes[i].plot(imf)
            axes[i].set_title(f'IMF {i}')

        # Plot residue
        if dates is not None:
            axes[-1].plot(dates, decomp['residue'])
        else:
            axes[-1].plot(decomp['residue'])
        axes[-1].set_title('Residue')

    # plt.tight_layout()
    # plt.show()
    return fig