# coding=utf-8

import os
from ctypes import c_double

import cpbd
import cv2
import numpy as np
from PIL import Image, ImageStat
from colorthief import ColorThief
from scipy.special import gamma
from scipy.stats import entropy
import svmutil
from svmutil import gen_svm_nodearray
from metric_tests import continuous_stats
from utils.utils import fix_path

def perform_test(s1, s2, test_type, config=None):
    report = continuous_stats.test(s1, s2, test_type, config)
    return report


class BRISQUE(object):
    def __init__(self):
        self._model = svmutil.svm_load_model('models/brisque_svm.txt')
        # if self._model is None:
        #     self._model = svmutil.svm_load_model('models/brisque_svm.txt')

        self._scaler = np.array([
            [-1, 1], [0.338, 10], [0.017204, 0.806612], [0.236, 1.642],
            [-0.123884, 0.20293], [0.000155, 0.712298], [0.001122, 0.470257],
            [0.244, 1.641], [-0.123586, 0.179083], [0.000152, 0.710456],
            [0.000975, 0.470984], [0.249, 1.555], [-0.135687, 0.100858],
            [0.000174, 0.684173], [0.000913, 0.534174], [0.258, 1.561],
            [-0.143408, 0.100486], [0.000179, 0.685696], [0.000888, 0.536508],
            [0.471, 3.264], [0.012809, 0.703171], [0.218, 1.046],
            [-0.094876, 0.187459], [1.5e-005, 0.442057], [0.001272, 0.40803],
            [0.222, 1.042], [-0.115772, 0.162604], [1.6e-005, 0.444362],
            [0.001374, 0.40243], [0.227, 0.996],
            [-0.117188, 0.09832299999999999], [3e-005, 0.531903],
            [0.001122, 0.369589], [0.228, 0.99], [-0.12243, 0.098658],
            [2.8e-005, 0.530092], [0.001118, 0.370399]])

    @staticmethod
    def preprocess_image(img):
        """Handle any kind of input for our convenience.

        :param img: The image path or array.
        :type img: str, np.ndarray
        """
        if isinstance(img, str):
            if os.path.exists(img):
                return cv2.imread(img, 0).astype(np.float64)
            else:
                raise FileNotFoundError('The image is not found on your '
                                        'system.')
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                image = img
            elif len(img.shape) == 3:
                if img.max() < 2:
                    img = (img * 255).astype('uint8')

                image = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError('The image shape is not correct.')

            return image.astype(np.float64)
        else:
            raise ValueError('You can only pass image to the constructor.')

    @staticmethod
    def _estimate_ggd_param(vec):
        """Estimate GGD parameter.

        :param vec: The vector that we want to approximate its parameter.
        :type vec: np.ndarray
        """
        gam = np.arange(0.2, 10 + 0.001, 0.001)
        r_gam = (gamma(1.0 / gam) * gamma(3.0 / gam) / (gamma(2.0 / gam) ** 2))

        sigma_sq = np.mean(vec ** 2)
        sigma = np.sqrt(sigma_sq)
        E = np.mean(np.abs(vec))
        rho = sigma_sq / E ** 2

        differences = abs(rho - r_gam)
        array_position = np.argmin(differences)
        gamparam = gam[array_position]

        return gamparam, sigma

    @staticmethod
    def _estimate_aggd_param(vec):
        """Estimate AGGD parameter.

        :param vec: The vector that we want to approximate its parameter.
        :type vec: np.ndarray
        """
        gam = np.arange(0.2, 10 + 0.001, 0.001)
        r_gam = ((gamma(2.0 / gam)) ** 2) / (
                gamma(1.0 / gam) * gamma(3.0 / gam))

        left_std = np.sqrt(np.mean((vec[vec < 0]) ** 2))
        right_std = np.sqrt(np.mean((vec[vec > 0]) ** 2))
        gamma_hat = left_std / right_std
        rhat = (np.mean(np.abs(vec))) ** 2 / np.mean((vec) ** 2)
        rhat_norm = (rhat * (gamma_hat ** 3 + 1) * (gamma_hat + 1)) / (
                (gamma_hat ** 2 + 1) ** 2)

        differences = (r_gam - rhat_norm) ** 2
        array_position = np.argmin(differences)
        alpha = gam[array_position]

        return alpha, left_std, right_std

    def get_feature(self, img):
        """Get brisque feature given an image.

        :param img: The path or array of the image.
        :type img: str, np.ndarray
        """
        imdist = self.preprocess_image(img)

        scale_num = 2
        feat = np.array([])

        for itr_scale in range(scale_num):
            mu = cv2.GaussianBlur(
                imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
            mu_sq = mu * mu
            sigma = cv2.GaussianBlur(
                imdist * imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
            sigma = np.sqrt(abs((sigma - mu_sq)))
            structdis = (imdist - mu) / (sigma + 1)

            alpha, overallstd = self._estimate_ggd_param(structdis)
            feat = np.append(feat, [alpha, overallstd ** 2])

            shifts = [[0, 1], [1, 0], [1, 1], [-1, 1]]
            for shift in shifts:
                shifted_structdis = np.roll(
                    np.roll(structdis, shift[0], axis=0), shift[1], axis=1)
                pair = np.ravel(structdis, order='F') * \
                       np.ravel(shifted_structdis, order='F')
                alpha, left_std, right_std = self._estimate_aggd_param(pair)

                const = np.sqrt(gamma(1 / alpha)) / np.sqrt(gamma(3 / alpha))
                mean_param = (right_std - left_std) * (
                        gamma(2 / alpha) / gamma(1 / alpha)) * const
                feat = np.append(
                    feat, [alpha, mean_param, left_std ** 2, right_std ** 2])

            imdist = cv2.resize(
                imdist,
                (0, 0),
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_NEAREST
            )
        return feat

    def get_score(self, img):
        """Get brisque score given an image.

        :param img: The path or array of the image.
        :type img: str, np.ndarray
        """
        feature = self.get_feature(img)
        scaled_feature = self._scale_feature(feature)

        return self._calculate_score(scaled_feature)

    def _scale_feature(self, feature):
        """Scale feature with svm scaler.

        :param feature: Brisque unscaled feature.
        :type feature: np.ndarray
        """
        y_lower = self._scaler[0][0]
        y_upper = self._scaler[0][1]
        y_min = self._scaler[1:, 0]
        y_max = self._scaler[1:, 1]
        scaled_feat = y_lower + (y_upper - y_lower) * ((feature - y_min) / (
                y_max - y_min))

        return scaled_feat

    def _calculate_score(self, scaled_feature):
        """Calculate score from scaled brisque feature.

        :param scaled_feature: Scaled brisque feature.
        :type scaled_feature: np.ndarray
        """
        x, idx = gen_svm_nodearray(
            scaled_feature.tolist(),
            isKernel=(self._model.param.kernel_type == 'PRECOMPUTED')
        )
        nr_classifier = 1
        prob_estimates = (c_double * nr_classifier)()

        return svmutil.libsvm.svm_predict_probability(
            self._model, x, prob_estimates)


def read_image(img):
    if isinstance(img, str):
        if os.path.exists(img):
            return cv2.imread(img).astype(np.uint8)
        else:
            raise FileNotFoundError('The image is not found on your '
                                    'system.')
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            image = img
        else:
            raise ValueError('The image shape is not correct.')

        return image.astype(np.uint8)
    else:
        raise ValueError('You can only pass image to the constructor.')


def brightness(filename):
    im = read_image(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = Image.fromarray(im)
    stat = ImageStat.Stat(im)
    return stat.mean[0]


def sharpness(filename):
    im = read_image(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    array = np.asarray(im, dtype=np.int32)

    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx ** 2 + gy ** 2)
    return np.average(gnorm)


def cpbd_metric(filename):
    im = read_image(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cpbd.compute(im)


def image_colorfulness(filename):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(np.array(read_image(filename)).astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def contrast(filename):
    im = read_image(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    input_image = np.array(im).reshape(-1)
    value, counts = np.unique(input_image, return_counts=True)
    return entropy(counts, base=2)


def rms_contrast(filename):
    brightness = 0
    contrast = 0
    (B, G, R) = cv2.split(np.array(read_image(filename)).astype("float"))
    m, n = B.shape
    B, G, R = B.flatten() / 255., G.flatten() / 255., R.flatten() / 255.
    for b, g, r in zip(B, G, R):
        brightness += (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
    brightness /= len(B)
    for b, g, r in zip(B, G, R):
        pxIntensity = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
        contrast += pow((brightness - pxIntensity), 2)
    contrast /= m
    contrast /= n
    return np.sqrt(contrast)


class MyColorThief(ColorThief):

    def __init__(self, file):
        """Create one color thief for one image.

        :param file: A filename (string) or a file object. The file object
                     must implement `read()`, `seek()`, and `tell()` methods,
                     and be opened in binary mode.
        """
        self.image = Image.fromarray(read_image(file))


def dominant_colors(filename, color_count=5):
    color_thief = MyColorThief(filename)
    return color_thief.get_palette(color_count=color_count)


if __name__ == '__main__':
    fix_path()
    brisq = BRISQUE()
    print(brisq.get_score('data/duck.jpg'))
    print(brightness('data/duck.jpg'))
    print(sharpness('data/duck.jpg'))
    print(cpbd_metric('data/duck.jpg'))
    print(image_colorfulness('data/duck.jpg'))
    print(contrast('data/duck.jpg'))
    print(rms_contrast('data/duck.jpg'))
    print(dominant_colors('data/duck.jpg'))
