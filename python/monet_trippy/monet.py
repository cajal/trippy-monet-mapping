import re
from .visual import Visual



class Monet2(Visual):

    _parameter_names = re.split(
        r',\s*', 'rng_seed, pattern_width, pattern_aspect, temp_kernel, temp_bandwidth, '
                 'ori_coherence, ori_fraction, ori_mix, n_dirs, speed, directions, onsets')

    def __init__(self, rng_seed, pattern_width, pattern_aspect, temp_kernel, temp_bandwidth,
                 ori_coherence, ori_fraction, ori_mix, n_dirs, speed, directions, onsets, movie):
        """
        :param rng_seed:
        :param pattern_width:
        :param pattern_aspect:
        :param temp_kernel:
        :param temp_bandwidth:
        :param ori_coherence:
        :param ori_fraction:
        :param ori_mix:
        :param n_dirs:
        :param speed:
        :param directions:
        :param onsets:
        :param movie: (generated from above parameters in MATLAB only for now)
        """
        self.rng_seed = rng_seed,
        self.pattern_width = pattern_width,
        self.pattern_aspect = pattern_aspect
        self.temp_kernel = temp_kernel
        self.temp_bandwidth = temp_bandwidth
        self.ori_coherence = ori_coherence
        self.ori_fraction = ori_fraction
        self.ori_mix = ori_mix
        self.n_dirs = n_dirs
        self.speed = speed
        self.directions = directions
        self.onsets = onsets
        self._movie = movie
        self.nframes = self._movie.shape[2]

    @property
    def params(self):
        return {k: getattr(self, k) for k in self._parameter_names}

    @classmethod
    def from_condition(cls, condition):
        assert condition['stimulus_version'] == '6', "This code matches only Monet2 version 6."
        assert condition['blue_green_saturation'] == 0, "This code is for grayscale only"
        return cls(**{k: v for k, v in condition.items() if k in cls._parameter_names},
                   movie=condition['movie'].squeeze())

    @property
    def movie(self):
        return self._movie
