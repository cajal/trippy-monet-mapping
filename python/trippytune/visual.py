from scipy import signal
import numpy as np


class Visual:
    """
    Abstract stimulus movie
    """

    def get_intensity_frame(self, frame):
        """
        :param frame: number (can be fractional)
        :return: intensity frame corresponding to time
        """


class Trippy(Visual):

    def __init__(self, rng_seed=0, fps=60, packed_phase_movie=None, tex_size=(160, 90), nodes=(12, 6), up_factor=24,
                 duration=15, temp_freq=4.0, temp_kernel_length=61, spatial_freq=0.08):
        """
        :param rng_seed:  random number generator seed
        :param fps:  (Hz) monitor refresh rate
        :param packed_phase_movie: random packed movie (obviates random seed)
        :param tex_size: (x,y) pixel size of the output
        :param nodes: (nx,ny) dimensions of packed_phase_movie frame
        :param up_factor: spatial upscale factor
        :param duration: (s) clip duration
        :param temp_freq: central temporal frequency
        :param temp_kernel_length: length of Hanning kernel used for temporal filtering of the phase movie
        :param spatial_freq: (cy/point) approximate max spatial frequency. Actual frequencies may be higher.
        """

        if packed_phase_movie is not None:
            self.packed_phase_movie = packed_phase_movie
            self.rng_seed = None  # override if the packed movie is provided
        else:
            self.rng_seed = rng_seed
            np.random.seed(np.uint64(rng_seed))
            assert temp_kernel_length >= 3 and temp_kernel_length % 2 == 1
            nframes = np.ceil(duration * np.float32(fps))
            k2 = np.ceil(temp_kernel_length / 4)
            compensator = 8.0
            scale = compensator * up_factor * spatial_freq
            self.packed_phase_movie = (
                    scale * np.random.rand(int(np.ceil((nframes + temp_kernel_length - 1) / k2)), nodes[0] * nodes[1]))
        self.fps = fps
        self.tex_size = list(tex_size)
        self.nodes = list(nodes)
        self.up_factor = up_factor
        self.duration = duration
        self.temp_freq = temp_freq
        self.temp_kernel_length = temp_kernel_length
        self.spatial_freq = spatial_freq
        self._phase_movie = None

    @staticmethod
    def from_condition(cond):
        # construct from a stimulus condition in the Cajal database
        return Trippy(
            **{k: v for k, v in cond.items() if k in {
                'fps', 'rng_seed', 'packed_phase_movie', 'up_factor', 'temp_freq',
                'temp_kernel_length', 'duration', 'spatial_freq'}},
            tex_size=(cond['tex_xdim'], cond['tex_ydim']),
            nodes=(cond['xnodes'], cond['ynodes']))

    def __hash__(self):
        return hash(
            (self.packed_phase_movie.tobytes(), self.fps, tuple(self.tex_size), tuple(self.nodes), self.up_factor,
             self.duration, self.temp_freq, self.temp_kernel_length, self.spatial_freq))

    @staticmethod
    def upsample(array, factor, axis=0, phase=0):
        """
        Upsamples arrays by factor along axis, filling with zeros.
        Emulates MATLAB upsample.
        """
        return signal.resample_poly(
            array, factor, 1, window=[0] * phase * 2 + [1 / factor], axis=axis)

    @staticmethod
    def interp_time(packed_phase_movie, temp_kernel_length, duration, fps, temp_freq):
        assert temp_kernel_length >= 3 and temp_kernel_length % 2 == 1
        k2 = int(np.ceil(temp_kernel_length / 4))
        phase = Trippy.upsample(packed_phase_movie, k2)
        temp_kernel = np.hanning(temp_kernel_length + 2)[1:-1]
        temp_kernel *= k2 / temp_kernel.sum()
        phase = signal.convolve(phase, temp_kernel[:, None], 'valid')  # lowpass in time
        nframes = phase.shape[0]
        assert nframes == int(np.ceil(np.float(duration) * np.float(fps)))
        return phase + (np.r_[:nframes][:, None] + 1) / np.float(fps) * np.float(temp_freq)  # add motion

    @staticmethod
    def frozen_upscale(img, factor, axis):
        """
        Upscale image efficiently. This function is designated as frozen because changes in its function
        will make some results backward incompatible.
        :param img: 2d image
        :param factor: upscale factor
        :param axis: axis to upscale
        :return: full-size movie image
        """
        img = Trippy.upsample(img, factor, axis, phase=factor // 2)
        length = img.shape[axis]
        sigma = (length - 1) / (2 * np.sqrt(0.5) * length / factor)
        k = signal.gaussian(length, sigma)
        k = np.fft.fft(np.fft.ifftshift(factor * k / k.sum()))
        k = k.reshape([1] * axis + list(k.shape) + [1] * (img.ndim - 1 - axis))
        return np.real(np.fft.ifft(np.fft.fft(img, axis=axis) * k, axis=axis))

    @property
    def phase_movie(self):
        """
        Reconstruct the phase movie from packed_phase_movie.
        """
        if self._phase_movie is None:
            phase = Trippy.interp_time(
                packed_phase_movie=self.packed_phase_movie, temp_kernel_length=self.temp_kernel_length, fps=self.fps,
                duration=self.duration, temp_freq=self.temp_freq)
            movie = np.rollaxis(phase.reshape([phase.shape[0]] + self.nodes[::-1], order='F'), 0, 3)
            movie = Trippy.frozen_upscale(movie, self.up_factor, axis=1)
            movie = Trippy.frozen_upscale(movie, self.up_factor, axis=0)
            self._phase_movie = 2 * np.pi * movie[:self.tex_size[1], :self.tex_size[0], :]

        return self._phase_movie

    @property
    def movie(self):
        """
        :return: final result presented on the screen
        """
        return np.uint8(np.cos(self.phase_movie)*127.5+128)

