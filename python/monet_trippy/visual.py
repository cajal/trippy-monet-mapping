
class Visual:
    """
    Abstract stimulus movie for computing receptive fields
    """
    @property
    def params(self):
        """
        :return: dict uniquely summarizing the stimulus
        """
        raise NotImplementedError("This property must be implemented by subclasses")

    def from_conditions(self):
        """ construct from a stimulus condition entry of the Cajal DataJoint pipeline """
        raise NotImplementedError("This methods must be implemented by subclasses")

    @property
    def movie(self):
        """
        :return: uint8 grayscale movie of size (nframes, ny, nx)
        """
        raise NotImplementedError("This property must be implemented by subclasses")

    @property
    def fps(self):
        raise NotImplementedError("This property must be implemented by subclasses")

    def export(self, filename=None, suffix='', **kwargs):
        """
        save as an mp4 file
        """
        import imageio, os
        filename = filename or ''.join((self.__class__.__name__, str(suffix), '.mp4'))
        print('save ' + os.path.abspath(filename))
        imageio.mimwrite(filename, self.movie, fps=float(self.fps), **kwargs)
