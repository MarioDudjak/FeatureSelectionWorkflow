class FixedBinariser:
    """
    Used for baseline results, standard binariser for constrained solutions in 0.0-1.0
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.name = "FixedBinariser"

    def binarise(self, x, binary_x=None):
        return x >= self.alpha
