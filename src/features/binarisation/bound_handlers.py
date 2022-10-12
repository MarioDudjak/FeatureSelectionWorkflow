class EmptyBoundHandler:

    def __init__(self, lower_bound=-1, upper_bound=1):
        """
        Used solely for initialising the population
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name = "EmptyBoundHandler"

    def handle_bounds(self, x):
        pass


class LimitBoundHandler:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name = "LimitBoundHandler"

    def handle_bounds(self, x):
        x[x > self.upper_bound] = self.upper_bound
        x[x < self.lower_bound] = self.lower_bound

        # Check if any are true. If note, set one to upper bound.
