class LinearSchedule(object):
    """Linear schedule"""

    def __init__(self, sched_timesteps, final_p, init_p=1.0):
        """Interpolate linearly between `init_p` and `final_p`, over `sched_timesteps`
        timesteps, after which final_p is returned.

        Args:
            sched_timesteps (int): Number of timesteps for which to linearly anneal
                `initial_p` to `final_p`
            init_p (float): Initial output value
            final_p (float): Final output value
        """
        self.sched_timesteps = sched_timesteps
        self.final_p = final_p
        self.init_p = init_p

    def value(self, t):
        """Return the value of the schedule at timestep `t`"""
        frac = min(float(t) / self.sched_timesteps, 1.0)
        return self.init_p + frac * (self.final_p - self.init_p)
