
class LinearDecayEpsilonGreedy:
    def __init__(self, eps_init, eps_final, eps_decay_steps):
        self.eps_init        = eps_init
        self.eps_final       = eps_final
        self.eps_decay_steps = eps_decay_steps

        self.eps_inc = (eps_final - eps_init) / eps_decay_steps
        self.eps_t   = 0

    def get_epsilon(self, mode):
        "Returns the current epsilon based on linear scheduling."

        if mode == "train":
            self.current_eps = max(self.eps_init + self.eps_inc * self.eps_t, self.eps_final)
            self.eps_t += 1
        
        else:
            self.current_eps = 0

        return self.current_eps
