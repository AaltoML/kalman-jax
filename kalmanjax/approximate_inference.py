

class EP(object):
    """
    Expectation propagation
    """
    def __init__(self, ep_fraction=1.0):
        self.ep_fraction = ep_fraction

    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses moment matching to update the site parameters
        """
        if site_params is not None:
            site_mean, site_var = site_params
            # --- Compute the cavity distribution ---
            # remove local likelihood approximation to obtain the marginal cavity distribution:
            var_cav = 1.0 / (1.0 / v - self.ep_fraction / site_var)  # cavity variance
            mu_cav = var_cav * (m / v - self.ep_fraction * site_mean / site_var)  # cav. mean
            # calculate the new sites via moment matching
        else:
            mu_cav, var_cav = m, v
        return likelihood.moment_match(y, mu_cav, var_cav, hyp, site_update, self.ep_fraction)


class GHKS(object):
    """
    Iterated Kalman smoother (using Gauss-Hermite quadrature if necessary)
    """
    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses moment matching to update the site parameters
        """
        return likelihood.moment_match(y, m, v, hyp, site_update, 1.0)


class PL(object):
    """
    Posterior linearisation
    """
    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses SLR to update the site parameters
        """
        return likelihood.statistical_linear_regression(y, m, v, hyp, site_update)
