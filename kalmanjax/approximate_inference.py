

class EP(object):
    """
    Expectation propagation
    """
    def __init__(self, site_params=None, power=1.0):
        self.site_params = site_params
        self.power = power

    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses moment matching to update the site parameters
        """
        if site_params is None:
            # if no site is provided, use the predictions/posterior as the cavity with ep_fraction=1
            return likelihood.moment_match(y, m, v, hyp, site_update, 1.0)  # calculate new sites via moment matching
        else:
            site_mean, site_var = site_params
            # --- Compute the cavity distribution ---
            # remove local likelihood approximation to obtain the marginal cavity distribution:
            var_cav = 1.0 / (1.0 / v - self.power / site_var)  # cavity variance
            mu_cav = var_cav * (m / v - self.power * site_mean / site_var)  # cav. mean
            # calculate the new sites via moment matching
            return likelihood.moment_match(y, mu_cav, var_cav, hyp, site_update, self.power)


class GHKS(object):
    """
    Iterated Kalman smoother (using Gauss-Hermite quadrature if necessary)
    """
    def __init__(self, site_params=None):
        self.site_params = site_params

    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses moment matching to update the site parameters
        """
        return likelihood.moment_match(y, m, v, hyp, site_update, 1.0)


class PL(object):
    """
    Posterior linearisation
    """
    def __init__(self, site_params=None):
        self.site_params = site_params

    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses SLR to update the site parameters
        """
        if site_update:
            site_mean, site_var = likelihood.statistical_linear_regression(y, m, v, hyp)
            return 0., site_mean, site_var
        else:
            return 0.


class CL(object):
    """
    Cavity linearisation
    """
    def __init__(self, site_params=None, power=1.0):
        self.site_params = site_params
        self.power = power

    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses SLR to update the site parameters
        """
        if site_update:
            if site_params is None:
                site_mean, site_var = likelihood.statistical_linear_regression(y, m, v, hyp)
            else:
                site_mean, site_var = site_params
                # --- Compute the cavity distribution ---
                # remove local likelihood approximation to obtain the marginal cavity distribution:
                var_cav = 1.0 / (1.0 / v - self.power / site_var)  # cavity variance
                mu_cav = var_cav * (m / v - self.power * site_mean / site_var)  # cav. mean
                site_mean, site_var = likelihood.statistical_linear_regression(y, mu_cav, var_cav, hyp)
            return 0., site_mean, site_var
        else:
            return 0.
