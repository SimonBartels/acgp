import logging
from acgp.backends.numpy_backend import NumpyBackend


class Bounds:
    def __init__(self, delta, N, min_noise, backend=NumpyBackend()):
        """
        Initializer for the bound computation. This method collects all constants which are static throughout the
        computation of the Cholesky.
        :param delta: failure chance for the bounds
        :param N: the overall size of the dataset
        :param min_noise: the lowest diagonal element of A that could occur
        """
        self.backend = backend
        self.N = N
        self.delta = delta
        self.min_noise = min_noise
        self.log_min_noise = self.backend.log(min_noise)
        self.zero = 0. * self.backend.ones(1)
        self.log_var_estimate = self.backend.log(min_noise)
        self.covar_estimate = self.backend.ones(1)
        self.worst_case_estimate = 0. * self.backend.ones(1)
        self.average_calibration = 0. * self.backend.ones(1)
        self.expected_worst_case_increase_rate = self.backend.ones(1)

    def get_bound_estimators(self, *args, **kwargs):
        """
        This function exists for backwards compatibility reasons. See #get_bound_estimators_and_auxilary_quantities for
        dcoumentation.
        :param args:
        :param kwargs:
        :return:
        """
        U_det, L_det, U_quad, L_quad, _ = self.get_bound_estimators_and_auxilary_quantities(*args, **kwargs)
        return U_det, L_det, U_quad, L_quad

    def get_bound_estimators_and_auxilary_quantities(self, t0: int, log_sub_det: float, sub_quad: float, A_diag, A_diag_off, noise_diag, y):
        """
        This function returns upper and lower bound estimators for log-determinant and quadratic form in a step t of the
        Cholesky decomposition.

        :param t0: the number of datapoints for which the Cholesky has been fully computed
        :param log_sub_det: the log-determinant of the kernel matrix of this subset
        :param sub_quad: the quadratic form of this subset
        :param A_diag: the diagonal of the partially computed Cholesky from t0 to the current time step
        :param A_diag_off: the 1-diagonal of the partially computed Cholesky
        :param noise_diag: scalar or diagonal matrix that contains the noise for datapoints t0+1 to t
        :param y: the partially solved linear equation system from t0 to t
        :return:
            upper and lower bound estimators for log-determinant and quadratic form
        """
        assert(len(A_diag.shape) == 2 and A_diag.shape[1] == 1)
        assert(len(A_diag_off.shape) == 2 and A_diag_off.shape[1] == 1)
        assert(len(y.shape) == 2 and y.shape[1] == 1)
        assert(A_diag.shape[0] == y.shape[0] == A_diag_off.shape[0] + 1)
        assert(noise_diag.shape == A_diag_off.shape or (noise_diag.shape[0] == 1 and len(noise_diag.shape) == 1))

        auxilary_variables = {}

        # utility variables which occur repeatedly
        t = t0 + A_diag.shape[0]  # the number of seen data points
        y2 = self.backend.square(y)  # square error of the GP regressor on the new batch of targets
        # the name can be confusing--we divide by sigma^4 here!
        square_noise_covar = self.backend.square(A_diag_off / noise_diag)
        calibration = y2 / A_diag

        # determinant bounds
        m = t - t0  # the number of most recent observations used for our estimators
        log_var_estimate = self.backend.sum(self.backend.log(A_diag)) / m  # estimator for E[log k_{t0}(x, x)+noise(x)|x_0, ..., x_{t_0}]
        # NOTE: this estimator deviates from the theory where we could take only half of the points
        covar_estimate = self.backend.sum(square_noise_covar) / (m - 1)
        #epsilon = 0  # TODO: this term depends on delta and worst-case constants
        #psi = t0 + self._get_advance(mean_estimate=log_var_estimate, estimated_change=covar_estimate,
        #                             deterministic_worst_case_constant=self.log_min_noise, epsilon=epsilon)
        # TODO: decide whether to go against theory here or not
        self.log_var_estimate = log_var_estimate
        self.covar_estimate = covar_estimate
        if covar_estimate > 0:
            # it can occur that the covariance estimate becomes numerically 0
            #psi = t0 + max(0, self.backend.floor(2 * (self.log_var_estimate - self.log_min_noise) / self.covar_estimate) - 1)
            psi = t0 + self.backend.floor((self.log_var_estimate - self.log_min_noise) / self.covar_estimate + 0.5)
        else:
            psi = self.N
        psi = min(self.N, psi)
        self.log_var_estimate = log_var_estimate
        self.covar_estimate = covar_estimate
        U_det = log_sub_det + (self.N - t0) * log_var_estimate
        L_det = log_sub_det + (psi - t0) * (log_var_estimate - (psi - t0 - 1) / 2 * covar_estimate) \
                + (self.N - psi) * self.log_min_noise

        assert(U_det >= L_det)

        #diff_to_deterministic_lower_bound = log_var_estimate - (psi - t0 + 1) / 2 * covar_estimate - self.log_min_noise
        #assert diff_to_deterministic_lower_bound >= -1e-7, f"Estimated lower bound - deterministic lower bound is {diff_to_deterministic_lower_bound}"

        # quadratic form bounds
        # NOTE: this estimator deviates from the theory where we could take only half of the points
        average_calibration = self.backend.sum(calibration) / m
        expected_worst_case_increase_rate = self.backend.sum(square_noise_covar * calibration[1:]) / (m - 1)
        #worst_case_estimate = self.backend.sum(y2[1:] / noise_diag) / (m - 1)
        # TODO: the implementation below requires that diag is a scalar!
        worst_case_estimate = self.backend.sum(y2 / noise_diag) / m
        # TODO: decide whether to go against theory here or not
        self.worst_case_estimate = worst_case_estimate
        self.average_calibration = average_calibration
        self.expected_worst_case_increase_rate = expected_worst_case_increase_rate
        if expected_worst_case_increase_rate > 0:
            p = self.backend.floor((self.worst_case_estimate - self.average_calibration) / self.expected_worst_case_increase_rate + 0.5)
        else:
            p = self.N - t0
        p = min(p, self.N - t0)
        self.worst_case_estimate = worst_case_estimate
        self.average_calibration = average_calibration
        self.expected_worst_case_increase_rate = expected_worst_case_increase_rate

        assert(p * worst_case_estimate >= p * (average_calibration + (p - 1) / 2 * expected_worst_case_increase_rate))

        U_quad = sub_quad + p * (average_calibration + (p - 1) / 2 * expected_worst_case_increase_rate) \
                 + (self.N - (t0 + p)) * worst_case_estimate
        #factor = self.backend.sum(y2[1:, :] * (noise_diag + self.backend.square(A_diag_off) / noise_diag) / A_diag[1:, :] / noise_diag) / (m - 1)
        #U_quad = sub_quad + (self.N - t0) * factor

        # average_square_error = self.backend.sum(y2) / m
        # average_var_times_square_error = self.backend.sum(y2 * A_diag) / m
        # # NOTE: this estimator deviates from the theory where we could take only half of the points
        # # NOTE: also we use only the positive part as a sign-switch makes things nasty
        # average_signed_covar = max(0, self.backend.sum(y[:-1] * y[1:] * A_diag_off) / (m - 1))
        # #average_signed_covar = self.backend.sum(y[:-1] * y[1:] * A_diag_off) / (m - 1)
        # c = (self.N - t0) * 2 * average_square_error
        # b = (self.N - t0) * (average_var_times_square_error + average_signed_covar * (self.N - t0 - 1))
        # a = c / (2 * b)  # a is the maximizer of the quadratic function below
        # add = a * (c - a * b)

        calibrated_error_correlation = self.backend.sum(y[:-1] / A_diag[:-1] * (y[1:] / A_diag[1:]) * A_diag_off) / (m - 1)
        # small fluctuations can cause the lower bound to be higher than the upper bound
        # by taking only positive error correlations we guard against that
        #calibrated_error_correlation = self.backend.abs(calibrated_error_correlation)
        calibrated_error_correlation = max(0, calibrated_error_correlation)
        #calibrated_error_correlation = self.backend.sum(y[::2] / A_diag[::2] * (y[1::2] / A_diag[1::2]) * A_diag_off) / (
        #            m - 1) * 2
        #average_calibration = self.backend.sum(calibration[:-1] / 2 + calibration[1:] / 2) / (m - 1)
        add = average_calibration - (self.N - t0 - 1) * calibrated_error_correlation
        #assert(add >= 0.)
        add = add * (self.N - t0)
        L_quad = sub_quad + max(0, add)

        assert(U_quad >= L_quad)

        return U_det, L_det, U_quad, L_quad, auxilary_variables

    def _get_advance(self, mean_estimate, estimated_change, deterministic_worst_case_constant, epsilon):
        """
        This function computes the number of steps how much an estimated quantity can change before it hits the worst
        case constant.
        """
        try:
            # TODO: this step can be numerically tricky
            x_plus, x_minus = self._solve_quadratic(-estimated_change / 2, mean_estimate + estimated_change / 2 - deterministic_worst_case_constant, epsilon)
            return int(self.backend.floor(max(x_plus, x_minus)))
        except Exception as e:
            logging.exception(e)
            logging.log(level=logging.ERROR, msg=f"mean_estimate: {mean_estimate}, estimated_change: {estimated_change}, deterministic_worst_case_constant: {deterministic_worst_case_constant}")
            # above fails either because estimated_change is too close to zero or for some other reasons
            # in both cases we want to terminate
            return self.N

    def _solve_quadratic(self, a, b, c):
        """
        Solves a quadratic equation ax^2+bx+c=0
        :param a:
        :param b:
        :param c:
        :return:
        """
        if a == self.zero:
            # the equation is actually linear--happens sometimes
            return -c / b, -c / b
        t = self.backend.sqrt(self.backend.square(b) - 4 * a * c)
        return (-b + t) / 2 / a, (-b - t) / 2 / a


class ExperimentalBounds(Bounds):
    def get_bound_estimators_and_auxilary_quantities(self, t0: int, log_sub_det: float, sub_quad: float, A_diag, A_diag_off, noise_diag, y):
        U_det, L_det, U_quad_, L_quad_, aux = super().get_bound_estimators_and_auxilary_quantities(
            t0=t0, log_sub_det=log_sub_det, sub_quad=sub_quad, A_diag=A_diag, A_diag_off=A_diag_off,
            noise_diag=noise_diag, y=y)
        m = A_diag.shape[0]  # the number of seen data points

        average_model_calibration = self.backend.sum(self.backend.square(y) / A_diag) / m
        L_quad__ = sub_quad + average_model_calibration * (self.N - t0)

        y2 = self.backend.square(y)
        # the name can be confusing--we divide by sigma^4 here!
        square_noise_covar = self.backend.square(A_diag_off / noise_diag)

        average_model_calibration = self.backend.sum(y2 / A_diag) / m
        aux["average_model_calibration"] = average_model_calibration.item()
        worst_case_estimate = self.backend.sum(y2 / self.min_noise) / m
        expected_worst_case_increase_rate = self.backend.sum(y2[1:, :] * square_noise_covar / A_diag[1:, :]) / (m - 1)

        aux["expected_worst_case_increase_rate"] = expected_worst_case_increase_rate.item()
        epsilon = 0  # TODO: this term depends on delta and worst-case constants
        psi = t0 + self._get_advance(mean_estimate=average_model_calibration,
                                    estimated_change=-expected_worst_case_increase_rate,
                                    deterministic_worst_case_constant=worst_case_estimate, epsilon=epsilon)
        psi = min(self.N, psi)
        expected_final_model_calibration = average_model_calibration + (psi - t0 + 1) / 2 * expected_worst_case_increase_rate
        aux["expected_final_model_calibration"] = expected_final_model_calibration
        U_quad = sub_quad + (psi - t0) * expected_final_model_calibration + (self.N - psi) * worst_case_estimate

        average_absolute_error = self.backend.sum(self.backend.abs(y)) / m
        average_var = self.backend.sum(A_diag) / m
        prediction_error_signs = self.backend.sign(y)
        # NOTE: this estimator deviates from the theory where we could take only half of the points
        # NOTE: also we use only the positive part as a sign-switch makes things nasty
        average_signed_covar = max(0, self.backend.sum(prediction_error_signs[:-1] * prediction_error_signs[1:] * A_diag_off) / (m - 1))
        c = (self.N - t0) * 2 * average_absolute_error
        b = (self.N - t0) * (average_var + average_signed_covar * ((self.N - t0) - 1))
        a = c / (2 * b)  # a is the maximizer of the quadratic function below
        add = a * (c - a * b)
        L_quad = sub_quad + max(0, add)


        aux["alternative_provable_probabilistic_bound"] = L_quad
        aux["original_quad_lower_bound"] = L_quad_
        aux["quad_lower_bound_assuming_worsening_model_calibration"] = L_quad__
        aux["original_quad_upper_bound"] = U_quad_
        return U_det, L_det, U_quad, L_quad, aux
