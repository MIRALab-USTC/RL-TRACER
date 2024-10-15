import torch
from torch.distributions import Distribution, Normal


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)

    @property
    def stddev(self):
        return self.normal_std

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def _log_prob_from_pre_tanh(self, pre_tanh_value):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        This formula is mathematically equivalent to log(1 - tanh(x)^2).

        Derivation:

        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = - 2. * (
            ptu.from_numpy(np.log([2.]))
            - pre_tanh_value
            - torch.nn.functional.softplus(-2. * pre_tanh_value)
        ).sum(dim=1)
        return log_prob + correction

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            # errors or instability at values near 1
            value = torch.clamp(value, -0.999999, 0.999999)
            pre_tanh_value = torch.log((1+value) / (1-value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )
        # return self._log_prob_from_pre_tanh(pre_tanh_value)

    def sample(self, return_pre_tanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pre_tanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                torch.zeros_like(self.normal_mean),
                torch.ones_like(self.normal_std)
            ).sample()
        )
        z.requires_grad_()

        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
