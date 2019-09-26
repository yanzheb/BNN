from torch.distributions.normal import Normal
import torch
from dataclasses import dataclass
import math
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
sns.set(style="whitegrid")


@dataclass
class Constant:
    batch_size = 100
    test_batch_size = 10
    classes = 10
    train_epochs = 1
    samples = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DistConstant:
    init_mu = (-0.2, 0.2)
    init_rho = (-5, -4)
    mixture_scale = 0.5
    sigma1 = torch.tensor([math.exp(-0)]).to(Constant.device)
    sigma2 = torch.tensor([math.exp(-6)]).to(Constant.device)


class Gaussian:
    """
    A gaussian density, shifted by a mean mu, and scaled by a standard deviation sigma.
    This is used for the variational posterior distributions.
    """

    def __init__(self, mu, rho):
        """
        Initialize a Gaussian distribution with the given parameters.

        :param mu: variational posterior parameter.
        :param rho: variational posterior parameter.
        """
        self._rho = rho
        self._mu = mu
        self._normal = Normal(0, 1)

    @property
    def mu(self):
        """Property for the mu variable."""
        return self._mu

    def _sigma(self):
        """A matrix of the same size as rho."""
        return torch.log1p(torch.exp(self._rho))

    def _epsilon(self):
        """Epsilon is point-wise multiplied with sigma, therefore it must be of the same size as sigma."""
        return self._normal.sample(self._rho.size()).to(Constant.device)

    def sample(self):
        """Sampling weights."""
        return self._mu + self._sigma() * self._epsilon()

    def log_prob(self, input_):
        """
        Calculate the probability of the input, assuming it has
        a normal distribution with mean mu and standard deviation sigma.

        :param input_: The input to the pdf.
        """
        two_pi = torch.empty(self._rho.size()).fill_(2 * math.pi).to(Constant.device)

        p1 = torch.log(torch.sqrt(two_pi))
        p2 = torch.log(self._sigma())
        p3 = ((input_ - self._mu) ** 2) / ((2 * self._sigma()) ** 2)

        return (-p1 - p2 - p3).sum()


class ScaleMixtureGaussian:
    """Implementation of the num_batchesScale mixture prior introduced in section 3.3 in BBB paper."""

    def __init__(self, pi, sigma1, sigma2):
        """
        Initialize a scale mixture of two gaussian densities.

        :param pi: The weighting of the gaussian densities.
        :param sigma1: The standard deviation for the first gaussian density.
        :param sigma2: The standard deviation for the second gaussian density.
        """
        self._pi = pi
        self._normal1 = Normal(0, sigma1)
        self._normal2 = Normal(0, sigma2)

    def log_prob(self, input_):
        """
        Calculate the probability of the input, assuming it has
        scaled mixture of two normal distribution with mean 0 and standard deviation sigma1 and sigma2.

        :param input_: The input to the pdf.
        """
        prob1 = torch.exp(self._normal1.log_prob(input_))
        prob2 = torch.exp(self._normal2.log_prob(input_))

        return (torch.log(self._pi * prob1 + (1 - self._pi) * prob2)).sum()


class BayesianLinear(torch.nn.Module):
    """Single Bayesian Network layer."""
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Weight parameters
        # mu
        weight_mu = torch.empty(output_size, input_size).uniform_(DistConstant.init_mu[0], DistConstant.init_mu[1])
        self.weight_mu = torch.nn.Parameter(weight_mu)
        # rho_pi
        weight_rho = torch.empty(output_size, input_size).uniform_(DistConstant.init_rho[0], DistConstant.init_rho[1])
        self.weight_rho = torch.nn.Parameter(weight_rho)
        # Initialize the weights to gaussian variational posteriors
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        # mu
        bias_mu = torch.empty(output_size).uniform_(DistConstant.init_mu[0], DistConstant.init_mu[1])
        self.bias_mu = torch.nn.Parameter(bias_mu)
        # rho
        bias_rho = torch.empty(output_size).uniform_(DistConstant.init_rho[0], DistConstant.init_rho[1])
        self.bias_rho = torch.nn.Parameter(bias_rho)
        # Initialize the biases to gaussian variational posteriors
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(DistConstant.mixture_scale, DistConstant.sigma1, DistConstant.sigma2)
        self.bias_prior = ScaleMixtureGaussian(DistConstant.mixture_scale, DistConstant.sigma1, DistConstant.sigma2)
        self.log_prior, self.log_variational_posterior = 0, 0

    def forward(self, input_, sample=False, calculate_log_prob=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu()
            bias = self.bias.mu()

        if self.training or calculate_log_prob:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input_.to(Constant.device), weight, bias)


class BayesianNetwork(torch.nn.Module):
    def __init__(self):
        super(BayesianNetwork, self).__init__()
        self.l1 = BayesianLinear(28 * 28, 400)
        self.l2 = BayesianLinear(400, 400)
        self.l3 = BayesianLinear(400, 10)

    def forward(self, _input, sample=False):
        """Propagate the input through the network."""
        _input = _input.view(-1, 28 * 28)
        _input = F.relu(self.l1(_input, sample))
        _input = F.relu(self.l2(_input, sample))
        _input = F.log_softmax(self.l3(_input, sample), dim=1)
        return _input

    def log_prior(self):
        """Return the summed log prior probabilities from all layers."""
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior

    def log_variational_posterior(self):
        """Return the summed log variational posterior probabilities from all layers."""
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior + self.l3.log_variational_posterior

    def sample_elbo(self, input_, target, num_batches, samples=Constant.samples):
        """Sample the elbo, implementing the minibatch version presented in section 3.4 in BBB paper."""
        outputs = torch.zeros(samples, Constant.batch_size, Constant.classes).to(Constant.device)
        log_priors = torch.zeros(samples).to(Constant.device)
        log_variational_posteriors = torch.zeros(samples).to(Constant.device)

        for i in range(samples):
            outputs[i] = self(input_, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()

        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target.to(Constant.device), size_average=False)

        loss = (log_variational_posterior - log_prior) / num_batches + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


def train(net, optimizer, train_loader):
    """
    Train the given Bayesian network using th given optimizer and training data.

    :param net: a bayesian neural network.
    :param optimizer: a optimizer from the Pytorch library.
    :param train_loader: training set, the Pytorch preferred format.
    :return: List of numerical values that form the elbo.
    """
    losses = [[], [], [], []]
    net.train()

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        net.zero_grad()
        values = net.sample_elbo(data, target, len(train_loader))

        for i, value in enumerate(values):
            losses[i].append(value.item())

        values[0].backward()
        optimizer.step()

    return losses


def load_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load the MNIST data set from drive, if not present, it will download it from  the internet.

    :return: A tuple where the first item is the training set, and the second is the test set.
    """

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=True, download=True,
                                                              transform=transforms.ToTensor()),
                                               batch_size=Constant.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=False, download=True,
                                                             transform=transforms.ToTensor()),
                                              batch_size=Constant.test_batch_size, shuffle=False)
    return train_loader, test_loader


def make_plots(losses: List[List[float]]) -> None:
    """
    Take in numerical values and makes plots out of it.

    :param losses: Consists of the loss, log prior, log variational posterior and negative log likelihood.
    """
    fig, ax = plt.subplots(2, 2)

    sns.lineplot(data=pd.DataFrame({'loss': losses[0]}), ax=ax[0, 0])
    sns.lineplot(data=pd.DataFrame({'log prior': losses[1]}), palette="tab10", linewidth=2.5, ax=ax[0, 1])
    sns.lineplot(data=pd.DataFrame({'log variational_posterior': losses[2]}), palette="tab10", linewidth=2.5,
                 ax=ax[1, 0])
    sns.lineplot(data=pd.DataFrame({'negative log likelihood': losses[3]}), palette="tab10", linewidth=2.5, ax=ax[1, 1])

    plt.show()


def training_procedure() -> None:
    """Run the training procedure epochs number of times."""
    # Load data
    train_loader, test_loader = load_data()

    # Set up a bayesian neural network
    net = BayesianNetwork().to(Constant.device)

    # Set up optimizer
    optimizer = optim.Adam(net.parameters())

    losses_epoch = [[], [], [], []]

    # Run training session for Constant.train_epochs number of epochs
    for _ in range(Constant.train_epochs):
        losses = train(net, optimizer, train_loader)

        for i in range(len(losses_epoch)):
            losses_epoch[i] += losses[i]

    # Make plot of the different components of the elbo
    make_plots(losses_epoch)


def main() -> None:
    training_procedure()


if __name__ == '__main__':
    main()
