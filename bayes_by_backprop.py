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
import numpy as np
from typing import List, Tuple
from torchvision.utils import make_grid
import torch.nn as nn
from constants import Constant, DistConstant
import joblib
from captum.attr import IntegratedGradients, DeepLift, Saliency, InputXGradient
from captum.attr import visualization as viz
from dataset import get_dataset
import matplotlib.pyplot as plt


# from dataset import get_dataset


class Gaussian:
    """
    A gaussian density, shifted by a mean mu, and scaled by a standard deviation sigma.
    This is used for the variational posterior distributions.
    """

    def __init__(self, mu, rho) -> None:
        """
        Initialize a Gaussian distribution with the given parameters.

        :param mu: variational posterior parameter.
        :param rho: variational posterior parameter.
        """
        self._rho = rho
        self._mu = mu
        self._normal = Normal(0, 1)
        self._normal.loc = self._normal.loc.to(Constant.device)
        self._normal.scale = self._normal.scale.to(Constant.device)

    @property
    def mu(self):
        """Property for the mu variable."""
        return self._mu

    def __sigma(self):
        """A matrix of the same size as rho."""
        return torch.log1p(torch.exp(self._rho))

    def __epsilon(self):
        """Epsilon is point-wise multiplied with sigma, therefore it must be of the same size as sigma."""
        return self._normal.sample(self._rho.size())

    def sample(self):
        """Sampling weights."""
        return self._mu + self.__sigma() * self.__epsilon()

    def log_prob(self, input_):
        """
        Calculate the probability of the input, assuming it has
        a normal distribution with mean mu and standard deviation sigma.

        :param input_: The input to the pdf.
        """
        two_pi = torch.empty(1)
        two_pi = two_pi.new_full(self._rho.size(), 2 * math.pi, device=Constant.device)

        p1 = torch.log(torch.sqrt(two_pi))
        p2 = torch.log(self.__sigma())
        p3 = ((input_ - self._mu) ** 2) / ((2 * self.__sigma()) ** 2)

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


class BayesianConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight parameters
        self._initialize_weight()

        # Bias parameters
        self._initialize_bias()

        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(DistConstant.mixture_scale, DistConstant.sigma1, DistConstant.sigma2)
        self.bias_prior = ScaleMixtureGaussian(DistConstant.mixture_scale, DistConstant.sigma1, DistConstant.sigma2)
        self.log_prior, self.log_variational_posterior = 0, 0

    def _initialize_weight(self):
        # mu
        weight_mu = torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)).uniform_(
            DistConstant.init_mu[0],
            DistConstant.init_mu[1])
        self.weight_mu = torch.nn.Parameter(weight_mu)

        # rho
        weight_rho = torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)).uniform_(
            DistConstant.init_rho[0],
            DistConstant.init_rho[1])
        self.weight_rho = torch.nn.Parameter(weight_rho)
        # Initialize the weights to gaussian variational posteriors
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def _initialize_bias(self):
        # mu
        bias_mu = torch.empty(self.out_channels).uniform_(DistConstant.init_mu[0], DistConstant.init_mu[1])
        self.bias_mu = torch.nn.Parameter(bias_mu)
        # rho
        bias_rho = torch.empty(self.out_channels).uniform_(DistConstant.init_rho[0], DistConstant.init_rho[1])
        self.bias_rho = torch.nn.Parameter(bias_rho)
        # Initialize the biases to gaussian variational posteriors
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

    def forward(self, input_, sample=False, calculate_log_prob=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training or calculate_log_prob:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return F.conv2d(input_, weight, bias, self.stride, self.padding)


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
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training or calculate_log_prob:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return F.linear(input_, weight, bias)


class BayesianNetwork(torch.nn.Module):
    def __init__(self):
        super(BayesianNetwork, self).__init__()
        self.conv1 = BayesianConvolution(3, 32, 3)
        self.c_relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = BayesianConvolution(32, 64, 3)
        self.c_relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = BayesianConvolution(64, 128, 3)
        self.c_relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = BayesianConvolution(128, 256, 3)
        self.c_relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = BayesianConvolution(256, 512, 3)
        self.c_relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = BayesianLinear(18432, 256)
        self.f_relu1 = nn.ReLU()

        self.fc2 = BayesianLinear(256, Constant.classes)

    def forward(self, _input, sample):
        """Propagate the input through the network."""
        _input = self.pool1(self.c_relu1(self.conv1(_input, sample)))
        _input = self.pool2(self.c_relu2(self.conv2(_input, sample)))
        _input = self.pool3(self.c_relu3(self.conv3(_input, sample)))
        _input = self.pool4(self.c_relu4(self.conv4(_input, sample)))
        _input = self.pool5(self.c_relu5(self.conv5(_input, sample)))

        _input = _input.view(_input.shape[0], -1)
        _input = self.f_relu1(self.fc1(_input, sample))
        _input = F.log_softmax(self.fc2(_input, sample), dim=1)
        return _input

    def log_prior(self):
        """Return the summed log prior probabilities from all layers."""
        return self.conv1.log_prior + self.conv2.log_prior + self.conv3.log_prior + self.conv4.log_prior + \
               self.conv5.log_prior + self.fc1.log_prior + self.fc2.log_prior

    def log_variational_posterior(self):
        """Return the summed log variational posterior probabilities from all layers."""
        return self.conv1.log_variational_posterior + self.conv2.log_variational_posterior + \
               self.conv3.log_variational_posterior + self.conv4.log_variational_posterior + \
               self.conv5.log_variational_posterior + \
               self.fc1.log_variational_posterior + self.fc2.log_variational_posterior

    def sample_elbo(self, input_, target, num_batches, samples=Constant.samples):
        """Sample the elbo, implementing the minibatch version presented in section 3.4 in BBB paper."""
        outputs = torch.zeros(samples, Constant.batch_size, Constant.classes, device=Constant.device)
        log_priors = torch.zeros(samples, device=Constant.device)
        log_variational_posteriors = torch.zeros(samples, device=Constant.device)

        for i in range(samples):
            outputs[i] = self(input_, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()

        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)

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

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to cuda
        data = data.to(Constant.device)
        target = target.to(Constant.device)

        # Zero the gradient buffers
        net.zero_grad()
        values = net.module.sample_elbo(data, target, len(train_loader))

        for i, value in enumerate(values):
            losses[i].append(value.item())

        # Calculate the gradients, but doesn't update
        values[0].backward()
        # Does the update
        optimizer.step()

    return losses


def load_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load the MNIST data set from drive, if not present, it will download it from  the internet.

    :return: A tuple where the first item is the training set, and the second is the test set.
    """
    train_loader, test_loader = get_dataset(Constant.batch_size, Constant.test_batch_size)
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


def training_procedure(train_loader: torch.utils.data.DataLoader) -> BayesianNetwork:
    """Run the training procedure epochs number of times."""
    net = nn.DataParallel(BayesianNetwork()).to(Constant.device)

    # Set up optimizer
    optimizer = optim.Adam(net.parameters())

    losses_epoch = [[], [], [], []]

    # Run training session for Constant.train_epochs number of epochs
    for epoch in tqdm(range(Constant.train_epochs)):
        losses = train(net, optimizer, train_loader)

        for i in range(len(losses_epoch)):
            losses_epoch[i] += losses[i]

        if (epoch + 1) % Constant.save_period == 0:
            joblib.dump(net, f"saves/save_network_{epoch + 1}.pickle", protocol=4)

    # Make plot of the different components of the elbo
    # make_plots(losses_epoch)

    return net


def test_ensemble(net: BayesianNetwork, test_loader: torch.utils.data.DataLoader) -> None:
    """Run test set on networks with different weights, and a ensemble of them."""
    correct = 0
    test_size = len(test_loader.dataset)
    corrects = np.zeros(Constant.num_networks + 1, dtype=int)

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            # Move data to cuda
            data = data.to(Constant.device)

            # A tensor where the first dimension is the predictions of the different networks
            # Second dimension is the the different samples in the batch
            # Third dimension is the probabilities for the different classes predicted
            outputs = torch.zeros(Constant.num_networks + 1, Constant.test_batch_size, Constant.classes)

            # Sample Constant.test_samples number of weight configurations, which means that we get
            # Constant.test_samples number of networks
            for i in range(Constant.num_networks):
                outputs[i] = net(data, sample=True)

            # Last network which uses the mean of the distributions as weights
            outputs[Constant.num_networks] = net.forward(data, sample=False)

            # Finds which class has the highest probability for each network
            preds = outputs.max(2, keepdim=True)[1]

            # Find the mean prediction of all the networks
            output = outputs.mean(0)

            # Finds which class has the highest probability based on the average of all the networks
            pred = output.max(1, keepdim=True)[1]

            # Finds out number of correct predictions
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()

    for index, num in enumerate(corrects):
        if index < Constant.num_networks:
            print(f"Network {index}'s accuracy: {num / test_size}")
        else:
            print(f"Network using the mean weight's accuracy: {num / test_size}")
            print(f'Ensemble Accuracy: {correct / test_size}')


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        Constant.device)
    return torch.index_select(a, dim, order_index)


def attribute_image_features(algorithm, input_, net, label, additional_forward_args):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input_, target=label, additional_forward_args=additional_forward_args)

    return tensor_attributions


def generate_explanation(net, image, label, sample=1, additional_forward_args=False):
    image.requires_grad = True
    ig = IntegratedGradients(net)
    shape = image.shape

    total = np.zeros((shape[-1], shape[-2], 3))

    for _ in range(sample):
        attr_ig = attribute_image_features(ig, image, net, label, additional_forward_args)
        attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        total += attr_ig
    attr_ig = total / sample

    return attr_ig


def display_explanation(explanation, image):
    original_image = np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0))

    viz.visualize_image_attr(explanation, original_image, method="blended_heat_map", sign="all",
                             show_colorbar=True, title="Overlayed Integrated Gradients")


def main() -> None:
    torch.cuda.empty_cache()
    # Load data
    train_loader, test_loader = load_data()

    # net = training_procedure(train_loader)
    net = joblib.load("saves/save_network_60.pickle")
    net = net.module.to(Constant.device)
    net.eval()

    # test_ensemble(net, test_loader)
    dataiter = iter(test_loader)

    for i in range(100):
        image, label = next(dataiter)

        image = image.to(Constant.device)
        label = label.to(Constant.device)

        attr_ig1 = generate_explanation(net, image, label, 1, True)
        attr_ig20 = generate_explanation(net, image, label, 20, True)
        attr_mean = generate_explanation(net, image, label, 1, False)

        original_image = np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0))

        fig_0 = viz.visualize_image_attr(None, original_image, method="original_image", title="Original image")[0]
        fig_1 = viz.visualize_image_attr(attr_ig1, original_image, method="masked_image", title="Masked DeepLift - 1")[
            0]
        fig_3 = \
            viz.visualize_image_attr(attr_ig20, original_image, method="masked_image", title="Masked DeepLift - 20")[0]
        fig_2 = \
        viz.visualize_image_attr(attr_mean, original_image, method="masked_image", title="Masked DeepLift - mean")[
            0]

        fig_0.savefig(f"images/image_{i}_0.png", dpi=fig_0.dpi)
        fig_1.savefig(f"images/image_{i}_1.png", dpi=fig_1.dpi)
        fig_2.savefig(f"images/image_{i}_2.png", dpi=fig_2.dpi)
        fig_3.savefig(f"images/image_{i}_3.png", dpi=fig_3.dpi)


if __name__ == '__main__':
    main()
