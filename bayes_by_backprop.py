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
from grad_cam import GradCam, GuidedBackpropReLUModel
from constants import Constant, DistConstant
import joblib


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

    @property
    def mu(self):
        """Property for the mu variable."""
        return self._mu

    def __sigma(self):
        """A matrix of the same size as rho."""
        return torch.log1p(torch.exp(self._rho))

    def __epsilon(self):
        """Epsilon is point-wise multiplied with sigma, therefore it must be of the same size as sigma."""
        return self._normal.sample(self._rho.size()).to(Constant.device)

    def sample(self):
        """Sampling weights."""
        return self._mu + self.__sigma() * self.__epsilon()

    def log_prob(self, input_):
        """
        Calculate the probability of the input, assuming it has
        a normal distribution with mean mu and standard deviation sigma.

        :param input_: The input to the pdf.
        """
        two_pi = torch.empty(self._rho.size()).fill_(2 * math.pi).to(Constant.device)

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
            weight = self.weight.mu()
            bias = self.bias.mu()

        if self.training or calculate_log_prob:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return F.conv2d(input_.to(Constant.device), weight, bias, self.stride, self.padding)


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
        self.conv1 = BayesianConvolution(3, 64, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = BayesianConvolution(64, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU()

        self.conv3 = BayesianConvolution(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.relu3 = nn.ReLU()

        self.fc1 = BayesianLinear(1024, 128)
        self.relu4 = nn.ReLU()

        self.fc2 = BayesianLinear(128, 256)
        self.relu5 = nn.ReLU()

        self.fc3 = BayesianLinear(256, 512)
        self.relu6 = nn.ReLU()

        self.fc4 = BayesianLinear(512, 10)

    def forward(self, _input, sample=False):
        """Propagate the input through the network."""
        _input = self.pool1(self.relu1(self.conv1(_input, sample)))
        _input = self.pool2(self.relu2(self.conv2(_input, sample)))
        _input = self.pool3(self.relu3(self.conv3(_input, sample)))

        _input = _input.view(_input.shape[0], -1)
        _input = self.relu4(self.fc1(_input, sample))
        _input = self.relu5(self.fc2(_input, sample))
        _input = self.relu6(self.fc3(_input, sample))
        _input = F.log_softmax(self.fc4(_input, sample), dim=1)
        return _input

    def log_prior(self):
        """Return the summed log prior probabilities from all layers."""
        return self.conv1.log_prior + self.conv2.log_prior + \
               self.fc1.log_prior + self.fc2.log_prior + self.fc3.log_prior

    def log_variational_posterior(self):
        """Return the summed log variational posterior probabilities from all layers."""
        return self.conv1.log_variational_posterior + self.conv2.log_variational_posterior + \
               self.fc1.log_variational_posterior + self.fc2.log_variational_posterior + \
               self.fc3.log_variational_posterior

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

    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero the gradient buffers
        net.zero_grad()
        values = net.sample_elbo(data, target, len(train_loader))

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
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./cifar10', train=True, download=True,
                                                              transform=transforms.ToTensor()),
                                               batch_size=Constant.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./cifar10', train=False, download=True,
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


def training_procedure(net: BayesianNetwork, train_loader: torch.utils.data.DataLoader) -> BayesianNetwork:
    """Run the training procedure epochs number of times."""
    # Set up optimizer
    optimizer = optim.Adam(net.parameters())

    losses_epoch = [[], [], [], []]

    # Run training session for Constant.train_epochs number of epochs
    for _ in tqdm(range(Constant.train_epochs)):
        losses = train(net, optimizer, train_loader)

        for i in range(len(losses_epoch)):
            losses_epoch[i] += losses[i]

    # Make plot of the different components of the elbo
    make_plots(losses_epoch)

    return net


def test_ensemble(net: BayesianNetwork, test_loader: torch.utils.data.DataLoader) -> None:
    """Run test set on networks with different weights, and a ensemble of them."""
    correct = 0
    test_size = len(test_loader.dataset)
    corrects = np.zeros(Constant.num_networks + 1, dtype=int)

    with torch.no_grad():
        for data, target in tqdm(test_loader):

            # A tensor where the first dimension is the predictions of the different networks
            # Second dimension is the the different samples in the batch
            # Third dimension is the probabilities for the different classes predicted
            outputs = torch.zeros(Constant.num_networks + 1, Constant.test_batch_size, Constant.classes)

            # Sample Constant.test_samples number of weight configurations, which means that we get
            # Constant.test_samples number of networks
            for i in range(Constant.num_networks):
                outputs[i] = net.forward(data, sample=True)

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
            print(f"Network {index}'s accuracy: {num/test_size}")
        else:
            print(f"Network using the mean weight's accuracy: {num/test_size}")
    print(f'Ensemble Accuracy: {correct/test_size}')


def show(img) -> None:
    """Given a image, display it."""
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)), interpolation='nearest')
    plt.show()


def test_sample(test_loader: torch.utils.data.DataLoader):
    mnist_sample = iter(test_loader).next()
    mnist_sample[0] = mnist_sample[0].to(Constant.device)
    show(make_grid(mnist_sample[0].cpu()))


def fuse_masks(grad_cam, guided_backprop):
    output = torch.empty(grad_cam.shape)
    for sample_i in range(grad_cam.shape[0]):
        output[sample_i] = grad_cam[sample_i] * guided_backprop[sample_i]

    return output


def ensemble_saliency(image, net, n_sample=5):
    grad_cam = GradCam(net, ["relu2"])
    grad_masks = [grad_cam(image) for _ in range(n_sample)]

    guided_backprop = GuidedBackpropReLUModel(net)
    guided_gradients = [guided_backprop(image) for _ in range(n_sample)]

    total = torch.zeros(image.shape)

    for i in tqdm(range(n_sample)):
        total += fuse_masks(grad_masks[i], guided_gradients[i])

    total /= n_sample

    # Normalize

    for sample in range(total.shape[0]):
        total[sample] -= torch.min(total[sample]).item()
        total[sample] /= (torch.max(total[sample]).item())
    return total


def main() -> None:
    # Load data
    train_loader, test_loader = load_data()

    """# Initialize a network
    net = BayesianNetwork().to(Constant.device)

    # Train the network on the training set
    net = training_procedure(net, train_loader)
    joblib.dump(net, "save_network.pickle", protocol=4)"""
    net = joblib.load("save_network.pickle")

    # Test the network on the test set
    # test_ensemble(net, test_loader)

    index = 17
    sample = next(iter(test_loader))
    explaination = ensemble_saliency(sample[0], net, 10)
    pred = net.forward(sample[0])

    show(make_grid(sample[0].detach()[index]))
    show(make_grid(explaination.detach()[index]))
    print(f"Predicted label {pred[index].max(0)[1]}")
    print(f"True label {sample[1][index]}")


if __name__ == '__main__':
    main()