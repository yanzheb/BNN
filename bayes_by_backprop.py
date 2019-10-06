from torch.distributions.normal import Normal
import torch
from constants import Constant, DistConstant
import math
import torch.nn.functional as F
from torch.autograd import Function
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
import matplotlib.pyplot as plt
# sns.set(style="whitegrid")


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
        weight_mu = torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
                                ).uniform_(DistConstant.init_mu[0], DistConstant.init_mu[1])
        self.weight_mu = torch.nn.Parameter(weight_mu)

        # rho
        weight_rho = torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
                                 ).uniform_(DistConstant.init_rho[0], DistConstant.init_rho[1])
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
        self.relu = GuidedBackpropRelu()

        self.conv1 = BayesianConvolution(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = BayesianConvolution(6, 16, 5)
        self.fc1 = BayesianLinear(16 * 4 * 4, 400)
        self.fc2 = BayesianLinear(400, 400)
        self.f3 = BayesianLinear(400, 10)

        self.backwards = []
        self.forwards = []

    def forward(self, _input, sample=False, saliency=False):
        """Propagate the input through the network."""

        # Conv layers
        _input = self.pool(F.relu(self.conv1(_input)))

        _input = F.relu(self.conv2(_input))
        _input.register_hook(self.save_backward)
        if saliency:
            self.forwards.append(_input)

        _input = self.pool(_input)

        # Feed forward layers
        _input = _input.view(-1, BayesianNetwork.num_flat_features(_input))
        _input = F.relu(self.fc1(_input, sample))
        _input = F.relu(self.fc2(_input, sample))
        _input = F.log_softmax(self.f3(_input, sample), dim=1)
        return _input

    def save_backward(self, grad):
        self.backwards.append(grad)

    @staticmethod
    def num_flat_features(input_):
        size = input_.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def log_prior(self):
        """Return the summed log prior probabilities from all layers."""
        return self.conv1.log_prior + self.conv2.log_prior + \
               self.fc1.log_prior + self.fc2.log_prior + self.f3.log_prior

    def log_variational_posterior(self):
        """Return the summed log variational posterior probabilities from all layers."""
        return self.conv1.log_variational_posterior + self.conv2.log_variational_posterior + \
               self.fc1.log_variational_posterior + self.fc2.log_variational_posterior + \
               self.f3.log_variational_posterior

    def sample_elbo(self, input_, target, num_batches, batch_size, samples=Constant.samples, saliency=False):
        """Sample the elbo, implementing the minibatch version presented in section 3.4 in BBB paper."""
        outputs = torch.zeros(samples, batch_size, Constant.classes).to(Constant.device)
        log_priors = torch.zeros(samples).to(Constant.device)
        log_variational_posteriors = torch.zeros(samples).to(Constant.device)

        for i in range(samples):
            if saliency:
                outputs[i] = self(input_, sample=True, saliency=saliency)
            else:
                outputs[i] = self(input_, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()

        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target.to(Constant.device), size_average=False)

        loss = (log_variational_posterior - log_prior) / num_batches + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


class GuidedBackpropRelu(Function):
    def forward(self, input_):
        """Do normal relu operation to the input, but in addition save the input and output of the operation"""
        positive_mask = (input_ > 0).type_as(input_)  # Zero out negative values

        # Do the relu operation
        output = torch.addcmul(torch.zeros(input_.size()).type_as(input_), input_, positive_mask)
        self.save_for_backward(input_, output)
        return output

    def backward(self, grad_output):
        input_, output = self.saved_tensors

        positive_mask_1 = (input_ > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_.size()).type_as(input_),
                                   torch.addcmul(torch.zeros(input_.size()).type_as(input_),
                                                 grad_output, positive_mask_1), positive_mask_2)

        return grad_input


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
        # Zero the gradient buffers
        net.zero_grad()
        values = net.sample_elbo(data, target, len(train_loader), Constant.batch_size)

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


def training_procedure(net: BayesianNetwork, train_loader: torch.utils.data.DataLoader) -> BayesianNetwork:
    """Run the training procedure epochs number of times."""
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
            outputs[Constant.num_networks] = net.forward(data)

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
    # sns.set_style("dark")
    show(make_grid(mnist_sample[0].cpu()))


def saliency(data, target, net):
    # Empty gradient, and all saved values
    net.zero_grad()
    net.backwards = []
    net.forwards = []

    # Make a forward and backward pass
    values = net.sample_elbo(data, target, 1, data.shape[0], saliency=True)

    # Calculate the gradients, but doesn't update
    values[0].backward()

    back = net.backwards[0]
    forward = net.forwards[0]

    weight_importance = (torch.sum(torch.sum(net.backwards[0], dim=2), dim=2)) / (back.shape[2] + back.shape[3])
    grad_cam = torch.zeros((forward.shape[0], forward.shape[2], forward.shape[3])).to(Constant.device)

    for sample_i in range(weight_importance.shape[0]):
        for filter_i in range(weight_importance.shape[1]):
            grad_cam[sample_i] += weight_importance[sample_i, filter_i] * forward[sample_i, filter_i]
        grad_cam[sample_i] = F.relu(grad_cam[sample_i])
        grad_cam[sample_i] = grad_cam[sample_i] / torch.max(grad_cam[sample_i]).item()

    return grad_cam


def main() -> None:
    # Load data
    train_loader, test_loader = load_data()

    # Initialize a network
    net = BayesianNetwork().to(Constant.device)

    # Train the network on the training set
    net = training_procedure(net, train_loader)

    # Test the network on the test set
    # test_ensemble(net, test_loader)

    # Saliency
    sample = next(iter(test_loader))
    sal = saliency(sample[0], sample[1], net)
    image = sample[0][3]
    show(make_grid(image))
    show(make_grid(sal[3].cpu().detach()))


if __name__ == '__main__':
    main()
