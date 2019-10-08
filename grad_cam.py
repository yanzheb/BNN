import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
import torch.nn.functional as F
import numpy as np
from constants import Constant
from skimage.transform import resize
import torch.nn as nn


class FeatureExtractor:
	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, input_):
		self.model.forward(input_)
		target_activations = []

		for name, module in self.model._modules.items():
			if name == "fc1":
				input_ = input_.view(input_.shape[0], -1)
			# Sample weights
			if "conv" in name.lower() or "fc" in name.lower():
				input_ = module(input_, sample=True)
			else:
				input_ = module(input_)

			if name in self.target_layers:
				input_.register_hook(self.save_gradient)
				target_activations += [input_]
		input_ = F.log_softmax(input_, dim=1)

		return target_activations, input_


class GradCam:
	def __init__(self, model, target_layers):
		self.model = model
		self.extractor = FeatureExtractor(model, target_layers)

	def __call__(self, input_):
		input_size = input_.shape
		target_activations, output = self.extractor(input_)
		index = np.argmax(output.cpu().detach().numpy(), axis=1)

		one_hot = np.zeros(output.size(), dtype=np.float32)
		for sample in range(one_hot.shape[0]):
			one_hot[sample, index[sample]] = 1
		one_hot = torch.from_numpy(one_hot)
		one_hot = torch.sum(one_hot * output.cpu())

		self.model.zero_grad()
		one_hot.backward()

		# Gradients
		grads_val = self.extractor.gradients[-1].cpu().numpy()

		# Calculate the filter weights
		weights = (np.sum(np.sum(grads_val, axis=2), axis=2)) / (grads_val.shape[2] + grads_val.shape[3])

		grad_cam = torch.zeros((target_activations[-1].shape[0], target_activations[-1].shape[2],
								target_activations[-1].shape[3])).to(Constant.device)
		mask = np.zeros(input_size)

		# Calculate grad cam and interpolate the result to original image size
		for sample_i in range(weights.shape[0]):
			for filter_i in range(weights.shape[1]):
				grad_cam[sample_i] += weights[sample_i, filter_i] * target_activations[-1][sample_i, filter_i]
			grad_cam[sample_i] = F.relu(grad_cam[sample_i])

			grad_cam[sample_i] -= torch.min(grad_cam[sample_i]).item()
			grad_cam[sample_i] /= torch.max(grad_cam[sample_i]).item()

			# Resize the filter
			mask[sample_i] = resize(grad_cam[sample_i].cpu().detach().numpy(), input_size[2:])

		mask = torch.from_numpy(mask).float()

		return mask


class ReluFunction(Function):
	@staticmethod
	def forward(ctx, input_):
		positive_mask = (input_ > 0).type_as(input_)
		output = torch.addcmul(torch.zeros(input_.size()).type_as(input_), input_, positive_mask)
		ctx.save_for_backward(input_)
		return output

	@staticmethod
	def backward(ctx, grad_output):
		input_ = ctx.saved_tensors[-1]
		positive_mask_1 = (input_ > 0).type_as(grad_output)
		positive_mask_2 = (grad_output > 0).type_as(grad_output)
		grad_input = torch.addcmul(torch.zeros(
			input_.size()).type_as(input_), torch.addcmul(
			torch.zeros(input_.size()).type_as(input_), grad_output, positive_mask_1), positive_mask_2)

		return grad_input


class GuidedBackpropReLU(nn.Module):
	def __init__(self):
		super(GuidedBackpropReLU, self).__init__()
		self.relu = ReluFunction.apply

	def forward(self, input_):
		return self.relu(input_)


class GuidedBackpropReLUModel:
	def __init__(self, model):
		self.model = model

		for name, module in self.model.named_children():
			if isinstance(module, nn.ReLU):
				setattr(model, name, GuidedBackpropReLU())

	def forward(self, input_):
		return self.model(input_, sample=True)

	def __call__(self, input_):
		input_.requires_grad = True

		output = self.forward(input_)
		index = np.argmax(output.cpu().detach().numpy(), axis=1)

		one_hot = np.zeros(output.size(), dtype=np.float32)
		for sample in range(one_hot.shape[0]):
			one_hot[sample, index[sample]] = 1
		one_hot = torch.from_numpy(one_hot)
		one_hot = torch.sum(one_hot * output.cpu())

		self.model.zero_grad()
		one_hot.backward()
		output = input_.grad.detach()

		return output
