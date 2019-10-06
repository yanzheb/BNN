import numpy as np
import cv2
import torch
from torch.autograd import Variable
from constants import Constant


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, input_):
        """
        :param input_:
        :return: outputs is/are the output of the target layer(s).
        input_ which is the output of the network.
        """
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            print(name, "----------", module)
            input_ = module(input_)
            if name in self.target_layers:
                input_.register_hook(self.save_gradient)
                outputs += [input_]
        return outputs, input_


class ModelOutputs:
    """ Class for making a forward pass, and getting:

    1. The network output.
    2. Activations from intermediate targeted layers.
    3. Gradients from intermediate targeted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names):
        self.model = model
        # self.model.eval()
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, _input):
        return self.model(_input)

    def __call__(self, input_, index=None):
        features, output = self.extractor(input_.to(Constant.device))

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.to(Constant.device) * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_variables=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam