import torch
from torch import nn

class GHU(nn.Module):
    def __init__(self, kernel_size, num_features, bias, tln=False):
        """Initialize the Gradient Highway Unit.
        """
        super(GHU, self).__init__()
        self.kernel_size  = kernel_size
        self.padding      = kernel_size[0] // 2, kernel_size[1] // 2
        self.num_features = num_features
        self.bias         = bias
        self.layer_norm   = tln

        self.conv_z = nn.Conv2d(in_channels=self.num_features,
                              out_channels=2 * self.num_features,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.conv_x = nn.Conv2d(in_channels=self.num_features,
                              out_channels=2 * self.num_features,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self,x,z):
        if z is None:
            z = self.init_state(x, self.num_features)
            z_concat = self.conv_z(z)
            #if self.layer_norm:
            #    z_concat = tensor_layer_norm(z_concat, 'state_to_state')

            x_concat = self.conv_x(x)
            #if self.layer_norm:
            #    x_concat = tensor_layer_norm(x_concat, 'input_to_state')

            gates = torch.add(x_concat, z_concat)
            p, u = torch.split(gates, self.num_features, dim=1)
            p = torch.tanh(p)
            u = torch.sigmoid(u)
            z_new = u * p + (1-u) * z
            return z_new

    def init_state(self, inputs, num_features):
        dims = len(inputs.shape)
        if dims == 4:
            batch = inputs.shape[0]
            height = inputs.shape[2]
            width = inputs.shape[3]
        else:
            raise ValueError('input tensor should be rank 4.')
        return torch.zeros((batch,  num_features, height, width)).cuda()
        
