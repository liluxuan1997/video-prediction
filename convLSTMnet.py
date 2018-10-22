import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from convLSTMCell import ConvLSTMCell

class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 bias=True, seq_length=20, input_length=10, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.seq_length = seq_length
        self.input_length = input_length
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias).cuda())

        self.cell_list = nn.ModuleList(cell_list)

        self.gen_image = nn.Conv2d(in_channels=self.hidden_dim[-1],
                              out_channels=self.input_dim,
                              kernel_size=1,
                              padding=0,
                              bias=self.bias)

    def forward(self, inputs):
        #inputs should have shape (batch_size, input_sequence_length, channels, img_width, img_height)
        hidden_state = self._init_hidden(batch_size=inputs.shape[0])
        ##TODO
        #add mask
        pred_imgs = torch.zeros(inputs.shape).cuda()

        for t in range(self.seq_length-1):
            if t < self.input_length:
                x = inputs[:, t, :, :, :]
            else:
                x = pred_img
            #x = np.transpose(x,(0,3,1,2))
            #x = torch.Tensor(x)
            for layer_idx in range(self.num_layers):
                h_in,c_in = hidden_state[layer_idx]
                h_out,c_out = self.cell_list[layer_idx](input_tensor = x, cur_state=(h_in, c_in))
                x = h_out
                hidden_state[layer_idx] = (h_out,c_out)

            pred_img = self.gen_image(x)
            pred_imgs[:, t, :, :, :] = pred_img
        return pred_imgs[:,self.input_length-1:self.seq_length-1,:,:,:]

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param