import torch.nn as nn
from torch.autograd import Variable
import torch


class CausualLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, mem_dim, hidden_dim, kernel_size, 
                 bias, forget_bias=1.0, tln=False):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(CausualLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        self.layer_norm = tln
        self._forget_bias = forget_bias

        
        self.conv_h = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.conv_m = nn.Conv2d(in_channels=self.mem_dim,
                              out_channels=3 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.conv_x = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=7 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.conv_c = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.conv_o = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.conv_cell = nn.Conv2d(in_channels=self.hidden_dim*2,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, x, h, c, m):
        h_cc = self.conv_h(h)
        m_cc = self.conv_m(m)
        #if self.layer_norm:
        #
        i_h, g_h, f_h, o_h = torch.split(h_cc, self.hidden_dim, dim=1)
        i_m, f_m, m_m = torch.split(m_cc, self.hidden_dim, dim=1)

        if x is None:
            i = torch.sigmoid(i_h)
            f = torch.sigmoid(f_h + self._forget_bias)
            g = torch.tanh(g_h)
        else:
            x_cc = self.conv_x(x)

            #if self.layer_norm:
            
            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.split(x_cc, self.hidden_dim, dim=1)

            i = torch.sigmoid(i_x + i_h)
            f = torch.sigmoid(f_x + f_h + self._forget_bias)
            g = torch.tanh(g_x + g_h)

        c_new = f * c + i * g

        c_cc = self.conv_c(c_new)

        #if self.layer_norm:
        #
        
        i_c, g_c, f_c, o_c = torch.split(c_cc, self.hidden_dim, dim=1)

        if x is None:
                ii = torch.sigmoid(i_c + i_m)
                ff = torch.sigmoid(f_c + f_m + self._forget_bias)
                gg = torch.tanh(g_c)
        else:
            ii = torch.sigmoid(i_c + i_x_ + i_m)
            ff = torch.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
            gg = torch.tanh(g_c + g_x_)

        m_new = ff * torch.tanh(m_m) + ii * gg

        o_m = self.conv_o(m_new)

        #if self.layer_norm:
        #
        
        if x is None:
                o = torch.tanh(o_h + o_c + o_m)
        else:
            o = torch.tanh(o_x + o_h + o_c + o_m)

        cell = torch.cat([c_new, m_new],dim=1)
        cell = self.conv_cell(cell)

        h_new = o * torch.tanh(cell)

        return h_new, c_new, m_new

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda())


