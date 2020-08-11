import torch 
import torch.nn as nn
import pdb
from strainDetector.bn_lstm.bnlstm import SeparatedBatchNorm1d

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, init_hidden_dim=None, batch_norm=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        max_length=120

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.init_hidden_dim=init_hidden_dim

        if not self.init_hidden_dim:
            hidden = self.hidden_dim
        else:
            hidden = self.init_hidden_dim

        self.conv = nn.Conv1d(in_channels=self.input_dim + hidden,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.batch_norm = batch_norm        

        if batch_norm:
            # BN parameters
            self.bn_ih = SeparatedBatchNorm1d(
                    num_features=4 * hidden_dim, 
                    max_length=max_length)
            self.bn_hh = SeparatedBatchNorm1d(
                    num_features=4 * hidden_dim, 
                     max_length=max_length)
            self.bn_c = SeparatedBatchNorm1d(
                    num_features=hidden_dim, 
                    max_length=max_length)



    def forward(self, input_tensor, cur_state, time):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        combined_conv_bn = self.bn_ih(combined_conv, time)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv_bn, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        if c_cur.shape[1] > 1:
            # in case of attention augmented state only 
            c_cur = c_cur[:,0].unsqueeze(1)
            h_cur = h_cur[:,0].unsqueeze(1)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):
        # Initialization of BN parameters.
        self.bn_ih.reset_parameters()
        self.bn_hh.reset_parameters()
        self.bn_c.reset_parameters()
        self.bn_ih.bias.data.fill_(0)
        self.bn_hh.bias.data.fill_(0)
        self.bn_ih.weight.data.fill_(0.1)
        self.bn_hh.weight.data.fill_(0.1)
        self.bn_c.weight.data.fill_(0.1)

        if not self.init_hidden_dim:
            hidden = self.hidden_dim
        else:
            hidden = self.init_hidden_dim
        return (torch.zeros(batch_size, 
                            hidden, 
                            spatial_size, 
                            device=self.conv.weight.device),
                torch.zeros(batch_size, 
                            hidden, 
                            spatial_size, 
                            device=self.conv.weight.device)
                )



