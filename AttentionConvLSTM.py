#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
from strainDetector.ConvLSTM_pytorch.ConvLSTMCell import ConvLSTMCell


class AttentionConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, S or T, B, C, S
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, static_features=0):
        super(AttentionConvLSTM, self).__init__()

        #self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.static_features = static_features
        self.return_all_layers = return_all_layers

        cell_list0 = []
        cell_list1 = []

        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list0.append(ConvLSTMCell(input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dim[i],
                                           kernel_size=self.kernel_size[i],
                                           bias=self.bias))

            cell_list1.append(ConvLSTMCell(input_dim=self.hidden_dim[-1]+1,
                                           hidden_dim=1,
                                           kernel_size=self.kernel_size[i],
                                           bias=self.bias,
                                           init_hidden_dim=1))

        self.encoder = nn.ModuleList(cell_list0)
        self.decoder = nn.ModuleList(cell_list1)


        self.attention_enc = nn.Linear(100,
                                   100,
                                   bias=True)

        self.attention_dec = torch.nn.Linear(100,
                                            100,
                                            bias=True)

        self.FC = torch.nn.Linear(2*100,
                                  1,
                                  bias=True)


    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, s) or (b, t, c, s)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, s) -> (b, t, c, s)
            input_tensor = input_tensor.permute(1, 0, 2, 3)

        if len(input_tensor.shape) == 3:
            # add singleton feature dimension
            input_tensor = input_tensor.unsqueeze(2)
        if self.static_features > 0:
            B, T, _, s = input_tensor.size()
            input_tensor, static_feat_tensor = torch.split(input_tensor,
                                                        [s-self.static_features, self.static_features], 
                                                        dim=-1)
        B, T, C, S = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state_enc = self._init_hidden_encoder(batch_size=B,
                                             spatial_size=(S))
            hidden_state_dec = self._init_hidden_decoder(batch_size=B,
                                             spatial_size=(S))


        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h_e, c_e = hidden_state_enc[layer_idx]
            h_d, c_d = hidden_state_dec[layer_idx]

            ones = torch.ones((B,1,S)).cuda()
            h_d = torch.cat((h_d, ones), dim=1)
            c_d = torch.cat((c_d, ones), dim=1)

            output_inner = []
            for t in range(seq_len):
                # iterate through time

                # encoder [B, 11, S]
                h_e, c_e = self.encoder[layer_idx](input_tensor=cur_layer_input[:, t, :, :],
                                                 cur_state=[h_e, c_e])
                # decoder  [B, 1, S]
                h_d, c_d = self.decoder[layer_idx](input_tensor=c_e,
                                                 cur_state=[h_d, c_d])

                # attention
                att_score = torch.tanh(self.attention_dec(c_d))

                att = F.softmax(att_score, dim=1)

                w_c_d = torch.mul(att, c_d)
                w_h_d = torch.mul(att, h_d)
                
                c_d = torch.cat((c_d, w_c_d),axis=1)
                h_d = torch.cat((h_d, w_h_d),axis=1)

                output_inner.append(h_d)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h_d, c_d])

        out = self.FC(layer_output_list[-1].view(B,T,-1))

        out = torch.sigmoid(out)
        return out

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        lstm_out = layer_output_list[-1].reshape(B,T,-1)
        if self.static_features > 0:
            lstm_out = torch.cat((lstm_out, static_feat_tensor), 
                                 dim=-1)

        out = self.linear(lstm_out)
        out = torch.sigmoid(out)
        
        return out#layer_output_list, last_state_list

    def _init_hidden_encoder(self, batch_size, spatial_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.encoder[i].init_hidden(batch_size, spatial_size))
        return init_states

    def _init_hidden_decoder(self, batch_size, spatial_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.decoder[i].init_hidden(batch_size, spatial_size))
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
