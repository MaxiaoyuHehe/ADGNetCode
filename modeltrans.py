import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt

class IQARegression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_enc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.config = config
        self.transformer = Transformer(self.config)
        self.projection = nn.Sequential(
            nn.Linear(self.config.d_hidn, self.config.d_MLP_head, bias=False),
            nn.ReLU(),
            nn.Linear(self.config.d_MLP_head, self.config.n_output, bias=False),
        )

    def forward(self, enc_inputs, x, y):
        x = self.conv_enc(x)
        y = self.conv_enc(y)
        b, c, h, w = x.size()
        max_val = torch.max(x) if torch.max(x) > torch.max(y) else torch.max(y)
        x, y =  x/max_val, y/max_val

        x = torch.reshape(x, (b, c, h * w))
        x = x.permute(0, 2, 1)
        # batch x 256 x (29*29) -> batch x (29*29) x 256
        y = torch.reshape(y, (b, c, h * w))
        y = y.permute(0, 2, 1)
        # (bs, n_dec_seq+1, d_hidn), [(bs, n_head, n_enc_seq+1, n_enc_seq+1)], [(bs, n_head, n_dec_seq+1, n_dec_seq+1)], [(bs, n_head, n_dec_seq+1, n_enc_seq+1)]
        enc_outputs = self.transformer(enc_inputs, x, y)
        enc_outputs = enc_outputs[:, 0, :]  # in the IQA paper
        pred = self.projection(enc_outputs)

        return pred


""" transformer """


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)

    def forward(self, enc_inputs, enc_inputs_embed, dec_inputs_embed):
        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs, enc_inputs_embed, dec_inputs_embed)
        # (bs, n_seq, d_hidn), [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        # dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, dec_inputs_embed, enc_inputs, enc_outputs)

        # (bs, n_dec_seq, n_dec_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        return enc_outputs


""" encoder """


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # fixed position embedding
        # sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_enc_seq+1, self.config.d_hidn))
        # self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        # learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.config.n_enc_seq + 1, self.config.d_hidn))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))
        self.dropout = nn.Dropout(self.config.emb_dropout)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, inputs, inputs_embed, inputs_embed_fix):
        # inputs: batch x (len_seq+1) / inputs_embed: batch x len_seq x n_feat
        b, n, _ = inputs_embed.shape
        b_fix, n_fix, _ = inputs_embed_fix.shape

        # positions: batch x (len_seq+1)
        # 这里不用看后面没用到

        # outputs: batch x (len_seq+1) x n_feat
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, inputs_embed), dim=1)
        x += self.pos_embedding
        outputs = self.dropout(x)

        # cls_tokens_fix = repeat(self.cls_token_fix, '() n d -> b n d', b=b_fix)
        x_fix = torch.cat((cls_tokens, inputs_embed_fix), dim=1)
        x_fix += self.pos_embedding
        outputs_fix = self.dropout(x_fix)

        # (bs, n_enc_seq+1, n_enc_seq+1)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:
            # (bs, n_enc_seq+1, d_hidn), (bs, n_head, n_enc_seq+1, n_enc_seq+1)
            outputs, attn_prob = layer(outputs, outputs_fix, attn_mask)
            attn_probs.append(attn_prob)

        # (bs, n_enc_seq+1, d_hidn), [(bs, n_head, n_enc_seq+1, n_enc_seq+1)]
        return outputs, attn_probs


""" encoder layer """


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, inputs, inputs_fix, attn_mask):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs_fix, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)

        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)

        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        # ？？？？？？？？？？？为什么不是29x29+1
        return ffn_outputs, attn_prob


def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)

    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table


""" attention pad mask """


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask


""" multi head attention """


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        # (bs, n_head, n_q_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        # (bs, n_head, n_k_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)

        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head)
        # (bs, n_head, n_q_seq, e_embd)
        output = self.linear(context)
        output = self.dropout(output)

        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob


""" scale dot product attention """


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores.mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)

        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob


""" feed forward """


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, d_ff, n_seq)
        output = self.conv1(inputs.transpose(1, 2))
        output = self.active(output)
        # (bs, n_seq, d_hidn)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        # (bs, n_seq, d_hidn)
        return output


""" decoder """


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.config.n_enc_seq + 1, self.config.d_hidn))  # xixi 1,29x29+1,256
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))  # xixi 1,1,256
        self.dropout = nn.Dropout(self.config.emb_dropout)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])  # 2层

    def forward(self, dec_inputs, dec_inputs_embed, enc_inputs, enc_outputs):
        # enc_inputs: batch x (len_seq+1) / enc_outputs: batch x (len_seq+1) x n_feat
        # dec_inputs: batch x (len_seq+1) / dec_inputs_embed: batch x len_seq x n_feat
        # xixi？？？？？？？？？
        b, n, _ = dec_inputs_embed.shape  # xixi b=B n=29x29

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # xixi B,1,256
        x = torch.cat((cls_tokens, dec_inputs_embed), dim=1)  # xixi B,29x29+1,256
        x += self.pos_embedding[:, :(n + 1)]  # xixi ????
        # (bs, n_dec_seq+1, d_hidn)
        dec_outputs = self.dropout(x)

        # (bs, n_dec_seq+1, n_dec_seq+1)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        # (bs, n_dec_seq+1, n_dec_seq+1)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
        # (bs, n_dec_seq+1, n_dec_seq+1)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        # (bs, n_dec_seq+1, n_enc_seq+1)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)

        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            # (bs, n_dec_seq+1, d_hidn), (bs, n_dec_seq+1, n_dec_seq+1), (bs, n_dec_seq+1, n_enc_seq+1)
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                                   dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)

        # (bs, n_dec_seq+1, d_hidn), [(bs, n_dec_seq+1, n_dec_seq+1)], [(bs, n_dec_seq+1, n_enc_seq+1)]
        return dec_outputs, self_attn_probs, dec_enc_attn_probs


""" decoder layer """


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq)
        self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_enc_seq)
        dec_enc_att_outputs, dec_enc_attn_prob = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs,
                                                                   dec_enc_attn_mask)
        dec_enc_att_outputs = self.layer_norm2(self_att_outputs + dec_enc_att_outputs)
        # (bs, n_dec_seq, d_hidn)
        ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
        ffn_outputs = self.layer_norm3(dec_enc_att_outputs + ffn_outputs)

        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq), (bs, n_head, n_dec_seq, n_enc_seq)
        return ffn_outputs, self_attn_prob, dec_enc_attn_prob


""" attention decoder mask """


def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1)  # upper triangular part of a matrix(2-D)
    return subsequent_mask
