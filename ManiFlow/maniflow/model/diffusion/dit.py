# Copyright (c) Sudeep Dasari, 2023
# Heavy inspiration taken from DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# and DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)


class _SelfAttnEncoder(nn.Module):
    def __init__(
        self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None] + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # create modulation layers
        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # process the conditioning vector first
        # print(f"cond shape before mean: {cond.shape}")
        cond = torch.mean(cond, axis=0)
        # print(f"cond shape after mean: {cond.shape}")
        cond = cond + t

        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = x * scale[None] + shift[None]
        x = self.linear(x)
        return x.transpose(0, 1)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )

        for l in self.layers:
            l.reset_parameters()

    def forward(self, src, pos):
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs


class _TransformerDecoder(_TransformerEncoder):
    def forward(self, src, t, all_conds):
        x = src
        for layer, cond in zip(self.layers, all_conds):
            x = layer(x, t, cond)
        return x

class DiT(nn.Module):
    def __init__(
        self,
        # ac_dim,
        # ac_chunk,
        # time_dim=256,
        # hidden_dim=512,
        # num_blocks=6,
        # dropout=0.1,
        # dim_feedforward=2048,
        # nhead=8,
        # activation="gelu",
        input_dim: int, # action dim if obs as global cond
        output_dim: int, # action dim
        horizon: int, # ac_chunk, action chunk length
        n_obs_steps: int = None, # number of observation steps
        cond_dim: int = 256, # input dim for visual condition observation
        visual_cond_len: int = 1024, # length of visual condition sequence
        diffusion_timestep_embed_dim: int = 256, # time_dim
        # omit diffusion_target_t_embed_dim, not used by DiT
        n_layer: int = 12, # num_blocks
        n_head: int = 8, # nhead
        n_emb: int = 768, # hidden_dim
        # dropout, dim_feedforward and activation are 3 args different from DiTXBlock
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        activation: str = "gelu",  
        # for language conditioning
        language_conditioned: bool=False,
        language_model: str = "t5-small",
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.visual_cond_len = visual_cond_len
        self.language_conditioned = language_conditioned

        # positional encoding blocks
        self.enc_pos = _PositionalEncoding(n_emb)
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(horizon, 1, n_emb), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)
        
        # input encoder mlps
        self.time_net = _TimeNetwork(diffusion_timestep_embed_dim, n_emb)
        self.ac_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(input_dim, n_emb),
        )

        # visual cond projection
        self.vis_cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(cond_dim, n_emb),
        )


        # Language conditioning, use T5-small as default
        if self.language_conditioned:
            # else:
            self.load_T5_encoder(
                model_name=language_model,
                freeze=True)
            self.lang_adaptor = self.build_condition_adapter(
                "mlp2x_gelu", 
                in_features=self.language_encoder_out_dim, 
                out_features=n_emb
            )
    
        # encoder blocks
        encoder_module = _SelfAttnEncoder(
            n_emb,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = _TransformerEncoder(encoder_module, n_layer)

        # decoder blocks
        decoder_module = _DiTDecoder(
            n_emb,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, n_layer)

        # turns predicted tokens into epsilons
        self.eps_out = _FinalLayer(n_emb, output_dim)

        print(
            "number of diffusion parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    # language encoder
    def load_T5_encoder(self, model_name, freeze=True):
        from transformers import (
            T5Config,
            T5EncoderModel,
            AutoTokenizer
        )
        T5_model_name = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
        assert model_name in T5_model_name, f"Model name {model_name} not in {T5_model_name}"
        encoder_name = model_name
        pretrained_model_id = f"google-t5/{encoder_name}"
        encoder_cfg = T5Config()
        self.language_encoder = T5EncoderModel(encoder_cfg).from_pretrained(
                pretrained_model_id
            )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
        if freeze:
            self.language_encoder.eval()
            # freeze the language encoder
            for param in self.language_encoder.parameters():
                param.requires_grad = False

        self.language_encoder_out_dim = 512
        cprint(f"Loaded T5 encoder: {encoder_name}", "green")

    def encode_text_input_T5(self,
                             lang_cond,
                             norm_lang_embedding=False,
                             output_type="sentence",
                             device="cuda" if torch.cuda.is_available() else "cpu"
                             ):
        language_inputs = self.tokenizer(
            lang_cond,
            return_tensors="pt",
            padding=True,
            truncation=True,
            )
        input_ids = language_inputs["input_ids"].to(device)
        attention_mask = language_inputs["attention_mask"].to(device)
        encoder_outputs = self.language_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            )
        token_embeddings = encoder_outputs.last_hidden_state
        if output_type == "token":
            return token_embeddings
        # obtain sentence embedding by averaging the token embeddings
        sentence_embedding = torch.mean(token_embeddings, dim=1).squeeze(1) # (B, 512)
        if norm_lang_embedding:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=-1)

        return sentence_embedding
    
    # def forward(self, noise_actions, time, obs_enc, enc_cache=None):
    #     if enc_cache is None:
    #         enc_cache = self.forward_enc(obs_enc)
    #     return enc_cache, self.forward_dec(noise_actions, time, enc_cache)
    def forward(self, sample, timestep, target_t, vis_cond, lang_cond=None, enc_cache=None, **kwargs):
        if enc_cache is None:
            enc_cache = self.forward_enc(vis_cond, lang_cond, device=sample.device)
        return self.forward_dec(sample, timestep, enc_cache)
    
    def forward_enc(self, vis_cond, lang_cond, device):
        # project visual cond
        # print(f"vis_cond shape: {vis_cond.shape}")
        vis_emb = self.vis_cond_proj(vis_cond)
        if self.language_conditioned:
            assert lang_cond is not None
            if isinstance(lang_cond, torch.Tensor):
                # print(lang_cond)
                # print(f"lang_cond is tensor with shape: {lang_cond.shape}")
                lang_emb = lang_cond.unsqueeze(1) # (B, D) -> (B, 1, D)
            else:
                lang_emb = self.encode_text_input_T5(lang_cond, output_type="token", device=device) # (B, L_lang, 512)
                # print(f"lang_emb shape before adaptor: {lang_emb.shape}")
                lang_emb = self.lang_adaptor(lang_emb) # (B, L, D) or (B, D)
        # print(f"vis_emb shape after proj: {vis_emb.shape}")
        # if self.language_conditioned:
            # print(f"lang_emb shape after adaptor: {lang_emb.shape}")
        obs_enc = torch.cat([vis_emb, lang_emb], dim=1)  # (B, L_vis + L_lang, D)
        # print(f"obs_enc shape after concat: {obs_enc.shape}")
        obs_enc = obs_enc.transpose(0, 1)
        pos = self.enc_pos(obs_enc)
        enc_cache = self.encoder(obs_enc, pos)
        # print(f"enc_cache shape: {len(enc_cache)} layers, each of shape {enc_cache[0].shape}")
        return enc_cache # (L, B, D)

    def forward_dec(self, noise_actions, time, enc_cache):
        time_enc = self.time_net(time)
        # print(f"noise_actions shape: {noise_actions.shape}") # (B, T, input_dim)
        ac_tokens = self.ac_proj(noise_actions)
        ac_tokens = ac_tokens.transpose(0, 1)
        # print(f"ac_tokens shape after proj and transpose: {ac_tokens.shape}") # (T, B, D)
        dec_in = ac_tokens + self.dec_pos

        # apply decoder
        dec_out = self.decoder(dec_in, time_enc, enc_cache)
        # print(dec_out.shape) # (T, B, D)
        # apply final epsilon prediction layer
        output = self.eps_out(dec_out, time_enc, enc_cache[-1])
        # print(f"output shape: {output.shape}") # (B, T, output_dim)
        return output # (B, T, output_dim)


