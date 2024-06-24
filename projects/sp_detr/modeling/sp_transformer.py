# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    ConditionalCrossAttention,
    ConditionalSelfAttention,
    MultiheadAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid


class SPDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        activation: nn.Module = nn.PReLU(),
        post_norm: bool = False,
        num_layers: int = 6,
        batch_first: bool = False,
    ):
        super(SPDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(normalized_shape=embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        for layer in self.layers:
            position_scales = self.query_scale(query)
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos * position_scales,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class SPDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.0,
        activation: nn.Module = nn.PReLU(),
        num_layers: int = None,
        modulate_hw_attn: bool = True,
        batch_first: bool = False,
        post_norm: bool = True,
        return_intermediate: bool = True,
    ):
        super(SPDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    ConditionalSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                    ConditionalCrossAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(
                    normalized_shape=embed_dim,
                ),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        self.bbox_embed = None

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.modulate_hw_attn = modulate_hw_attn

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

        self.roi_x_min = torch.tensor(0.1) ## define it
        self.roi_y_min = torch.tensor(0.2)
        self.roi_x_max = torch.tensor(0.8)
        self.roi_y_max = torch.tensor(0.8)

        self.w_min = torch.tensor(0.1)
        self.w_max = torch.tensor(0.5)
        self.h_min = torch.tensor(0.1)
        self.h_max = torch.tensor(0.6)

    def ROI(self, reference_boxes):
        x= reference_boxes[:,:,0]
        y= reference_boxes[:,:,1]
        w= reference_boxes[:,:,2]
        h= reference_boxes[:,:,3]

        self.roi_x_min = self.roi_x_min.cuda()
        self.roi_y_min = self.roi_y_min.cuda()
        self.roi_x_max = self.roi_x_max.cuda()
        self.roi_y_max = self.roi_y_max.cuda()
        self.w_min = self.w_min.cuda()
        self.w_max = self.w_max.cuda()
        self.h_min = self.h_min.cuda()
        self.h_max = self.h_max.cuda()

        x = torch.clamp(x, self.roi_x_min,self.roi_x_max)
        y = torch.clamp(y, self.roi_y_min,self.roi_y_max)
        max_w = torch.clamp(self.roi_x_max - x, min=self.w_min, max=self.w_max)
        max_h = torch.clamp(self.roi_y_max - y, min=self.h_min, max=self.h_max)
        w = torch.clamp(w, self.w_min, max_w)
        h = torch.clamp(h, self.h_min, max_h)

        reference_boxes = torch.stack([x,y,w,h], dim=-1)
        return reference_boxes

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        anchor_box_embed=None,
        **kwargs,
    ):
        intermediate = []

        reference_boxes = anchor_box_embed.sigmoid()
        reference_boxes = self.ROI(reference_boxes)
        intermediate_ref_boxes = [reference_boxes]

        for idx, layer in enumerate(self.layers):
            obj_center = reference_boxes[..., : self.embed_dim]
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)

            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # apply position transform
            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2 :] *= (
                    ref_hw_cond[..., 0] / obj_center[..., 2]
                ).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dim // 2] *= (
                    ref_hw_cond[..., 1] / obj_center[..., 3]
                ).unsqueeze(-1)

            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                is_first_layer=(idx == 0),
                **kwargs,
            )

            # update anchor boxes after each decoder layer using shared box head.
            if self.bbox_embed is not None:
                # predict offsets and added to the input normalized anchor boxes.
                offsets = self.bbox_embed(query)
                offsets[..., : self.embed_dim] += inverse_sigmoid(reference_boxes)
                new_reference_boxes = offsets[..., : self.embed_dim].sigmoid()
                new_reference_boxes = self.ROI(new_reference_boxes)

                if idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_boxes)
                reference_boxes = new_reference_boxes.detach()

            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(intermediate_ref_boxes).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_boxes.unsqueeze(0).transpose(1, 2),
                ]

        return query.unsqueeze(0)


class AttentionFusionModel(nn.Module):
    def __init__(self):
        super(AttentionFusionModel, self).__init__()

        self.downsample_x3_4x = nn.Conv2d(256, 2048, kernel_size=4, stride=4, padding=0)
        self.downsample_x3_2x = nn.Conv2d(256, 1024, kernel_size=2, stride=2, padding=0)


        self.upsample1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)


        self.lateral_conv1 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(768, 256, kernel_size=1)


        self.attention_4x = nn.Conv2d(2048, 2048, kernel_size=1)
        self.attention_2x = nn.Conv2d(1024, 1024, kernel_size=1)
        self.attention_1x = nn.Conv2d(256, 256, kernel_size=1)


        self.final_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3):

        x3_downsampled_4x = self.downsample_x3_4x(x3)  
        x3_downsampled_2x = self.downsample_x3_2x(x3)  


        attention_weights_4x = torch.sigmoid(self.attention_4x(x3_downsampled_4x))  
        attention_weights_2x = torch.sigmoid(self.attention_2x(x3_downsampled_2x))  
        attention_weights_1x = torch.sigmoid(self.attention_1x(x3))  


        x1_weighted = x1 * attention_weights_4x
        x1_weighted = x1_weighted + x1


        x1_upsampled = self.upsample1(x1_weighted)  


        x1_x2_combined = torch.cat([x1_upsampled, x2], dim=1)  


        x1_x2_combined = self.lateral_conv1(x1_x2_combined) 
        x1_x2_weighted = x1_x2_combined * attention_weights_2x
        x1_x2_weighted = x1_x2_weighted + x1_x2_combined


        x1_x2_upsampled = self.upsample2(x1_x2_weighted)  


        x3_combined = torch.cat([x1_x2_upsampled, x3], dim=1)  
        x3_combined = self.lateral_conv2(x3_combined)  

        x3_weighted = x3_combined * attention_weights_1x
        x3_weighted = x3_weighted + x3_combined


        output = self.final_conv1(x3_weighted)  
        output = F.relu(output)
        output = self.final_conv2(output)  
        output = F.relu(output)

        return output


class SPDetrTransformer(nn.Module):
    def __init__(self, encoder=None, decoder=None, num_patterns=0):
        super(SPDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim

        # using patterns designed as AnchorDETR
        assert isinstance(num_patterns, int), "num_patterns should be int but got {}".format(type(num_patterns))
        self.num_patterns = num_patterns
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.embed_dim)

        self.init_weights()
        self.bbox_props = {
            "1": {
           
                "attn_area": [
                    0.29804517027743144,
                    0.22020573202524696,
                    0.8041343647091109,
                    0.8462242830423939,
                ]
                
            }
        }
        self.input_shape = torch.tensor([64, 64]) 
        self.attn_mask = self.generate_attn_masks(0).cuda() 
        self.glff = AttentionFusionModel()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_attn_masks(self, i, padding=0):
        attn_volumes = []
        for props in self.bbox_props.values():
            attn_volume = torch.tensor(props['attn_area'])  
            attn_volumes.append(attn_volume[None])
        attn_volumes = torch.repeat_interleave(torch.cat(attn_volumes), 300, dim=0)


        attn_volumes = ((attn_volumes * self.input_shape.repeat(2)) - padding).clamp(
            min=torch.zeros(4, dtype=torch.int), max=self.input_shape.repeat(2))
        attn_volumes[:, :2] = torch.floor(attn_volumes[:, :2])
        attn_volumes[:, 2:] = torch.ceil(attn_volumes[:, 2:])
        attn_volumes = attn_volumes.int()

        attn_mask = torch.ones(300, *self.input_shape.tolist()).bool()


        for q in range(300):
            attn_mask[q, attn_volumes[q, 0]:attn_volumes[q, 2], attn_volumes[q, 1]:attn_volumes[q, 3]] = False

        return attn_mask.flatten(1)

    def forward(self, x, mask, anchor_box_embed, pos_embed, features_s4, features_s5):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)  
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        anchor_box_embed = anchor_box_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )

        memory = memory.permute(0, 2, 1).reshape(-1, c, h, w)
        memory = self.glff(features_s5, features_s4, memory)
        memory = memory.view(bs, c, -1).permute(2, 0, 1)


        num_queries = anchor_box_embed.shape[0]


        if self.num_patterns == 0:
            target = torch.zeros(num_queries, bs, self.embed_dim, device=anchor_box_embed.device)
        else:
            target = self.patterns.weight[:, None, None, :].repeat(1, num_queries, bs, 1).flatten(0, 1)
            anchor_box_embed = anchor_box_embed.repeat(self.num_patterns, 1, 1)

        hidden_state, reference_boxes = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            anchor_box_embed=anchor_box_embed,
            attn_masks = self.attn_mask.float() 
        )

        return hidden_state, reference_boxes
