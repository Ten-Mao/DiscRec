import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack, logger
from transformers.models.t5.configuration_t5 import T5Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.utils.import_utils import is_torchdynamo_compiling


from layers.FeedForward import FeedForward
from layers.LayerNorm import LayerNorm
from layers.MultiHeadAttention import MultiHeadAttention
from util.util import Trie, get_prefix_allowed_tokens_fn


class DiscRecAttention(MultiHeadAttention):
    def __init__(self, d_model, n_heads, dropout=0.1, causal=False, q_mask=False):
        super().__init__(d_model, n_heads, dropout=dropout, causal=causal, q_mask=q_mask)
    
    def forward(self, q, k, v, key_atten_mask=None, query_padding_mask=None):
        def proj_and_split(x, proj_layer):
            x = proj_layer(x)
            bsz, sqlen, _ = x.shape
            head_dim = self.d_model // self.n_heads
            x = x.reshape(bsz, sqlen, self.n_heads, head_dim).permute(0, 2, 1, 3)
            return x
        
        Q = proj_and_split(q, self.q_proj)
        K = proj_and_split(k, self.k_proj)
        V = proj_and_split(v, self.v_proj)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)

        if key_atten_mask is not None:
            attn_weights += key_atten_mask
        
        if self.causal:
            causal_mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2]), diagonal=1).to(q.device) > 0
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.q_mask and query_padding_mask is not None:
            query_padding_mask = ~query_padding_mask[:, None, :, None].to(attn_weights.device)
            attn_weights = attn_weights.masked_fill(query_padding_mask, 0.0)
        

        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[1], -1)

        return out + q


class DiscRecT5Stack(T5Stack):

    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        self.code_embedding = nn.Embedding(6, 128)
        self.collaborate_attn_norm = LayerNorm(128, eps=1e-8)
        self.collaborate_attention = DiscRecAttention(
            d_model=128, 
            n_heads=2,
            dropout=0.1,
            causal=True if self.is_decoder else False,
            q_mask=True
        )
        self.collaborate_ffn_norm = LayerNorm(128, eps=1e-8)
        self.collaborate_ffn = FeedForward(
            d_model=128,
            inner_dim=1024,
            dropout=0.1,
            activation="relu"
        )

        self.gated_w = nn.Embedding(2, 128)

    
    def moe4co(self, input_ids, attention_mask):
        if not self.config.is_decoder:
            semantic_emb = self.embed_tokens(input_ids)
            code_pos = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.long) 
            code_pos = code_pos.remainder(4)+1  # 等价于 base % 4  
            code_pos[-1] = torch.tensor(5, dtype=torch.long, device=input_ids.device)
            code_pos_emb = self.code_embedding(code_pos).unsqueeze(0)
            collaborate_emb = semantic_emb + code_pos_emb
            # collaborate_emb = semantic_emb
            collaborate_emb = self.collaborate_attn_norm(collaborate_emb)
            q, k, v = collaborate_emb, collaborate_emb, collaborate_emb
            # 1. 创建一个 N x N 的单位矩阵 I
            I = torch.eye(math.ceil(input_ids.shape[-1]/4), device=input_ids.device, dtype=torch.long)  # :contentReference[oaicite:0]{index=0}
            # 2. 构造一个 4x4 的全 1 块
            ones_block = torch.ones((4, 4), device=input_ids.device, dtype=torch.long)  # :contentReference[oaicite:1]{index=1}
            # 3. 利用 Kronecker 积生成大小为 4N x 4N 的块对角掩码矩阵（对角块为 1，其他为 0）
            #    mask[k,l] = 1 当且仅当 floor(k/4)==floor(l/4) 且同属第 i 块
            mask = torch.kron(I, ones_block).to(torch.bool)  # :contentReference[oaicite:2]{index=2}

            # 4. 先创建一个全 -inf 的矩阵
            M = torch.full((4*math.ceil(input_ids.shape[-1]/4), 4*math.ceil(input_ids.shape[-1]/4)), float('-inf'), device=input_ids.device, dtype=torch.float32)  # :contentReference[oaicite:3]{index=3}

            # 5. 利用掩码将对角子块位置填为 0
            M = M.masked_fill_(mask, 0.0)[:input_ids.shape[-1], :input_ids.shape[-1]]  # :contentReference[oaicite:4]{index=4}
            M[-5, -1] = torch.tensor(0.0, dtype=torch.float32, device=M.device)
            M[-4, -1] = torch.tensor(0.0, dtype=torch.float32, device=M.device)
            M[-3, -1] = torch.tensor(0.0, dtype=torch.float32, device=M.device)
            M[-2, -1] = torch.tensor(0.0, dtype=torch.float32, device=M.device)
            M[-1, -5] = torch.tensor(0.0, dtype=torch.float32, device=M.device)
            M[-1, -4] = torch.tensor(0.0, dtype=torch.float32, device=M.device)
            M[-1, -3] = torch.tensor(0.0, dtype=torch.float32, device=M.device)
            M[-1, -2] = torch.tensor(0.0, dtype=torch.float32, device=M.device)

            mask_for_co = attention_mask.bool() if attention_mask is not None else None
            collaborate_emb = self.collaborate_attention(q, k, v, key_atten_mask=M.unsqueeze(0).unsqueeze(0), query_padding_mask=mask_for_co)
            collaborate_emb = self.collaborate_ffn_norm(collaborate_emb)
            collaborate_emb = self.collaborate_ffn(collaborate_emb)

            EMB = torch.cat([semantic_emb.unsqueeze(-2), collaborate_emb.unsqueeze(-2)], dim=-2)
            score = torch.softmax((EMB * self.gated_w.weight).sum(-1), dim=-1)
            inputs_embeds = (EMB * score.unsqueeze(-1)).sum(-2)
        else:
            semantic_emb = self.embed_tokens(input_ids)
            assert input_ids.shape[-1] <= 5
            code_pos = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.long) 
            code_pos_emb = self.code_embedding(code_pos).unsqueeze(0)
            collaborate_emb = semantic_emb + code_pos_emb
            # collaborate_emb = semantic_emb
            collaborate_emb = self.collaborate_attn_norm(collaborate_emb)
            q, k, v = collaborate_emb, collaborate_emb, collaborate_emb
            collaborate_emb = self.collaborate_attention(q, k, v)
            collaborate_emb = self.collaborate_ffn_norm(collaborate_emb)
            collaborate_emb = self.collaborate_ffn(collaborate_emb)

            EMB = torch.cat([semantic_emb.unsqueeze(-2), collaborate_emb.unsqueeze(-2)], dim=-2)
            score = torch.softmax((EMB * self.gated_w.weight).sum(-1), dim=-1)
            inputs_embeds = (EMB * score.unsqueeze(-1)).sum(-2) 

        return inputs_embeds


    def _forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        return_dict=None,
    ):
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            # inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.moe4co(input_ids, attention_mask) 

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values
        return_legacy_cache = False
        return_self_attention_cache = False
        if self.is_decoder and (use_cache or past_key_values is not None):
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache if past_key_values is not None else None,
                output_attentions,
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    causal_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                    return_dict,
                    cache_position,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, next_decoder_cache = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )



class DiscRec(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):
        super().__init__(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = DiscRecT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = DiscRecT5Stack(decoder_config, self.shared)

        self.loss_func = nn.CrossEntropyLoss()

        self.post_init()

    def _forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,                   # [batch_size, seq_len + 1]
        attention_mask: Optional[torch.FloatTensor] = None,             # [batch_size, seq_len + 1]
        decoder_input_ids: Optional[torch.LongTensor] = None,           # [batch_size, 1 + seq_len_tgt]
        inputs_embeds: Optional[torch.FloatTensor] = None,              # [batch_size, seq_len, hidden_size]
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,      # [batch_size, 1 + seq_len_tgt, hidden_size]
        labels: Optional[torch.LongTensor] = None,                      # [batch_size, seq_len_tgt + 1]
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert labels is not None, "labels are required"

        encoder_outputs = self.encoder._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        ) # (last_hidden_state)

        hidden_states = encoder_outputs[0] # [batch_size, seq_len, hidden_size]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels) # [batch_size, 1 + seq_len_tgt]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder._forward(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        ) # (last_hidden_state)

        sequence_output = decoder_outputs[0] # [batch_size, seq_len_tgt + 1, hidden_size]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output) # [batch_size, seq_len_tgt + 1, vocab_size]

        loss = None
        # move labels to correct device to enable PP
        labels = labels.to(lm_logits.device)
        loss = self.loss_func(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output)

        return loss

    def _inference(
        self,
        input_ids,                   
        attention_mask,  
        item_indices,
        beam_size      
    ):
        item_indices = [[0] + x + [1] for x in item_indices.tolist()]
        trie = Trie(item_indices)
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(trie)

        output = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=10,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            early_stopping=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

        batch_size = input_ids.shape[0]
        output_ids = output["sequences"]
        output_ids = torch.stack([x[1:-1] for x in output_ids])
        output_ids = torch.reshape(
            output_ids, [batch_size, beam_size, -1]
        )
        return output_ids




