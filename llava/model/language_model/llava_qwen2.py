#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# import sys
# sys.path.insert(0, '/opt/conda/lib/python3.10')
# print(sys.path)
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         Qwen2ForCausalLM, Qwen2Config, Qwen2Model \

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen2"


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config, init_vision_encoder_from_ckpt=False):
        # config._attn_implementation = "flash_attention_2"
        # config._flash_attn_2_enabled = True
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwen2Model(config)
        # assert self.model._use_flash_attention_2 == True
        # self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path,padding_side="left")
        self.tokenizer = None
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # if getattr(config, 'init_vision_encoder_from_ckpt', True):
        if init_vision_encoder_from_ckpt:
            vision_tower = self.get_vision_tower()
            print(f'loading from CLIP first. This should only be used at inference!!!')
            vision_tower.load_model() # 
            
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            # ) = self.prepare_inputs_labels_for_multimodal(
            ) = self.prepare_inputs_labels_for_multimodal_new(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) :
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal_new(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # print(inputs_embeds.shape)
        return super().generate(
            position_ids=None,
            attention_mask=None,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

# AutoConfig.register("llava", LlavaQwenConfig)
AutoConfig.register("llava_qwen2", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwen2ForCausalLM)