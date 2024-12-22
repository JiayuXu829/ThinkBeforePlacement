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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from .clip_model import CLIPVisionTower

MAX_IMAGE_NUM = 60

class AccNet(nn.Module):
    def __init__(self, input_channels, hidden_units, output_dim, vision_tower):
        super(AccNet, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_dim)
        self.vision_tower = vision_tower


    def forward(self, mask_fg_img, bg_img, trans_label):
        mask_fg_img_feat = self.vision_tower(mask_fg_img)[:,0,:]
        bg_img_feat = self.vision_tower(bg_img)[:,0,:]

        # Concatenate the images along the channel axis and flatten
        trans_label = trans_label.squeeze(1)
        fused_feat = torch.cat((mask_fg_img_feat, bg_img_feat, trans_label), dim=1)
        x1 = torch.relu(self.fc1(fused_feat))
        x2 = torch.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        x3_softmax = F.softmax(x3, dim=1)
        return x3_softmax

class ModelArguments:
    def __init__(self, vision_tower, mm_vision_select_layer,mm_vision_select_feature):
        self.vision_tower = vision_tower
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature

# 创建ModelArguments实例
vision_tower_cfg = ModelArguments(
    vision_tower='openai/clip-vit-large-patch14',
    mm_vision_select_layer=-2,
    mm_vision_select_feature='cls_patch'
)

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        self.token_num = 1
        self.pred_num = 10
        
        self.comp_regs = nn.ModuleList([nn.Sequential(
            nn.Linear(config.hidden_size, 1024, bias=True),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 3)
        ) for _ in range(self.pred_num)])

        self.acc_pred = nn.ModuleList([nn.Sequential(
            nn.Linear(config.hidden_size, 1024, bias=True),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        ) for _ in range(self.pred_num)])
        
        self.reg_token = nn.Embedding(self.token_num, config.hidden_size) #1*256

        acc_net_vision_tower = CLIPVisionTower(vision_tower=vision_tower_cfg.vision_tower, args=vision_tower_cfg)
        self.acc_net = AccNet(input_channels=1024*2+3, hidden_units=1024, output_dim=2, vision_tower=acc_net_vision_tower)
        
        self.recons_loss_no_reduction = torch.nn.MSELoss(reduction='none')
        
        self.print_num = 1
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        fg_imgs:Optional[torch.FloatTensor] = None,
        bg_imgs:Optional[torch.FloatTensor] = None,
        trans_labels_pos:Optional[torch.FloatTensor] = None,
        trans_labels_neg:Optional[torch.FloatTensor] = None,
        valid_trans_label_nums_pos:Optional[torch.IntTensor] = None,
        valid_trans_label_nums_neg:Optional[torch.IntTensor] = None,
        bboxes:Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bs = input_ids.shape[0]
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        if inputs_embeds is not None:
            tgt_src = self.reg_token.weight.unsqueeze(0).repeat(bs, 1, 1).to(inputs_embeds.device)
            inputs_embeds = torch.cat((inputs_embeds, tgt_src), dim=1)
            new_col = torch.ones((bs, self.token_num), dtype=torch.bool).to(attention_mask.device)
            attention_mask = torch.cat((attention_mask, new_col), dim=1)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if inputs_embeds is not None: # training process
            hidden_states = outputs[0]
            # regression
            reg_token_states = hidden_states[:,-self.token_num:,:]
            reg_token_states = torch.mean(reg_token_states, dim=1)
            pred_trans_list = list()
            for comp_reg in self.comp_regs:
                pred_trans_list.append(comp_reg(reg_token_states))
            pred_trans = torch.stack(pred_trans_list, dim=1)
            pred_trans = torch.tanh(pred_trans) / 2.0 + 0.5  #(16,5,3)
            
            # caption
            caption_states = hidden_states[:,:-self.token_num,:]
            caption_logits = self.lm_head(caption_states)
        
        else: # generate process
            hidden_states = outputs[0]
            caption_states = hidden_states
            caption_logits = self.lm_head(caption_states)

        # claculate caption and trans label and acc loss
        caption_loss = 0
        trans_label_loss = 0
        each_tran_loss = 0
        distance_loss = 0
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = caption_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            caption_loss = loss_fct(shift_logits, shift_labels)
            
            # caculate trans label loss
            #batch_pos_cnt = 0
            for i in range(bs):
                margin = 0.3

                if(valid_trans_label_nums_pos[i] == 0):
                    loss_pos = 0.0
                    each_mse_loss_pos = 0.0
                else:
                    single_pred_trans_pos = pred_trans[i].unsqueeze(0).repeat(valid_trans_label_nums_pos[i], 1, 1)
                    valid_trans_labels_pos = trans_labels_pos[i][:valid_trans_label_nums_pos[i]].unsqueeze(1).repeat(1,self.pred_num,1)
                    single_mse_loss_pos = self.recons_loss_no_reduction(single_pred_trans_pos, valid_trans_labels_pos).mean(dim=2)
                    min_mse_loss_pos, _ = torch.min(single_mse_loss_pos, dim=1) 
                    loss_pos = torch.mean(min_mse_loss_pos)
                
                if(valid_trans_label_nums_neg[i] == 0):
                    loss_neg = 0.0
                else:
                    single_pred_trans_neg = pred_trans[i].unsqueeze(0).repeat(valid_trans_label_nums_neg[i], 1, 1)
                    valid_trans_labels_neg = trans_labels_neg[i][:valid_trans_label_nums_neg[i]].unsqueeze(1).repeat(1,self.pred_num,1)
                    single_mse_loss_neg = self.recons_loss_no_reduction(single_pred_trans_neg[:,:,1:], valid_trans_labels_neg[:,:,1:]).mean(dim=2)
                    min_mse_loss_neg, _ = torch.min(single_mse_loss_neg, dim=1) 
                    margin_mse_loss_neg = torch.clamp(margin - min_mse_loss_neg, min=0)
                    loss_neg = torch.mean(margin_mse_loss_neg)

                # if valid_trans_label_nums_pos[i] == 0:
                #     continue
                # else:
                #     batch_pos_cnt += 1
                      #single_loss = loss_pos * valid_trans_label_nums_pos[i] / valid_trans_label_nums_pos[i]
                single_loss = (loss_pos * valid_trans_label_nums_pos[i] + 0.2 * loss_neg * valid_trans_label_nums_neg[i]) / (valid_trans_label_nums_neg[i] + valid_trans_label_nums_pos[i])
                trans_label_loss += single_loss

            trans_label_loss = trans_label_loss/bs

            # calculate distance loss            
            #epsilon = 1e-6  # 避免除以0
            penalty = 0.0
            dist_margin = 0.5
            dist_sum = 0
            penalty_num = (self.pred_num)*(self.pred_num-1)/2
            for i in range(self.pred_num):
                for j in range(i+1, self.pred_num):
                    distance = torch.norm(pred_trans[:, i, 1:] - pred_trans[:, j, 1:], dim=1)
                    dist_sum += distance
                    #penalty += torch.sum(1.0 / distance)
            distance_loss = max(dist_margin - torch.sum(dist_sum)/(penalty_num*bs),0.0)

        loss = 4.0*trans_label_loss + 0.1*caption_loss + 0.01*distance_loss
        #print(f"trans_label_loss:{4.0*trans_label_loss}, caption_loss:{0*caption_loss}, distance_loss:{0.01*distance_loss}")
        if not return_dict:
            output = (caption_logits,) + outputs[1:]
            return (caption_loss,) + output if caption_loss is not None else output


        return CausalLMOutputWithPast(
            loss=loss,
            logits=caption_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def generate_trans_label(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        token_idx:Optional[torch.LongTensor] = None
    ):
        bs = input_ids.shape[0]
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        tgt_src = self.reg_token.weight.unsqueeze(0).repeat(bs, 1, 1).to(inputs_embeds.device)
        inputs_embeds = torch.cat((inputs_embeds, tgt_src), dim=1)
        new_col = torch.ones((bs, self.token_num), dtype=torch.bool).to(attention_mask.device)
        attention_mask = torch.cat((attention_mask, new_col), dim=1)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
            
        hidden_states = outputs[0]
        # regression
        reg_token_states = hidden_states[:,-self.token_num:,:]
        reg_token_states = torch.mean(reg_token_states, dim=1)
        pred_trans_list = list()
        for comp_reg in self.comp_regs:
            pred_trans_list.append(comp_reg(reg_token_states))
        pred_trans = torch.stack(pred_trans_list, dim=1)
        pred_trans = torch.tanh(pred_trans) / 2.0 + 0.5  #(16,5,3)
        
        #predict the accurancy of our results
        pred_acc_list = list()
        for acc_pred_net in self.acc_pred:
             pred_acc_list.append(acc_pred_net(reg_token_states))
        pred_acc = torch.stack(pred_acc_list, dim=1)
        pred_acc = torch.sigmoid(pred_acc).squeeze(2) #(16,5,2)
        
        if self.print_num:
            print(f"pred_num:{self.pred_num}")
            self.print_num -= 1
        #import pdb;pdb.set_trace()
        if token_idx is None:
             _ , max_acc_score_idx = torch.max(pred_acc, dim=1) # (16,1)
             max_pred_trans = pred_trans[torch.arange(pred_trans.size(0)), max_acc_score_idx]
             return max_pred_trans
        else:
             return pred_trans[:,token_idx]
    
    def generate_trans_label_all(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
    ):
        bs = input_ids.shape[0]
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        tgt_src = self.reg_token.weight.unsqueeze(0).repeat(bs, 1, 1).to(inputs_embeds.device)
        inputs_embeds = torch.cat((inputs_embeds, tgt_src), dim=1)
        new_col = torch.ones((bs, self.token_num), dtype=torch.bool).to(attention_mask.device)
        attention_mask = torch.cat((attention_mask, new_col), dim=1)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
            
        hidden_states = outputs[0]
        # regression
        reg_token_states = hidden_states[:,-self.token_num:,:]
        reg_token_states = torch.mean(reg_token_states, dim=1)
        pred_trans_list = list()
        for comp_reg in self.comp_regs:
            pred_trans_list.append(comp_reg(reg_token_states))
        pred_trans = torch.stack(pred_trans_list, dim=1)
        pred_trans = torch.tanh(pred_trans) / 2.0 + 0.5  #(16,5,3)
        
        if self.print_num:
            print(f"pred_num:{self.pred_num}")
            self.print_num -= 1
        #import pdb;pdb.set_trace() 
        return pred_trans
    
    def generate_acc_score(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        token_idx:Optional[torch.LongTensor] = None
    ):
        bs = input_ids.shape[0]
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        tgt_src = self.reg_token.weight.unsqueeze(0).repeat(bs, 1, 1).to(inputs_embeds.device)
        inputs_embeds = torch.cat((inputs_embeds, tgt_src), dim=1)
        new_col = torch.ones((bs, self.token_num), dtype=torch.bool).to(attention_mask.device)
        attention_mask = torch.cat((attention_mask, new_col), dim=1)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        # regression
        reg_token_states = hidden_states[:,-self.token_num:,:]
        reg_token_states = torch.mean(reg_token_states, dim=1)
        pred_trans_list = list()
        for comp_reg in self.comp_regs:
            pred_trans_list.append(comp_reg(reg_token_states))
        pred_trans = torch.stack(pred_trans_list, dim=1)
        pred_trans = torch.tanh(pred_trans) / 2.0 + 0.5  #(16,5,3)
        

        #predict the accurancy of our results
        pred_acc_list = list()
        for acc_pred_net in self.acc_pred:
            pred_acc_list.append(acc_pred_net(reg_token_states))
        pred_acc = torch.stack(pred_acc_list, dim=1)
        pred_acc = torch.sigmoid(pred_acc).squeeze(2) #(16,5,2)
        

        return pred_acc
    
AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
