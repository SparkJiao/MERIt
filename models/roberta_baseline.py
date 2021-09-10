from abc import ABC

from typing import Union
import torch
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaConfig, RobertaLMHead, \
    MaskedLMOutput

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules import layers

logger = get_child_logger("RoBERTa")


class RobertaForMultipleChoice(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 re_init_cls: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 freeze_encoder: bool = False,
                 no_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.re_init_cls = re_init_cls
        if self.re_init_cls:
            self.classifier_i = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, 1)

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.no_pooler = no_pooler
        self.freeze_encoder = freeze_encoder
        print(self.freeze_encoder)
        if freeze_encoder:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.init_weights()

        self.init_metric("loss", "acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            sentence_index: Tensor = None,
            sentence_mask: Tensor = None,
            sent_token_mask: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.no_pooler:
            pooled_output = outputs[0][:, 0]
        else:
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        if self.re_init_cls:
            logits = self.classifier_i(pooled_output)
        else:
            logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        choice_mask = (attention_mask.sum(dim=-1) == 0).reshape(-1, num_choices)
        reshaped_logits = reshaped_logits + choice_mask * -10000.0

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(reshaped_logits, labels)

            if mlm_labels is not None:
                mlm_scores = self.lm_head(outputs[0])
                mlm_loss = loss_fct(mlm_scores.reshape(-1, self.config.vocab_size), mlm_labels.reshape(-1))
                loss += mlm_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoiceForPreTrain(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 fs_checkpoint_start_layer_id: int = 0):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vocab_size = config.vocab_size

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 1)

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.init_weights()

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        # choice_mask = (attention_mask.sum(dim=-1) == 0).reshape(-1, num_choices)
        # reshaped_logits = reshaped_logits + choice_mask * -10000.0

        loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None:
                if mlm_attention_mask is None:
                    mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]

                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMaskedLM(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]

    def __init__(self, config: RobertaConfig,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.vocab_size = config.vocab_size

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.init_weights()

        self.init_metric("loss", "acc")

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))

            if not self.training:
                acc, true_label_num = layers.get_accuracy(prediction_scores, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", masked_lm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoiceForZeroShot(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 freeze_encoder: bool = False,
                 freeze_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        # self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.vocab_size = config.vocab_size

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 1)

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.freeze_pooler = freeze_pooler
        if self.freeze_pooler:
            for param in self.pooler.parameters():
                param.requires_grad = False
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.init_weights()

        self.init_metric("loss", "acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = cls_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoiceLReasoner(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 freeze_encoder: bool = False,
                 no_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.no_pooler = no_pooler
        if not self.no_pooler:
            self.pooler_1 = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )
            self.pooler_2 = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler_1 = None
            self.pooler_2 = None

        self.cls_1 = nn.Linear(config.hidden_size, 1)
        self.cls_2 = nn.Linear(config.hidden_size, 1)

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.freeze_encoder = freeze_encoder
        print(self.freeze_encoder)
        if freeze_encoder:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.loss_fct = CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

        self.init_metric("loss", "cls_loss", "cls_acc", "contras_loss", "contras_acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def cls_forward(self, input_ids, attention_mask, token_type_ids, pooler: Union[nn.Module, None], cls: nn.Module,
                    output_attentions, output_hidden_states, return_dict):
        num_choice = input_ids.size(1)

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.no_pooler:
            logits = cls(self.dropout(outputs[0][:, 0]))
        else:
            logits = cls(self.dropout(pooler(outputs[0][:, 0])))
        return logits.reshape(-1, num_choice)

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            con_input_ids: Tensor = None,
            con_attention_mask: Tensor = None,
            con_token_type_ids: Tensor = None,
            con_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            logits = self.cls_forward(input_ids, attention_mask, token_type_ids, self.pooler_1, self.cls_1,
                                      output_attentions, output_hidden_states, return_dict)
        else:
            logits = None

        if con_input_ids is not None:
            con_logits = self.cls_forward(con_input_ids, con_attention_mask, con_token_type_ids, self.pooler_2, self.cls_2,
                                          output_attentions, output_hidden_states, return_dict)
        else:
            con_logits = None

        loss = 0.
        if con_labels is not None:
            loss2 = self.loss_fct(con_logits, con_labels)
            loss = loss + loss2

            if not self.training:
                acc, true_label_num = layers.get_accuracy(con_logits, con_labels)
                self.eval_metrics.update("contras_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("contras_loss", val=loss2.item(), n=true_label_num)

        if labels is not None:
            loss1 = self.loss_fct(logits, labels)
            loss = loss + loss1

            if not self.training:
                acc, true_label_num = layers.get_accuracy(logits, labels)
                self.eval_metrics.update("cls_acc", val=acc, n=true_label_num)
                self.eval_metrics.update("cls_loss", val=loss1.item(), n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class RobertaForMultipleChoicePrompt(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 prompt_mlp_hidden_size: int = 768,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 freeze_encoder: bool = False,
                 freeze_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.prompt_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, prompt_mlp_hidden_size),
            nn.Tanh(),
            nn.Linear(prompt_mlp_hidden_size, config.hidden_size),
        )

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 1)

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.freeze_pooler = freeze_pooler
        if self.freeze_pooler:
            for param in self.pooler.parameters():
                param.requires_grad = False
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for name, param in self.roberta.named_parameters():
                if 'embeddings.word_embeddings' not in name:
                    param.requires_grad = False

        self.init_weights()

        self.init_metric("loss", "acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            prefix_pos: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)
        prefix_pos = self.fold_tensor(prefix_pos)

        embed_layer = self.roberta.embeddings.word_embeddings
        input_embeds = embed_layer(input_ids)

        ex_prefix_pos = prefix_pos.unsqueeze(-1).expand(-1, -1, input_embeds.size(-1))
        prefix_embed = torch.gather(input_embeds, index=ex_prefix_pos, dim=1)
        prefix_embed = self.prompt_mlp(prefix_embed)
        input_embeds = torch.scatter(input_embeds, dim=1, index=ex_prefix_pos, src=prefix_embed)

        if self.freeze_encoder:
            input_embeds = layers.keep_grad_prompt(input_embeds, prefix_pos)

        outputs = self.roberta(
            inputs_embeds=input_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = cls_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
