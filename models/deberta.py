from abc import ABC
from typing import List

import torch
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, LayerNorm
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model, DebertaV2Config, \
    ContextPooler, StableDropout, ACT2FN, DebertaV2Encoder

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules import layers

logger = get_child_logger("DeBERTaV2")


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

        self.LayerNorm = LayerNorm(self.embedding_size, config.layer_norm_eps, elementwise_affine=True)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


# Copied from https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/apps/models/masked_language_model.py#L30-L82
class EnhancedMaskDecoder(torch.nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.add_enhanced_encoder = getattr(config, "add_enhanced_decoder", True)
        self.position_biased_input = getattr(config, 'position_biased_input', True)
        self.lm_head = BertLMPredictionHead(config)

    def forward(self, encoded_layers: List[Tensor], z_states: Tensor, attention_mask: Tensor, encoder: DebertaV2Encoder,
                mlm_labels: Tensor, relative_pos=None):
        if self.add_enhanced_encoder:
            mlm_ctx_layers = self.emd_context_layer(encoded_layers, z_states, attention_mask, encoder, relative_pos=relative_pos)
        else:
            mlm_ctx_layers = encoded_layers[-1]
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

        mlm_labels = mlm_labels.reshape(-1)
        mlm_index = (mlm_labels > 0).nonzero().view(-1)
        mlm_ctx_states = mlm_ctx_layers[-1].reshape(-1, mlm_ctx_layers[-1].size(-1)).index_select(0, index=mlm_index)
        mlm_target_ids = mlm_labels.index_select(0, index=mlm_index)
        mlm_logits = self.lm_head(mlm_ctx_states)
        mlm_loss = loss_fct(mlm_logits.reshape(-1, self.vocab_size), mlm_target_ids)

        return mlm_logits, mlm_target_ids, mlm_loss

    def emd_context_layer(self, encoded_layers: List[Tensor], z_states: Tensor, attention_mask: Tensor,
                          encoder: DebertaV2Encoder, relative_pos=None):
        attention_mask = encoder.get_attention_mask(attention_mask)
        hidden_states = encoded_layers[-2]
        if not self.position_biased_input:
            enc_layers = [encoder.layer[-1] for _ in range(2)]
            z_states = z_states + hidden_states
            query_states = z_states
            query_mask = attention_mask
            outputs = []
            rel_embeddings = encoder.get_rel_embedding()

            for layer in enc_layers:
                # TODO: pass relative pos ids
                output = layer(hidden_states, query_mask, query_states=query_states, relative_pos=relative_pos,
                               rel_embeddings=rel_embeddings)
                query_states = output
                outputs.append(query_states)
        else:
            outputs = [encoded_layers[-1]]
            raise RuntimeError()  # For debug. ``position_biased_input==False``

        return outputs


# Copied from transformers.models.deberta.modeling_deberta.ContextPooler and modify the pooler hidden size
class ContextPoolerE(nn.Module):
    def __init__(self, config, mlp_hidden_size: 768):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, mlp_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


def wrap_activation_checkpoint(encoder: DebertaV2Encoder, config: DebertaV2Config, checkpoint: bool = False,
                               fs_checkpoint: bool = False, fs_checkpoint_cpu_offload: bool = False):
    if checkpoint:  # Requires ``transformers >= 4.12.0``
        # return ActivationCheckpointHelper(config)
        encoder.gradient_checkpointing = True
        return encoder

    if fs_checkpoint:
        wrapped_layers = nn.ModuleList([
            checkpoint_wrapper(encoder.layer[i], offload_to_cpu=fs_checkpoint_cpu_offload)
            for i in range(config.num_hidden_layers)
        ])
        encoder.layer = wrapped_layers
        return encoder

    return encoder


class DebertaV2ForMultipleChoicePreTrain(DebertaV2PreTrainedModel, LogMixin, ABC):
    def __init__(self, config: DebertaV2Config,
                 mlp_hidden_size: int = 768,
                 add_enhanced_decoder: bool = True,
                 mlm_alpha: float = 1.0,
                 use_stable_embedding: bool = False,
                 activation_checkpoint: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0):
        super().__init__(config)

        config.update({
            "mlp_hidden_size": mlp_hidden_size,
            "add_enhanced_decoder": add_enhanced_decoder,
            "mlm_alpha": mlm_alpha,
            "use_stable_embedding": use_stable_embedding
        })

        self.config = config

        self.deberta = DebertaV2Model(config)
        # Hack here. Since ``position_based_input==False``, the weights won't be loaded.
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # self.deberta.embeddings.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        self.lm_predictions = EnhancedMaskDecoder(config)

        if use_stable_embedding:
            import bitsandbytes as bnb

            pad_token_id = getattr(config, "pad_token_id", 0)
            self.deberta.embeddings.word_embeddings = bnb.nn.StableEmbedding(config.vocab_size,
                                                                             self.embedding_size,
                                                                             padding_idx=pad_token_id)
            self.deberta.embeddings.position_embeddings = bnb.nn.StableEmbedding(config.max_position_embeddings, self.embedding_size)

        self.pooler = ContextPoolerE(config, mlp_hidden_size=mlp_hidden_size)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(mlp_hidden_size, 1)

        self.deberta.encoder = wrap_activation_checkpoint(self.deberta.encoder, config,
                                                          checkpoint=activation_checkpoint,
                                                          fs_checkpoint=fs_checkpoint,
                                                          fs_checkpoint_cpu_offload=fs_checkpoint_offload_to_cpu)

        self.init_weights()
        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss")

        self.mlm_alpha = mlm_alpha

    def get_output_embeddings(self):
        return self.lm_predictions.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_predictions.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.deberta.embeddings.word_embeddings

    def get_position_embeddings(self, seq_length):
        position_ids = self.deberta.embeddings.position_ids[:, :seq_length].to(self.position_embeddings.weight.device)
        position_emb = self.position_embeddings(position_ids)
        return position_emb

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def mlm_forward(self, mlm_input_ids: Tensor, mlm_attention_mask: Tensor, mlm_labels: Tensor = None, return_dict=None):

        mlm_outputs = self.deberta(
            mlm_input_ids,
            attention_mask=mlm_attention_mask,
            output_hidden_states=True,
            return_dict=return_dict
        )

        encoded_layers = mlm_outputs[1]
        z_states = self.get_position_embeddings(mlm_input_ids.size(1))

        mlm_logits, mlm_target_ids, mlm_loss = self.lm_predictions(encoded_layers=encoded_layers,
                                                                   z_states=z_states,
                                                                   attention_mask=mlm_attention_mask,
                                                                   encoder=self.deberta.encoder,
                                                                   mlm_labels=mlm_labels,
                                                                   relative_pos=None)
        return mlm_logits, mlm_target_ids, mlm_loss

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor = None,
                token_type_ids: Tensor = None,
                labels: Tensor = None,
                mlm_input_ids: Tensor = None,
                mlm_attention_mask: Tensor = None,
                mlm_labels: Tensor = None,
                return_dict=None,
                **kwargs):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )

        logits = self.cls(self.dropout(self.pooler(outputs[0])))
        reshaped_logits = logits.view(-1, num_choices)

        loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None:
                if mlm_attention_mask is None:
                    mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]

                mlm_scores, mlm_labels, mlm_loss = self.mlm_forward(mlm_input_ids, mlm_attention_mask, mlm_labels, return_dict=return_dict)
                loss = loss + self.mlm_alpha * mlm_loss
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


class DebertaV2ForMultipleChoice(DebertaV2PreTrainedModel, LogMixin, ABC):
    def __init__(self, config: DebertaV2Config, override_pooler: bool = False,
                 activation_checkpoint: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_cpu_offload: bool = False):
        super().__init__(config)

        self.config = config
        self.override_pooler = override_pooler

        self.deberta = DebertaV2Model(config)
        self.deberta.encoder = wrap_activation_checkpoint(self.deberta.encoder, config, checkpoint=activation_checkpoint,
                                                          fs_checkpoint=fs_checkpoint, fs_checkpoint_cpu_offload=fs_checkpoint_cpu_offload)

        if self.override_pooler:
            mlp_hidden_size = getattr(config, "mlp_hidden_size", config.hidden_size)
            self.pooler = ContextPoolerE(config, mlp_hidden_size=mlp_hidden_size)
            self.cls = nn.Linear(mlp_hidden_size, 1)
        else:
            self.n_pooler = ContextPooler(config)
            output_dim = self.n_pooler.output_dim
            self.classifier = nn.Linear(output_dim, 1)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

        self.init_metric("loss", "acc")

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        if self.override_pooler:
            pooled_output = self.pooler(encoder_layer)
            pooled_output = self.dropout(pooled_output)
            logits = self.cls(pooled_output).reshape(-1, num_choices)
        else:
            pooled_output = self.n_pooler(encoder_layer)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output).reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        else:
            return MultipleChoiceModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
