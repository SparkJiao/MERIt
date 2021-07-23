from abc import ABC

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaConfig, RobertaLMHead

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules import layers

logger = get_child_logger("RoBERTa.Sent")


class RobertaSentForMultipleChoice(RobertaPreTrainedModel, LogMixin, ABC):
    """
    - Add supervision to attention scores.
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig, sup_start_layer_id: int = 10):
        super().__init__(config)

        self.sup_start_layer_id = sup_start_layer_id

        self.roberta = RobertaModel(config)

        self.s_sum_w = nn.Linear(config.hidden_size, 1)
        self.q_sum_w = nn.Linear(config.hidden_size, 1)

        self.s_att_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.s_att_k = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

        self.init_metric("loss", "acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        shape = x.size()[2:]
        return x.reshape(-1, *shape)

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            sentence_index: Tensor = None,
            sentence_mask: Tensor = None,
            sent_token_mask: Tensor = None,
            path: Tensor = None,
            path_mask: Tensor = None,
            rev_path: Tensor = None,
            rev_path_mask: Tensor = None,
            q_op_sent_index: Tensor = None,
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
        batch, num_choices, seq_len = input_ids.size()

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)
        # [batch * num_choices, max_sent_num, max_sent_len]
        sentence_index = self.fold_tensor(sentence_index)
        # [batch * num_choices, 2]
        q_op_sent_index = self.fold_tensor(q_op_sent_index)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        seq_hidden = outputs.sequence_output
        h = seq_hidden.size(-1)
        ex_batch, max_sent_num, max_sent_len = sentence_index.size()

        sent_token_hidden = torch.gather(seq_hidden, dim=1,
                                         index=sentence_index.unsqueeze(-1).expand(
                                             -1, -1, -1, h).reshape(ex_batch, max_sent_num * max_sent_len, h))
        sent_token_hidden = sent_token_hidden.reshape(ex_batch, max_sent_num, max_sent_len, h)
        sent_scores = self.s_sum_w(sent_token_hidden).squeeze(-1)
        sent_alpha = (sent_scores + (1 - sent_token_mask.to(seq_hidden.dtype)) * -10000.0).softmax(dim=-1)
        sent_hidden = torch.einsum("est,esth->esh", sent_alpha, sent_token_hidden)

        q_op_token_hidden = torch.gather(sent_token_hidden, dim=1,
                                         index=q_op_sent_index.unsqueeze(-1).expand(
                                             -1, -1, h).reshape(ex_batch, 2 * max_sent_len, h))
        q_op_token_mask = torch.gather(sent_token_mask, dim=1,
                                       index=q_op_sent_index.)
        q_op_sent_scores = self.q_sum_w(q_op_token_hidden).squeeze(-1)

        q_op_sent_hidden = torch.einsum("e")

        _att_q = self.s_att_q(self.dropout(q_op_sent_hidden))
        _att_kv = self.s_att_k(self.dropout(sent_hidden))

        beta = torch.einsum("eah,ebh->eab", _att_q, _att_kv).softmax(dim=-1)
        sum_hidden = torch.einsum("eab,ebh->eah")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        choice_mask = (attention_mask.sum(dim=-1) == 0).reshape(-1, num_choices)
        reshaped_logits = reshaped_logits + choice_mask * -10000.0

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(reshaped_logits, labels)

            # attention scores supervision
            sup_att_scores = torch.stack(attentions[self.sup_start_layer_id:], dim=0)
            ex_batch = input_ids.size(0)
            assert sup_att_scores.size(0) == (self.config.num_hidden_layers - self.sup_start_layer_id), sup_att_scores.size()
            # Only supervise the probabilities of [CLS] since we use its representation for classification.
            sup_att_scores = sup_att_scores[:, :, :, 0]
            assert len(sup_att_scores.size()) == 4
            # Average pooling over each layer.
            sup_att_scores = sup_att_scores.mean(dim=0)
            assert len(sup_att_scores.size()) == 3
            # Average pooling over each head.
            sup_att_scores = sup_att_scores.mean(dim=1)
            assert len(sup_att_scores.size()) == 2

            # Only supervise the correct option's path.
            masked_labels = labels.clone()
            masked_labels = masked_labels.masked_fill(labels == -1, 0).unsqueeze(1)  # [batch, 1]
            sup_att_scores = sup_att_scores.reshape(batch, num_choices, seq_len)
            sup_att_scores = torch.gather(sup_att_scores, dim=1, index=masked_labels.unsqueeze(-1).expand(-1, -1, seq_len)).squeeze(1)

            index_labels = masked_labels[:, :, None, None].expand(-1, -1, sentence_index.size(2), sentence_index.size(3))
            sentence_index = torch.gather(sentence_index, dim=1, index=index_labels).squeeze(1)
            sent_token_mask = torch.gather(sent_token_mask, dim=1, index=index_labels).squeeze(1)

            path_labels = masked_labels.unsqueeze(-1).expand(-1, -1, rev_path.size(-1))
            rev_path = torch.gather(rev_path, dim=1, index=path_labels).squeeze(1)
            rev_path_mask = torch.gather(rev_path_mask, dim=1, index=path_labels).squeeze(1)

            gathered_scores, gathered_mask = layers.extract_sent_tokens(sup_att_scores, sentence_index, sent_token_mask,
                                                                        rev_path, rev_path_mask)
            # Process mask. Reset as 0.
            gathered_scores = gathered_scores.masked_fill(~gathered_mask, 0)
            # if torch.any(torch.isnan(gathered_scores)) or torch.any(torch.isinf(gathered_scores)):
            #     print(gathered_scores)
            true_score_num = gathered_mask.sum().item()
            if true_score_num > 0:
                sup_loss = (-((1 - gathered_scores + 1e-7).log())).sum() / true_score_num
            # sup_loss = (-((1 - gathered_scores + 1e-7).log())).sum()
                loss += sup_loss
            # sup_loss = sup_loss.item()
            # else:
            #     sup_loss = 0.

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                # self.eval_metrics.update("att_loss", val=sup_loss, n=true_score_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
