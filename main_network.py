import math
import os

from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor
from configuration import Config
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.modeling_outputs import BaseModelOutput
from sklearn.metrics.pairwise import cosine_similarity
from transformers.activations import ACT2FN
from PIL import Image
import torchvision.transforms as transforms
from unixcoder import UniXcoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        seq_len = hidden_states.shape[1]

        query_tensor = hidden_states.split([1, seq_len - 1], dim=1)[0]
        # query_tensor = hidden_states
        mixed_query_layer = self.query(query_tensor)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        context_layer = torch.cat((context_layer, hidden_states.split([1, seq_len - 1], dim=1)[1]), dim=1)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class AttentionOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention = CrossAttention(config)
        self.output = AttentionOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class Intermediate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class Output(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class Layer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # seq_len = hidden_states.shape[1]
        # t = torch.zeros(1, seq_len - 1, 768).to(device)
        # first_hidden_state = torch.cat((hidden_states[:, :1, :], t), dim=1)
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        # hidden_states = first_hidden_state
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states
        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class Pooler(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Classifier(nn.Module):
    def __init__(self, config, vit_model, vit_feature_extractor, unixcoder, output_size=(224, 224)):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.vit_model = vit_model
        self.vit_feature_extractor = vit_feature_extractor
        self.unixcoder = unixcoder
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = Pooler(config)
        self.intermediate = nn.Linear(768, 3072)
        self.dense = nn.Linear(3072, 768)
        self.act = nn.GELU()
        # self.position = nn.Parameter(torch.randn(1280, config.hidden_size))
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 使用自适应池化将特征图调整为224x224
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def get_img_output(self, image):
        img_feature = torch.tensor(image).unsqueeze(dim=0).to(device)
        img_feature = self.relu(self.conv1(img_feature))
        img_feature = self.pool(img_feature)
        img_feature = self.relu(self.conv2(img_feature))
        img_feature = self.pool(img_feature)
        img_feature = self.relu(self.conv3(img_feature))
        img_feature = self.pool(img_feature)
        # 使用自适应池化调整输出大小
        img_feature = self.adaptive_pool(img_feature)
        img_feature = self.final_conv(img_feature)
        vit_output = self.vit_model(img_feature)["last_hidden_state"].squeeze()
        vit_output = vit_output.split([1, vit_output.shape[0] - 1], dim=0)[-1]  # remove [CLS]

        return vit_output

    def get_code_embedding(self, func):
        func = torch.tensor(func).to(device)
        tokens_embeddings, func_embedding = self.unixcoder(func)
        tokens_ids = None
        source_ids = None
        tokens_embeddings = None
        return func_embedding

    def layers_output(self,
                      hidden_states: torch.Tensor,
                      head_mask: Optional[torch.Tensor] = None,
                      output_attentions: bool = True,
                      output_hidden_states: bool = False,
                      return_dict: bool = True, ):
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

                hidden_states = layer_outputs[0]

        hidden_states = layer_outputs[0]

        attention_probs = layer_outputs[1]
        length = attention_probs.shape[3]
        attention_probs = attention_probs.view(12, -1) / 12
        attention_probs = torch.sum(attention_probs, dim=0)

        hidden_states = hidden_states[:, :1, :]

        return (hidden_states, attention_probs, length)

    # def calprob(self, sequence):
    #     mean = np.mean(sequence)
    #
    #     # 放大比例（可以调整）
    #     k = 100
    #
    #     # 放大与平均值的差异
    #     amplified_sequence = mean + k * (sequence - mean)
    #
    #     # 计算原序列和放大序列的总和
    #     original_sum = np.sum(sequence)
    #     new_sum = np.sum(amplified_sequence)
    #
    #     # 重新归一化
    #     normalized_sequence = amplified_sequence * (original_sum / new_sum)
    #
    #     return normalized_sequence

    def Intermediate(self, hidden_state):
        hidden_state = self.intermediate(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.dense(hidden_state)

        return hidden_state

    def hotmap(self, pair, length, prob):
        # 示例Attention分数，假设shape为 (sequence_length, sequence_length)
        attention_scores = prob[0].view(-1, 14).cpu().detach().numpy()
        # attention_scores = self.calprob(attention_scores)
        # 使用seaborn的heatmap函数绘制热力图
        plt.figure(figsize=(length[0] / 7, 14))
        # sns.heatmap(attention_scores, annot=True, cmap='viridis')
        plt.title('Attention Heatmap')

        plt.show()
        plt.savefig("heatmap/1.png")

    def forward(self, code1, code2, label, is_train = False):
        code_rep_1, code_rep_2 = self.get_code_embedding(code1[0]), self.get_code_embedding(code2[0])
        img_rep_1, img_rep_2 = self.get_img_output(code1[1]).squeeze(), self.get_img_output(code2[1]).squeeze()
        code1, code2 = torch.cat((code_rep_1, img_rep_1), dim=0).unsqueeze(0), torch.cat((code_rep_2, img_rep_2.squeeze()),
                                                                                        dim=0).unsqueeze(0)
        output1, output2 = self.layers_output(code1), self.layers_output(code2)
        output1, output2 = self.Intermediate(output1[0]), self.Intermediate(output2[0])
        output1, output2 = self.layernorm(output1), self.layernorm(output2)
        cos_sim = self.cosine_similarity(output1, output2).squeeze(dim=0)
        if is_train:
            return cos_sim, (label-cos_sim)**2
        else:
            return cos_sim


""" code search """


def code_search(id):
    files = os.listdir("dataset/OJClone/SourceCode")
    files.sort(key=lambda x: int(x.split(".")[0]))
    id = str(id) + ".c"
    pos = 0
    for file in files:
        if file == id:
            return pos
        pos += 1

    return -1

# """ dataloader: for old version """
# code_pairs = []
# with open("dataset/pair_25000.txt", "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip().split(' ')
#         result = []
#         for l in line:
#             result.append(int(l))
#         code_pairs.append(result)
#
# X = []
# y = []
# for code1, code2, label in code_pairs:
#     X.append((code1, code2))
#     y.append(label)
# # 划分训练集和测试集
# dataset_len = len(X)
# train_rate = int(0.7*dataset_len)
# X_train, X_test, y_train, y_test = X[0: train_rate], X[train_rate:], y[0: train_rate], y[train_rate:]
