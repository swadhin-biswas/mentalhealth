import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import math
import copy

from transformers import AutoModel, AutoConfig


class MultiHeadAttention(nn.Module):
    """Multi-head attention with proper causal masking."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        q = (
            self.q_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if causal:
            seq_len = query.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        )

        output = self.out_proj(context)

        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)

        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention_output, _ = self.attention(x, x, x, mask)
        x = self.attention_norm(x + attention_output)

        ff_output = self.feed_forward(x)
        x = self.ffn_norm(x + ff_output)

        return x


class HierarchicalAttention(nn.Module):
    """Hierarchical attention for processing different levels of context."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size, num_heads, hidden_size * 4, dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return self.output_norm(hidden_states)


class MultiEncoder(nn.Module):
    """Multi-encoder system combining different pretrained embeddings."""

    def __init__(
        self,
        encoder_config: Dict[str, Any],
        hidden_size: int = 768,
        pooling_strategy: str = "weighted",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy

        self.encoders = nn.ModuleDict()
        self.projection_layers = nn.ModuleDict()

        for name, config in encoder_config.items():
            encoder = AutoModel.from_pretrained(config["model_name"])
            encoder_dim = encoder.config.hidden_size

            self.encoders[name] = encoder

            if encoder_dim != hidden_size:
                self.projection_layers[name] = nn.Linear(encoder_dim, hidden_size)
            else:
                self.projection_layers[name] = None

        if pooling_strategy == "weighted":
            self.attention_weights = nn.Linear(hidden_size, len(encoder_config))

    def get_encoder_output(
        self, encoder_name: str, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get output from a specific encoder."""
        encoder = self.encoders[encoder_name]

        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state

        projection = self.projection_layers[encoder_name]
        if projection is not None:
            hidden_states = projection(hidden_states)

        return hidden_states

    def forward(
        self, inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process inputs through multiple encoders.
        inputs: Dict[encoder_name -> (input_ids, attention_mask)]
        """
        encoder_outputs = {}

        for name, (input_ids, attention_mask) in inputs.items():
            encoder_outputs[name] = self.get_encoder_output(
                name, input_ids, attention_mask
            )

        return encoder_outputs

    def get_pooled_output(
        self, encoder_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Pool outputs from multiple encoders."""

        if self.pooling_strategy == "concat":
            outputs_list = list(encoder_outputs.values())
            return torch.cat(outputs_list, dim=-1)

        elif self.pooling_strategy == "weighted":
            hidden_states = list(encoder_outputs.values())
            stacked = torch.stack(hidden_states, dim=1)

            pooled = hidden_states[0].mean(dim=1)
            weights = F.softmax(self.attention_weights(pooled).mean(dim=0), dim=-1)

            weighted = (stacked * weights.view(1, -1, 1, 1)).sum(dim=1)
            return weighted

        elif self.pooling_strategy == "mean":
            outputs_list = list(encoder_outputs.values())
            return torch.mean(torch.stack(outputs_list), dim=0)

        else:
            return list(encoder_outputs.values())[0]


class RetrievalAugmentedEncoder(nn.Module):
    """Retrieval-Augmented Generation encoder with knowledge integration."""

    def __init__(
        self,
        base_encoder: nn.Module,
        hidden_size: int,
        num_retrieved: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.hidden_size = hidden_size
        self.num_retrieved = num_retrieved

        self.retrieval_projection = nn.Linear(hidden_size, hidden_size)
        self.context_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion_layer = TransformerEncoderLayer(
            hidden_size, 4, hidden_size * 4, dropout
        )

        self.retrieval_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.retrieval_beta = nn.Sigmoid()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        retrieved_context: Optional[torch.Tensor] = None,
        retrieved_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with retrieval augmentation."""

        base_output = self.base_encoder(input_ids, attention_mask)
        query_hidden = base_output

        if retrieved_context is not None:
            retrieved_projected = self.retrieval_projection(retrieved_context)

            combined = torch.cat([query_hidden, retrieved_projected], dim=-1)
            fused = self.context_projection(combined)

            gate = self.retrieval_beta(self.retrieval_gate(combined))
            fused = gate * fused + (1 - gate) * query_hidden

            fused = self.fusion_layer(fused, retrieved_mask)

            return fused

        return query_hidden


class KnowledgeGraphIntegration(nn.Module):
    """Knowledge Graph integration for mental health context."""

    def __init__(
        self,
        hidden_size: int,
        num_relations: int = 10,
        num_entities: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.entity_embeddings = nn.Embedding(num_entities, hidden_size)
        self.relation_embeddings = nn.Embedding(num_relations, hidden_size)

        self.entity_norm = nn.LayerNorm(hidden_size)
        self.relation_norm = nn.LayerNorm(hidden_size)

        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        knowledge_indices: Optional[torch.Tensor] = None,
        relation_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate knowledge graph information."""

        if knowledge_indices is None:
            return hidden_states

        batch_size = hidden_states.size(0)

        entity_embeds = self.entity_embeddings(knowledge_indices)
        entity_embeds = self.entity_norm(entity_embeds)

        query = self.query_projection(hidden_states)
        key = self.key_projection(entity_embeds)
        value = self.value_projection(entity_embeds)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.hidden_size
        )
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value)

        combined = torch.cat([hidden_states, context], dim=-1)
        output = self.output_projection(combined)

        return output


class MentalHealthEncoder(nn.Module):
    """Main encoder combining all components for mental health support."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 768)

        encoder_config = config.get(
            "encoder_config",
            {
                "clinical_bert": {
                    "model_name": "./models/biobert-base-cased-v1.1",
                },
                "general": {"model_name": "./models/all-MiniLM-L6-v2"},
            },
        )

        self.multi_encoder = MultiEncoder(
            encoder_config,
            hidden_size=self.hidden_size,
            pooling_strategy=config.get("pooling_strategy", "weighted"),
        )

        self.hierarchical_attention = HierarchicalAttention(
            self.hidden_size,
            num_layers=config.get("num_attention_layers", 2),
            num_heads=config.get("num_attention_heads", 8),
            dropout=config.get("dropout", 0.1),
        )

        self.knowledge_integration = KnowledgeGraphIntegration(
            self.hidden_size,
            num_relations=config.get("num_relations", 10),
            num_entities=config.get("num_entities", 1000),
            dropout=config.get("dropout", 0.1),
        )

        self.num_retrieved = config.get("num_retrieved", 5)

    def forward(
        self,
        inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        knowledge_indices: Optional[torch.Tensor] = None,
        relation_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the full encoder."""

        encoder_outputs = self.multi_encoder(inputs)

        pooled_output = self.multi_encoder.get_pooled_output(encoder_outputs)

        if pooled_output.dim() == 2:
            hidden_states = pooled_output.unsqueeze(1)
            attended = self.hierarchical_attention(hidden_states)
            integrated = self.knowledge_integration(
                attended, knowledge_indices, relation_indices
            )
            return integrated.squeeze(1)
        elif pooled_output.dim() == 3:
            attended = self.hierarchical_attention(pooled_output)
            integrated = self.knowledge_integration(
                attended, knowledge_indices, relation_indices
            )
            if integrated.dim() == 3:
                return integrated.mean(dim=1)
            return integrated
        else:
            integrated = self.knowledge_integration(
                pooled_output, knowledge_indices, relation_indices
            )
            return integrated

    def get_encoder_names(self) -> List[str]:
        """Get list of available encoder names."""
        return list(self.multi_encoder.encoders.keys())


class CrisisDetectionHead(nn.Module):
    """Safety head for detecting crisis situations."""

    def __init__(self, hidden_size: int, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()

        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense2 = nn.Linear(hidden_size // 2, hidden_size // 4)

        self.classifier = nn.Linear(hidden_size // 4, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect crisis level from hidden states."""

        x = self.dense1(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.activation(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        probs = F.softmax(logits, dim=-1)

        crisis_level = torch.argmax(probs, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "crisis_level": crisis_level,
            "is_crisis": crisis_level > 1,
        }


class ResponseQualityHead(nn.Module):
    """Head for evaluating response quality."""

    def __init__(self, hidden_size: int, num_metrics: int = 5, dropout: float = 0.2):
        super().__init__()

        self.metrics = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_metrics)
            ]
        )

    def forward(
        self, hidden_states: torch.Tensor, return_details: bool = False
    ) -> torch.Tensor:
        """Evaluate response quality metrics."""

        metric_scores = []
        for metric_head in self.metrics:
            score = metric_head(hidden_states)
            metric_scores.append(score)

        scores = torch.cat(metric_scores, dim=-1)

        if return_details:
            return scores
        return scores.mean(dim=-1, keepdim=True)


class SafetyVerifier(nn.Module):
    """Comprehensive safety verification system."""

    def __init__(
        self,
        hidden_size: int,
        num_crisis_classes: int = 4,
        num_safety_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.crisis_detector = CrisisDetectionHead(
            hidden_size, num_crisis_classes, dropout
        )

        self.harm_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_safety_classes),
        )

        self.resource_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        self.crisis_keywords = [
            "suicide",
            "kill myself",
            "end my life",
            "self harm",
            "hurt myself",
            "overdose",
            "want to die",
            "no reason to live",
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_text: Optional[str] = None,
        return_full: bool = False,
    ) -> Dict[str, Any]:
        """Verify safety of input and generate appropriate responses."""

        crisis_output = self.crisis_detector(hidden_states)

        harm_logits = self.harm_classifier(hidden_states)
        harm_probs = F.softmax(harm_logits, dim=-1)

        resource_need = self.resource_predictor(hidden_states)

        keyword_crisis = False
        if input_text:
            text_lower = input_text.lower()
            for keyword in self.crisis_keywords:
                if keyword in text_lower:
                    keyword_crisis = True
                    break

        needs_crisis_response = (
            crisis_output["is_crisis"].any() or keyword_crisis or resource_need > 0.7
        )

        safety_level = torch.argmax(harm_probs, dim=-1)

        result = {
            "crisis_level": crisis_output["crisis_level"],
            "crisis_probs": crisis_output["probs"],
            "is_crisis": crisis_output["is_crisis"],
            "safety_level": safety_level,
            "harm_probs": harm_probs,
            "resource_need": resource_need,
            "needs_crisis_response": needs_crisis_response,
            "keyword_crisis": keyword_crisis,
        }

        if return_full:
            result["crisis_output"] = crisis_output

        return result

    def get_response_type(self, safety_output: Dict[str, Any]) -> str:
        """Determine appropriate response type based on safety analysis."""

        if safety_output["keyword_crisis"] or safety_output["crisis_level"].max() > 2:
            return "crisis"
        elif safety_output["crisis_level"].max() > 1:
            return "elevated"
        elif safety_output["resource_need"] > 0.5:
            return "resource"
        else:
            return "standard"


class MentalHealthModel(nn.Module):
    """Complete mental health support model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 768)

        self.encoder = MentalHealthEncoder(config)

        self.safety_verifier = SafetyVerifier(
            self.hidden_size,
            num_crisis_classes=config.get("num_crisis_classes", 4),
            num_safety_classes=config.get("num_safety_classes", 3),
            dropout=config.get("dropout", 0.3),
        )

        self.response_quality = ResponseQualityHead(
            self.hidden_size,
            num_metrics=config.get("num_metrics", 5),
            dropout=config.get("dropout", 0.2),
        )

    def forward(
        self,
        inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        return_safety: bool = True,
        return_quality: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass through the full model."""

        hidden_states = self.encoder(inputs)

        outputs = {"hidden_states": hidden_states}

        if return_safety:
            safety_output = self.safety_verifier(hidden_states)
            outputs["safety"] = safety_output

        if return_quality:
            quality_output = self.response_quality(hidden_states)
            outputs["quality"] = quality_output

        return outputs

    def encode(
        self, inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Encode inputs without safety verification."""
        return self.encoder(inputs)

    def verify_safety(
        self, hidden_states: torch.Tensor, input_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify safety of encoded states."""
        return self.safety_verifier(hidden_states, input_text)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config


def create_model(
    model_name: str = "mental_health_support",
    hidden_size: int = 768,
    num_attention_heads: int = 8,
    num_attention_layers: int = 2,
    dropout: float = 0.1,
    use_clinical_bert: bool = True,
    use_knowledge_graph: bool = True,
) -> MentalHealthModel:
    """Factory function to create a mental health support model."""

    encoder_config = {"general": {"model_name": "./models/all-MiniLM-L6-v2"}}

    if use_clinical_bert:
        encoder_config["clinical_bert"] = {
            "model_name": "./models/biobert-base-cased-v1.1"
        }

    config = {
        "hidden_size": hidden_size,
        "encoder_config": encoder_config,
        "num_attention_heads": num_attention_heads,
        "num_attention_layers": num_attention_layers,
        "dropout": dropout,
        "pooling_strategy": "weighted",
        "num_retrieved": 5,
        "num_crisis_classes": 4,
        "num_safety_classes": 3,
        "num_metrics": 5,
        "num_relations": 10,
        "num_entities": 1000,
    }

    return MentalHealthModel(config)


class ModelOutput:
    """Container for model outputs."""

    def __init__(
        self,
        hidden_states: torch.Tensor,
        safety: Optional[Dict[str, Any]] = None,
        quality: Optional[torch.Tensor] = None,
    ):
        self.hidden_states = hidden_states
        self.safety = safety
        self.quality = quality

    def to_dict(self) -> Dict[str, Any]:
        result = {"hidden_states": self.hidden_states}
        if self.safety:
            result["safety"] = self.safety
        if self.quality is not None:
            result["quality"] = self.quality
        return result
