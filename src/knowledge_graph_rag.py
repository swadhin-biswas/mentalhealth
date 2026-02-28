import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import json
from pathlib import Path


class MentalHealthKnowledgeGraph:
    """Knowledge graph for mental health concepts and resources."""

    def __init__(self):
        self.entities = {}
        self.relations = {}
        self.triplets = []

        self.entity_idx = 0
        self.relation_idx = 0

        self._build_mental_health_graph()

    def _build_mental_health_graph(self):
        """Build the mental health knowledge graph."""

        concepts = [
            ("depression", "condition", "mental_health"),
            ("anxiety", "condition", "mental_health"),
            ("PTSD", "condition", "mental_health"),
            ("ADHD", "condition", "mental_health"),
            ("BPD", "condition", "mental_health"),
            ("OCD", "condition", "mental_health"),
            ("eating_disorder", "condition", "mental_health"),
            ("schizophrenia", "condition", "mental_health"),
            ("addiction", "condition", "mental_health"),
            ("self_harm", "behavior", "concern"),
            ("suicidal_thoughts", "behavior", "concern"),
            ("isolation", "factor", "risk"),
            ("substance_use", "factor", "risk"),
            ("trauma", "factor", "risk"),
            ("therapy", "treatment", "support"),
            ("medication", "treatment", "support"),
            ("support_group", "treatment", "support"),
            ("crisis_line", "resource", "emergency"),
            ("emergency_services", "resource", "emergency"),
            ("hotline", "resource", "support"),
        ]

        for entity, rel_type, category in concepts:
            self.add_entity(entity, {"type": rel_type, "category": category})

        treatments = [
            ("cognitive_behavioral_therapy", "type_of", "therapy"),
            ("dialectical_behavior_therapy", "type_of", "therapy"),
            ("exposure_therapy", "type_of", "therapy"),
            ("medication_management", "type_of", "medication"),
            ("mindfulness", "practice", "self_help"),
            ("meditation", "practice", "self_help"),
            ("exercise", "practice", "self_help"),
            ("journaling", "practice", "self_help"),
        ]

        for entity, rel_type, target in treatments:
            self.add_entity(entity, {"type": rel_type})
            self.add_triplet(entity, rel_type, target)

        resources = [
            ("988 Suicide & Crisis Lifeline", "phone", "988"),
            ("Crisis Text Line", "text", "HOME"),
            ("National Suicide Prevention", "phone", "988"),
            ("SAMHSA National Helpline", "phone", "1-800-662-4357"),
        ]

        for entity, resource_type, contact in resources:
            self.add_entity(entity, {"type": resource_type, "contact": contact})
            self.add_triplet(entity, "provides", "support")

        coping_strategies = [
            ("reaching_out", "action", "connect_with_others"),
            ("professional_help", "action", "seek_support"),
            ("self_care", "action", "maintain_wellness"),
            ("grounding_techniques", "action", "manage_anxiety"),
            ("breathing_exercises", "action", "manage_stress"),
        ]

        for entity, action_type, benefit in coping_strategies:
            self.add_entity(entity, {"type": action_type})
            self.add_triplet(entity, "helps_with", benefit)

    def add_entity(self, entity_name: str, attributes: Optional[Dict] = None) -> int:
        """Add an entity to the graph."""
        if entity_name not in self.entities:
            self.entities[entity_name] = {
                "id": self.entity_idx,
                "attributes": attributes or {},
            }
            self.entity_idx += 1
        return self.entities[entity_name]["id"]

    def add_triplet(
        self, subject: str, relation: str, obj: str
    ) -> Tuple[int, int, int]:
        """Add a triplet to the knowledge graph."""
        sub_id = self.add_entity(subject)
        obj_id = self.add_entity(obj)

        if relation not in self.relations:
            self.relations[relation] = self.relation_idx
            self.relation_idx += 1

        rel_id = self.relations[relation]

        self.triplets.append((sub_id, rel_id, obj_id))

        return (sub_id, rel_id, obj_id)

    def get_entity_id(self, entity_name: str) -> Optional[int]:
        """Get entity ID by name."""
        return self.entities.get(entity_name, {}).get("id")

    def get_neighbors(self, entity_name: str) -> List[Tuple[str, str]]:
        """Get all neighbors of an entity."""
        entity_id = self.get_entity_id(entity_name)
        if entity_id is None:
            return []

        neighbors = []
        for sub_id, rel_id, obj_id in self.triplets:
            if sub_id == entity_id:
                for name, info in self.entities.items():
                    if info["id"] == obj_id:
                        rel_name = self._get_relation_name(rel_id)
                        neighbors.append((name, f"->{rel_name}"))
            elif obj_id == entity_id:
                for name, info in self.entities.items():
                    if info["id"] == sub_id:
                        rel_name = self._get_relation_name(rel_id)
                        neighbors.append((name, f"<-{rel_name}"))

        return neighbors

    def _get_relation_name(self, rel_id: int) -> str:
        """Get relation name from ID."""
        for name, idx in self.relations.items():
            if idx == rel_id:
                return name
        return "unknown"

    def get_support_resources(self) -> List[Dict]:
        """Get all support resources."""
        resources = []
        for entity_name, info in self.entities.items():
            if info["attributes"].get("type") in ["phone", "text"]:
                resources.append(
                    {
                        "name": entity_name,
                        "contact": info["attributes"].get("contact", "N/A"),
                        "type": info["attributes"].get("type"),
                    }
                )
        return resources

    def get_treatments_for(self, condition: str) -> List[str]:
        """Get treatments for a specific condition."""
        condition_id = self.get_entity_id(condition)
        if condition_id is None:
            return []

        treatments = []
        for sub_id, rel_id, obj_id in self.triplets:
            if sub_id == condition_id:
                for name, info in self.entities.items():
                    if info["id"] == obj_id:
                        treatments.append(name)

        return treatments


class SemanticRetriever:
    """Semantic retrieval for relevant context."""

    def __init__(
        self, embedding_dim: int = 768, num_indices: int = 10000, device: str = "cuda"
    ):
        self.embedding_dim = embedding_dim
        self.device = device

        self.context_embeddings = nn.Embedding(num_indices, embedding_dim)
        self.context_texts = {}
        self.context_metadata = {}

        self.index = None

    def add_contexts(self, contexts: List[Dict[str, Any]]):
        """Add contexts to the retrieval index."""
        for i, ctx in enumerate(contexts):
            if i >= self.context_embeddings.num_embeddings:
                break

            self.context_texts[i] = ctx.get("text", "")
            self.context_metadata[i] = {
                "subreddit": ctx.get("subreddit", "unknown"),
                "type": ctx.get("type", "comment"),
                "score": ctx.get("score", 0),
                "has_crisis": ctx.get("has_crisis", False),
            }

    def retrieve(
        self, query_embedding: torch.Tensor, top_k: int = 5, filter_crisis: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant contexts."""

        all_embeddings = self.context_embeddings.weight[: len(self.context_texts)]

        similarities = torch.matmul(
            query_embedding.unsqueeze(0), all_embeddings.T.unsqueeze(0)
        ).squeeze()

        top_scores, top_indices = torch.topk(
            similarities, min(top_k, len(self.context_texts))
        )

        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            ctx = {
                "text": self.context_texts.get(idx, ""),
                "metadata": self.context_metadata.get(idx, {}),
                "relevance_score": score.item(),
            }

            if filter_crisis and ctx["metadata"].get("has_crisis", False):
                continue

            results.append(ctx)

        return results

    def build_index(self, model, tokenizer):
        """Build retrieval index using the model."""
        if not self.context_texts:
            return

        texts = list(self.context_texts.values())

        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**{k: v for k, v in encoded.items()})
            embeddings = outputs.last_hidden_state[:, 0, :]

        for i in range(min(len(embeddings), self.context_embeddings.num_embeddings)):
            self.context_embeddings.weight[i] = embeddings[i]


class RAGSystem:
    """Retrieval-Augmented Generation system for mental health support."""

    def __init__(
        self,
        knowledge_graph: MentalHealthKnowledgeGraph,
        retriever: SemanticRetriever,
        device: str = "cuda",
    ):
        self.kg = knowledge_graph
        self.retriever = retriever
        self.device = device

        self.response_templates = self._load_response_templates()

    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates for different scenarios."""
        return {
            "crisis": [
                "I hear that you're going through an incredibly difficult time. Your life has real value and meaning.",
                "I'm concerned about you right now. Would you like to talk about what's happening?",
                "You don't have to face this alone. There are people who want to help you get through this.",
            ],
            "empathy": [
                "Thank you for sharing this with me. What you're experiencing sounds really challenging.",
                "I can only imagine how difficult this must be for you.",
                "Your feelings are completely valid. It makes sense that you're feeling this way given what you've been through.",
            ],
            "support": [
                "You don't have to go through this alone. There are people who care about you.",
                "I'm here with you in this moment. You're not alone.",
                "It takes courage to reach out. I'm glad you did.",
            ],
            "resources": [
                "If you're in immediate danger, please call 988 (Suicide & Crisis Lifeline).",
                "Would you like me to share some resources that might help?",
                "Talking to a mental health professional can really help. Would you like information about how to access support?",
            ],
            "coping": [
                "Have you tried any grounding techniques? Sometimes focusing on your senses can help.",
                "Taking things one moment at a time can make challenges feel more manageable.",
                "Self-care is important. Have you been able to do any things that help you feel better?",
            ],
        }

    def get_retrieved_context(
        self, query_embedding: torch.Tensor, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query."""
        return self.retriever.retrieve(query_embedding, top_k=top_k)

    def get_related_concepts(self, query: str) -> Dict[str, List[str]]:
        """Get related concepts from knowledge graph."""
        query_lower = query.lower()

        related = {"conditions": [], "treatments": [], "resources": []}

        condition_keywords = {
            "depression": "depression",
            "anxious": "anxiety",
            "stress": "anxiety",
            "trauma": "PTSD",
            "flashback": "PTSD",
            "mood": "BPD",
            "impulsive": "BPD",
            "repetitive": "OCD",
            "obsess": "OCD",
            "food": "eating_disorder",
            "weight": "eating_disorder",
            "hearing": "schizophrenia",
            "voices": "schizophrenia",
            "substance": "addiction",
            "drink": "addiction",
            "cut": "self_harm",
            "hurt": "self_harm",
        }

        for keyword, condition in condition_keywords.items():
            if keyword in query_lower:
                related["conditions"].append(condition)
                treatments = self.kg.get_treatments_for(condition)
                related["treatments"].extend(treatments)

        return related

    def generate_supportive_response(
        self,
        user_message: str,
        context: Optional[torch.Tensor] = None,
        response_type: str = "empathy",
    ) -> Dict[str, Any]:
        """Generate a supportive response based on context."""

        related = self.get_related_concepts(user_message)

        crisis_indicators = self._check_crisis_indicators(user_message)

        response_parts = []

        if crisis_indicators["is_crisis"]:
            response_parts.extend(self.response_templates["crisis"])
            response_parts.extend(self.response_templates["resources"])
        else:
            response_parts.extend(self.response_templates[response_type])

        if related["treatments"]:
            response_parts.append(
                f"Some approaches that might help include: {', '.join(related['treatments'][:3])}"
            )

        resources = self.kg.get_support_resources()

        return {
            "response_parts": response_parts,
            "related_concepts": related,
            "crisis_indicators": crisis_indicators,
            "resources": resources[:3],
            "response_type": "crisis"
            if crisis_indicators["is_crisis"]
            else response_type,
        }

    def _check_crisis_indicators(self, text: str) -> Dict[str, bool]:
        """Check for crisis indicators in text."""
        text_lower = text.lower()

        crisis_keywords = [
            "suicide",
            "kill myself",
            "end my life",
            "want to die",
            "self harm",
            "hurt myself",
            "overdose",
            "no reason to live",
            "better off dead",
            "permanent solution",
            "final way",
        ]

        immediate_crisis = any(kw in text_lower for kw in crisis_keywords)

        warning_keywords = [
            "hopeless",
            "worthless",
            "burden",
            "empty",
            "numb",
            "alone",
            "nobody cares",
            "better without me",
        ]

        warning_signs = any(kw in text_lower for kw in warning_keywords)

        return {
            "is_crisis": immediate_crisis,
            "warning_signs": warning_signs,
            "immediate_need": immediate_crisis,
        }

    def format_response(
        self, response_data: Dict[str, Any], include_resources: bool = True
    ) -> str:
        """Format the response data into a readable response."""

        parts = response_data["response_parts"]

        if include_resources and response_data.get("resources"):
            resources_text = "\n\nIf you need immediate support:\n"
            for r in response_data["resources"]:
                resources_text += f"- {r['name']}: {r['contact']}\n"
            parts.append(resources_text)

        return "\n\n".join(parts)


class ContextProcessor:
    """Process and manage conversation context."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history = []

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation history."""
        message = {"role": role, "content": content, "metadata": metadata or {}}

        self.conversation_history.append(message)

        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_context_string(self) -> str:
        """Get the conversation history as a string."""
        context_parts = []

        for msg in self.conversation_history:
            role = msg["role"]
            content = msg["content"]
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_last_k_messages(self, k: int) -> List[Dict]:
        """Get the last k messages."""
        return self.conversation_history[-k:]


class MentalHealthRAG:
    """Complete RAG system for mental health support."""

    def __init__(
        self,
        model,
        knowledge_graph: MentalHealthKnowledgeGraph,
        retriever: SemanticRetriever,
        device: str = "cuda",
    ):
        self.model = model
        self.kg = knowledge_graph
        self.rag = RAGSystem(knowledge_graph, retriever, device)
        self.context = ContextProcessor()
        self.device = device

    def process_user_message(
        self, user_message: str, return_context: bool = False
    ) -> Dict[str, Any]:
        """Process a user message and generate a response."""

        self.context.add_message("user", user_message)

        response_data = self.rag.generate_supportive_response(user_message)

        self.context.add_message(
            "assistant",
            "\n".join(response_data["response_parts"]),
            {"response_type": response_data["response_type"]},
        )

        if return_context:
            return {
                "response": response_data,
                "conversation_history": self.context.get_last_k_messages(5),
            }

        return response_data

    def format_final_response(self, response_data: Dict[str, Any]) -> str:
        """Format the final response for the user."""
        return self.rag.format_response(response_data)


def initialize_rag_system(model, device: str = "cuda") -> MentalHealthRAG:
    """Initialize the complete RAG system."""

    kg = MentalHealthKnowledgeGraph()

    retriever = SemanticRetriever(embedding_dim=768, num_indices=10000, device=device)

    rag_system = MentalHealthRAG(
        model=model, knowledge_graph=kg, retriever=retriever, device=device
    )

    return rag_system


if __name__ == "__main__":
    kg = MentalHealthKnowledgeGraph()

    print("Mental Health Knowledge Graph:")
    print(f"  Entities: {len(kg.entities)}")
    print(f"  Relations: {len(kg.relations)}")
    print(f"  Triplets: {len(kg.triplets)}")

    print("\nSupport Resources:")
    resources = kg.get_support_resources()
    for r in resources:
        print(f"  - {r['name']}: {r['contact']}")

    print("\nTreatments for depression:")
    treatments = kg.get_treatments_for("depression")
    for t in treatments:
        print(f"  - {t}")
