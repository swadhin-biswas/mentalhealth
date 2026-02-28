import torch
import argparse
from typing import Dict, Any, Optional
import logging
import json

from model_architecture import create_model
from data_preprocessing import MentalHealthDataProcessor
from knowledge_graph_rag import (
    MentalHealthKnowledgeGraph,
    SemanticRetriever,
    RAGSystem,
    MentalHealthRAG,
    ContextProcessor,
)
from evaluation_safety import (
    CrisisDetector,
    ResponseEvaluator,
    SafetyMonitor,
    ResponseSafetyFilter,
    create_evaluation_system,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MentalHealthAssistant:
    """Main interface for mental health support assistant."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        logger.info(f"Initializing MentalHealthAssistant on {device}")

        logger.info("Creating model...")
        self.model = self._create_model()

        logger.info("Initializing knowledge graph...")
        self.knowledge_graph = MentalHealthKnowledgeGraph()

        logger.info("Initializing retriever...")
        self.retriever = SemanticRetriever(device=device)

        logger.info("Initializing RAG system...")
        self.rag = RAGSystem(self.knowledge_graph, self.retriever, device)

        logger.info("Initializing safety systems...")
        self.crisis_detector = CrisisDetector()
        self.response_evaluator = ResponseEvaluator()
        self.safety_monitor = SafetyMonitor()
        self.safety_filter = ResponseSafetyFilter()

        logger.info("Initializing context processor...")
        self.context = ContextProcessor()

    def _create_model(self):
        """Create the mental health model."""
        config = {
            "hidden_size": 768,
            "num_attention_heads": 8,
            "num_attention_layers": 2,
            "dropout": 0.1,
            "use_clinical_bert": True,
            "use_knowledge_graph": True,
        }
        return create_model(**config)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def generate_response(
        self, user_message: str, temperature: float = 0.7, max_length: int = 256
    ) -> Dict[str, Any]:
        """Generate a supportive response."""

        self.context.add_message("user", user_message)

        crisis_analysis = self.crisis_detector.analyze_text(
            user_message, return_details=True
        )

        recommended = self.crisis_detector.get_recommended_response(crisis_analysis)

        response_data = self.rag.generate_supportive_response(
            user_message, response_type=recommended["response_type"]
        )

        response = self.rag.format_response(response_data)

        response_evaluation = self.response_evaluator.evaluate_response(
            user_message, response
        )

        filtered_response, was_filtered = self.safety_filter.filter_response(
            response, crisis_analysis["crisis_level"], response_evaluation
        )

        if was_filtered:
            self.safety_monitor.log_safety_intervention(
                user_message, response, "content_filter"
            )

        if crisis_analysis["crisis_level"] >= 2:
            self.safety_monitor.log_crisis_detection(
                user_message, crisis_analysis["crisis_level"], filtered_response
            )

        self.context.add_message("assistant", filtered_response)

        return {
            "response": filtered_response,
            "crisis_analysis": crisis_analysis,
            "response_evaluation": response_evaluation,
            "was_filtered": was_filtered,
            "recommended_response_type": recommended["response_type"],
        }

    def get_crisis_resources(self) -> list:
        """Get crisis resources."""
        return self.knowledge_graph.get_support_resources()

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return self.safety_monitor.get_session_summary()

    def reset_context(self):
        """Reset conversation context."""
        self.context.clear_history()


def interactive_mode():
    """Run the assistant in interactive mode."""

    print("=" * 60)
    print("Mental Health Support Assistant")
    print("=" * 60)
    print("\nThis is a supportive AI assistant. Type 'quit' to exit.")
    print("Type 'resources' to see crisis resources.")
    print("Type 'stats' to see session statistics.")
    print("Type 'reset' to reset conversation history.")
    print("=" * 60)

    assistant = MentalHealthAssistant()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nThank you for using Mental Health Support Assistant.")
                print(
                    "If you're in crisis, please reach out to 988 (Suicide & Crisis Lifeline)."
                )
                break

            if user_input.lower() == "resources":
                resources = assistant.get_crisis_resources()
                print("\nCrisis Resources:")
                for r in resources:
                    print(f"  - {r['name']}: {r['contact']}")
                continue

            if user_input.lower() == "stats":
                stats = assistant.get_session_stats()
                print("\nSession Statistics:")
                print(json.dumps(stats, indent=2))
                continue

            if user_input.lower() == "reset":
                assistant.reset_context()
                print("\nConversation history reset.")
                continue

            result = assistant.generate_response(user_input)

            print(f"\nAssistant: {result['response']}")

            if result["crisis_analysis"]["crisis_level"] >= 2:
                print("\n[System: Crisis indicators detected - resources provided]")

        except KeyboardInterrupt:
            print(
                "\n\nSession ended. Remember: You are not alone. Reach out for support."
            )
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print("\nI'm here to help. Please try again.")


def evaluate_mode(test_data_path: str):
    """Run evaluation on test data."""
    from evaluation_safety import create_evaluation_system

    logger.info("Initializing evaluation system...")
    assistant = MentalHealthAssistant()
    evaluator, monitor = create_evaluation_system(assistant.model, assistant.device)

    logger.info(f"Loading test data from {test_data_path}")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    logger.info(f"Evaluating {len(test_data)} samples...")
    results = evaluator.evaluate_batch(test_data)

    logger.info("\nEvaluation Results:")
    logger.info(json.dumps(results["summary"], indent=2))

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Mental Health Support Assistant")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "evaluate"],
        default="interactive",
        help="Mode to run the assistant",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-data", type=str, default=None, help="Path to test data for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        interactive_mode()
    elif args.mode == "evaluate":
        if not args.test_data:
            parser.error("--test-data is required for evaluate mode")
        evaluate_mode(args.test_data)


if __name__ == "__main__":
    main()
