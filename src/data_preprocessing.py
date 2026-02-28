import os
import json
import glob
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import random
import numpy as np
from collections import defaultdict
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
    logging as transformers_logging,
)
from transformers.modeling_outputs import BaseModelOutput
import warnings

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

SPECIAL_SUBREDDITS = {
    "SuicideWatch",
    "selfharm",
    "depression",
    "Anxiety",
    "BPD",
    "OCD",
    "PTSD",
    "CPTSD",
    "EatingDisorders",
    "schizophrenia",
    "BipolarReddit",
    "addiction",
    "ADHD",
    "autism",
    "lonely",
}

SUPPORTIVE_SUBREDDITS = {
    "MentalHealthSupport",
    "MMFB",
    "stopdrinking",
    "selfhelp",
    "SeriousConversation",
    "offmychest",
    "nosurf",
    "mentalhealth",
    "socialanxiety",
}


@dataclass
class MentalHealthSample:
    """Represents a training sample for mental health support."""

    input_text: str
    target_text: str
    subreddit: str
    score: int
    is_post: bool
    has_crisis_indicators: bool = False
    response_type: str = "supportive"


@dataclass
class ProcessedDataset:
    """Container for processed mental health dataset."""

    samples: List[MentalHealthSample]
    subreddit_distribution: Dict[str, int]
    crisis_samples: List[MentalHealthSample]
    total_posts: int
    total_comments: int


class MentalHealthDataProcessor:
    """Processes Reddit mental health data for training."""

    CRISIS_KEYWORDS = [
        "suicide",
        "kill myself",
        "end my life",
        "want to die",
        "self-harm",
        "cutting myself",
        "hurt myself",
        "overdose",
        "hang myself",
        "slit my wrists",
        "jump off",
        "no reason to live",
        "better off dead",
        "permanent solution",
        "final way out",
        "make it stop",
        "cant go on",
        "wont wake up",
    ]

    RESPONSE_PATTERNS = {
        "empathetic": [
            "i understand",
            "i hear you",
            "that sounds really hard",
            "thank you for sharing",
            "i can imagine how difficult",
        ],
        "validation": [
            "your feelings are valid",
            "its okay to feel",
            "what you feel makes sense",
            "its not your fault",
            "you deserve to feel better",
        ],
        "supportive": [
            "you are not alone",
            "im here for you",
            "we care about you",
            "there is hope",
            "things can get better",
            "you matter",
        ],
        "resource": [
            "have you talked to",
            "have you considered",
            "professional help",
            "crisis line",
            "therapist",
            "counselor",
            "resources available",
        ],
        "gentle_challenge": [
            "i want to gently offer",
            "have you thought about",
            "im wondering if",
            "what would it be like if",
        ],
    }

    def __init__(self, data_dir: str, min_score: int = 2, max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.min_score = min_score
        self.max_length = max_length
        self.subreddit_samples = defaultdict(list)

    def load_all_data(self) -> ProcessedDataset:
        """Load and process all JSONL files from data directory."""
        all_samples = []
        subreddit_dist = defaultdict(int)
        crisis_samples = []
        total_posts = 0
        total_comments = 0

        jsonl_files = list(self.data_dir.glob("*.jsonl"))

        for filepath in jsonl_files:
            subreddit = filepath.stem.split("_")[0]
            samples = self.load_jsonl_file(filepath, subreddit)

            for sample in samples:
                all_samples.append(sample)
                subreddit_dist[sample.subreddit] += 1

                if sample.has_crisis_indicators:
                    crisis_samples.append(sample)

                if sample.is_post:
                    total_posts += 1
                else:
                    total_comments += 1

        print(f"Loaded {len(all_samples)} total samples")
        print(f"Posts: {total_posts}, Comments: {total_comments}")
        print(f"Crisis samples: {len(crisis_samples)}")

        return ProcessedDataset(
            samples=all_samples,
            subreddit_distribution=dict(subreddit_dist),
            crisis_samples=crisis_samples,
            total_posts=total_posts,
            total_comments=total_comments,
        )

    def load_jsonl_file(
        self, filepath: Path, subreddit: str
    ) -> List[MentalHealthSample]:
        """Load a single JSONL file."""
        samples = []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        sample = self.process_entry(data, subreddit)
                        if sample:
                            samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

        return samples

    def process_entry(self, data: Dict, subreddit: str) -> Optional[MentalHealthSample]:
        """Process a single JSONL entry into a training sample."""
        entry_type = data.get("type", "comment")
        text = data.get("text", data.get("title", ""))
        score = data.get("score", 0)

        if not text or len(text) < 20:
            return None

        text = self.clean_text(text)

        if len(text) > self.max_length * 4:
            text = text[: self.max_length * 4]

        has_crisis = self.check_crisis_indicators(text)

        is_post = entry_type == "post"

        response_type = self.categorize_response(text, subreddit)

        return MentalHealthSample(
            input_text=text if is_post else "",
            target_text=text,
            subreddit=subreddit,
            score=score,
            is_post=is_post,
            has_crisis_indicators=has_crisis,
            response_type=response_type,
        )

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\[deleted\]|\[removed\]", "", text)
        text = re.sub(r"\*+", "", text)
        text = re.sub(r">+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        text = self.remove_pii(text)

        return text

    def remove_pii(self, text: str) -> str:
        """Remove personally identifiable information."""
        text = re.sub(r"u/\w+", "[USER]", text)
        text = re.sub(r"r/\w+", "[SUBREDDIT]", text)

        phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        text = re.sub(phone_pattern, "[PHONE]", text)

        email_pattern = r"\b[\w.-]+@[\w.-]+\.\w+\b"
        text = re.sub(email_pattern, "[EMAIL]", text)

        return text

    def check_crisis_indicators(self, text: str) -> bool:
        """Check for crisis indicators in text."""
        text_lower = text.lower()

        for keyword in self.CRISIS_KEYWORDS:
            if keyword in text_lower:
                return True

        return False

    def categorize_response(self, text: str, subreddit: str) -> str:
        """Categorize the type of response/comment."""
        text_lower = text.lower()

        for category, patterns in self.RESPONSE_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return category

        if subreddit in SPECIAL_SUBREDDITS:
            return "crisis_support"

        return "general_support"

    def create_training_pairs(
        self,
        dataset: ProcessedDataset,
        include_posts: bool = True,
        include_comments: bool = True,
    ) -> List[Tuple[str, str, Dict]]:
        """Create input-target pairs for training."""
        pairs = []

        for sample in dataset.samples:
            if sample.is_post and not include_posts:
                continue
            if not sample.is_post and not include_comments:
                continue

            if sample.is_post:
                prompt = self.create_supportive_prompt(sample.subreddit)
                pairs.append(
                    (
                        prompt,
                        sample.target_text,
                        {
                            "subreddit": sample.subreddit,
                            "score": sample.score,
                            "has_crisis": sample.has_crisis_indicators,
                            "response_type": sample.response_type,
                        },
                    )
                )
            else:
                pairs.append(
                    (
                        "",
                        sample.target_text,
                        {
                            "subreddit": sample.subreddit,
                            "score": sample.score,
                            "has_crisis": sample.has_crisis_indicators,
                            "response_type": sample.response_type,
                        },
                    )
                )

        return pairs

    def create_supportive_prompt(self, subreddit: str) -> str:
        """Create a supportive prompt for the model."""
        prompts = {
            "depression": "Someone is sharing about their experience with depression. Offer empathetic, validating support:",
            "Anxiety": "Someone is expressing anxiety. Provide calm, understanding support:",
            "SuicideWatch": "Someone is reaching out in crisis. Offer hope and resources while validating their pain:",
            "selfharm": "Someone is struggling with self-harm urges. Show compassion and provide resources:",
            "default": "Someone is reaching out for support. Provide empathetic, helpful response:",
        }

        return prompts.get(subreddit, prompts["default"])

    def balance_dataset(
        self, pairs: List[Tuple[str, str, Dict]], crisis_ratio: float = 0.3
    ) -> List[Tuple[str, str, Dict]]:
        """Balance dataset to ensure adequate crisis samples."""
        crisis_pairs = [p for p in pairs if p[2].get("has_crisis", False)]
        non_crisis_pairs = [p for p in pairs if not p[2].get("has_crisis", True)]

        target_crisis = int(len(non_crisis_pairs) * crisis_ratio)

        if len(crisis_pairs) > target_crisis:
            crisis_pairs = random.sample(crisis_pairs, target_crisis)

        balanced = crisis_pairs + non_crisis_pairs
        random.shuffle(balanced)

        return balanced

    def get_data_statistics(self, dataset: ProcessedDataset) -> Dict:
        """Get comprehensive statistics about the dataset."""
        return {
            "total_samples": len(dataset.samples),
            "total_posts": dataset.total_posts,
            "total_comments": dataset.total_comments,
            "crisis_samples": len(dataset.crisis_samples),
            "crisis_ratio": len(dataset.crisis_samples) / len(dataset.samples)
            if dataset.samples
            else 0,
            "subreddit_distribution": dict(dataset.subreddit_distribution),
            "avg_text_length": np.mean([len(s.target_text) for s in dataset.samples]),
            "response_types": self.count_response_types(dataset),
        }

    def count_response_types(self, dataset: ProcessedDataset) -> Dict[str, int]:
        """Count different response types in dataset."""
        counts = defaultdict(int)
        for sample in dataset.samples:
            counts[sample.response_type] += 1
        return dict(counts)


class MentalHealthDataset(Dataset):
    """PyTorch Dataset for mental health training data."""

    def __init__(
        self,
        pairs: List[Tuple[str, str, Dict]],
        tokenizer,
        max_input_length: int = 256,
        max_target_length: int = 256,
        training: bool = True,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.training = training

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, target, metadata = self.pairs[idx]

        if prompt:
            input_text = f"{prompt}\n\n{target}"
        else:
            input_text = target

        source_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(0),
            "attention_mask": source_encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "metadata": metadata,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
            "metadata": [x["metadata"] for x in batch],
        }


def create_data_loaders(
    dataset: ProcessedDataset,
    processor: MentalHealthDataProcessor,
    tokenizer,
    batch_size: int = 4,
    train_ratio: float = 0.9,
    max_input_length: int = 256,
    max_target_length: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""

    pairs = processor.create_training_pairs(dataset)

    pairs = processor.balance_dataset(pairs, crisis_ratio=0.3)

    random.shuffle(pairs)

    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_dataset = MentalHealthDataset(
        train_pairs, tokenizer, max_input_length, max_target_length, training=True
    )
    val_dataset = MentalHealthDataset(
        val_pairs, tokenizer, max_input_length, max_target_length, training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=2,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    data_dir = "./data"
    processor = MentalHealthDataProcessor(data_dir, min_score=2)
    dataset = processor.load_all_data()

    stats = processor.get_data_statistics(dataset)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        if key != "subreddit_distribution":
            print(f"  {key}: {value}")

    print("\nTop Subreddits:")
    sorted_subs = sorted(
        stats["subreddit_distribution"].items(), key=lambda x: x[1], reverse=True
    )
    for sub, count in sorted_subs[:10]:
        print(f"  {sub}: {count}")
