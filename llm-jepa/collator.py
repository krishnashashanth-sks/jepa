import torch
from dataclasses import dataclass
from typing import Any, Dict, List,  Tuple

@dataclass
class DataCollatorForLLMJEPA:
    tokenizer: Any
    mlm_probability: float = 0.15  # Probability of masking a token
    mask_token_id: int = None # Token ID for masking
    pad_token_id: int = None # Token ID for padding

    def __post_init__(self):
        if self.mask_token_id is None:
            self.mask_token_id = self.tokenizer.mask_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id
        if self.mask_token_id is None:
            raise ValueError("Tokenizer does not have a mask_token_id. Please provide one or use a suitable tokenizer.")
        if self.pad_token_id is None:
            raise ValueError("Tokenizer does not have a pad_token_id. Please provide one or use a suitable tokenizer.")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        # Create the uncorrupted target input
        target_input_ids = batch["input_ids"].clone()
        target_attention_mask = batch["attention_mask"].clone()

        # Create the masked encoder input
        encoder_input_ids, labels = self.torch_mask_tokens(batch["input_ids"].clone())
        encoder_attention_mask = batch["attention_mask"].clone()

        return {
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "target_input_ids": target_input_ids,
            "target_attention_mask": target_attention_mask,
            # 'labels' are needed for traditional MLM, but for JEPA, the target_input_ids are the 'labels' for prediction
            # We include it here for compatibility or if we want to combine with MLM loss.
            "labels": labels # Masked positions' original tokens
        }

    def torch_mask_tokens(self, inputs: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True
            ) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.mask_token is not None:
            padding_mask = labels.eq(self.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The remaining 10% of the time (100 - 80 - 10 = 10%), we keep the masked input tokens unchanged
        return inputs, labels