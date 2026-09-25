from trl import SFTTrainer, SFTConfig
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class MinistralTrainer:
    def __init__(self, model: Any, tokenizer: Any, config: 'TrainingConfig'):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.trainer = None

    def setup_trainer(self, dataset: Any) -> SFTTrainer:
        """Setup the SFT trainer with given configuration"""
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("The selected tokenizer must define a chat template.")
        if len(dataset) == 0 or "messages" not in dataset.column_names:
            raise ValueError("Training requires nonempty conversational messages.")
        training_args = SFTConfig(
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            max_length=self.config.max_seq_length,
            report_to="none",
            push_to_hub=False,
            logging_steps=10,
            output_dir=self.config.output_dir,
            optim="adamw_8bit",
            seed=3407,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=dataset.select_columns(["messages"]),
            args=training_args,
        )

        return self.trainer

    def train(self) -> None:
        """Execute the training process"""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer() first.")

        logger.info(f"Starting training for {self.config.max_steps} steps...")
        self.trainer.train()
        logger.info("Training completed!")
