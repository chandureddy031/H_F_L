import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import os

class ChatModelFineTuner:
    def __init__(self, base_model="microsoft/DialoGPT-small"):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='right')
        
        # Load model with proper configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        
        # Configure tokenizer properly
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        
        # Ensure model has correct pad token id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def load_data(self, data_path="data/training_data.jsonl"):
        """Load and preprocess training data"""
        texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # Create proper conversation format
                    text = f"{item['instruction']}{self.tokenizer.eos_token}{item['output']}{self.tokenizer.eos_token}"
                    texts.append(text)
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(texts)} training examples")
        return texts
    
    def create_dataset(self, texts, max_length=128):
        """Create properly tokenized dataset"""
        def tokenize_function(examples):
            # Tokenize texts with padding
            tokenized = self.tokenizer(
                examples,
                truncation=True,
                padding='max_length',  # Pad to max_length
                max_length=max_length,
                return_tensors=None
            )
            
            # Create labels (copy of input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Create dataset first
        dataset = Dataset.from_dict({"text": texts})
        
        # Tokenize in batches
        dataset = dataset.map(
            lambda examples: tokenize_function(examples["text"]),
            batched=True,
            batch_size=100,
            remove_columns=["text"],
        )
        
        print(f"Tokenized dataset created with {len(dataset)} examples")
        return dataset
    
    def setup_lora(self):
        """Setup LoRA for efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn"],
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        return self.model
    
    def train(self, dataset, output_dir="./fine_tuned_model"):
        """Fine-tune the model"""
        # Setup LoRA
        self.setup_lora()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            logging_steps=25,
            learning_rate=3e-4,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final model
        print("Saving model...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model successfully saved to {output_dir}")

def main():
    try:
        # Initialize fine-tuner
        print("Initializing fine-tuner...")
        tuner = ChatModelFineTuner()
        
        # Load data
        print("Loading training data...")
        texts = tuner.load_data()
        
        if len(texts) == 0:
            print("❌ No training data found! Please check data/training_data.jsonl")
            return
        
        # Create dataset
        print("Creating dataset...")
        dataset = tuner.create_dataset(texts)
        
        # Fine-tune
        print("Starting fine-tuning process...")
        tuner.train(dataset)
        
        print("🎉 Fine-tuning completed successfully!")
        print("You can now run: python app.py")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()