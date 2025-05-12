import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayLightningEnvironment, RayTrainReportCallback
from ray.train.lightning import RayDDPStrategy, prepare_trainer

# Load manually with pandas
ami_data_dir = os.getenv("AMI_DATA_DIR", "data")

# === Config ===
config = {
    'per_device_train_batch_size':2,
    'per_device_eval_batch_size':2,
    'learning_rate':2e-4,
    'num_train_epochs':5,
    'logging_steps':10,
    'eval_strategy':"epoch",
    'save_strategy':"epoch",
    'summarization_dataset_path':ami_data_dir}

model_name='google/flan-t5-large'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)



# === Dataset ===
class AMISummarizationDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_input_length, max_target_length):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8')]
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        source = "summarize: " + record["input_text"]
        target = record["target_summary"]

        source_enc = self.tokenizer(
            source,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt"
        )

        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze()
        }

# === LightningModule ===
class T5SummarizationModule(L.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# === Ray Lightning Trainer Wrapper ===
def train_func():
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    train_dataset = AMISummarizationDataset(config["summarization_dataset_path"], tokenizer,
                                            config["max_input_length"], config["max_target_length"])
    val_dataset = train_dataset.train_test_split(test_size=0.1)['test']
    train_dataset = train_dataset.train_test_split(test_size=0.1)['train']

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    model = T5SummarizationModule(model_name=config["model_name"], lr=config["lr"])

    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        strategy=RayDDPStrategy(),
        accelerator="auto",
        devices="auto",
        plugins=[RayLightningEnvironment()],
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config["early_stopping_patience"], mode="min"),
            RayTrainReportCallback()
        ]
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# === Launch Ray Trainer ===
run_config = RunConfig(storage_path="s3://ray")  # Optional: for Checkpoints
scaling_config = ScalingConfig(num_workers=1, use_gpu=True)

trainer = TorchTrainer(
    train_func,
    run_config=run_config,
    scaling_config=scaling_config
)

result = trainer.fit()
