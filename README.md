# Practical Fine-Tuning Lab: Instruction Tuning with LoRA & PEFT

This repository contains a lightweight, hands-on lab demonstrating end-to-end fine-tuning of a small language model using **PEFT (LoRA)** and Hugging Face's **TRL (`SFTTrainer`)** in Google Colab. 

The purpose of this lab is to understand the end-to-end pipeline: 
Base Model -> Dataset -> Tokenization -> LoRA Adapter -> SFT Training -> Fine-Tuned Behavior -> Evaluation

---

## Stack & Technologies
* **Environment:** Google Colab (T4 GPU)
* **Base Model:** `HuggingFaceTB/SmolLM2-135M-Instruct`
* **Libraries:** `transformers`, `datasets`, `peft`, `trl`, `accelerate`

---

## Implementation Code

### Step 1 & 2: Setup & Environment
```bash
!pip install -q transformers datasets peft trl accelerate
