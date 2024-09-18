---
base_model: /module/Tongyi-Finance-14B-Chat
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: '1212'
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 1212

This model is a fine-tuned version of [/module/Tongyi-Finance-14B-Chat](https://huggingface.co//module/Tongyi-Finance-14B-Chat) on the qwen dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 4
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 4.0

### Training results



### Framework versions

- Transformers 4.34.1
- Pytorch 2.0.1+cu117
- Datasets 2.14.6
- Tokenizers 0.14.1
