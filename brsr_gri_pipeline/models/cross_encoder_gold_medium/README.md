---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:116
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: validation
      type: validation
    metrics:
    - type: accuracy
      value: 0.8333333333333334
      name: Accuracy
    - type: accuracy_threshold
      value: -0.4814453125
      name: Accuracy Threshold
    - type: f1
      value: 0.9090909090909091
      name: F1
    - type: f1_threshold
      value: -0.4814453125
      name: F1 Threshold
    - type: precision
      value: 1.0
      name: Precision
    - type: recall
      value: 0.8333333333333334
      name: Recall
    - type: average_precision
      value: 1.0
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision 233902d25c440f23af6f7d6e94d2946bac0bee0a -->
- **Maximum Sequence Length:** 256 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Percentage coverage by training and awareness programmes on any of the principles during the financial year: GRI Disclosure 2-17 requires disclosing measures taken to advance the collective knowledge, skills, and experience of the highest governance body on sustainable development, which can include training. direct', 'S2-1_06 ESRS S2 S2 S2-1 Undertaking has supplier code of conduct semi-narrative'],
    ['Provide details related to waste management by the entity, in the following format: Total Waste generated (in metric tonnes) Plastic waste (A) E-waste (B) Bio-medical waste © Construction and demolition waste (D) Battery waste (E) Radioactive waste (F) Other Hazardous waste. Please specify, if any. (G) Other Non-hazardous waste generated (H). Please specify, if any. (Break-up by composition i.e. by materials relevant to the sector) Total (A+B + C + D + E + F + G + H) For each category of waste generated, total waste recovered through recycling, re-using or other recovery operations (in metric tonnes) (I) Recycled (ii) Re-used (iii) Other recovery operations Total GRI standard allows the organization to present the waste composition at their convenience, when BRSR specifies the composition of waste information to be presented in the report. direct', 'E5-5_07 ESRS E5 E5 E5-5 Total Waste generated Mass'],
    ['List stakeholder groups identified as key for your entity and the frequency of engagement with each stakeholder group. The frequency of engagement is not specifically mentioned in any of these GRI disclosures, however, it can be a part of the ‘methods used’ as defined in the Guidance 3- 1-b. direct', 'IRO-1_05 ESRS 2 ESRS 2 IRO-1 Description of how process includes consultation with affected stakeholders to understand how they may be impacted and with external experts narrative'],
    ['Employees and workers who have been provided training on human rights issues and policy(ies) of the entity, in the following format: For current and previous financial year for - Permanent and other employees Permanent and other workers Employees and workers who have been provided training on human rights issues and policy(ies) of the entity, in the following format: For current and previous financial year for - Permanent and other employees Permanent and other workers GRI Disclosure 404-1 requires average training hours and not on number of employees / workers trained. GRI Disclosure 2-24, Disclosure 403-5 and Disclosure 410-1 do not require presenting the number of employees/workers trained by gender or by category. GRI Disclosure 205-2 does not require presenting the number of employees/workers trained gender wise. To be continued on next page... GRI Disclosure 404-1 requires average training hours and not on number of employees / workers trained. GRI Disclosure 2-24, Disclosure 403-5 and Disclosure 410-1 do not require presenting the number of employees/workers trained by gender or by category. GRI Disclosure 205-2 does not require presenting the number of employees/workers trained gender wise. direct', 'G1-3_08 ESRS G1 G1 G1-3 Information about members of administrative, supervisory and management bodies relating to anti-corruption or anti-bribery training narrative'],
    ['Do you have a focal point (Individual/ Committee) responsible for addressing human rights impacts or issues caused or contributed to by the business? direct', 'GOV-1_12 ESRS 2 ESRS 2 GOV-1 Information about reporting lines to administrative, management and supervisory bodies narrative'],
]
scores = model.predict(pairs)
print(scores)
# [-2.7129  2.9004  0.854   7.1367 -2.1426]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Percentage coverage by training and awareness programmes on any of the principles during the financial year: GRI Disclosure 2-17 requires disclosing measures taken to advance the collective knowledge, skills, and experience of the highest governance body on sustainable development, which can include training. direct',
    [
        'S2-1_06 ESRS S2 S2 S2-1 Undertaking has supplier code of conduct semi-narrative',
        'E5-5_07 ESRS E5 E5 E5-5 Total Waste generated Mass',
        'IRO-1_05 ESRS 2 ESRS 2 IRO-1 Description of how process includes consultation with affected stakeholders to understand how they may be impacted and with external experts narrative',
        'G1-3_08 ESRS G1 G1 G1-3 Information about members of administrative, supervisory and management bodies relating to anti-corruption or anti-bribery training narrative',
        'GOV-1_12 ESRS 2 ESRS 2 GOV-1 Information about reporting lines to administrative, management and supervisory bodies narrative',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `validation`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value   |
|:----------------------|:--------|
| accuracy              | 0.8333  |
| accuracy_threshold    | -0.4814 |
| f1                    | 0.9091  |
| f1_threshold          | -0.4814 |
| precision             | 1.0     |
| recall                | 0.8333  |
| **average_precision** | **1.0** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 116 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                         | label                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                             | string                                                                             | float                                                          |
  | modality | text                                                                               | text                                                                               |                                                                |
  | details  | <ul><li>min: 3 tokens</li><li>mean: 98.38 tokens</li><li>max: 244 tokens</li></ul> | <ul><li>min: 22 tokens</li><li>mean: 35.39 tokens</li><li>max: 57 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.62</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | sentence_1                                                                                                                                                                                       | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Percentage coverage by training and awareness programmes on any of the principles during the financial year: GRI Disclosure 2-17 requires disclosing measures taken to advance the collective knowledge, skills, and experience of the highest governance body on sustainable development, which can include training. direct</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>S2-1_06 ESRS S2 S2 S2-1 Undertaking has supplier code of conduct semi-narrative</code>                                                                                                     | <code>0.0</code> |
  | <code>Provide details related to waste management by the entity, in the following format: Total Waste generated (in metric tonnes) Plastic waste (A) E-waste (B) Bio-medical waste © Construction and demolition waste (D) Battery waste (E) Radioactive waste (F) Other Hazardous waste. Please specify, if any. (G) Other Non-hazardous waste generated (H). Please specify, if any. (Break-up by composition i.e. by materials relevant to the sector) Total (A+B + C + D + E + F + G + H) For each category of waste generated, total waste recovered through recycling, re-using or other recovery operations (in metric tonnes) (I) Recycled (ii) Re-used (iii) Other recovery operations Total GRI standard allows the organization to present the waste composition at their convenience, when BRSR specifies the composition of waste information to be presented in the report. direct</code> | <code>E5-5_07 ESRS E5 E5 E5-5 Total Waste generated Mass</code>                                                                                                                                  | <code>1.0</code> |
  | <code>List stakeholder groups identified as key for your entity and the frequency of engagement with each stakeholder group. The frequency of engagement is not specifically mentioned in any of these GRI disclosures, however, it can be a part of the ‘methods used’ as defined in the Guidance 3- 1-b. direct</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | <code>IRO-1_05 ESRS 2 ESRS 2 IRO-1 Description of how process includes consultation with affected stakeholders to understand how they may be impacted and with external experts narrative</code> | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `fp16`: True
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | validation_average_precision |
|:-----:|:----:|:----------------------------:|
| 1.0   | 8    | 1.0                          |
| 2.0   | 16   | 1.0                          |
| 3.0   | 24   | 1.0                          |


### Training Time
- **Training**: 2.1 seconds

### Framework Versions
- Python: 3.10.20
- Sentence Transformers: 5.7.0
- Transformers: 5.14.1
- PyTorch: 2.7.1+cu118
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->