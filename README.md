# PsyLLM
<p align="center">
  <img src="https://github.com/Emo-gml/PsyLLM/blob/main/logo.jpg" alt="project logo" width="200px" />
</p>
<p align="center">
    <a href="https://arxiv.org/pdf/2505.15715">
        <img src="https://img.shields.io/badge/arXiv-2502.04424-b31b1b.svg" alt="arXiv">
    </a>
    <a href="https://huggingface.co/GMLHUHE/PsyLLM-8B">
        <img src="https://img.shields.io/badge/🧠-PsyLLM--8B-4682B4" alt="Model Weights 8B">
    </a>
    <a href="https://huggingface.co/GMLHUHE/PsyLLM-4B">
        <img src="https://img.shields.io/badge/🧠-PsyLLM--4B-5A67D8" alt="Model Weights 4B">
    </a>
    <a href="https://huggingface.co/GMLHUHE/PsyLLM-1.7B">
        <img src="https://img.shields.io/badge/🧠-PsyLLM--1.7B-6B46C1" alt="Model Weights 1.7B">
    </a>
    <a href="https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy">
        <img src="https://img.shields.io/badge/📖-Dataset-8A2BE2" alt="Dataset">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
    </a>
</p>




</p>

<p align="center">
    <a href="#-about">🌸 About</a> •
    <a href="#-news">📰 News</a> •
    <a href="#-dataset">📦 Dataset</a> •
    <a href="#-psyllm">🧠 PsyLLM</a> •
    <a href="#-quick-start">🔥 Quick Start</a> •
    <a href="#-citation">📜 Citation</a>
</p>

## 🌸 About
This repository contains the official evaluation code and data for the paper "**Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with Large Language Models for Mental Health Counseling**". See more details in our [paper](https://arxiv.org/abs/2505.15715). If you find this project helpful, feel free to ⭐ it!

> PsyLLM is the first large language model explicitly designed to combine diagnostic and therapeutic reasoning for mental health counseling. Unlike traditional LLM-based systems that mainly provide empathetic or surface-level responses, PsyLLM simulates the reasoning process of professional therapists — assessing symptoms, applying international diagnostic standards (DSM/ICD), and selecting suitable therapeutic strategies (such as CBT, ACT, and psychodynamic approaches) to produce clinically grounded, context-sensitive counseling dialogues.

## 📰 News
- **[2025-11-02]** We released a new work on arXiv — [**TheraMind: A Strategic and Adaptive Agent for Longitudinal Psychological Counseling**](https://arxiv.org/abs/2510.25758). We warmly welcome everyone to check it out and join the discussion!
- **[2025-10-28]** Created the official project website: https://github.com/Emo-gml/PsyLLM.
- **[2025-10-21]** We open-sourced the model weights and dataset on Hugging Face!
- **[2025-05-12]** Paper submitted to arXiv: https://arxiv.org/abs/2505.15715.

## 📦 Dataset
<p align="center">
  <img src="https://github.com/Emo-gml/PsyLLM/blob/main/openR1-psy.jpg" alt="OpenR1-Psy Dataset Overview" width="75%">
  <br>
  <em>Figure: Overview of the OpenR1-Psy dataset construction pipeline.</em>
</p>

**OpenR1-Psy** is a large-scale psychological counseling dataset that integrates **diagnostic reasoning** and **therapeutic reasoning** to train and evaluate large language models for mental health dialogue generation. It goes beyond empathy-focused corpora by incorporating explicit reasoning traces grounded in **DSM/ICD diagnostic standards** and diverse psychotherapy frameworks such as **CBT**, **ACT**, **psychodynamic**, and **humanistic therapy**.

The original Reddit posts used in **OpenR1-Psy** are derived from three publicly available research datasets:  
[LRF](https://github.com/drmuskangarg/Irf), [Identifying Depression](https://github.com/Inusette/Identifying-depression), and [Dreaddit](http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip). All datasets are used strictly for research purposes in accordance with their original licenses and usage guidelines.



## 🧠 PsyLLM

**PsyLLM** is a large language model specialized in **psychological counseling** and **mental health dialogue generation**. It unifies **diagnostic reasoning** and **therapeutic reasoning**, grounded in established clinical frameworks such as **DSM** and **ICD**, and integrates diverse therapeutic paradigms including **CBT (Cognitive Behavioral Therapy)**, **ACT (Acceptance and Commitment Therapy)**, and **psychodynamic therapy**.

PsyLLM is trained on the [**OpenR1-Psy**](https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy) dataset, which features multi-turn counseling dialogues enriched with **explicit reasoning traces**. These traces enable **clinically informed**, **empathetic**, and **interpretable** AI-assisted therapeutic interactions.

The model training and fine-tuning pipeline are implemented using the open-source framework [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory). For more details, please refer to the [Code](https://github.com/Emo-gml/PsyLLM/tree/main/Code).

## 🔥 Quick Start
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "GMLHUHE/PsyLLM"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "I have participated in big group sessions before where I was left to find my own safe place, but it hasn't worked for me."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True 
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("PsyLLM thinking content:", thinking_content)
print("PsyLLM content:", content)
```

## 📜 Citation
```bibtex
@article{hu2025beyond,
  title={Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with Large Language Models for Mental Health Counseling},
  author={Hu, He and Zhou, Yucheng and Si, Juzheng and Wang, Qianning and Zhang, Hengheng and Ren, Fuji and Ma, Fei and Cui, Laizhong},
  journal={arXiv preprint arXiv:2505.15715},
  year={2025}
}
```

## 🧩 License

For **research and educational use only.**

Please ensure compliance with **ethical and legal standards** in mental health AI research.

🔥Please contact huhe@gml.ac.cn
 if you encounter any issues.
