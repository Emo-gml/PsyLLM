# PsyLLM

<p align="center">
  <img src="https://github.com/Emo-gml/PsyLLM/blob/main/logo.jpg" alt="project logo" width="200px" />
</p>

<p align="center">
    <a href="https://arxiv.org/pdf/2505.15715">
        <img src="https://img.shields.io/badge/arXiv-2502.04424-b31b1b.svg" alt="arXiv">
    </a>
    <a href="https://huggingface.co/GMLHUHE/PsyLLM">
        <img src="https://img.shields.io/badge/%F0%9F%A7%A0-Model%20Weights-4682B4" alt="Model Weights">
    </a>
    <a href="https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy">
        <img src="https://img.shields.io/badge/%F0%9F%93%96-Dataset-8A2BE2" alt="Dataset">
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
    <a href="#-quick-start">🔥 Quick Start</a> •
    <a href="#-citation">📜 Citation</a>
</p>

## 🌸 About
This repository contains the official evaluation code and data for the paper "**Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with Large Language Models for Mental Health Counseling**". See more details in our [paper](https://arxiv.org/abs/2505.15715).

> PsyLLM is the first large language model explicitly designed to combine diagnostic and therapeutic reasoning for mental health counseling. Unlike traditional LLM-based systems that mainly provide empathetic or surface-level responses, PsyLLM simulates the reasoning process of professional therapists — assessing symptoms, applying international diagnostic standards (DSM/ICD), and selecting suitable therapeutic strategies (such as CBT, ACT, and psychodynamic approaches) to produce clinically grounded, context-sensitive counseling dialogues.

## 📰 News
- **[2025-10-28]** Created the official project website: https://emo-gml.github.io/.
- **[2025-07-08]** We open-sourced the model weights and dataset on Hugging Face!
- **[2025-05-12]** Paper submitted to arXiv: https://arxiv.org/abs/2505.15715.

## 📦 Dataset
**OpenR1-Psy** is a large-scale psychological counseling dataset that integrates **diagnostic reasoning** and **therapeutic reasoning** to support the training and evaluation of large language models for mental health dialogue generation.  
Unlike traditional empathy-focused datasets, **OpenR1-Psy** incorporates explicit reasoning traces aligned with **DSM/ICD diagnostic standards** and diverse psychotherapy frameworks, including **CBT**, **ACT**, **psychodynamic**, and **humanistic therapy**.  

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
@article{hu2025emobench,
  title={EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models},
  author={Hu, He and Zhou, Yucheng and You, Lianzhong and Xu, Hongbo and Wang, Qianning and Lian, Zheng and Yu, Fei Richard and Ma, Fei and Cui, Laizhong},
  journal={arXiv preprint arXiv:2502.04424},
  year={2025}
  }
```

## 🧩 License

For **research and educational use only.**

Please ensure compliance with **ethical and legal standards** in mental health AI research.

🔥Please contact huhe@gml.ac.cn
 if you encounter any issues.
