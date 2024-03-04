# LoRA


Lora微调实际上是指一种特定的模型微调技术，称为"LoRA"，全称为"Low-Rank Adaptation"（低秩适配）。核心思想是在模型的预训练权重基础上，

通过引入额外的、较小的、可训练的参数矩阵来实现微调，这些矩阵作为原始权重的低秩更新。



## 原理



## 流程


+ 准备数据集

+ 选择预训练模型

+ 构建模型

+ 定义损失函数

+ 定义优化器

+ 训练模型

+ 评估模型


## 实战


### 依赖库导入


```bash

pip install -q peft transformers datasets

```


### 准备数据集


+ 加载数据集

```python


from datasets import load_dataset   # 导入数据集加载函数

ds= load_dataset('imdb')           # 加载IMDB数据集


```


+ 数据集预处理

```python

labels = ds["train"].features["label"].names

label2id, id2label = dict(), dict()

for i, label in enumerate(labels):

    label2id[label] = i

    id2label[i] = label


id2label[2]

"baklava"

```

+ 特征缩放标准化

```python

from transformers import AutoImageProcessor


image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")



from torchvision.transforms import (

    CenterCrop,

    Compose,

    Normalize,

    RandomHorizontalFlip,

    RandomResizedCrop,

    Resize,

    ToTensor,

)


normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

train_transforms = Compose(

    [

        RandomResizedCrop(image_processor.size["height"]),

        RandomHorizontalFlip(),

        ToTensor(),

        normalize,

    ]

)


val_transforms = Compose(

    [

        Resize(image_processor.size["height"]),

        CenterCrop(image_processor.size["height"]),

        ToTensor(),

        normalize,

    ]

)


def preprocess_train(example_batch):

    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]

    return example_batch


def preprocess_val(example_batch):

    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]

    return example_batch


```


+ 数据集划分

```python


train_ds = ds["train"]

val_ds = ds["validation"]


train_ds.set_transform(preprocess_train)

val_ds.set_transform(preprocess_val)


``` 


+ 微调数据整理器

```python


import torch


def collate_fn(examples):

    pixel_values = torch.stack([example["pixel_values"] for example in examples])

    labels = torch.tensor([example["label"] for example in examples])

    return {"pixel_values": pixel_values, "labels": labels}

```


### 选择预训练模型


+ 构建模型

```python


from transformers import AutoModelForImageClassification, TrainingArguments, Trainer


model = AutoModelForImageClassification.from_pretrained(

    "google/vit-base-patch16-224-in21k",

    label2id=label2id,

    id2label=id2label,

    ignore_mismatched_sizes=True,

)

```
+ PEFT微调器配置





## 训练


+ 定义损失函数

```python

from transformers import TrainingArguments, Trainer


account = "stevhliu"

peft_model_id = f"{account}/google/vit-base-patch16-224-in21k-lora"

batch_size = 128


args = TrainingArguments(

    peft_model_id,

    remove_unused_columns=False,

    evaluation_strategy="epoch",

    save_strategy="epoch",

    learning_rate=5e-3,

    per_device_train_batch_size=batch_size,

    gradient_accumulation_steps=4,

    per_device_eval_batch_size=batch_size,

    fp16=True,

    num_train_epochs=5,

    logging_steps=10,

    load_best_model_at_end=True,

    label_names=["labels"],

)

```

+   定义优化器

```python

trainer = Trainer(

    model,

    args,

    train_dataset=train_ds,

    eval_dataset=val_ds,

    tokenizer=image_processor,

    data_collator=collate_fn,

)

trainer.train()

```


## 评估



## 模型发布


```python


from huggingface_hub import notebook_login


notebook_login()

model.push_to_hub(peft_model_id)

```

## 模型部署


```python


from peft import PeftConfig, PeftModel

from transfomers import AutoImageProcessor

from PIL import Image

import requests


config = PeftConfig.from_pretrained("stevhliu/vit-base-patch16-224-in21k-lora")

model = AutoModelForImageClassification.from_pretrained(

    config.base_model_name_or_path,

    label2id=label2id,

    id2label=id2label,

    ignore_mismatched_sizes=True,

)

model = PeftModel.from_pretrained(model, "stevhliu/vit-base-patch16-224-in21k-lora")


url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"

image = Image.open(requests.get(url, stream=True).raw)

image


```
