---
layout: post
title:  Financial Phrases Classification (part 1)
date:   2024-02-02 
description: Overview of HuggingFace NLP course and applied to financial phrases dataset
tags: HuggingFace, transformers, part1
categories: 
---

In last week's post I said one of the challenges I had in understanding the code for weak-to-strong models was my lack of familiarity with PyTorch and the HuggingFace Transformers library. To address this I went through the [introductory course](https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt) that HuggingFace includes in it's docs, and then I applied the Transformers library to a mini project which I'll show here. 

## HuggingFace Course Overview

The mini course in the HuggingFace Docs are very easy to follow. There are currently 9 sections and each section covers a different part of the transformers library. That is, every section except the first one which gives an overview of what transformers are and how they work. This was pretty useful for me because even though I have used language models frequently in the last 1 year, I don't have a very deep understanding of exactly how they work. I think within the next couple weeks I will do a detailed read of the *Attention is All You Need* paper which first introduced the transformer model. 

They go over a brief history of Transformer models. The first was GPT in June of 2018, which was shortly followed by BERT, GPT-2, BART, and GPT-3 in May of 2020. They then go over the architecture of a transformer. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/financial-phrases/transformer.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
credit: Google AI
    
</div>

But only high-level concepts and to get a full understanding I'll have to dig deeper. The two main components of a transformer are the encoder and the decoder. The encoder uses a full attention layer grasp all of the information contained in the input. The decoder uses a partial attention layer -- only considering words which have already been outputted by the transformer. The authors of *Attention is All You Need* created the model in this way because this model was intially proposed for language translation. The idea was that by using both an encoder and a decoder, the model will be able to understand the full statement in the first language, and then be able to generate words in the translated language using the full encoder knowledge and the decoder knowledge of words generated so far. 

Some models are encoder only(auto-encoding), some are decoder only(auto-regressive), and some are both encoder and decoder(sequence to sequence). Encoder only models are good for tasks which require full understanding of input because it uses a full attention mask. These models are good at things like sentiment analysis, Q&A, and sentence classification. BERT is a popular example of a encoder-only model. Decoder only models on the other hand are good at tasks which work best when the model is only aware of previously generated words. These models are good at text-generation. Popular examples are GPT and GPT-2. Sequence to Sequence models work by using the encoder once on the input text and then use the decoder repeatedly to generate more and more words to be appended to the response. These models are good for text-generation and language translation. BART and T5 are examples of sequence to sequence models. 

That pretty much covers the first section of the course. The remaining sections cover the specific classes and uses cases of the transformers library. Instead of going into detail about each one of these sections, I will describe the mini-project I did. This project touches all the sections of the course so it will give a good overview of what is possible with the transformers library. 

## Finetuning on Financial Phrases

The mini project is as follows: To finetune a model that outputs the sentiment of a financial phrase. After creating this model I will create a hugging face space where users can input a name of a company on twitter, and the app uses the twitter API to gather the last 1000 mentions of the company and gives a summary of the sentiments of the 1000 tweets. In this part 1 post I am only going over the finetuning part. 

Sentiment analysis we aim this model to do is classify a sentence as 0-negative, 1-neutral, or 2-postiive. The first step is to choose a model to finetune. Based on the fact that encoder only models are suitable for understanding the entire input which is what sentiment analysis should do, I choose DiistilBert. This model is encoder only, and it is also much fewer parameters than BERT with similar performance so it can be trained and evaluated much faster. The next step is choosing a suitable dataset. [This dataset](https://huggingface.co/datasets/financial_phrasebank) has just what we are looking for -- there is a sentence column with the financial phrase and then a label column with the human labelled sentiment. This dataset is found on Hugging Face datasets

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/financial-phrases/dataset.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    
</div>

Before we begin we need the following packages:

```python
# Importing the libraries needed
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset, DatasetDict, Dataset

```

To finetune the model we need to download the dataset and then modify it so that it can be trained. the first modification is creating a training split, validation split, and test split. In the original state there is only a train split. 

```python
import datasets

raw_datasets = load_dataset("financial_phrasebank", 'sentences_50agree')
data_splits = DatasetDict({})
train_df = pd.DataFrame(raw_datasets['train'][:3876])
train_df['label'] = train_df['label'].astype(float)
valid_df = pd.DataFrame(raw_datasets['train'][3876:3876+485])
valid_df['label'] = valid_df['label'].astype(float)
test_df = pd.DataFrame(raw_datasets['train'][3876+485:])
test_df['label'] = test_df['label'].astype(float)
training_data = Dataset.from_pandas(pd.DataFrame(raw_datasets['train'][:3876]))
validation_data = Dataset.from_pandas(pd.DataFrame(raw_datasets['train'][3876:3876+485]))
test_data = Dataset.from_pandas(pd.DataFrame(raw_datasets['train'][3876+485:]))

data_splits['train'] = Dataset.from_pandas(train_df)
data_splits['valid'] = Dataset.from_pandas(valid_df)
data_splits['test'] = Dataset.from_pandas(test_df)


```

This code creates a new DatasetDict called `data_splits` which has a different split for training, validation, and testing. I used 80% of the dataset for training and 10% for validation and testing. The validation split is useful for determining whether the model is overfitting or underfitting. It can also be used for evaluation. The test split is used for evaluation. After creating the different splits of the database, the next step is to pre-process the data so that it can be fed into the pretrained model. This primarily involves turning the sentences into sequences of numbers, aka tokenizing. To do this we use the `DistillBertTokenizer` module from the transformers library. Trying it out on a smaple sentence:

```python
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
inputs = tokenizer("Hello, this is a example sentence", "Hi im Bob")
inputs
```
returns
```
{'input_ids': [101, 1188, 1110, 1103, 1148, 5650, 119, 102, 1188, 1110, 1103, 1248, 1141, 119, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

It returns an attention mask as well as the tokenized sentence -- this just allows the model to know which tokens to pay attention to. Since BERT is an auto-encoding model it will pay attention to all of the input.

Another crucial pre-processing step is padding. When inputs are fed into the model during fin-tuning, they are loaded in batches. In order for the gpu to compute the new weights of the model efficiently, all of the inputs must have the same size. This allows the gpu to use its parallel computing capabilities. Below is how I did the padding:

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

```

This defines a function data_collator which is passed a parameter to the PyTorch Trainer API. The last step of finetuning is training and evaluation. We can use the evaluate library to pass in a parameter into the Trainer API which tells the Trainer API about which metrics to keep track of. I defined a metric which keeps track of training loss and validation loss, and accuracy.

```python
import evaluate

metric = evaluate.load("accuracy")
metric.compute(predictions=preds, references=predictions.label_ids)
```

For training I used the PyTorch training API:

```python
checkpoint = 'distilbert-base-cased'

def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

and the results were as follows:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/financial-phrases/results.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    
</div>

The final accuracy on the test split is 80% which is relatively good considering that the labels were also not unanimously -- meaning the labels were voted on by a group of people and the dataset includes labels with >60% consensus among the voters. The part that is concerning is that the validation loss goes up on the last epoch despite the training loss going down. This is a common sign of over-fitting. Over-fitting can be solved by increasing the dataset or by using methods such as regularization. Since I am getting 80% accuracy on the test split which is pretty good I will continue to making the app with this model and I may modify the model later if I find that over-fitting is a significant problem.

## Next Steps

This mini-project shows many of the core functionalities of the transformers library: using the datasets hub, using tokenizers, using pretrained model checkpoints, using the evluate library. Now that I have a finetuned model that is scoring 80% on the test split, I will use this model inference to make a app on HuggingFace spaces which uses this inference on the last 1000 mentions of a name to summarize the sentiment. 

Later on I also hope to dig deeper into the details of transformers and tokenizers. Another thing I was very curious about was how exactly DistilBERT was created? I am interested how a model with similar performance but much fewer parameters was achieved. 
