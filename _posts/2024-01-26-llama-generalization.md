---
layout: post
title:  Weak to Strong with LLaMas
date:   2024-01-26 
description: Weak to Strong Generalization experiment with LLaMa v1 and v2
tags: llama, colab, llm, openai
categories: 
---


## Recap 

In the previous post I went into detail about this OpenAI paper. A full summary of the experiment and results are found in that post but here is a quick recap. Superhuman AI alignment is somewhat analogous to weak-to-strong generalization. This is because is superhuman alignment, an inferior intelligence(humans) aims to supervise a superior intelligence(the AI). Since superhuman AI does not exist yet, a close analogy is weak-to-strong generalization in which a weak AI aims to supervise a stronger AI. In the paper, the researchers discuss an experiment where they test weak-to-strong generalization where GPT-2 supervises GPT-4 on 3 different tasks(NLP benchmarks, chess puzzles, GPT reward modeling). One of the weaknesses of the experiment which the researchers noted is that the saliency of the tasks in the stronger model is not clearly known. For example, if GPT-4's pretraining dataset included the tasks, this would artificially inflate the perceived effectiveness of weak-to-strong generalization. 

For this reason my goal here was to repeat the experiment with llama v1 supervising llama v2 on a task which is not salient in the strong model. The idea here is that llama v1 and llama v2 are opensource so it should be possible to choose a task which is not salient in llama. In addition to choosing opensource models in which there is more knowledge about the pretraining dataset, simply choosing a more complicated task reduces the likelihood of pretraining leakage.

## OpenAI repo

The OpenAI weak-to-strong repo has code similar to what is used in the experiment the paper covers. The repo however uses the models GPT-2 at varying sizes and some other language models like QWEN instead of GPT-2 and GPT-4 which were used in the experiment discussed in the paper. I didn't really understand why the code repo is so different from the code which must have been used for the actual experiment in the paper. It may be because GPT-4 needs API use and it would be an expensive experiment for an average person to run if the strong model used was GPT-4. 

The first thing I did was test out the code in the repo using GPT-2 small as the weak model and GPT-2 medium as the strong model. This runs GPT-2 small on ground truth and GPT-2 medium on ground truth as the two benchmarks, and then the GPT-2 medium finetuned on GPT-2 small labels as the weak-to-strong model. I ran this experiment to see what I get and I get the following results:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/weak-strong-llamas/sciq.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
GPT-2 small, and GPT-2 medium on ground truth, and GPT-2 medium on weak labels
    
</div>

Based on these results, it is clear that a very small PGR is achieved by the 

Despite looking very different, they are pretty similar to the expected results given by the repo authors:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/weak-strong-llamas/amazon_polarity.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
credit: OpenAI
    
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/weak-strong-llamas/anthropic_hh.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
credit: OpenAI
    
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/weak-strong-llamas/boolq.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
credit: OpenAI
    
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/weak-strong-llamas/sciq expected.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
credit: OpenAI
    
</div>



These results show in all the datasets the weak-to-strong generalization typically achieves a PGR >= 0.2 . With the highest PGR being 0.8 and lowest PGR being negative. However in both of these outliers the distance between ground truth weak and ground truth strong was very small. These PGRs are aligned with the papers findings the weak-to-strong generalization has some promise but the current PGRs acheived are not close to adequate fro superalignment or any alignment.  


I forked this repo and used the same code to run my experiment using llama v1 as the weak model, llama v2 as the strong model. The task I choose was chess puzzles again because it doesn't seem to be included in the llama pretraining datasets according to here. Additionally, of the three tasks used in the original experiment, chess is the most complicated and least likely to be leaked from the pretraining. 

## Weak to strong Generalization with LLaMas

Similarly to the GPT-2 small and GPT-2 medium experiment, In the LLaMa experiment I trained v1 on ground truth and v2 on ground truth as the benchmarks. I then trained v2 on the weakly generated labels by v1 to produce the weak-to-strong model. Below are the results of this experiment:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/weak-strong-llamas/sciq.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
LLaMa v1, and LLaMa v2 on ground truth, and LLaMa v2 on weak labels
    
</div>

With the task being chess puzzles we can see that v1 and v2 both do much poorly compared to GPT-2 and GPT-4 repectively. The distance between v1's ground truth training and v2's ground truth training is 0.2 which is less than the difference between GPT-2 and GPT-4. The weak-to-strong model is 0.015 above v1's ground truth. This leads to a PGR of around 0.07. This PGR is lower than the PGR for chess puzzles of the weakly supervised GPT-4 model. This difference could be caused by multiple factors:

* The performance gap between GPT-2 and GPT-4 is greater than v1 vs v2. This allows a weakly supervised GPT-4 to make more gains
* There could be pretraining leakage of the task in GPT which the authors acknowledge.
* The researchers used larger models which have more latent capabilities and this could have allowed the weak GPT to elicit more capability from the strong GPT. The LLaMa v1 and v2 I used are 3.5B parameters and 7B parameters respectively and the llama.cpp version as well. 

The challenge with deep learning is that it is difficult to impossible to scientifically show which of these factors cause this difference and how much. The lack of scientific understanding surrounding weak-to-strong generalization is also something the paper addresses as future work. One of the goals was to see if the pretraining leakage disanalogy revealed itself in this experiment, but it is unclear at this point because numerous other factors could have caused a lower PGR in the LLaMa weak-to-strong model. 




## Challenges

One major challenge throughout this experiment was working with colab. Running the experiments took a very long time and there was a risk of the runtime being disconnected in the middle of the run which forces you to start over. Testing just GPT2 and GPT2-medium took 2 hours and testing LLaMa v1 and v2, and the weak-to-strong model took around 3 hours. I may consider upgrading my colab to save runtimes or to train models faster, esepcially as I continue to do more deepl learning. I may also look into other cloud computing platforms like lambda labs or Nvidia.

Another challenge was my lack of familiarity with PyTorch and the HuggingFace Transformers library. It was difficult to understand the code that produced the experiment in the OpenAI repository because it was all using PyTorch and Transformers. I was able to gain a good enough understanding to use it and modify it just by looking at the comments and my general programming experience. I have been going through the HuggingFace Transformers introductory mini course and I think my next post will be an application of what I learned through that. I am also going through the stable diffusion series on FastAI and that has given me some very interesting projects to do as well. 

I was not able to run an experiment on the same level as the researchers or even the truncated experiment done in the repo due to limitations in compute and time access to GPUs imposed by Colab. I think as I continue deep learning projects I will need to find a way to solve this common problem. 



