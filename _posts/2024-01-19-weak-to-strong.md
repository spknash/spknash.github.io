---
layout: post
title:  Weak to Strong Generalization
date:   2024-01-19 
description: Review of new paper on super aligment and weak to strong generalization
tags: AI, Alignment, openai
categories: 
---

# Weak-to-Strong Generalization (OpenAI)

OpenAI recently came out with a paper titled *Weak-to-Strong Generalization: Eliciting Strong Capabilities using Weak Supervision*. The paper tackles the broad problem of super human alignment -- how will humans supervise superhuman artificial intelligence. Since superintelligent AI does not exist yet, this alignment problem can not be tackled directly. The authors choose to tackle an analogous problem -- can a weak model supervise a strong model(both of whichare sub-human intelligence)? I found this paper to be very interesting. It shows that weak-to-strong generalisation is tractable based on the experiments they ran, however a significant percentage of strong capabilities are not elicitied using weak supervision.

In this post I'll summarize methods used, results, and conclusions of the paper. Then I'll look ahead to experiments that I am doing which were inspired by this paper.

## Aligning Superintelligence

Aligning Superintelligence is one of the grand challenges of AI. Modern day sota models use human supervision for alignment. This is fine because humans are still stronger than the current models. However when superintelligent AI exists, they will be stronger than humans and humans can only weakly supervise these models. Superintelligent AI does not exist yet, so a close analogous situation is weak-to-strong generalization -- a weak model supervising a strong model. The authors aim to show that weak-to-strong generalazation and take the first steps towards super alignment.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/weak_strong.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
credit: OpenAI
    
</div>

## Methodology

GPT-2 is used as the weak supervisor, and GPT-4 is used as the strong student in the experiments

The experiment the authors conducted is structured in the following way. A weak supervisor is finetuned on the ground truth, and this model makes predictions on test data(Call the score of the weak model WM). A strong student is trained with weak supervision, ie the strong model is finetuned on labels produced by the weak model. This model then makes predictions on the test data(Call the score of this weakly trained strong model WTS). Finally, as a benchmark, the strong model is trained on the ground truth and labels the test data(Call the score of this model SM). The effectiveness of the weak-to-strong generalization is measured by the Performance Gap Recovered(PGR) which is the percent of the performance gap between the weak model and strong model covered by the weak to strong model. So the PGR = (WTS-WM)/(SM-WM). In other words if the PGR is 0% the weak to strong generalization was completely innefective and the weakly supervised strong model performed the same as the weak model. If PGR is 100% then it was very effective and the weak to strong generalization is just as good as the strong model.

Some advantages of this setup include:

* Any weak and strong models acn be used in this experiment, as long as there are task and measurement methods to assess performance on tasks.
* Similarly, any set of tasks can be used
* The authors say success in weak-to-strong generalization will help with alignment of current models.

There are also some limits of this setup. Specifically, there are disanalogies to the problem of humans supervising super intelligent AI.

* The first is Imitation Saliency: The errors weak models make may be differenct than the errors humans make.
* The second is Pretraining Leakage: The pretraining data in the strong models used in these experiments contains supervision from humans. So in the experiment, the weakly trained strong models may be artificially producing the correct answer in some cases because of their human supervised pretraining. Howerver, super-intelligence may not be created using human supervision, eg self-supervised training. This would make it harder to elicit from super human models.

## Results

The tasks used in the experiment are the following:

* Popular NLP benchmarks: all datasets converted to binary classification, these tasks include sentiment analysis, natural language inference and more
* Chess Puzzles: data set of puzzles which is a position on the chess board and the label is the best next move. This is the only generative task used in this experiment
* GPT reward modeling: a binary dataset of pairs of gpt responses and the label is the better completion result

First the authors did plain weak-to-strong generalization with no additional mathods used and tested to get a benchmark. On the NLP benchmarks test, the PGR was between 20% and 50% with the PGR increasing as the model size of the strong model increases. On the chess puzzles task, the PGR is close to 0% for small weak model size and it increases to 40% for the largest weak model size. The PGR does not increase with an increase of strong model size. On the GPT reward modeling task, the PGR is always quite low never crossing 25% PGR. On all these tasks the PGR was greater than 0% but still very far from the ceiling set by the strong model trained on the ground truth. The highest PGR in any task was around 50%. These results show that weak-to-strong generalization is tractable but many improvements need to be made before it is a viable alignment technique.

The authors then show results from experiments using weak-to-strong generalization but with different improvements. The first is bootstrapping.

Bootstrapping is a method that has been talked about for super alignment before. The idea is to train a model slightly stronger than the weak model, then use this model to train a slightly stronger model than itself, and so on until you train the strong model you are trying to align. In this experiment, the authors used two intermediary models(I1 and I2) between GPT2 and GPT4. So GPT-2 trained I1 which trained I2 which trained GPT-4. Bootstrapping showed a significant improvement in the chess puzzles task, the largest improvement being around a 25% increase in the PGR. The other tasks, however, did not show much improvement.

The second method which caused gains for weak-to-strong supervision was the auxillary confidence loss term. One of the main things causing errors in the weak-to-strong training is the strong model will sometimes imitate the mistakes of the weak supervisor. Giving the strong model the ability to disagree with the weak models label using its pretraining knowledge would help improve the weak-to-strong generalization. That is what adding the auxillary confidence loss term to standard cross entropy does. During training, if the strong model disagrees with the weak models labels, this label will not affec the model weights very much. This method dramatically improves PGR expecially in large gaps between the size of weak and strong models.

## Understanding Weak-to-Strong Generalization

The authors identify two major phenomena related to weak-to-strong generalization: Imitation of weak supervisor mistakes, and saliency of tasks in the strong model.

### Weak Supervisor Imitation

The strong model repeating the weak models mistakes reduces the effectiveness of weak to strong generalization. One cause of mistake imitation is the strong model overfitting with the weak model. This can be reduced by using techniques commonly used to fix over-fitting like regularization or early stopping of training. Another reason for imitation is high supervisor-student agreement. From the experiment they saw that introducing confidence loss significantly decreased model agreement and the PGR increased as well.

### Saliency of Strong model representations

Some of the PGR acheived by the strong model could be due to the inherent saliency of the tasks in the strong model. For example, if GPT-4 was trained using data similar to the popular NLP tasks it may be able to acheive better results than the weak model without any finetuning. The authors tested this by seeing how accurate the strong models were with zero-shot or few shot prompting. For smaller strong models the PGR acheived was minimal but for larger students the PGR was significantly higher and comparable to results of the weakly supervised model. However, weakly supervised models with confidence loss generally outperforms zero-shot and few-shot prompting even for large models.

If salient representations of the task are ineherent in the pretrained strong model, then weak-to-strong generalization could be improved by using unsupervised finetuning in order to bring out the salient abilities of the strong model. Unsupervised finetuning improves the PGR of the GPT reward modeling task from  less than 20% to 30-40%.

Finally, the authors discuss the possibility of using linear probing to improve weak-to-strong. They found that weak supervision increases the linearity of the labels so finetuning using weak labels and then linear probing increases the PGR of the weak-to-strong generalization.

## Discussion

The first potential avenue for future work discussed is changes to the analogous setup. As mentioned before, there are two main disanalogies with this setup:

* Imitation Saliency
* Pretraining Leakage

The authors mention the following ways to improve the setup:
* fixing disanalogies
* showing that the disanalogies are not severe
* generalizing tasks to more complicated tasks

The next avenue future work is scalable methods for weak-to-strong generalization. In a desired weak-to-strong generalization:

* the strong model should disagree with weak models errors
* the generalization should be salient to the strong model
* The gneralization should be consistent.

The authors also believe there is a lot of potential for gains to be made in weak-to-strong generalization by using more ML techniques like confidence loss which dramatically improved PGR.

Finally, the authors say that super human alignment is a critical problem, and if we are to use super human AI for important things there should be a clear scientific understanding of when and why the alignment works. They mention the following questions among others:

* Why is there a significant difference between PGR obtained for different tasks?
* What makes a task or concept easy or hard to elicit?
* How much to errors in weak labels affect the strong models behavior?

## What I am working on

The authors mention that the saliency of the strong model representation for the tasks used is not clearly known and could have effected the results. Specifically, there may be overlap between the GPT-4 pretraining data and the popular NLP tasks data which was used in the experiment. I plan to recreate the experiment but instead of using GPT-2 and GPT-4 as the weak and strong models I will use 2 opensource models llama v1 and llama v2. And I will test on data that is disjoint from the pretraining data of these models. The idea behind this is to choose a task which the strong model does not have pretraining data on, to show I will share all details and code in my next post.





