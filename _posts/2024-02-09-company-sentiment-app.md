---
layout: post
title:  Financial Phrases Classification (part 2)
date:   2024-02-09 
description: Showing the app I made
tags: HuggingFace, transformers, spaces, part2
categories: 
---


## The App

I used the model I made trained in the previous post to make an app which shows the recent sentiment of a company on Twitter when you input its stock tag on twitter. Check it out [here!](https://huggingface.co/spaces/suhaaspk/Company-Sentiment). You can find the code [here](https://huggingface.co/spaces/suhaaspk/Company-Sentiment/tree/main). Sometime HuggingFace spaces go to sleep or I may have to disconnect this app if I make more apps in the future so I will leave a screenshot here to show what it looks like.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/financial-phrases/app.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">

    
</div>

I use the Twitter API to fetch a 1000 of the most recent tweets and use the sentiment classifying pipeline to classify each tweet.

## Improvements

One way this app can be improved is: The app currently assesses the overall sentiment of the tweet and not the sentiment of the tweet with regards to stock in question. For example, the statement: "Google did good last quarter, but Paypal did bad" may be classified as neutral since it assesses the whole tweet even though it is clearly negative for PayPal. I still think a large enough sample size gives a pretty good heuristic of the current sentiment of a company on Twitter.

Next week, I may look to fix this issue and improve this app, or go onto other projects. 