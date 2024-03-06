---
layout: post
title:  AI generated profile picture
date:   2024-02-16 
description: My approach trying to create a AI-generated profile picture, overview of tools used
tags: HuggingFace, diffusers, stable-diffusion, dreambooth, controlnet
categories: 
---
I want to create a profile picture for myself using stable diffusion. I've seen people online such as Abhishek Thakur and Max Howell use AI-generated profile pictures. The good ones that I've seen make the photo more interesting but preserve the subject's core facial features. Below are some that I like.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/mahowellreall.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/maxhowellAI.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Max Howell
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/AbhiREAL.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/AbhiAI.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Abhishek Thakur
</div>


To create my profile picture the first thing I did was do some research into what models are used to create these images. The most popular open-source models are the diffusion models released by [Stability AI](https://stability.ai/) and [runway ML](https://app.runwayml.com/). The most recent of which is Stable Diffusion 3. Their older model Stable Diffusion XL is still being updated by the open-source community and is still considered the SOTA model by most. 

## Stable Diffusion

The first thing I tried out was vanilla stable diffusion text-to-image. The model I am using is Stable Diffusion XL (SDXL) I simply described my appearance to see what stable diffusion returned. Some results are below.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/zeroshot0.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/zeroshot2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/zeroshot3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: indian boy with curly hair, jawline, trimmed beard, smiling, dimples, pimple on cheek, in a forest, pixel art, colorful, vibrant, cheerful
</div>

I obviously did not expect this method to give any output close to desirable because the model has no information about my appearance. Next, I tried using
the image-to-image pipeline of the model. The input image is an image of myself. Some results are below.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/im2im1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/im2im2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/im2im3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: smiling man, comic book style
</div>

I expected these images to be better since the model has access to my face. However, the out-of-the-box image to image method is also not what I am looking for here.
The model seems to change many of my core facial features and it does not look like me at all. The model still does not retain information about the features of my face.
Since the out-of-box pipelines for SDXL do not give me what I want, I began researching how a model can learn a subject's features. This research brought me to
Dreambooth.

## Dreambooth

Dreambooth is a method to finetune a pretrained text-to-image model with a set of images of a subject along with captions including the subject's unique identifier. The final model will
then be able to generate images including the subject when the unique identifier is used in the prompt. Below is the example used in the 2022 [paper](https://arxiv.org/abs/2208.12242) which introduced Dreambooth.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/dreambooth.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
source: Google Research
</div>

I first tried using the example included in the HuggingFace Diffusers library. I inputted my images along with their captions and began the training. Below are some of the results.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/bad1.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/bad2.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/bad3.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, formal (added pencil sketch to last one)
</div>

I was surprised by the low quality of these images, and I was initially unsure why these images were such low quality. I realized what the issue was when I used a different training script. I then used TheLastBen's training script
and there were several things different about this training script. It encouraged creating a class images folder. The base class images are images of the class of the subject but not the subject itself. For this use case, I am the subject, so the class
would be a person. I generated class images using the prompt "portrait of a man, photorealistic". Below are some of the results of this model.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/frizzy.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, smiling
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/jungle.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, in jungle
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/indian.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, indian
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/punk.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/punk5.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/sajh.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, wearing sweatshirt
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/punk9.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/punk7.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, wearing sweatshirt, bubblegum background, punk
</div>



These are much better! These images clearly show the core facial features of my face, unlike the previous scripts. The base classification images seem to help the model quite a bit. TheLastBen's scripts also have many comments to understand what each of the parameters does.
Increasing the number of epochs of the training improved the quality of the images significantly as well. 

I wanted to see if I could get the model to output a portrait of me in different styles. Below are some examples of pencil sketches, anime style, and comic book style. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/image.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, smiling, anime style
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/punk3.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, smiling, comic book style
</div>


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/another.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, smiling, pencil sketch
</div>

These are not what I was hoping for! It seems the model is unable to switch to a different style and maintain the subject's core features. When I prompt for anime style it simply outputs a typical anime character that does not look like me at all. 
I thought a possible reason may be the base class images. The class images show the model what kind of images it should produce. I trained a new model using pencil sketch pictures as the base class images. The results are below.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/closeup.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/muscular.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, wearing sweatshirt, bubblegum background, punk
</div>

As you can see, these images are still not what I was hoping for. The subject does not look like me at all. Despite the model not working well for particular styles like "anime style" and "pencil sketch", it does work very well for other styles like "punk", "impressionist", and "oil painting". 
Some examples are below.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/dramatic2.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/dramatic3.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/dramatic5.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, oil painting, dramatic
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/punk4.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, wearing sweatshirt, animated, punk
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/impressionist1.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/impressionist2.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/impressionist4.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of jrdnpl man, 20 year old man, oil painting, impressionist
</div>

I think these styles work because they are much closer to the styles present in the input images. All of the input images could be described as "photorealistic, hd". To produce images in a specific style, it is possible to train stable diffusion on a style as well as a subject.
This could be promising and also very useful -- If you see a style of animation you like you could potentially create your image in that style by training stable diffusion. Despite, some of these pictures turning out very well, there are some issues which are present in many of them.
In many photos, I am given elephant ears and my hair is usually very bushy. For a professional artist or someone who wants their image to be exactly a specific kind of way, it would be nice if you could take an image produced by stable diffusion and then modify it to your needs.
This is where a tool called ControlNet has a lot of potential.

## ControlNet

ControlNet is a method for controlling image diffusion models by providing the model with an image as a reference in the last few resolution steps. This method was originally introduced in a [paper](https://arxiv.org/abs/2302.05543) published in February 2023. The most common use of ControlNet is with a Canny image. There are several other types of images you can feed into a ControlNet model
which are all very interesting like depth, segmentation, and human pose. But for my use case, I think Canny is the most appropriate.
A canny image is an image with only the edges turned on, all the other pixels are black. Below is an example of a canny image.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/Canny.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Source: Sofiane Sahir on medium
</div>

The Canny image is fed into the model in the last few steps of resolution causing the output image to be very similar to the original image. By feeding in the Canny, the idea is that the model must conform to the core features of the original images, but has the freedom to create something original between the edges.
Below is an example where I use my image and then give the model the prompt "Mona Lisa". The model will try to show me the Mona Lisa but it must conform to features of the canny image.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/ControlNetmona.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: Mona Lisa
</div>

That worked decently well! Now I will try the thing I was interested in earlier. Try to make the portrait of myself but in an interesting animation style. Below are some initial results.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/ControlNetinit1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of a man, anime style
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/ControlNetinit2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of a man, anime style
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/ControlNetinit3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of a man, comic book style
</div>


Those are ok, and certainly much better than the Dreambooth results. Since many of these seem underdeveloped and without color I will increase the number of resolution steps to see if that makes an improvement.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/comicbook1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of a man, comic book style
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/Controlnet2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of a man, anime style
</div>


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/Controlnet3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of a man, anime style
</div>


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/Controlnet4.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Prompt: portrait of a man, comic book style
</div>

Ok! These are much better and quite good. There are many other ControlNet techniques I could try using but I'll stop here for now and explore ControlNet further at a later time. There is no question that ControlNet is a very powerful technique.

## AUTOMATIC1111

At the beginning of this process, I was calling the inference of the model using the HuggingFace Diffusers library in Colab. Although this was serviceable, it caused some difficulties and inconveniences. I had to check what parameters I could put into the pipeline function
and had to manually change the parameters in a way that was not very user-friendly. Eventually, I learned of different user interfaces that can be used for image generation workflows. The most popular one is called AUTOMATIC1111 or A1111 and it is a web-based UI that 
makes producing images and altering parameters much easier. There are other similar open-source UIs available but A1111 is by far the most popular. I had not heard of it before diving into this topic, but this project was one of the biggest open-source successes in the last year
with over 500 different contributors. 

## Conclusions

Throughout this process of trying to create an interesting profile picture for myself, I became familiar with several very powerful tools and techniques. I used stable diffusion (SDXL) as the pretrained image generation model. I used Dreambooth to 
train stable diffusion on my face so I could use my face as a concept in new images. I also began using ControlNet to make modifications to generated images and have more control over the output. Simultaneously, I began using the A1111 user interface which is 
immensely useful in the image generation workflow. Based on my research and asking questions online, this is a very common stack of tools, and workflow to use among many AI art people. Of course, new techniques and tools are being added as we speak because the 
field is advancing so quickly. There is another popular technique called Segment Anything (SAM) from 2023 which is also very popular. Diffusion models for video generation and video editing are also advancing rapidly. 

The advantage of all of the tools that I used is that they are open source. As long as you have access to a gpu you can use any of these techniques for free. A downside to these tools is their disjointedness and ease of use. It is certainly not straightforward how to proceed
if you are trying to create an AI image using only open-source software. There are closed-source solutions for AI art as well. The two major models which compete with Stable Diffusion are Midjourney and DALL-E. I have not tried either of these paid options and I do not know
if these platforms allow users to use Dreambooth and ControlNet in a user-friendly way. I would doubt it since both of these methods are still fairly new and there has not been enough time to integrate these methods into a smooth UI. It is also relevant to note that
many AI images that are customized or have specific requirements use open-source software because open-source methods allow for much more customization. They also allow you to use the newest image generation techniques such as ControlNet, and SAM without having to wait for a closed-source
platform to offer support. However, for general-purpose AI images -- images that do not need to be in a specific style, or contain a specific subject -- Midjourney and DALL-E can perform on par or better than Stable Diffusion. They are also much easier to use. For general use I would say they are all in the
same range of capability. Below are some images fromMidjourney v6, DALL-E 3, and Stable Diffusion 3 to compare.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/midjourneyv6.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Modjourney v6 is known for the amount of detail, and realism of its images (Source: Ars Technica)
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/DallE3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
DallE3 showing nuclear war (Source: stable-diffusion-art.com)
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/profile-pic/sd3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Stable Diffusion 3 came out this week and touts it capability to include text in the image. The prompt for this image was: "Epic anime artwork of a wizard atop a mountain at night casting a cosmic spell into the dark sky that says 'Stable Diffusion 3' made out of colorful energy" (Source: Stability AI)
</div>

Another important thing to note is that creating these images was not completely free of charge for me. I do not have a GPU so I bought 100 GPU hours for $10 on Google Colab. I trained a total of 4 models and did inference many times. After my usage the last
couple weeks I am left with 65.4 GPU hours. So my total cost for this post was $3.46.

For future work, I am very interested in diving deeper into the architecture of Stable Diffusion models, as well as exploring further how Dreambooth and ControlNet work. ControlNet especially seems to be a very promising technique since it gives so much control to the user. 
I will certainly continue playing around with ControlNet as I learn more about diffusers and image generation. 

