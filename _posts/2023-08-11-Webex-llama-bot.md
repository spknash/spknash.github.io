---
layout: post
title:  Webex llama bot
date:   2023-08-11 
description: Webex chatbot using llama.cpp and ngrok
tags: chat-bot, llama, networking, colab
categories: 
---

# Webex chatbot to increase office interaction

This project started off as the project my team did for the intern hackathon at my summer internship. The topic of the hackathon was build something that improves remote and hybrid work. Our team decided that one of the biggest things lost during remote work is in person interaction and social interaction. It is very hard to become friends with someone when you don't see them in person. Even harder to become friends when the only topic of conversation between you and them is a specific work task within the context of a work meeting. In person there are a lot more casual interactions in the hallway and especially during lunch that allow co-workers to become much closer to each other.

Our idea was to create a application which pairs two random employees in the company for a 30 minute chat each week. I won't talk about the implementation of the full application but only about what I did. I worked on all parts related to Webex. We needed to use the Webex API to automatically schedule meetings between two people in the org. We also decided to incorporate a webex chatbot which could do things like provide background information about each participant in the meeting and also do things like offer a potential fun meeting topic if the participants don't know what to talk about. As I was making the chatbot I realized its not a very good chatbot if it only responds to certain commands. Like below:  

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/bot-help.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    
</div>

I wanted to use a Large language model to allow the bot to respond to messages more dynamically. The only issue is I don't want to spend any money on this because this a purely for fun project and many llms require payment to either use their apis or require the use of cloud credits to host on a cloud platform. I arrived at an idea based on something I used to build the webex chatbot.



## Webex chatbot

The webex chatbot was made using this template: [github link](https://github.com/WebexSamples/webex-bot-starter). I cloned this repo to my local and began to make changes. In order for the bot to work it needs to be able to communicate over http, so I used ngrok to expose the bot application to the internet.
###What is ngrok and how does it work?
When developing a web application, you typically run a local server that is only accessible on your personal computer. However, there might be instances where you need to share this local server with someone else on the internet, perhaps for testing, demonstrations, or external integrations like webhooks.

This is where ngrok comes in. By running a simple command in your command line interface, ngrok provides you with a public URL (HTTP/HTTPS) that forwards incoming requests to your local server. This URL can be shared with anyone, and they'll be able to access your locally running application as if it were hosted online.

### General Use Cases:
Testing on Different Devices: If you want to test how your application appears on different devices, ngrok makes it easy by allowing those devices to access your local server through the public URL.

Collaborating with Team Members: If you're working with team members who are not on the same local network, you can use ngrok to give them access to your development environment.

Webhook Development and Testing: Many third-party services use webhooks to communicate with your application. Ngrok allows these services to connect with your local server, simplifying the process of developing and testing webhooks.

Sharing a Demo with Clients: If you want to share a live demo of an application that's still in development, ngrok enables you to do so without having to deploy it to a public server.

Temporary Hosting for Hackathons or Prototyping: Quick prototyping or participation in hackathons often requires temporary public access to your local development. Ngrok provides a swift solution for these scenarios.

In conclusion, ngrok is an essential tool for modern web development, offering a convenient way to share your local development environment with others. Its ease of use and wide range of applications make it an indispensable resource for developers seeking to streamline their workflow and collaboration efforts.


## Trying to use llama.cpp
So my webex bot at this point works well with limited structured commands but I wanted the catch-all case to give a best response using a large language model instead of not being able to handle a novel response. The first LLM I thought of using was llama because it seems to be the best performing open source langauge model at the moment but also because of llama.cpp which is a implementation of llama in C which is compact enough thata it can be run on Mac M1 or M2 chips. I forked llama.cpp and tried using it on local but the speed at which the response was very slow and it is also probably not great to run something so intensive on my Mac even if it was just for a hackathon/demo.

The next idea was to use llama.cpp on colab -- and use ngrok to expose llama to the internet so it can be used essentially as an inference api for any application I want to build.

##Using llama.cpp and ngrok on Colab

Now I can use llama.cpp to create an app on colab and expose it to the internet to my webex bot can use it as a inference API to generate responses. To use llama.cpp on colab I need to use llama-cpp-python which is python bindings for llama.cpp. I also use langchains to do some minor prompt templating.

The imports:


```python
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
```

The simple prompt template using langchains:


```python
template = """{question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])
```

This command declares a callback manager which allows the llm to produce tokens one by one chat-gpt style.


```python
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
```

This command gets the ggml file needed to use llama.cpp


```python
!wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin
```

This next block of code sets the model parameters and creates the inference function: llm


```python
n_gpu_layers = 100  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="llama-2-13b-chat.ggmlv3.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,
)
```

Now that we have the inference function we can create a app and expose to the internet using ngrok. First I test llm to see how fast it is.


```python
prompt = """
Are there hills in Peru?
"""
llm(prompt)
```



```
Yes, Peru is home to a variety of landscapes including hills, mountains,
 plains, and coastal areas. The Andes Mountains run through Peru, giving rise
 to many high peaks and rolling hills. These geographical features create
 diverse ecosystems throughout the country, from the high-altitude grasslands
 of the Andean Plateau to the arid coastal plains.
```



It runs much faster than on local and it is a necesary improvement because it was way too slow on local. Now onto the app.  Initially I had trouble using ngrok, this was because I hadn't authenticated my ngrok token and to do that I needed to run these commands:


```python
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
!./ngrok authtoken *my-token*
```

I made a flask app and which on a POST request to /generate, retrieves the prompt and generates the response, and packages the response in json and sends it back.


```python
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
run_with_ngrok(app)  # Start ngrok when app is run

# Assuming llm is already defined and loaded elsewhere in your code
# and you can get a response by calling llm(prompt)
@app.route('/')
def home():
    print("home page")
    return 'Hello, World!'
@app.route('/generate', methods=['POST'])
def generate():
    print("generating response ...")
    data = request.json
    prompt = data['prompt']
    print(f"Received prompt: {prompt}")  # Print the received prompt
    response = llm(prompt)
    print(f"Generated response: {response}")  # Print the generated response

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()
```

Lets say this app runs on abc.ngrok-free.app. Now since this app is exposed to the internet I can make a POST request to this app in the app hosting the Webex bot.


```python
import requests
import json
import sys

prompt = sys.argv[1] # Retrieve the prompt from command-line arguments
url = "abc.ngrok-free.app/generate"
payload = {"prompt": prompt}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(payload), headers=headers)

if response.status_code == 200:
    print(json.dumps(response.json()))
else:
    print(f"Failed to make request. Status code: {response.status_code}")
```

This script communicates with the main javascript file which produces the bot responses


```python
const { exec } = require('child_process');

    const prompt = trigger.text;
    const scriptPath = './make_post.py'; // Make sure to provide the correct path to the script

    exec(`python ${scriptPath} "${prompt}"`, (error, stdout, stderr) => {
      if (error) {
        console.error(`An error occurred: ${error}`);
        return;
      }
      const res = stdout.trim();
      console.log(`Response from Python: ${res.response}`);

      bot.say(`Sorry, I don't know how to respond to "${trigger.text}" but llama might`)
        .then(() => bot.say("markdown", res))
        // .then(() => sendHelp(bot))
        .catch((e) => console.error(`Problem in the unexepected command hander: ${e.message}`));
    });
```

And lets see if that works:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/bot-llama-response.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    
</div>

Yes it does!
## Final thoughts
So this project started off as a fun way to get more interaction in a remote workplace, but I think the most important thing I found here is how to create my own inference api using a very high quality LLM in llama v2 using ngrok and colab. I think soon this may not even be an issue, people are finding ways to compact these opensource language models so that they can be run on local computers and even micro controllers like Raspberry Pis. I am excited to see where that goes.

This exposed me to some interesting concepts like ngrok for application development and the trend towards models being made light weight to run on devices is also very interesting -- tiny ML as they call it. And I'll probably go deeper into these topics later.
