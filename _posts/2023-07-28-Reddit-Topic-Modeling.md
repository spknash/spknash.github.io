---
layout: post
title:  Topic Modeling in Reddit
date:   2023-07-28 
description: Supervised and Unsupervised Topic modeling on r/changemyview
tags: intro
categories: 
---

# Most Common Topic in r/changemyview

In this post I am tackling the problem: What is the most talked about issue/topic on r/changemyview right now?

r/changemyview is a a subreddit where users post an opinion they have about a certain issue or topic and people post replies which try to pursuade the original poster to change their view. The original poster can then award delta points to the reply which pursuades them to change their view, or comes close by presenting a very good argument. Due to moderation and community rules, every single post has this same format. A opinion, and then replies which try to pursuade away from that opinion, and the replies which do an exceptional job receive "delta points". Due to the structure and high quality of posts in this subreddit, r/changemyview is an ideal source of data for machine learning projects. Below is picture of what a r/changemyview post looks like.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/r:changemyview.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    
</div>


For the problem I am trying to solve we only need the post titles and not the full scope of the data provided in the sub. The post title contains an opinion on a topic/issue and this is enough to determine the what topic the post is addressing. The first step to solve this problem no matter the strategy chosen is to fetch post titles over the timeframe you are interested in. Below is code for fetching titles from the top 50 post titles since last year as a sample. ```reddit``` is a read/write reddit instance that has been initialized with my reddit api credentials.




```python
for submission in reddit.subreddit("changemyview").top(limit=50):
      print(submission.title)
```

From here the question becomes how do we use this collection of post titles to determine the most common topic. I try a few different methods.
##Manual Labels & Zero-shot Classification

The idea behind this first method is to manually provide several potential topics which we think are popular and use zero-shot classification to classify each post title. Either zero shot classification or trained classifier could be used but I thought zero shot classification(specifically the hugging face pipeline) might work reasonably well especially when the categories are very diverse. After manually scanning through some of the r/changemyview titles, here are the categories I came up with:


```python
candidate_labels=["education", "taxes", "Trump", "healthcare", "Religion", "elections", "race", "LGBTQ"]
```

After that I used zero-shot classification in hugging-face pipelines to classify each title and here were the results of 5 posts to see if it was working as well as I wanted it to:



```
CMV: Mike Bloomberg's campaign is proof that the ultra wealthy in the US can afford a higher tax rate with no ill effect on them
[0.8761507272720337, 0.039841409772634506, 0.028353260830044746, 0.025303443893790245, 0.010511195287108421, 0.007160380948334932, 0.006719440221786499, 0.005960128735750914]
CMV: Kanye West is a shill for president Trump and running to syphon off young voters from voting for Biden.
[0.3772038221359253, 0.30558013916015625, 0.2326604723930359, 0.038126397877931595, 0.012778099626302719, 0.012653850950300694, 0.011820374056696892, 0.009176867082715034]
CMV: Most Americans who oppose a national healthcare system would quickly change their tune once they benefited from it.
[0.8781680464744568, 0.03057781793177128, 0.028276970610022545, 0.018674472346901894, 0.012455514632165432, 0.012006503529846668, 0.010698405094444752, 0.009142286144196987]
CMV: Donald Trump has not made a single lasting positive impact on the USA during his term as president.
[0.9717963933944702, 0.008201655931770802, 0.006410819478332996, 0.0034016254357993603, 0.0033515936229377985, 0.0030098608694970608, 0.0021523970644921064, 0.0016757362755015492]
CMV: being a conservative is the least Christ-like political view
[0.24276088178157806, 0.15054336190223694, 0.12963169813156128, 0.1104813814163208, 0.10187441110610962, 0.09882443398237228, 0.08833231031894684, 0.07755151391029358]
```



A quick look at how zero-shot classification is doing in these few examples reveals that it is not working that well. The topic "CMV: Donald Trump has not made a single lasting positive impact on the USA during his term as president." is classified as 0.97 towards the topic ```education``` even though this topic has the word "Trump" in it and ```Trump``` is one of the categories. That is concerning but could be because the model doesn't have embeddings for the word "Trump" as in president Trump instead of just the dictionary word. The same thing happens for "CMV: Most Americans who oppose a national healthcare system would quickly change their tune once they benefited from it.". The classifier chooses ```education``` with 0.87 probability even though ```healthcare``` exists as an option.

Clearly, this strategy of manual labels and zero-shot classification is not effective. The next strategy I could explore is still within supervised topic modeling, but instead of zero-shot classification we use a classifier trained specifically for this classification task. The big problem with this method is there is no obvious way to get labels for each post title in the training dataset -- manually would take very long. So instead of going in this route I will explore unsupervised topic modeling because there is no clear path forward within supervised modeling, and unsupervised modeling seems more interesting anyways because we can let the model identify distinctions between topics. So I will look at a Bag of Words model first.

## Bag of Words/LDA


I'll walk through how I used the bag of words method and LDA to perform unsupervised topic classification. Afterwards I'll provide a brief overview of how these methods work and the motivation behind it.

The first step is to create a bag of words matrix, which will contain  For each document(each post title in this case) the frequency of each word in the dictionary is recorded in a row vector. The idea behind the bag of words matrix is to capture where words are repeating to provide information about which documents are covering the similar topics. For example if document 1 and document 3 both contain high frequency of the word "cat" they likely cover similar topics. This type of analysis clearly does not depend on many words which appear very frequently in the english language such as "as", "to", or "the". These words are called stop words. Additionally, in this type of analysis there is no real difference between root words and extended words such as "happy" and "happiest". Therefore, all words should be shortened to their shortest stem. Below is the function used to pre-process every post title. It tokenizes, deletes stop words, and shortens to smallest stem. This implementation uses the nltk library which is common for text preprocessing.









```python
def process_document(text):
  tokens = word_tokenize(text)

  # Removing Stop words
  stop_words = set(stopwords.words('english'))
  tokens = [word for word in tokens if word not in stop_words]

  # Stemming
  stemmer = PorterStemmer()
  stemmed_tokens = [stemmer.stem(word) for word in tokens]
  return stemmed_tokens
```

Below is an example to show exactly what the preprocessing is doing:


```python
process_document("hello, my name is Suhaas and I really like frisbee and tennis. What sports do you like?")
```

```
['hello',
 ',',
 'name',
 'suhaa',
 'i',
 'realli',
 'like',
 'frisbe',
 'tenni',
 '.',
 'what',
 'sport',
 'like',
 '?']
```


After preprocessing of the text is complete the bag of words matrix can be made.

The row vector of every document in the corpus are stacked on top of each other to form the Bag of words matrix. The row vector contains the frequency of each word. Below is a simple example. Say the corpus is
```
corpus = ["the dog is wet, the dog is angry",
  "the cat is upset and very hungry",
  "who let the dogs out? They are making a mess",
  "I can't believe the cat drank the milk"]
```
Then the bag of words matrix will be:
```
   angri  believ  ca  cat  dog  drank  hungri  let  make  mess  milk  upset  \
0      1       0   0    0    2      0       0    0     0     0     0      0   
1      0       0   0    1    0      0       1    0     0     0     0      1   
2      0       0   0    0    1      0       0    1     1     1     0      0   
3      0       1   1    1    0      1       0    0     0     0     1      0   

   wet  
0    1  
1    0  
2    0  
3    0
```

This matrix is pretty useful but can be made even more meaningful if rare words are given a greater weight and prevalence in the model. For example, in r/changemyview topics the word "abortion" is very rare. But when it does appear it is almost certainly the topic of the entire post. There are many words like this: words which are rare in the english language but determine the topic of the document when they do appear. This is where TF-IDF comes in. TF-IDF is a method to alter the bag of words matrix to increase the weight on these important words which are rare in the corpus. TF-IDF is composed of two parts:

Term Frequency (TF): This is simply the frequency of a word in a document. It's based on the idea that the importance of a word is proportional to its frequency. However, some words like 'the', 'is', and 'and' appear frequently in all sorts of contexts, so high frequency doesn't always mean high importance. That's where the second part of TF-IDF comes in.

Inverse Document Frequency (IDF): This reduces the weight of words that are common in the corpus. IDF is calculated as the logarithm of the total number of documents in the corpus divided by the number of documents containing the term. Thus, it increases for rare words and decreases for common words.

The overall TF-IDF score for a word in a document is the product of its TF and IDF scores.

Here's the mathematical formula for TF-IDF:
```
TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)
```

where:

```t``` is the term or word
```d``` is the document
```D``` is the corpus
```TF(t, d)``` is the term frequency of t in d (usually normalized by dividing by the total number of words in d)
```IDF(t, D)``` is the inverse document frequency of t in D, calculated as ```log(N / df(t))```, where N is the total number of documents and df(t) is the number of documents that contain t (to prevent division by zero if a word is not in the corpus, it's common to add 1 to the denominator)

This calculation increases the weight of words which are rare in the corpus but frequent in a particular document. Such a word is probably very relevant to the topic of that document. Below is a implementation of creating a bag of words matrix using TF-IDF(this implementation uses the sci-kit library):


```python
def BAG_matrix(corpus):
  # Apply preprocessing to each document in the corpus
  corpus = [" ".join(process_document(doc)) for doc in corpus]

  # Initialize CountVectorizer
  vectorizer = CountVectorizer(ngram_range=(1, 1))

  # Tokenize and build vocab
  X = vectorizer.fit_transform(corpus)

  # Now, we initialize the TfidfTransformer and transform our count-matrix to tf-idf representation
  transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
  tfidf = transformer.fit_transform(X)

  # Output the shape of X
  print(X.shape)

  # To get feature names
  feature_names = vectorizer.get_feature_names_out()

  # To view the matrix as a DataFrame
  import pandas as pd
  df = pd.DataFrame(tfidf.toarray(), columns=feature_names)
  print(df)
  return tfidf, vectorizer
```

And below is the matrix of the same corpus from earlier but using tf-idf now:
```
     angri    believ        ca       cat       dog     drank    hungri  \
0  0.47212  0.000000  0.000000  0.000000  0.744450  0.000000  0.000000   
1  0.00000  0.000000  0.000000  0.486934  0.000000  0.000000  0.617614   
2  0.00000  0.000000  0.000000  0.000000  0.414289  0.000000  0.000000   
3  0.00000  0.465162  0.465162  0.366739  0.000000  0.465162  0.000000   

        let      make      mess      milk     upset      wet  
0  0.000000  0.000000  0.000000  0.000000  0.000000  0.47212  
1  0.000000  0.000000  0.000000  0.000000  0.617614  0.00000  
2  0.525473  0.525473  0.525473  0.000000  0.000000  0.00000  
3  0.000000  0.000000  0.000000  0.465162  0.000000  0.00000  
```


The next step is to use this tf-idf matrix to construct topics. This can be done using Latent Dirichlet Allocation.
We will be using the LDA method in sci-kit but here is an overview of how LDA works:

Initialize LDA: First, initialize the LDA model. One key parameter to set here is the number of topics you want the model to identify. This is a hyperparameter that might need to be tuned to get the best results.

Assign topics to words: LDA starts by randomly assigning each word in each document to one of the K topics (where K is the number of topics you decided on). This random assignment already gives you both topic representations of all the documents and word distributions of all the topics (albeit not very good ones because it is random).

Iteratively update topic assignments: Then, LDA iteratively updates the topic assignments for each word in each document, based on two criteria:

a. How prevalent is the topic in the document? The more often the topic occurs in the document, the more likely it is that the word belongs to this topic.

b. How prevalent is the word across topics? If a word is already often assigned to a topic, it's likely that it will be assigned to this topic again.

Each iteration of this step is done using a method called Gibbs Sampling, which is a type of Markov Chain Monte Carlo (MCMC) algorithm.

The process continues until the model's estimates of the topics stabilize, or after a set number of iterations. Once finished, you'll evaluate the topics that the model has learned. After this iterative process, LDA will represent each document in the corpus as a mixture of different topics(learns a distribution of topics for each document), and represents each topic as a set of top words(learns a distribution of words for each topic). More information about LDA can be found in the paper by Blie, Jordan, and Ng [here.](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

Below is an implementattion using the bag of words matrix from earlier and an LDA method from sci-kit. It produces ```num_topics``` and prints the top 5 words for each topic to allow us to see what each topic is really about.


```python
def LDA_topics(tfidf, vectorizer, num_topics):
  # Initialize LDA
  # n_components specifies the number of topics
  lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)

  # Fit LDA to BoW data
  lda.fit(tfidf)

  # For each topic, print the top 10 most representative words
  for index, topic in enumerate(lda.components_):
      print(f"Top 5 words for Topic #{index+1}")
      print(
          [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]])
      print("\n")
```

This function takes the tfidf matrix and vectorizer as inputs from the bag of words function -- The function below creates the tf-idf matrix and topics using LDA.


```python
def make_topics(corpus, num_topics):
  tfidf, vectorizer = BAG_matrix(corpus)
  LDA_topics(tfidf,vectorizer,num_topics)
```

Now we are almost ready to use this function to generate topics from r/changemyview posts! The last thing left to do is to produce the corpus which will be a list post title from the subreddit. The code below creates the corpus:


```python
reddit_corpus = []
subreddit = reddit.subreddit("changemyview")
after = None
for _ in range(10):
    if after:
        new_posts = subreddit.top(limit=500, params={'after': after})
    else:
        new_posts = subreddit.top(limit=500)

    last_post = None
    for post in new_posts:
        reddit_corpus.append(post.title[4:])
        last_post = post

    after = last_post  # ID of the last post

for submission in reddit.subreddit("changemyview").top(limit=1):
      #print(submission.title[4:])
      reddit_corpus.append(submission.title[4:])
```

For each post I append ```post.title[4:]``` instead of the full post because each post starts with "CMV:" and we don't want CMV to be one of the words which define a topic since it is present in every title. Additionally, the reddit API only allows 60 queries per minute and on 500 posts per query so I had to use the ```params={'after': after})``` parameter to start the next query where the previous one left off.

Now lets try out how the LDA topic generation did.


```python
make_topics(reddit_corpus,15)
```

```
Top 5 words for Topic #1
['work', 'licens', 'conserv', 'the', 'religion']


Top 5 words for Topic #2
['charact', 'the', 'race', 'us', 'racist']


Top 5 words for Topic #3
['noth', 'there', 'sex', 'peopl', 'gender']


Top 5 words for Topic #4
['tri', 'cover', 'parent', 'world', 'sub']


Top 5 words for Topic #5
['joke', 'eat', 'women', 'citizenship', 'the']


Top 5 words for Topic #6
['offens', 'help', 'place', 'polic', 'the']


Top 5 words for Topic #7
['anim', 'much', 'small', 'commun', 'peopl']


Top 5 words for Topic #8
['child', 'flag', 'the', 'peopl', 'chang']


Top 5 words for Topic #9
['parti', 'make', 'peopl', 'vote', 'we']


Top 5 words for Topic #10
['wealth', 'donald', 'if', 'bodi', 'posit']


Top 5 words for Topic #11
['realiti', 'answer', 'appropri', 'cultur', 'thing']


Top 5 words for Topic #12
['hire', 'incom', 'job', 'includ', 'social']


Top 5 words for Topic #13
['consid', 'there', 'health', 'it', 'peopl']


Top 5 words for Topic #14
['homeless', 'there', 'it', 'see', 'wrong']


Top 5 words for Topic #15
['like', 'need', 'cultur', 'us', 'get']
```

Ok! so definetly an improvement from zero-shot learning, but still room for improvement as well. Some topics have top words that are very connected and make sense with how humans would define topics in this corpus. For example topic 3
```
'charact', 'the', 'race', 'us', 'racist'
```
seems to be about race and culture.

Topic 12 is
```
['hire', 'incom', 'job', 'includ', 'social']
```
This topic seems to be about employment and jobs.

And this topic:
```
['noth', 'there', 'sex', 'peopl', 'gender']
```
seems to be about gender and sexuality issues. On that same note there are some topics which don't have a common theme like these ones:
```
['homeless', 'there', 'it', 'see', 'wrong']
['joke', 'eat', 'women', 'citizenship', 'the']
```
This is to be expected and highlights the limitations of LDA. Because it is a unsupervised topic model, the topics may not be formed in the same way a human would. The algorithms may form a topic based on some similarities or frequency of a particular word which we don't see as important. Additionally, LDA does not take into account the order and semantics fo the words. For example, LDA would think "this post is about police" and "this post is not about police" are both about police simply because it contains the word "police". Despite not taking into account semantics and word order it is pretty cool to see it produce some topics the similar to how a human would.

There are also some hyper parameters which could be tuned to produce better results like: number of topics, n-grams during bag of words matrix, number of top words for each topic. Perhaps the results of this topic modeling could be improved a little more just by further tuning these hyper parameters.

I am really interested to see how a model which does take into account the order of words could be used for topic modeling. An example includes BERTopic. There are also many extensions to LDA such as Dynamic LDA, Hierarchical Dirichlet Process, and GuidedLDA. I may explore these other techniques for topic modeling in future posts.

