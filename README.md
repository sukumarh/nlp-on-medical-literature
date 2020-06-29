# Utilizing Untagged Medical Literature for Diagnoses
This project aims to establish and evaluate a methodology to computationally consume medical literature and draw certain results from it. We intend to construct the project around a symptom-disease paradigm, employing NLP techniques to traverse through large quantities of textual data and extract a disease diagnosis from a given set of symptoms.


#### ABOUT
We intend to utilize untagged, unstructured literature and extract information and evaluate our findings on them. We incorporated the concepts of Word Embeddings and the working of Word2Vec to vectorize the texts.


#### Related Work
We referred to the following literature when considering the possible methodologies, we could adopt to approach the problem.
Nye B, Jessy Li J, Patel R, et al. A Corpus with Multi-Level Annotations of Patients, Interventions and Outcomes to Support Language Processing for Medical Literature. Proc Conf Assoc Comput Linguist Meet. 2018;2018:197‐207. ([Read here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6174533/))


#### METHODOLOGY
We used the word2vec algorithm provided in the Gensim library for training. Using a subsection of the literature we trained several Word2Vec models to create word embeddings and tested the resulting vectors by hand, manually checking similarities between different words and assessing which model provided the most accurate embeddings. Training multiple models allowed us to tune the hyper-parameters which could then be used over the entire dataset.

> During our work we noticed the need to perform some formatting such as centering the different references to COVID-19 (such as Coronavirus) to one simple string ‘covid19’. We noticed improvements in our findings once we re-trained our model using the cleaned data.

However, this incurred a dependency in the dataset, whenever a symptom would be associated with the disease, the occurrence of the word ‘positive’ or something alike was essential. Additionally, using only a single word is not very useful in describing symptoms of a disease. When describing symptoms, saying a patient has high temperature instead of temperature gives much more value to the input. Including descriptors such as these could prove useful in our tool.

Our next goal was to then create not word embeddings, but phrase embeddings. Our first approach to this task was to create bi-grams with our unsupervised corpus and take it as an input to Word2Vec.


#### DATASET
[COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)


#### REFERENCES
1. [Word2Vec For Phrases — Learning Embeddings For More Than One Word](https://towardsdatascience.com/word2vec-for-phrases-learning-embeddings-for-more-than-one-word-727b6cf723cf)
2. [Nye B, Jessy Li J, Patel R, et al. A Corpus with Multi-Level Annotations of Patients, Interventions and Outcomes to Support Language Processing for Medical Literature. Proc Conf Assoc Comput Linguist Meet. 2018;2018:197‐207.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6174533/)
3. [Clinical Data Science](https://clinicaldatascience.mountsinai.org/)


#### BUILT WITH
* [Jupyter Notebook](https://jupyter.org/)
