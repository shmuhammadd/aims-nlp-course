# Natural Language Processing & Large Language Models  


This course introduces Natural Language Processing (NLP) and transformer-based Large Language Models (LLMs). Students will explore foundational NLP concepts, including tokenization, word embeddings, and language modelling. They will learn the core mechanics of LLMs, such as architecture, training, fine-tuning, reasoning, evaluation, and deployment strategies. The curriculum includes practical applications such as text classification, machine translation, summarization, and zero-/few-shot prompting. 

Through hands-on work with real-world datasets, students will design NLP pipelines and evaluate model performance in multilingual settings, with particular emphasis on low-resource and under-represented languages. By the end of the course, students will also build a simple language model from scratch.



## **Part  A: Natural Language Processing**


| Lecture | Title                                   | Resources                                                                                                                                                                                                 | Suggested Readings                                                                                                                                                                             |
|---------|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1**   | Introduction to NLP and LLMs              | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/01_NLP_Lecture1..pdf)                                                                                                          | 1. [Natural Language Processing: State of The Art, Current Trends and Challenges](https://arxiv.org/pdf/1708.05148) <br> 2. [The Rise of AfricaNLP: Contributions, Contributors, and Community Impact (2005–2025) ](https://arxiv.org/pdf/2509.25477)                                             |
| **2**   | How Language Modelling Started (N-grams)   | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/02_NLP_Lecture2.pdf) <br><br> [Practical ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/ngram_language_models_class.ipynb) <br><br> [Exercise ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/ngram_practice_huggingface.ipynb) | 1. [Jurafsky & Martin, Chapter 3: N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/) <br> 2. [Two decades of statistical language modeling: Where do we go from here?](https://www.cs.cmu.edu/~roni/papers/survey-slm-IEEE-PROC-0004.pdf) |
| **3**   | Text Classification                       | —                                                                                                                                                                                                          | 1. [Zhang et al. (2015) — Character-level CNN](https://arxiv.org/pdf/1509.01626) <br> 2. [Joulin et al. (2017) — Bag of Tricks for Text Classification](https://arxiv.org/pdf/1607.01759)         |
| **4**   | Word Vectors                              | —                                                                                                                                                                                                          | 1. [Mikolov et al. (2013) — Efficient Estimation of Word Representations](https://arxiv.org/pdf/1301.3781) <br> 2. [Mikolov et al. (2013) — Compositionality](https://arxiv.org/pdf/1310.4546)   |
| **5**   | Neural Networks                           | —                                                                                                                                                                                                          | 1. [Goodfellow et al. — Deep Learning (Chapter 6)](https://www.deeplearningbook.org/) <br> 2. [Goldberg (2016) — Neural Network Models for NLP](https://arxiv.org/pdf/1510.00726)               |



## **Part B: Large Language Models**

| Lecture | Title                                   | Resources | Suggested Readings |
|---------|-------------------------------------------|-----------|---------------------|
| **6**   | Introduction to Transformers              | —         | 1. [Vaswani et al. (2017) — Attention is All You Need](https://arxiv.org/pdf/1706.03762) <br> 2. [Alammar — Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| **7**   | Pretraining Objectives (MLM, CLM, etc.)   | —         | 1. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805) <br> 2. [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165) |
| **8**   | Fine-tuning and Instruction Tuning        | —         | 1. [FLAN: Finetuned Language Models](https://arxiv.org/pdf/2109.01652) <br> 2. [T0: Multitask Prompted Training](https://arxiv.org/pdf/2110.08207) |
| **9**   | Prompting, Reasoning, and CoT             | —         | 1. [Wei et al. (2022) — Chain-of-Thought Prompting](https://arxiv.org/pdf/2201.11903) <br> 2. [Kojima et al. (2022) — Zero-Shot CoT](https://arxiv.org/pdf/2205.11916) |
| **10**  | Evaluation of LLMs (Safety, Bias, Multilinguality) | — | 1. [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/pdf/2211.09110) <br> 2. [Adelani et al. — MasakhaEval](https://arxiv.org/pdf/2307.09982) |
| **11**  | LLMs for African Languages                | —         | 1. [AfriLLM Survey](https://arxiv.org/pdf/2402.05852) <br> 2. [IrokoBench: Cultural & Safety Evaluation for African Languages](https://arxiv.org/pdf/2503.04518) |
| **12**  | Building an LLM from Scratch              | —         | 1. [nanoGPT Codebase](https://github.com/karpathy/nanoGPT) <br> 2. [Transformer Math & Implementation Notes](https://arxiv.org/pdf/2312.00455) |





## Practical Sessions

#### **Lecture 2: N-grams and Language Modelling**
- Practical: Build Unigram/Bigram Models · Compute Probabilities · Perplexity  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/ngram_language_models_class.ipynb)
- Exercise: Practice and build N-Gram models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/ngram_practice_huggingface.ipynb)


## Projects and Quizzes


#  Resources  

1.  **Speech and Language Processing** – Jurafsky & Martin ([Online Draft](https://web.stanford.edu/~jurafsky/slp3/))  
2.  [Hands-On Large Language Models: Language Understanding and Generation](https://www.amazon.in/Hands-Large-Language-Models-Understanding/dp/935542552X/ref=pd_sbs_d_sccl_1_1/521-7549942-9569643?pd_rd_w=Ueibj&content-id=amzn1.sym.6d240404-f8ea-42f5-98fe-bf3c8ec77086&pf_rd_p=6d240404-f8ea-42f5-98fe-bf3c8ec77086&pf_rd_r=Z9BASYAF4RW1MVP0D173&pd_rd_wg=ZUKds&pd_rd_r=95ab3bb8-4c74-458a-8089-fa654d4b720c&pd_rd_i=935542552X&psc=1) 
3.  [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
4.  [LLM-course](https://github.com/mlabonne/llm-course)  
5. **Natural Language Processing with Python** – Steven Bird, Ewan Klein, Edward Loper ([Free Online](https://www.nltk.org/book/))  
6. **Transformers for Natural Language Processing** – Denis Rothman  
7. **Deep Learning for NLP** – Palash Goyal, Sumit Pandey, Karan Jain  
8. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** – Aurélien Géron  

