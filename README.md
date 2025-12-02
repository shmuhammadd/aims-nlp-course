# Natural Language Processing & Large Language Models  


This course introduces Natural Language Processing (NLP) and transformer-based Large Language Models (LLMs). Students will explore foundational NLP concepts, including tokenization, word embeddings, and language modelling. They will learn the core mechanics of LLMs, such as architecture, training, fine-tuning, reasoning, evaluation, and deployment strategies. The curriculum includes practical applications such as text classification, machine translation, summarization, and zero-/few-shot prompting. 

Through hands-on work with real-world datasets, students will design NLP pipelines and evaluate model performance in multilingual settings, with particular emphasis on low-resource and under-represented languages. By the end of the course, students will also build a simple language model from scratch.



## **Part  A: Natural Language Processing**


| Lecture | Title                                   | Resources                                                                                                                                                                                                  | Suggested Readings                                                                                                                                                                                                 |
|---------|-------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1**   | Introduction to NLP and LLMs              | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/01_NLP_Lecture.pdf)                                                                                                                | 1. [Natural Language Processing: State of the Art, Current Trends and Challenges](https://arxiv.org/pdf/1708.05148) <br> 2. [The Rise of AfricaNLP: Contributions, Contributors, and Community Impact (2005–2025)](https://arxiv.org/pdf/2509.25477) |
| **2**   | How Language Modelling Started (N-grams)  | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/02_NLP_Lecture.pdf) <br><br> [Practical ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/ngram_language_models_class.ipynb) <br><br> [Exercise ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/ngram_practice_huggingface.ipynb) | 1. [Jurafsky & Martin — *Speech and Language Processing*, Chapter 3](https://web.stanford.edu/~jurafsky/slp3/ed3book_aug25.pdf) <br> 2. Rosenfeld (2000) — [Two Decades of Statistical Language Modeling: Where Do We Go from Here?](https://www.cs.cmu.edu/~roni/papers/survey-slm-IEEE-PROC-0004.pdf) |
| **3**   | Text Classification                       | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/03_NLP_Lecture.pdf) <br><br> [Intro to PyTorch ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/pytorch_intro_notebook.ipynb) | 1. [Jurafsky & Martin — Speech and Language Processing, Chapter 4](https://web.stanford.edu/~jurafsky/slp3/ed3book_aug25.pdf) <br> 2. Muhammad et al. (2022) — [AfriSenti: Sentiment Analysis for African Languages](https://arxiv.org/pdf/2302.08956)  <br> 3. [Learn PyTorch for Deep Learning: Zero to Mastery](https://www.learnpytorch.io) |
| **4**   | Word Vectors                              | [Slide](https://github.com/shmuhammadd/aims-nlp-course/blob/main/slides/04_NLP_Lecture.pdf) <br><br>[Training Embeddings ![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/word_embeddings_lab.ipynb)                                                                                                                                                                                                         | 1. Mikolov et al. (2013) — [Efficient Estimation of Word Representations](https://arxiv.org/pdf/1301.3781) <br> 2. Mikolov et al. (2013) — [Linguistic Regularities in Continuous Space Word Representations](https://arxiv.org/pdf/1310.4546) |
| **5**   | Sequence Modelling                 | —                                                                                                                                                                                                           | 1. Goodfellow et al. — *Deep Learning*, Chapter 6 <br> 2. Goldberg (2016) — [Neural Network Models for NLP](https://arxiv.org/pdf/1510.00726) |

## **Part B: Large Language Models**

| Lecture | Title                                   | Resources | Suggested Readings |
|---------|-------------------------------------------|-----------|---------------------|
| **6**   | Introduction to Transformers              | —         | 1. [Vaswani et al. (2017) — Attention is All You Need](https://arxiv.org/pdf/1706.03762) <br> 2. [Alammar — Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| **7**   | Pretraining Objectives (MLM, CLM, etc.)   | —         | 1. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805) <br> 2. [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165) |
| **8**   | Fine-tuning and Instruction Tuning        | —         | 1. [FLAN: Finetuned Language Models](https://arxiv.org/pdf/2109.01652) <br> 2. [T0: Multitask Prompted Training](https://arxiv.org/pdf/2110.08207) |
| **9**   | Prompting, Reasoning, and CoT             | —         | 1. [Wei et al. (2022) — Chain-of-Thought Prompting](https://arxiv.org/pdf/2201.11903) <br> 2. [Kojima et al. (2022) — Zero-Shot CoT](https://arxiv.org/pdf/2205.11916) |
| **10**  | Evaluation of LLMs (Safety, Bias, Multilinguality) | — | 1. [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/pdf/2211.09110)  |
| **11**  | LLMs for African Languages                | —         | 1.  |
| **12**  | Building an LLM from Scratch              | —         | 1. [nanoGPT Codebase](https://github.com/karpathy/nanoGPT) <br> 2. [Transformer Math & Implementation Notes](https://arxiv.org/pdf/2312.00455) |



## Practical Sessions

#### **Lecture 2: N-grams and Language Modelling**
- Practical: Build Unigram/Bigram Models · Compute Probabilities · Perplexity  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/ngram_language_models_class.ipynb)
- Exercise: Practice and build N-Gram models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/ngram_practice_huggingface.ipynb)


#### **Lecture 3: Introduction to PyTorch**
- Practical: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shmuhammadd/aims-nlp-course/blob/main/practicals/ngram_language_models_class.ipynb)
- What PyTorch is and why it's popular
- Tensors - the fundamental data structure
- Automatic differentiation (autograd)
- Building neural networks
- Training models with a typical training loop
- Working with GPUs
- Build your first classifier


## Course Project


The course project will be aligned with [SemEval 2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization](https://polar-semeval.github.io). This shared task includes **three subtasks**, and **each student is required to participate in all three** as part of the project.

### Project Requirements  
- **Individual Work:** Each student will work independently on the entire task.  
- **System Development:** You are expected to design, implement, and evaluate your own model(s) for all three subtasks.  
- **System Description Paper:** You must write a **minimum 4-page system description paper** detailing your methodology, experiments, and results.  
- **Evaluation and Ranking:** Students will be ranked based on the **accuracy** of their submitted systems in the competition.

After the course, students may continue refining their system and paper. You are encouraged to further improve your work and submit the final version to the SemEval Workshop, which will be co-located with ACL2025 in San Diego, USA. 

Follow the following guidelines for writting SemEval system papers: 

 1. System paper guide: https://semeval.github.io/system-paper-template.html
 2. Writing a System description Paper: https://github.com/nedjmaou/Writing_a_task_description_paper

Below are sample papers for previous shared task: 

  1. [AILS-NTUA at SemEval-2025 Task 4: Parameter-Efficient Unlearning for Large Language Models using Data Chunking](https://aclanthology.org/2025.semeval-1.184.pdf)
  2. [AAdaM at SemEval-2024 Task 1: Augmentation and Adaptation for Multilingual Semantic Textual Relatedness](https://aclanthology.org/2024.semeval-1.114.pdf)
  3. [DAMO-NLP at SemEval-2023 Task 2: A Unified Retrieval-augmented System for Multilingual Named Entity Recognition](https://aclanthology.org/2023.semeval-1.277.pdf)

#  Resources  

1.  **Speech and Language Processing** – Jurafsky & Martin ([Online Draft](https://web.stanford.edu/~jurafsky/slp3/))  
2.  [Hands-On Large Language Models: Language Understanding and Generation](https://www.amazon.in/Hands-Large-Language-Models-Understanding/dp/935542552X/ref=pd_sbs_d_sccl_1_1/521-7549942-9569643?pd_rd_w=Ueibj&content-id=amzn1.sym.6d240404-f8ea-42f5-98fe-bf3c8ec77086&pf_rd_p=6d240404-f8ea-42f5-98fe-bf3c8ec77086&pf_rd_r=Z9BASYAF4RW1MVP0D173&pd_rd_wg=ZUKds&pd_rd_r=95ab3bb8-4c74-458a-8089-fa654d4b720c&pd_rd_i=935542552X&psc=1) 
3.  [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
4.  [LLM-course](https://github.com/mlabonne/llm-course)  
5. **Natural Language Processing with Python** – Steven Bird, Ewan Klein, Edward Loper ([Free Online](https://www.nltk.org/book/))  
6. **Transformers for Natural Language Processing** – Denis Rothman  
7. **Deep Learning for NLP** – Palash Goyal, Sumit Pandey, Karan Jain  
8. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** – Aurélien Géron  
