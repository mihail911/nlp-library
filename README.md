# nlp-library
This is a curated list of papers that I have encountered in some capacity and deem worth including in the NLP practitioner's library. Some papers may appear in multiple sub-categories, if they don't fit easily into one of the boxes.

**PRs are absolutely welcome!** Direct any correspondence/questions to [@mihail_eric](https://twitter.com/mihail_eric).

Some special designations for certain papers:

:bulb: LEGEND: This is a game-changer in the NLP literature and worth reading.

:vhs: RESOURCE: This paper introduces some dataset/resource and hence may be useful for application purposes.


## Part-of-speech Tagging
* (2000) [A Statistical Part-of-Speech Tagger](https://arxiv.org/pdf/cs/0003055.pdf)
  - **TLDR**: Seminal paper demonstrating a powerful HMM-based POS tagger. Many tips and tricks for building such classical systems included. 
* (2003) [Feature-rich part-of-speech tagging with a cyclic dependency network](https://nlp.stanford.edu/pubs/tagging.pdf)
  - **TLDR**: Proposes a number of powerful linguistic features for building a (then) SOTA POS-tagging system
* (2015) [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)
  - **TLDR**: Proposes an element sequence-tagging model combining neural networks with conditional random fields, achieving SOTA in POS-tagging, NER, and chunking. 

## Parsing
* (2003) [Accurate unlexicalized parsing](https://people.eecs.berkeley.edu/~klein/papers/unlexicalized-parsing.pdf) :bulb:
  - **TLDR**: Beautiful paper demonstrating that unlexicalized probabilistic context free grammars can exceed the performance of lexicalized PCFGs.
* (2006) [Learning Accurate, Compact, and Interpretable Tree Annotation](https://pdfs.semanticscholar.org/d84b/9507ff9687a900fde451f27106d930c1b838.pdf)
  - **TLDR**: Fascinating result showing that using expectation-maximization you can automatically learn accurate and compact latent nonterminal symbols for tree annotation, achieving SOTA. 
* (2014) [A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf)
  - **TLDR**: Very important work ushering in a new wave of neural network-based parsing architectures, achieving SOTA performance as well as blazing parsing speeds. 
* (2014) [Grammar as a Foreign Language](https://arxiv.org/pdf/1412.7449.pdf)
  - **TLDR**: One of the earliest demonstrations of the effectiveness of seq2seq architectures with attention on constituency parsing, achieving SOTA on the WSJ corpus. Also showed the importance of data augmentation for the parsing task. 
* (2015) [Transition Based Dependency Parsing with Stack Long Short Term Memory](https://arxiv.org/pdf/1505.08075.pdf)
  - **TLDR**: Presents stack LSTMs, a neural parser that successfully neuralizes the traditional push-pop operations of transition-based dependency parsers, achieve SOTA in the process. 

## Named Entity Recognition
* (2005) [Incorporating Non-local Information into Information Extraction Systems by Gibbs Sampling](http://nlp.stanford.edu/~manning/papers/gibbscrf3.pdf)
  - **TLDR**: Using cool Monte Carlo methods combined with a conditional random field model, this work achieves a huge error reduction in certain information extraction benchmarks.
* (2015) [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)
  - **TLDR**: Proposes an element sequence-tagging model combining neural networks with conditional random fields, achieving SOTA in POS-tagging, NER, and chunking. 

## Coreference Resolution
* (2010) [A multi-pass sieve for coreference resolution](https://nlp.stanford.edu/pubs/conllst2011-coref.pdf) :bulb:
  - **TLDR**: Proposes a sieve-based approach to coreference resolution that for many years (until deep learning approaches) was SOTA.
* (2015) [Entity-Centric Coreference Resolution with Model Stacking](http://cs.stanford.edu/~kevclark/resources/clark-manning-acl15-entity.pdf) 
  - **TLDR**: This work offers a nifty approach to building coreference chains iteratively using entity-level features.
* (2016) [Improving Coreference Resolution by Learning Entity-Level Distributed Representations](https://cs.stanford.edu/~kevclark/resources/clark-manning-acl16-improving.pdf)
  - **TLDR**: One of the earliest effective approaches to using neural networks for coreference resolution, significantly outperforming the SOTA.

## Sentiment Analysis
* (2012) [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://www.aclweb.org/anthology/P12-2018)
  - **TLDR**: Very elegant paper, illustrating that simple Naive Bayes models with bigram features can outperform more sophisticated methods like support vector machines on tasks such as sentiment analysis.
* (2013) [Recursive deep models for semantic compositionality over a sentiment treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) :vhs:
  - **TLDR**: Introduces the Stanford Sentiment Treebank, a wonderful resource for fine-grained sentiment annotation on sentences. Also introduces the Recursive Neural Tensor Network, a neat linguistically-motivated deep learning architecture. 
* (2014) [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
  - **TLDR**: Introduces *ParagraphVector* an unsupervised that learns fixed representations of paragraphs, using ideas inspired from *Word2Vec*. Achieves then SOTA on sentiment analysis on Stanford Sentiment Treebank and the IMDB dataset. 

* (2019) [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/pdf/1904.12848.pdf)
  - **TLDR**: Introduces *Unsupervised Data Augmentation*, a method for efficient training on a small number of training examples. Paper applies UDA to IMDB sentiment analysis dataset, achieving SOTA with only 30 training examples.


## Natural Logic/Inference
* (2007) [Natural Logic for Textual Inference](https://nlp.stanford.edu/pubs/natlog-wtep07.pdf)
  - **TLDR**: Proposes a rigorous logic-based approach to the problem of textual inference called natural logic. Very cool mathematically-motivated transforms are used to deduce the relationship between phrases. 
* (2008) [An Extended Model of Natural Logic](https://dl.acm.org/citation.cfm?id=1693772)
  - **TLDR**: Extends previous work on natural logic for inference, adding phenomena such as semantic exclusion and implicativity to enhance the premise-hypothesis transform process.
* (2014) [Recursive Neural Networks Can Learn Logical Semantics](https://arxiv.org/abs/1406.1827)
  - **TLDR**: Demonstrates that deep learning architectures such as neural tensor networks can effectively be applied to natural language inference. 
* (2015) [A large annotated corpus for learning natural language inference](http://nlp.stanford.edu/pubs/snli_paper.pdf) :vhs:
  - **TLDR**: Introduces the Stanford Natural Language Inference corpus, a wonderful NLI resource larger by two orders of magnitude over previous datasets. 

## Machine Translation
* (1993) [The Mathematics of Statistical Machine Translation](https://www.aclweb.org/anthology/J93-2003) :bulb:
  - **TLDR**: Introduces the IBM machine translation models, several seminal models in statistical MT. 
* (2002) [BLEU: A Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf) :vhs:
  - **TLDR**: Proposes BLEU, the defacto evaluation technique used for machine translation (even today!)
* (2003) [Statistical Phrase-Based Translation](http://dl.acm.org/citation.cfm?id=1073462)
  - **TLDR**: Introduces a phrase-based translation model for MT, doing nice analysis that demonstrates why phrase-based models outperform word-based ones. 
* (2014) [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf) :bulb:
  - **TLDR**: Introduces the sequence-to-sequence neural network architecture. While only applied to MT in this paper, it has since become one of the cornerstone architectures of modern natural language processing. 
* (2015) [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) :bulb:
  - **TLDR**: Extends previous sequence-to-sequence architectures for MT by using the attention mechanism, a powerful tool for allowing a target word to softly search for important signal from the source sentence. 
* (2015) [Effective approaches to attention-based neural machine translation](https://arxiv.org/abs/1508.04025)
  - **TLDR**: Introduces two new attention mechanisms for MT, using them to achieve SOTA over existing neural MT systems.
* (2016) [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)
  - **TLDR**: Introduces byte pair encoding, an effective technique for allowing neural MT systems to handle (more) open-vocabulary translation.
* (2016) [Pointing the Unknown Words](https://www.aclweb.org/anthology/P16-1014)
  - **TLDR**: Proposes a copy-mechanism for allowing MT systems to more effectively copy words from a source context sequence.
* (2016) [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144) 
  - **TLDR**: A wonderful case-study demonstrating what a production-capacity machine translation system (in this case that of Google) looks like. 

## Semantic Parsing
* (2013) [Semantic Parsing on Freebase from Question-Answer Pairs](https://aclweb.org/anthology/D13-1160) :bulb: :vhs:
  - **TLDR**: Proposes an elegant technique for semantic parsing that learns directly from question-answer pairs, without the need for annotated logical forms, allowing the system to scale up to Freebase. 
* (2014) [Semantic Parsing via Paraphrasing](http://aclweb.org/anthology/P14-1133)
  - **TLDR**: Develops a unique paraphrase model for learning appropriate candidate logical forms from question-answer pairs, improving SOTA on existing Q/A datasets. 
* (2015) [Building a Semantic Parser Overnight](https://cs.stanford.edu/~pliang/papers/overnight-acl2015.pdf) :vhs:
  - **TLDR**: Neat paper showing that a semantic parser can be built from scratch starting with no training examples!
* (2015) [Bringing Machine Learning and Computational Semantics Together](http://www.stanford.edu/~cgpotts/manuscripts/liang-potts-semantics.pdf)
  - **TLDR**: A nice overview of a computational semantics framework that uses machine learning to effectively learn logical forms for semantic parsing. 

## Question Answering/Reading Comprehension
* (2016) [A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task](https://arxiv.org/abs/1606.02858)
  - **TLDR**: A great wake-up call paper, demonstrating that SOTA performance can be achieved on certain reading comprehension datasets using simple systems with carefully chosen features. Don't forget non-deep learning methods!
* (2017) [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) :vhs:
  - **TLDR**: Introduces the SQUAD dataset, a question-answering corpus that has become one of the defacto benchmarks used today. 
* (2019) [Look before you Hop: Conversational Question Answering over Knowledge Graphs Using Judicious Context Expansion](https://arxiv.org/abs/1910.03262)
  - **TLDR**: Introduces an unsupervised method that can answer incomplete questions over Knowledge Graph by maintaining conversation context using entities and predicates seen so far and automatically inferring missing or ambiguous pieces for follow-up questions.
* (2019) [Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://arxiv.org/abs/1911.10470)
  - **TLDR**: Introduces a new graph based recurrent retrieval approach, which retrieves reasoning paths over the Wikipedia graph to answer multi-hop open-domain questions.
* (2019) [Abductive Commonsense Reasoning](https://arxiv.org/abs/1908.05739) 
  - **TLDR**: Introduces a dataset and conceptualizes two new tasks for Abductive Reasoning: Abductive Natural Language Inference and Abductive Natural Language Generation.
* (2020) [Differentiable Reasoning over a Virtual Knowledge Base](https://arxiv.org/abs/2002.10640)
  - **TLDR**: Introduces a neural module for multi-hop Question Answering, which is differentiable and can be trained end-to-end.

* (2020) [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282)
  - **TLDR**: Presents an approach to open-domain question answering that relies on retrieving support passages before processing them with a generative model

* (2020) [DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering](https://www.aclweb.org/anthology/2020.acl-main.411.pdf)
  - **TLDR**: Presents a decomposed transformer, which substitutes the full self-attention with question-wide and passage-wide self-attentions in the lower layers reducing runtime compute.

* (2020) [Unsupervised Alignment-based Iterative Evidence Retrieval for Multi-hop Question Answering](https://www.aclweb.org/anthology/2020.acl-main.414.pdf)
  - **TLDR**: Presents introduce a simple, fast, and unsupervised iterative evidence retrieval method for multi-hop Question Answering.

* (2020) [Learning to Ask More: Semi-Autoregressive Sequential Question Generation under Dual-Graph Interaction](https://www.aclweb.org/anthology/2020.acl-main.21.pdf)
  - **TLDR**: Presents approach to generate Question in semi-autoregressive using two graphs based on passages and answers.

* (2020) [What Question Answering can Learn from Trivia Nerds](https://www.aclweb.org/anthology/2020.acl-main.662.pdf)
  - **TLDR**: Presents insights into what Question Answering task can learn from Trivia tournaments.

* (2020) [Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://malllabiisc.github.io/publications/papers/final_embedkgqa.pdf)
  - **TLDR**: Presents an approach effective in performing multi-hop KGQA over sparse Knowledge Graphs.


## Natural Language Generation/Summarization
* (2004) [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013) :vhs:
  - **TLDR**: Introduces ROUGE, an evaluation metric for summarization that is used to this day on a variety of sequence transduction tasks. 
* (2004) [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
  - **TLDR**: Applying graph-based text analysis techniques based on PageRank, the authors achieve SOTA results on keyword extraction and very strong extractive summarization results in an unsupervised fashion.
* (2015) [Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems](https://arxiv.org/abs/1508.01745)
  - **TLDR**: Proposes a neural natural language generator that jointly optimises sentence planning and surface realization, outperforming other systems on human eval. 
* (2016) [Pointing the Unknown Words](https://arxiv.org/abs/1603.08148)
  - **TLDR**: Proposes a copy-mechanism for allowing MT systems to more effectively copy words from a source context sequence.
* (2017) [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
  - **TLDR**: This work offers an elegant soft copy mechanism, that drastically outperforms the SOTA on abstractive summarization. 
* (2020) [A Generative Model for Joint Natural Language Understanding and Generation](https://www.aclweb.org/anthology/2020.acl-main.163.pdf)
  - **TLDR**: This work presents a generative model which couples NLU and NLG through a shared latent variable, achieving state-of-the-art performance on two dialogue datasets with both flat and tree-structured formal representations
* (2020) [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://www.aclweb.org/anthology/2020.acl-main.703.pdf)
  - **TLDR**: This work presents a generative model which couples NLU and NLG through a shared latent variable, achieving state-of-the-art performance on two dialogue datasets with both flat and tree-structured formal representations.

## Dialogue Systems
* (2011) [Data-drive Response Generation in Social Media](http://dl.acm.org/citation.cfm?id=2145500)
  - **TLDR**: Proposes using phrase-based statistical machine translation methods to the problem of response generation. 
* (2015) [Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems](https://arxiv.org/abs/1508.01745)
  - **TLDR**: Proposes a neural natural language generator that jointly optimises sentence planning and surface realization, outperforming other systems on human eval.
* (2016) [How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](https://arxiv.org/abs/1603.08023) :bulb:
  - **TLDR**: Important work demonstrating that existing automatic metrics used for dialogue woefully do not correlate well with human judgment. 
* (2016) [A Network-based End-to-End Trainable Task-oriented Dialogue System](https://arxiv.org/abs/1604.04562)
  - **TLDR**: Proposes a neat architecture for decomposing a dialogue system into a number of individually-trained neural network components. 
* (2016) [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/abs/1510.03055)
  - **TLDR**: Introduces a maximum mutual information objective function for training dialogue systems. 
* (2016) [The Dialogue State Tracking Challenge Series: A Review](https://pdfs.semanticscholar.org/4ba3/39bd571585fadb1fb1d14ef902b6784f574f.pdf)
  - **TLDR**: A nice overview of the dialogue state tracking challenges for dialogue systems. 
* (2017) [A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue](https://arxiv.org/abs/1701.04024)
  - **TLDR**: Shows that simple sequence-to-sequence architectures with a copy mechanism can perform competitively on existing task-oriented dialogue datasets. 
* (2017) [Key-Value Retrieval Networks for Task-Oriented Dialogue](https://arxiv.org/abs/1705.05414) :vhs:
  - **TLDR**: Introduces a new multidomain dataset for task-oriented dataset as well as an architecture for softly incorporating information from structured knowledge bases into dialogue systems. 
* (2017) [Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings](https://arxiv.org/abs/1704.07130) :vhs:
  - **TLDR**: Introduces a new collaborative dialogue dataset, as well as an architecture for representing structured knowledge via knowledge graph embeddings. 
* (2017) [Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning](https://arxiv.org/abs/1702.03274)
  - **TLDR**: Introduces a hybrid dialogue architecture that can be jointly trained via supervised learning as well as reinforcement learning and combines neural network techniques with fine-grained rule-based approaches. 

## Interactive Learning
* (1971) [Procedures as a Representation for Data in a Computer Program for Understanding Natural Language](http://hci.stanford.edu/~winograd/shrdlu/AITR-235.pdf)
  - **TLDR**: One of the seminal papers in computer science, introducing SHRDLU an early system for computers understanding human language commands. 
* (2016) [Learning language games through interaction](http://arxiv.org/abs/1606.02447)
  - **TLDR**: Introduces a novel setting for interacting with computers to accomplish a task where only natural language can be used to communicate with the system!
* (2017) [Naturalizing a programming language via interactive learning](https://arxiv.org/abs/1704.06956)
  - **TLDR**: Very cool work allowing a community of workers to iteratively naturalize a language starting with a core set of commands in an interactive task. 

## Language Modelling
* (1996) [An Empirical Study of Smoothing Techniques for Language Modelling](https://aclweb.org/anthology/P96-1041)
  - **TLDR**: Performs an extensive survey of smoothing techniques in traditional language modelling systems.
* (2003) [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) :bulb:
  - **TLDR**: A seminal work in deep learning for NLP, introducing one of the earliest effective models for neural network-based language modelling. 
* (2014) [One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling](https://arxiv.org/abs/1312.3005) :vhs:
  - **TLDR**: Introduces the Google One Billion Word language modelling benchmark. 
* (2015) [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)
  - **TLDR**: Proposes a language model using convolutional neural networks that can employ character-level information, performing on-par with word-level LSTM systems. 
* (2016) [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410)
  - **TLDR**: Introduces a mega language model system using deep learning that uses a variety of techniques and significantly performs the SOTA on the One Billion Words Benchmark. 
* (2018) [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) :bulb: :vhs:
  - **TLDR**: This paper introduces ELMO, a super powerful collection of word embeddings learned from the intermediate representations of a deep bidirectional LSTM language model. Achieved SOTA on 6 diverse NLP tasks. 
* (2018) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) :bulb:
  - **TLDR**: One of the most important papers of 2018, introducing BERT a powerful architecture pretrained using language modelling which is then effectively transferred to other domain-specific tasks.
* (2019) [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) :bulb:
  - **TLDR**: Generalized autoregressive pretraining method that improves upon BERT by maximizing the expected likelihood over all permutations of the factorization order.

## Miscellanea
* (1997) [Long Short-Term Memory](https://bioinf.jku.at/publications/older/2604.pdf) :bulb:
  - **TLDR**: Introduces the LSTM recurrent unit, a cornerstone of modern neural network-based NLP
* (2000) [Maximum Entropy Markov Models for Information Extraction and Segmentation](https://www.seas.upenn.edu/~strctlrn/bib/PDF/memm-icml2000.pdf) :bulb:
  - **TLDR**: Introduces Markov Entropy Markov models for information extraction, a commonly used ML technique in classical NLP. 
* (2010) [From Frequency to Meaning: Vector Space Models of Semantics](https://arxiv.org/pdf/1003.1141.pdf)
  - **TLDR**: A wonderful survey of existing vector space models for learning semantics in text. 
* (2012) [An Introduction to Conditional Random Fields](http://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)
  - **TLDR**: A nice, in-depth overview of conditional random fields, a commonly-used sequence-labelling model. 
* (2013) [Distributed Representation of Words and Phrases and Their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - **TLDR** Introduced word2vec, a collection of distributed vector representations that have been commonly used for initializing word embeddings in basically every NLP architecture of the last five years. :bulb: :vhs:
* (2014) [Glove: Global vectors for word representation](https://nlp.stanford.edu/pubs/glove.pdf) :bulb: :vhs:
  - **TLDR**: Introduces Glove word embeddings, one of the most commonly used pretrained word embedding techniques across all flavors of NLP models
* (2014) [Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors](http://www.aclweb.org/anthology/P14-1023) 
  - **TLDR**: Important paper demonstrating that context-predicting distributional semantics approaches outperform count-based techniques.
* (2015) [Improving Distributional Similarity with Lessons Learned From Word Embeddings](https://www.aclweb.org/anthology/Q15-1016) :bulb:
  - **TLDR**: Demonstrates that traditional distributional semantics techniques can be enhanced with certain design choices and hyperparameter optimizations that make their performance rival that of neural network-based embedding methods. 
* (2018) [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)
  - **TLDR**: Provides a smorgasbord of nice techniques for finetuning language models that can be effectively transferred to text classification tasks. 
* (2019) [Analogies Explained: Towards Understanding Word Embeddings](https://arxiv.org/pdf/1901.09813.pdf)
  - **TLDR**: Very nice work providing a mathematical formalism for understanding some of the paraphrasing properties of modern word embeddings.
