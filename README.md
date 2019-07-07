# nlp-library
This is a curated list of papers that I have encountered in some capacity and deem worth including in the NLP practitioner's library. Some papers may appear in multiple sub-categories, if they don't fit easily into one of the boxes.

**PRs are absolutely welcome!**

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
* (2014) [A Fast and Accurate Dependency Parser using Neural Networks](cs.stanford.edu/~danqi/papers/emnlp2014.pdf)
  - **TLDR**: Very important work ushering in a new wave of neural network-based parsing architectures, achieving SOTA performance as well as blazing parsing speeds. 

## Named Entity Recognition
* (2005) [Incorporating Non-local Information into Information Extraction Systems by Gibbs Sampling](http://nlp.stanford.edu/~manning/papers/gibbscrf3.pdf)
  - **TLDR**: Using cool Monte Carlo methods combined with a conditional random field model, this work achieves a huge error reduction in certain information extraction benchmarks.
* (2015) [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)
  - **TLDR**: Proposes an element sequence-tagging model combining neural networks with conditional random fields, achieving SOTA in POS-tagging, NER, and chunking. 

## Coference Resolution
* (2010) [A multi-pass sieve for coreference resolution](https://nlp.stanford.edu/pubs/conllst2011-coref.pdf) :bulb:
* (2015) [Entity-Centric Coreference Resolution with Model Stacking](http://cs.stanford.edu/~kevclark/resources/clark-manning-acl15-entity.pdf) 

## Sentiment Analysis
* (2012) [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://www.aclweb.org/anthology/P12-2018)
* (2013) [Recursive deep models for semantic compositionality over a sentiment treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) :vhs:

## Natural Logic/Inference
* (2007) [Natural Logic for Textual Inference](https://nlp.stanford.edu/pubs/natlog-wtep07.pdf)
* (2008) [An Extended Model of Natural Logic](dl.acm.org/citation.cfm?id=1693772)
* (2014) [Recursive Neural Networks Can Learn Logical Semantics](https://arxiv.org/abs/1406.1827)
* (2015) [A large annotated corpus for learning natural language inference](http://nlp.stanford.edu/pubs/snli_paper.pdf) :vhs:

## Machine Translation
* (1993) [The Mathematics of Statistical Machine Translation](www.aclweb.org/anthology/J93-2003) :bulb:
* (2002) [BLEU: A Method for Automatic Evaluation of Machine Translation](www.aclweb.org/anthology/P02-1040.pdf) :vhs:
* (2003) [Statistical Phrase-Based Translation](http://dl.acm.org/citation.cfm?id=1073462)
* (2011) [Statistical Machine Translation: IBM Models 1 and 2](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/ibm12.pdf)
* (2014) [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf) :bulb:
* (2015) [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) :bulb:
* (2015) [Effective approaches to attention-based neural machine translation](https://arxiv.org/abs/1508.04025)
* (2016) [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)
* (2016) [Pointing the Unknown Words](www.aclweb.org/anthology/P16-1014)
* (2016) [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144)

## Semantic Parsing
* (2013) [Semantic Parsing on Freebase from Question-Answer Pairs](www.aclweb.org/anthology/D13-1160) :bulb: :vhs:
* (2014) [Semantic Parsing via Paraphrasing](http://aclweb.org/anthology/P14-1133)
* (2015) [Building a Semantic Parser Overnight](https://cs.stanford.edu/~pliang/papers/overnight-acl2015.pdf) :vhs:
* (2015) [Bringing Machine Learning and Computational Semantics Together](http://www.stanford.edu/~cgpotts/manuscripts/liang-potts-semantics.pdf)

## Question Answering/Reading Comprehension
* (2016) [A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task](https://arxiv.org/abs/1606.02858)
* (2017) [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) :vhs:

## Natural Language Generation/Summarization
* (2004) [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013) :vhs:
* (2015) [Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems](https://arxiv.org/abs/1508.01745)
* (2016) [Pointing the Unknown Words](https://arxiv.org/abs/1603.08148)
* (2017) [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

## Dialogue Systems
* (2011) [Data-drive Response Generation in Social Media](http://dl.acm.org/citation.cfm?id=2145500)
* (2015) [Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems](https://arxiv.org/abs/1508.01745)
* (2016) [How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation](https://arxiv.org/abs/1603.08023) :bulb:
* (2016) [A Network-based End-to-End Trainable Task-oriented Dialogue System](https://arxiv.org/abs/1604.04562)
* (2016) [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/abs/1510.03055)
* (2016) [The Dialogue State Tracking Challenge Series: A Review](https://pdfs.semanticscholar.org/4ba3/39bd571585fadb1fb1d14ef902b6784f574f.pdf)
* (2017) [A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue](https://arxiv.org/abs/1701.04024)
* (2017) [Key-Value Retrieval Networks for Task-Oriented Dialogue](https://arxiv.org/abs/1705.05414) :vhs:
* (2017) [Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings](https://arxiv.org/abs/1704.07130) :vhs:
* (2017) [Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning](https://arxiv.org/abs/1702.03274)

## Interactive Learning
* (1971) [Procedures as a Representation for Data in a Computer Program for Understanding Natural Language](http://hci.stanford.edu/~winograd/shrdlu/AITR-235.pdf)
* (2016) [Learning language games through interaction](http://arxiv.org/abs/1606.02447)
* (2017) [Naturalizing a programming language via interactive learning](https://arxiv.org/abs/1704.06956)

## Language Modelling
* (1996) [An Empirical Study of Smoothing Techniques for Language Modelling](https://aclweb.org/anthology/P96-1041)
* (2003) [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) :bulb:
* (2014) [One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling](https://arxiv.org/abs/1312.3005) :vhs:
* (2015) [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)
* (2016) [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410)
* (2018) [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) :bulb: :vhs:

## Miscellanea
* (1997) [Long Short-Term Memory](www.bioinf.jku.at/publications/older/2604.pdf) :bulb:
  - **TLDR**: Introduces the LSTM recurrent unit, a cornerstone of modern neural network-based NLP
* (2000) [Maximum Entropy Markov Models for Information Extraction and Segmentation](https://www.seas.upenn.edu/~strctlrn/bib/PDF/memm-icml2000.pdf) :bulb:
* (2010) [From Frequency to Meaning: Vector Space Models of Semantics](https://arxiv.org/pdf/1003.1141.pdf)
* (2012) [An Introduction to Conditional Random Fields](http://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)
* (2014) [Glove: Global vectors for word representation](https://nlp.stanford.edu/pubs/glove.pdf) :bulb: :vhs:
  - **TLDR**: Introduces Glove word embeddings, one of the most commonly used pretrained word embedding techniques across all flavors of NLP models
* (2014) [Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors](http://www.aclweb.org/anthology/P14-1023) 
* (2015) [Improving Distributional Similarity with Lessons Learned From Word Embeddings](https://www.aclweb.org/anthology/Q15-1016) :bulb:
* (2018) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) :bulb:
