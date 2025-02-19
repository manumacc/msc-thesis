# Explaining black-box models in deep active learning in the context of image classification

<img width="600" alt="image" src="https://github.com/user-attachments/assets/19bb5ddf-b216-4764-895c-429266061cf2" />

## Abstract
Active learning limits manual labeling costs by selecting a small number of samples from a large pool of unlabeled data in an iterative fashion. At each step, active learning aims to query the unlabeled samples that maximize the performance gain of the model. Active learning can be applied in deep learning to limit the amount of data required by deep neural networks. This work focuses on a supervised learning problem, namely multi-class image classification, in the context of deep active learning (DAL) using convolutional neural networks (CNNs). Deep neural networks are inherently black-box. Explainable AI (xAI) provides tools to locally explain the reasons behind predictions and understand how the model operates globally. This increases trust in black-box models and encourages their conscious adoption. We envision a complete xAI framework applied to different parts of the DAL architecture.

Our work focuses on the design of an unsupervised query strategy that makes use of the rich information contained in indices extracted by EBAnO, an explanation framework able to analyze the decision-making process of CNNs through the unsupervised mining of knowledge contained in convolutional layers. To the best of our knowledge, no existing study injects knowledge extracted from explanations into a query strategy in an unsupervised manner. To efficiently apply this explanation framework into the DAL loop, we introduce BatchEBAnO, an optimized version of EBAnO that delivers a tenfold decrease in execution time while retaining good cluster quality.

We propose a query strategy that selects samples whose most influential interpretable feature is not precisely focused on the predicted class, as measured by indices extracted from the unlabeled pool. We introduce another version of this query strategy that selects samples in a ranked fashion. Additionally, we augment the training dataset at each DAL iteration by partially obfuscating a subset of samples selected by our query strategy. Our experiments are carried out on multiple subsets of the ImageNet dataset using ResNet-50 and VGG-16. We compare our results to uncertainty-based DAL baselines.

## Thesis
→ [Read thesis (PDF)](thesis.pdf).

The thesis is also available at Politecnico di Torino's digital thesis library [Webthesis](https://webthesis.biblio.polito.it/21084/).

## License
CC BY-NC-ND 2.5 (see [LICENSE](LICENSE)).
