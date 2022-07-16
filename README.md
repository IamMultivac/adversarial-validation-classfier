# Adversarial Validation

Based  on [this](https://arxiv.org/abs/2004.03045) paper, the idea is to detect features drift.

The method proposes to train a binary classification model that tries to discriminate between train and test samples under the hypotesis that if such classfier has performance >> .5 (say in some metric like ROC AUC) then the most important features for that model are also the features that have changed between the training and the test/production stage.

So under the assumption that a change on the distribution of a feature or a group of features, it is reccomended to eliminate the features recursively until we achive the classifer to have a performance close to random.

Diagram below shows the general idea. 

The original source of the image may be found [here](https://ing-bank.github.io/probatus/tutorials/nb_sample_similarity.html)

![diagram](img/adversarial_validation.png)
