# BERT is Robust! A Case Against Synonym-Based Adversarial Examples in Text Classification 
This repository contains the code to reproduce the results of the paper *BERT is Robust! A Case Against Synonym-Based Adversarial Examples in Text Classification*.

The reproduce the results: 
  1. Fine-tune BERT using defense_fine_tune.py
  2. Collect a dataset of adversarial examples using collect_adv_dataset.py
  3. Use defense_post.py on the created adversarial dataset
