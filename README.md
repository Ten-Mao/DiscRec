# DiscRec

This is the pytorch implementation of our paper:

> DiscRec: Disentangled Semantic–Collaborative Modeling for Generative Recommendation

# Requirements

```
torch==2.6.0+cu126
transformers==4.46.3
accelerate==1.6.0
bitsandbytes==0.45.5
numpy==1.26.3
peft==0.15.2
scikit-learn==1.6.1
tqdm==4.67.1
```

# Data

you can download the corresponding datasets through the following links:

https://nijianmo.github.io/amazon/index.html

# Train & Eval

To train **DiscRec** on the **Beauty** dataset, simply run the following command after downloading and preprocessing the data according to the steps outlined in our paper:

```
bash run_discrec.sh  
```

