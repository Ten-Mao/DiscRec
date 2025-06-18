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

We release the **Beauty** dataset along with the pretrained **DiscRec-T** checkpoint on **Beauty**. Link: 

https://drive.google.com/drive/folders/1I7uCqLteAfUIUsmhv2wGIeZtnn_eoxv1?usp=sharing

# Train & Eval

To train **DiscRec** on **Beauty**, simply run:

```
bash run_discrec.sh  
```

