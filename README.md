# Text Classification

 Multi-Class and Multi-Label text classification on feedback comments.

Supports BERT and XLNet for both Multi-Class and Multi-Label text classification.

Functions:
1. Train BERT, XLNet text classification models on Disney dataset.
2. Tune model hyper-parameters such as epochs, learning rate, batch size, optimiser schedule and more.
3. Save and deploy trained model for inference.
4. Traditional machine learning algorithms.

## Installation

```bash
pip install -r requirements.txt
```

You will also need to install NVIDIA Apex.

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Dataset


## Pretrained Japanese BERT models
- BERT-base models (12-layer, 768-hidden, 12-heads, 110M parameters)
    - **[`BERT-base_mecab-ipadic-bpe-32k.tar.xz`](https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-bpe-32k.tar.xz)** (2.1GB)
        - MeCab + WordPiece tokenization.
    - **[`BERT-base_mecab-ipadic-bpe-32k_whole-word-mask.tar.xz`](https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-bpe-32k_whole-word-mask.tar.xz)** (2.1GB)
        - MeCab + WordPiece tokenization. Whole Word Masking is enabled during training.
    - **[`BERT-base_mecab-ipadic-char-4k.tar.xz`](https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-char-4k.tar.xz)** (1.6GB)
        - Character tokenization.
    - **[`BERT-base_mecab-ipadic-char-4k_whole-word-mask.tar.xz`](https://www.nlp.ecei.tohoku.ac.jp/~m-suzuki/bert-japanese/BERT-base_mecab-ipadic-char-4k_whole-word-mask.tar.xz)** (1.6GB)
        - Character tokenization. Whole Word Masking is enabled during training (word boundaries are determined by MeCab).

All the model archives include following files.
`pytorch_model.bin` and `tf_model.h5` are compatible with [Transformers](https://github.com/huggingface/transformers).

```
.
├── config.json
├── model.ckpt.data-00000-of-00001
├── model.ckpt.index
├── model.ckpt.meta
├── pytorch_model.bin
├── tf_model.h5
└── vocab.txt
```

At present, only `BERT-base` models are available.


## Usage

### 1. Traditional machine learning solutions
Please refer to and run jupyter file:
```
ml/eda_train_analysis.ipynb
```

### 2. Japanese BERT solutions
Please refer to and run jupyter file:
```
bert_pytorch.ipynb
```


## Refernence
- HuggingFace [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) 
- **[BERT](https://github.com/google-research/bert)** (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
- **[XLNet](https://github.com/zihangdai/xlnet/)** (from Google/CMU) released with the paper [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
- **[RoBERTa](https://arxiv.org/abs/1907.11692)** (from Facebook), a Robustly Optimized BERT Pretraining Approach by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du et al.
- **DistilBERT (from HuggingFace)**, released together with the blogpost [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5) by Victor Sanh, Lysandre Debut and Thomas Wolf.
