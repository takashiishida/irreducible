# A direct approach to estimating the Bayes error in binary classification

[![arXiv](https://img.shields.io/badge/arXiv-2202.00395-b31b1b.svg)](https://arxiv.org/abs/2202.00395)

We provide implementation of ["Is the Performance of My Deep Network Too Good to Be True? A Direct Approach to Estimating the Bayes Error in Binary Classification" (Notable-top-5% paper at ICLR 2023)](https://openreview.net/forum?id=FZdJQgy05rz) by [Takashi Ishida](https://takashiishida.github.io), [Ikko Yamane](https://i-yamane.github.io), [Nontawat Charoenphakdee](https://nolfwin.github.io), [Gang Niu](https://niug1984.github.io/), and [Masashi Sugiyama](http://www.ms.k.u-tokyo.ac.jp/sugi/).

## Introduction

From the abstract of our [paper](https://openreview.net/forum?id=FZdJQgy05rz):

> There is a fundamental limitation in the prediction performance that a machine learning model can achieve due to the inevitable uncertainty of the prediction target. In classification problems, this can be characterized by the Bayes error, which is the best achievable error with any classifier. The Bayes error can be used as a criterion to evaluate classifiers with state-of-the-art performance and can be used to detect test set overfitting. We propose a simple and direct Bayes error estimator, where we just take the mean of the labels that show uncertainty of the class assignments. Our flexible approach enables us to perform Bayes error estimation even for weakly supervised data. In contrast to others, our method is model-free and even instancefree. Moreover, it has no hyperparameters and gives a more accurate estimate of the Bayes error than several baselines empirically. Experiments using our method suggest that recently proposed deep networks such as the Vision Transformer may have reached, or is about to reach, the Bayes error for benchmark datasets. Finally, we discuss how we can study the inherent difficulty of the acceptance/rejection decision for scientific articles, by estimating the Bayes error of the ICLR papers from 2017 to 2023.

## Estimating the Bayes error

See `requirements.txt` for requirements.

For synthetic experiments (in Section 5.1):

```sh
python synthetic.py
```

For benchmark experiments (Section 5.2 and 5.3), first prepare CIFAR-10H data:

```sh
git clone https://github.com/jcpeterson/cifar-10h.git
```

Then, run the following for CIFAR-10H and Fashion-MNIST-H:

```sh
python be_c10h.py
python be_fmh.py
```

For real-world experiments (in Section 5.4), run the following:

```sh
python be_iclr.py
```

We thank the [OpenReview Python library](https://github.com/openreview/openreview-py).
We used this library when we collected the data for ICLR reviews.

## Fashion-MNIST-H

In the `data/fmh_counts.csv` file, we offer labels for the Fashion-MNIST-H dataset.
This includes multiple human labels for each test image from the original [Fashion-MNIST](https://arxiv.org/abs/1708.07747) dataset.
By following the naming convention of CIFAR-10H, we refer to this dataset as Fashion-MNIST-H.
The file comprises 10,000 rows and 10 columns. Each row corresponds to a test image from Fashion-MNIST.
The columns represent the following categories: 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', and 'Ankle boot'.

We also provide a file named`data/fmh_probs.csv`, which shows the proportion of each class opposed to the raw counts.

Our `data/fmh_raw.csv` file offers the raw annotation data. The 2nd column is the test image ID (ranging from `0000.jpg` to `9999.jpg`), the 3rd column presents the annotation provided by the annotator, and the 4th column contains the pseudo worker ID.

See [Section 5.3 and Appendix E in our paper](https://openreview.net/forum?id=FZdJQgy05rz) for further details of Fashion-MNIST-H.
Note that this dataset is used to derive the Bayes error in `be_fmh.py`.

Although we collected these labels for the purpose of Bayes error estimation, we hope Fashion-MNIST-H is helpful for other applications that require soft labels or probabilistic labels.

## Contact

If you have any questions, please feel free to send an email to ishi at k.u-tokyo.ac.jp.
