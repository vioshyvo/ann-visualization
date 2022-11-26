# ANN visualization

Toy examples illustrating the [Multilabel classification framework for approximate nearest neighbor search](https://arxiv.org/abs/1910.08322).

## Illustration of toy data set

To generate a figure of corpus, query point, 5-nn of the query and the partition element, run
```plot-toy-data
./knn_demo.py
```

![Illustration of corpus, query point, nearest neighbors & a partition element](fig/fig2-new.png)

## Illustration of consistency results

To generate partitions and candidate sets with different training set sizes, run
```plot-consistency
./consistency.py
```
PCA tree with `n=50`:

![Illustration of corpus, query point, nearest neighbors & a partition element](fig/fig-PCA-n_0-8-n-50-consistency-cell-candidate-set.png)

PCA tree with `n=250`:

![Illustration of corpus, query point, nearest neighbors & a partition element](fig/fig-PCA-n_0-8-n-250-consistency-cell-candidate-set.png)

PCA tree with `n=1000`:

![Illustration of corpus, query point, nearest neighbors & a partition element](fig/fig-PCA-n_0-8-n-1000-consistency-cell-candidate-set.png)
