# GEOMETRY-AWARE HIERARCHICAL REPRESENTATION 

---



## Usage

Run the following command to train the model.

```bash
python -m source --multirun datasz=100p model.name=BrainNetworkTransformer dataset=ABIDE n_splits=10  cv_epochs=200 cv_lr=0.0005 preprocess=mixup
```

- **datasz**, default=(10p, 20p, 30p, 40p, 50p, 60p, 70p, 80p, 90p, 100p). How much data to use for training. The value is a percentage of the total number of samples in the dataset. For example, 10p means 10% of the total number of samples in the training set.


- **preprocess**, default=(mixup, non_mixup). Which preprocess to applied. The value is a list of preprocess names. For example, mixup means mixup, non_mixup means the dataset is feeded into models without preprocess.



```






