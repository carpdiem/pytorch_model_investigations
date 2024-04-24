import datasets as ds

def setup_mnist():
    mnist = ds.load_dataset("mnist")
    test_valid = mnist["test"].train_test_split(test_size=0.2)
    mnist = ds.DatasetDict({
        "train": mnist["train"],
        "test": test_valid["train"],
        "valid": test_valid["test"]
    })
    return mnist