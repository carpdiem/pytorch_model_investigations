import datasets as ds
import torchvision.transforms.functional as TF

def setup_mnist():
    mnist = ds.load_dataset("mnist")
    test_valid = mnist["test"].train_test_split(test_size=0.2)
    mnist = ds.DatasetDict({
        "train": mnist["train"],
        "test": test_valid["train"],
        "valid": test_valid["test"]
    })
    
    def tf(b):
        b['image'] = [TF.to_tensor(o) for o in b['image']]
        return b
    
    mnist = mnist.with_transform(tf)

    return mnist

