"""
main.py
"""

from mnist import load_mnist
from model import LeNet5
from trainer import Trainer


# Load MNIST Dataset
train_imgs, train_labels, test_imgs, test_labels = load_mnist(normalize=True, one_hot_label=True)

# LeNet-5 Model
model = LeNet5()

# Model Trainer
trainer = Trainer(
    model, train_imgs, train_labels, test_imgs, test_labels,
    optimizer='SGD', optimizer_params={'lr': 0.01, },
    epoch=20, batch_size=64, shuffle=True
)

# Train and Test
trainer.train(dump_eval=True)
