"""
main.py
"""

from mnist import load_mnist
from model import LeNet5
from trainer import Trainer
import pickle


# Load MNIST Dataset
train_imgs, train_labels, test_imgs, test_labels = load_mnist(normalize=True, one_hot_label=True)

# LeNet-5 Model
model = LeNet5()

# Model Trainer
trainer = Trainer(
    model, train_imgs, train_labels, test_imgs, test_labels,
    # optimizer='SGD', optimizer_params={'lr': 0.01, },
    optimizer='Adam', optimizer_params={'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
    epoch=20, batch_size=64, shuffle=True
)

# Train and Test
trainer.train(dump_eval=True)

# Get Model Parameters
model_params = model.get_params()
# print('Model Parameters:\n{}'.format(model_params))

# Save Model Parameters
with open('model_params.pkl', 'wb') as f:
    pickle.dump(model_params, f)
