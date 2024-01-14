"""
main.py
"""

from mnist import load_mnist
from model import LeNet5
from trainer import Trainer
from visualization import display_loss, display_accuracy, display_confusion_matrix
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
    epoch=50, batch_size=64, shuffle=True
)

# Train and Test
trainer.train(dump_eval=True)

# Get Evaluation Results
train_loss_list, train_acc_list, test_acc_list = trainer.get_eval_results()
# Get Confusion Matrix
train_confusion_matrix, test_confusion_matrix = trainer.get_confusion_matrix()

# Display Loss
display_loss(train_loss_list)
# Display Accuracy
display_accuracy(train_acc_list, test_acc_list)
# Display Confusion Matrix
display_confusion_matrix(train_confusion_matrix, is_train=True)
display_confusion_matrix(test_confusion_matrix, is_train=False)

# Get Model Parameters
model_params = model.get_params()
# print('Model Parameters:\n{}'.format(model_params))

# Save Model Parameters
with open('model_params.pkl', 'wb') as f:
    pickle.dump(model_params, f)
