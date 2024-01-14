"""
Visualization of Evaluation Results
"""

import matplotlib.pyplot as plt


def display_loss(train_loss_list):
    # 绘制训练过程损失曲线
    plt.figure()
    plt.plot(train_loss_list)
    plt.title('Loss per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # 保存图像到本地
    plt.savefig('train_loss.png')
    print('save train_loss.png successfully!')

    # 显示图片
    plt.show()


def display_accuracy(train_accuracy_list, test_accuracy_list):
    # 绘制准确率曲线
    plt.figure()
    plt.plot(train_accuracy_list, label='train')
    plt.plot(test_accuracy_list, label='test')
    plt.title('Accuracy per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图像到本地
    plt.savefig('accuracy.png')
    print('save accuracy.png successfully!')

    # 显示图片
    plt.show()


def display_confusion_matrix(confusion_matrix, is_train: bool = True):
    if is_train:
        title = 'Train Confusion Matrix'
        path = 'train_confusion_matrix.png'
    else:
        title = 'Test Confusion Matrix'
        path = 'test_confusion_matrix.png'
    # 绘制混淆矩阵
    plt.figure()
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 保存图像到本地
    plt.savefig(path)
    print('save ' + path + ' successfully!')

    # 显示图片
    plt.show()


if __name__ == '__main__':
    # import pickle
    #
    # with open('train_loss_list.pkl', 'rb') as f:
    #     train_loss_list = pickle.load(f)
    # with open('train_acc_list.pkl', 'rb') as f:
    #     train_acc_list = pickle.load(f)
    # with open('test_acc_list.pkl', 'rb') as f:
    #     test_acc_list = pickle.load(f)
    #
    # display_loss(train_loss_list)
    # display_accuracy(train_acc_list, test_acc_list)
    pass
