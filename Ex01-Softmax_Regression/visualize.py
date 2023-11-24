import matplotlib.pyplot as plt


def display_loss(loss_list):
    # 绘制损失曲线
    plt.figure()
    plt.plot(loss_list)
    plt.title('Loss per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # 保存图像到本地
    plt.savefig('loss_plot.png')
    print('save loss_plot.png successfully!')

    # 显示图片
    plt.show()


def display_accuracy(train_accuracy_list, test_accuracy_list):
    # 绘制准确率曲线
    plt.figure()
    plt.plot(train_accuracy_list, label='train')
    plt.plot(test_accuracy_list, label='test')
    plt.title('Accuracy per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图像到本地
    plt.savefig('accuracy_plot.png')
    print('save accuracy_plot.png successfully!')

    # 显示图片
    plt.show()
