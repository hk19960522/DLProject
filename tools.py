import os
import matplotlib.pyplot as plt

def read_all_loss(filename):
    data = []
    with open(filename) as file:
        data = [[], [], []]
        for line in file:
            batch_data = line.split()
            # format: epoch, L2 loss, ADE, FDE
            for idx in range(3):
                data[idx].append(float(batch_data[idx+1]))
    return data


def drawplot(filename):
    data = read_all_loss(filename)
    if len(data) > 0:
        label = ['L2 loss', 'ADE', 'FDE']
        plt.title('Train Result')
        plt.xlabel('Epoch')
        for idx in range(0, 1):
            plt.plot(range(len(data[idx])), data[idx], label=label[idx])
        plt.legend()
        plt.show()
        pass


if __name__ == '__main__':
    drawplot('Result_1.txt')
    print('Tools Test.')