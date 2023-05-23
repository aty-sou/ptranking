from matplotlib.pylab import plt
import torch
import numpy as np
import os


#dirname = "/Users/kanazawaatsuya/Dropbox/ptrfl/ptranking/ltr_federation/result/default/"
#os.makedirs(dirname, exist_ok=True)


def draw_line(epoch, ndcg_list, user_model):
    """
    epoch: serverのepoch数
    ndcg: serverのndcgリスト
    color: グラフの色を指定する
    fold_k: 図の区別をするため
    user_model:
    """
    dirname = "/Users/kanazawaatsuya/Dropbox/ptrfl/ptranking/ltr_federation/result/default/"

    fig = plt.figure(facecolor="white")

    #plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # タイトル
    fig.suptitle("user model:{}".format(user_model))

    # x軸
    x = torch.arange(epoch)

    # x軸とy軸のタイトル
    ax = fig.add_subplot(111, xlabel="epoch", ylabel='ndcg')
    ax.set_ylim(0.0, 1.0)

    e = [1.2, 2.3, 4.5, 10]
    #COLORS = ['darkblue', 'darkslateblue', 'indigo', 'thistle']

    # es毎にplot
    for i in range(len(ndcg_list)):
        label="epsilon={}".format(e[i])
        ax.plot(x, ndcg_list[i], label=label)

    #print(ax)
    #print(ndcg_list)
    filename = dirname + "UM:{}.png".format(user_model)

    ax.legend()  # 凡例表示
    fig.savefig(filename)  # 画像保存
    #print(filename)


#path = os.getcwd()
#print(path)