from matplotlib.pylab import plt
import torch
import numpy as np
import os


#dirname = "/Users/kanazawaatsuya/Dropbox/ptrfl/ptranking/ltr_federation/result/default/"
#os.makedirs(dirname, exist_ok=True)


def draw_line(epoch, ndcg_list, fold_k):
    """
    epoch: serverのepoch数
    ndcg: serverのndcgリスト
    color: グラフの色を指定する
    fold_k: 図の区別をするため
    """
    dirname = "/Users/kanazawaatsuya/Dropbox/ptrfl/ptranking/ltr_federation/result/default/"

    fig = plt.figure(facecolor="white")

    # タイトル
    #fig.suptitle("epsilon={}, fold={}".format(es, fold_k))

    # x軸
    x = torch.arange(epoch)

    # x軸とy軸のタイトル
    ax = fig.add_subplot(111, xlabel="epoch", ylabel='ndcg')

    e = [1.2, 2.3, 4.5, 10]
    COLORS = ['darkblue', 'darkslateblue', 'indigo', 'thistle']

    # es毎にplot
    for i in range(len(ndcg_list)):
        label="epsilon={}".format(e[i])
        ax.plot(x, ndcg_list[i], marker="o", label=label, color=COLORS[i])

    #print(ax)
    #print(ndcg_list)
    filename = dirname + "fed_{}.png".format(fold_k)

    ax.legend()  # 凡例表示
    fig.savefig(filename)  # 画像保存
    #print(filename)


#path = os.getcwd()
#print(path)