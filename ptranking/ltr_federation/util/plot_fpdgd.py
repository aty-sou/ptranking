from matplotlib.pylab import plt
import matplotlib.cm as cm
import torch
import numpy as np
import os


import datetime


dirname = "/Users/kanazawaatsuya/Dropbox/ptrfl/ptranking/ltr_federation/result/default/"
#os.makedirs(dirname, exist_ok=True)


def draw_line(epoch, ndcg_list, user_model):
    """
    epoch: serverのepoch数
    ndcg: serverのndcgリスト
    color: グラフの色を指定する
    fold_k: 図の区別をするため
    user_model:
    """
    dirname = "/Users/kanazawaatsuya/Dropbox/ptrfl/ptranking/ltr_federation/result/img/"
    dt_now = datetime.datetime.now()
    fig = plt.figure(figsize = [10,5],facecolor="white")
    # figsize パラメータのデフォルト値は [6.4, 4.8]
    plt.figure(figsize=[50, 4.2])

    #plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # タイトル
    #fig.suptitle("user model:{}, {}".format(user_model, dt_now))

    # x軸
    x = torch.arange(epoch)

    # x軸とy軸のタイトル
    ax = fig.add_subplot(111, xlabel="Number of updates", ylabel='nDCG')
    ax.set_ylim(0.0, 1.0)

    e = [1.2, 2.3, 4.5, 10]
    #COLORS = ['darkblue', 'darkslateblue', 'indigo', 'thistle']

    # es毎にplot
    for i in range(len(ndcg_list)):
        label="epsilon={}".format(e[i])
        ax.plot(x, ndcg_list[i], label=label, color=cm.hsv(i/4))
    #print(ax)
    #print(ndcg_list)
    filename = dirname + "{}.png".format(user_model)

    ax.legend() # 凡例表示
    fig.savefig(filename)  # 画像保存
    #print(filename)

# オンラインのスコア計算に行う
def cumulative_online_score(ys):
    cndcg = 0
    for i, score in enumerate(ys):
        cndcg += 0.9995 ** i * score
    return cndcg



#path = os.getcwd()
#print(path)