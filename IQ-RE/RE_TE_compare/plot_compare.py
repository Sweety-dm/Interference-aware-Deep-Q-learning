import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
COLORS = ['#6140ef', '#ff000d']

font_legend = {'weight': 'normal',
               'size': 23}
font_label = {'weight': 'normal',
              'size': 25}

def plot_results(results, game_name, y_lim, linestyle, linewidth, method_name, baseRL,
                 shaded_std=True, shaded_err=True):
    fig = plt.figure(figsize=(9, 6))

    num_methods = np.shape(results)[0]

    for i in range(num_methods):
        ymean = np.mean(results[i], axis=0)
        ystd = np.std(results[i], axis=0)
        ystderr = ystd / np.sqrt(np.shape(results[i])[0])

        plt.plot(np.arange(len(ymean)), ymean, color = COLORS[i], label = method_name[i], alpha = 0.7,
                linewidth=linewidth[i], linestyle=linestyle[i])

        if shaded_err:
            plt.fill_between(np.arange(len(ymean)), ymean - ystderr, ymean + ystderr, color=COLORS[i],
                             alpha=.2, linewidth=0)
        if shaded_std:
            plt.fill_between(np.arange(len(ymean)), ymean - ystd, ymean + ystd, color=COLORS[i],
                             alpha=.2, linewidth=0)
    plt.grid(linestyle='-', linewidth=0.8)
    plt.title(game_name, font_label)
    plt.tick_params(axis='both', width=1, labelsize=20)
    plt.xticks(range(0, 51, 10), ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    plt.xlabel(r'Environment Steps ($\times 10^7$)', font_label)
    plt.ylabel('Train Episode Returns', font_label)
    plt.xlim(0, 50)
    plt.ylim(bottom=y_lim[0], top=y_lim[1])
    plt.legend(loc='upper left', prop=font_legend, edgecolor='black')
    if game_name == 'Carnival':
        plt.text(-0.55, 6.1, '1e3', size=20)
    plt.savefig(str('./{}/RE_TE_compare_{}.pdf').format(game_name, baseRL), bbox_inches='tight', pad_inches=0)

    plt.show()

if __name__ == '__main__':
    game_name = 'Carnival'      #Breakout:[0, 100]  Carnival:[0, 6]
    replay_buffer = 10000
    smooth_interval = 4

    baseRL = 'DQN'  # 'DQN', 'Rainbow'

    y_lim = [0, 6]
    linestyle = ['-', '-']
    linewidth = [3.0, 3.0]

    method_name = [str('TE_{}').format(baseRL), str('RE_{}').format(baseRL)]
    method_name_ = ['Trained Encoder', 'Random Encoder']

    results = []
    for num_method in range(len(method_name)):
        method = []
        for seed in range(1,6):
            training_performance = np.load(str('./{}/{}_{}/training_performance.npy').
                                           format(game_name, method_name[num_method], seed),
                                           allow_pickle=True)
            training_performance = training_performance.tolist()
            training_performance_smooth = []
            for i in range(int(200/smooth_interval)):
                training_performance_smooth.append(np.mean(training_performance[
                                                           i * smooth_interval : (i + 1) * smooth_interval])/1000)
            method.append(training_performance_smooth)
        results.append(method)

    plot_results(results, game_name, y_lim, linestyle, linewidth, method_name_, baseRL,
                 shaded_std=True, shaded_err=False)