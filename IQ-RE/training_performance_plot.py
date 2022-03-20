import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

matplotlib.use('TkAgg')
COLORS = ['orange', 'tomato', 'blueviolet', 'green', 'royalblue', 'red']

font_legend = {'weight': 'normal',
               'size': 19}
font_label = {'weight': 'normal',
              'size': 25}

def plot_results(results, game_name, replay_buffer, y_lim, linestyle, linewidth, method_name,
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
    plt.grid(linestyle='--', linewidth=1.0)
    plt.title(game_name, font_label)
    plt.tick_params(axis='both', width=1, labelsize=20)
    plt.xticks(range(0, 51, 10), ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    plt.xlabel(r'Environment Steps ($\times 10^7$)', font_label)
    plt.ylabel('Train Episode Returns', font_label)
    plt.xlim(0, 50)
    plt.ylim(bottom=y_lim[0], top=y_lim[1])
    plt.legend(loc='lower right', prop=font_legend, frameon=True, shadow=False, edgecolor='black')

    if game_name == 'Carnival':
        plt.text(-0.55, 6.1, '1e3', size=20)
    plt.savefig(str('./experiments/{}/RBS={}/{}(RBS={}).pdf').format(
        game_name, replay_buffer, game_name, replay_buffer), bbox_inches='tight', pad_inches=0)

    plt.show()

def caculate_p(results, method_name):
    for i in [2, 5]:
        IQ_result = results[i]
        IQ_result_mean = np.mean(IQ_result)
        for j in [i-2, i-1]:
            baseline_result = results[j]
            baseline_result_mean = np.mean(baseline_result)
            stat, p = ks_2samp(IQ_result, baseline_result)

            if IQ_result_mean >= baseline_result_mean:
                winner = method_name[i]
            else:
                winner = method_name[j]
            print(str('{} vs {}: {} --- winner: {}').format(method_name[i], method_name[j], p, winner))


if __name__ == '__main__':
    game_name = 'Pong'      # Pong:[-21, 21]     Freeway:[-2, 40]  Tennis:[-25, 5]
                                # Breakout:[0, 100]  Carnival:[0, 6]   FishingDerby:[-100, 40]
    replay_buffer = 1000000
    smooth_interval = 4

    y_lim = [-21, 21]
    linestyle = ['-', '-', '-', '-', '-', '-']
    linewidth = [3.0, 3.0, 5.0, 3.0, 3.0, 5.0]

    method_name = ['DQN', 'SRNN_DQN', 'IQ_DQN', 'Rainbow', 'SRNN_Rainbow', 'IQ_Rainbow']
    method_name_ = ['DQN', 'DQN + SRNN', 'DQN + IQ-RE (ours)', 'Rainbow', 'Rainbow + SRNN', 'Rainbow + IQ-RE (ours)']

    results_ori = []
    results = []
    for num_method in range(len(method_name)):
        method_ori = []
        method = []
        for seed in range(1,6):
            training_performance = np.load(str('./experiments/{}/RBS={}/{}_{}/training_performance.npy').
                                           format(game_name, replay_buffer, method_name[num_method], seed),
                                           allow_pickle=True)
            training_performance = training_performance.tolist()
            training_performance_smooth = []
            for i in range(int(200/smooth_interval)):
                if game_name == 'Carnival':
                    training_performance_smooth.append(
                        np.mean(training_performance[i * smooth_interval:(i + 1) * smooth_interval]) / 1000)
                else:
                    training_performance_smooth.append(
                        np.mean(training_performance[i * smooth_interval:(i + 1) * smooth_interval]))
            method.append(training_performance_smooth)
            method_ori.extend(training_performance_smooth)
        results_ori.append(method_ori)
        results.append(method)

    plot_results(results, game_name, replay_buffer, y_lim, linestyle, linewidth, method_name_,
                 shaded_std=True, shaded_err=False)

    # caculate_p(results_ori, method_name_)


