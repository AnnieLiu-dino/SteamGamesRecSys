import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def plot_precision_recall_bar(data_labels, avg_precision, avg_recall, title='Average Precision and Recall Comparison'):
    recall_color = '#006da8' 
    precision_color = '#ff7224' 
    x = np.arange(len(data_labels))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, avg_precision, width, label='Mean Precision', color=precision_color)
    bars2 = ax.bar(x + width/2, avg_recall, width, label='Mean Recall', color=recall_color)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height * 100:.2f}%',  # 转换为百分比
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 文字偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color=precision_color)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height * 100:.2f}%',  # 转换为百分比
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 文字偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color=recall_color)

    ax.set_xlabel('Data Labels')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(data_labels)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, borderpad=1, labelspacing=1.2)

    fig.tight_layout()

    plt.show()


def plot_interactoin_distribution(total_records, top5_records, top10_records):

    other_records = total_records - top10_records
    top5_only_records = top5_records - (top10_records - top5_records)
    top10_only_records = top10_records - top5_records

    def thousands_formatter(x, pos):
        return f'{int(x/1000):,}K'

    fig, ax = plt.subplots(figsize=(10, 7))

    bars1 = ax.bar('Top5 Active User', top5_only_records, label='Top5 Only', color='lightblue')
    bars2 = ax.bar('Top10 Active User', top10_only_records, bottom=top5_only_records, label='Top10 Only', color='lightgreen')
    bars3 = ax.bar('Other User', other_records, bottom=top10_records, label='Other Records', color='lightcoral')

    ax.set_xlabel('Categories')
    ax.set_ylabel('Number of Records (in thousands)')
    ax.set_title('Distribution: Top5 Active Users vs. Top10 vs. Other Users Interaction Records')
    ax.legend()

    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(

            bar.get_x() + bar.get_width() / 2, height,
            f'{height / 1000:,.0f}K \n{top5_records/total_records * 100:.2f}%',
            ha='center', va='bottom'
        )

    for bar in bars2:
        height = bar.get_height() + top5_records
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{bar.get_height() / 1000:,.0f}K \n{top10_records/total_records * 100:.2f}%',
            ha='center', va='bottom'
        )

    for bar in bars3:
        height = bar.get_height() + top10_records
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{bar.get_height() / 1000:,.0f}K \n{other_records/total_records * 100:.2f}%',
            ha='center', va='bottom'
        )

    plt.show()


def plot_map_comparison(data_labels, map_0_8_training, map_full_training, title='MAP Comparison for Different Training Data'):
    color_0_8_training = '#1f77b4'
    color_full_training = '#ff7f0e' 
    x = np.arange(len(data_labels))
    width = 0.35 

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, map_0_8_training, width, label='0.8 Dataset Trained Model', color=color_0_8_training)
    rects2 = ax.bar(x + width/2, map_full_training, width, label='Full Dataset Trained Model', color=color_full_training)

    ax.set_xlabel('K Value')
    ax.set_ylabel('MAP (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(data_labels)
    ax.legend()

    def autolabel(rects, color):
        """在每个柱形图上方添加数值标签"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height * 100:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color=color)

    autolabel(rects1, color_0_8_training)
    autolabel(rects2, color_full_training)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))

    fig.tight_layout()

    plt.show()
