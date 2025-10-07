import sys

# sys.path.append('../ADATIME')

import os
import pandas as pd
import numpy as np
import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter
from abstract_trainer import AbstractTrainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


import time  # 导入时间模块


class Trainer(AbstractTrainer):
    """
    This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def train(self):
        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            start_time = time.time()  # 记录每个scenario循环的开始时间
            all_labels_total = []  # 定义一个空列表来存储所有标签
            all_predictions_total = []  # 定义一个空列表来存储所有预测
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                # Average meters
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)

                # Train model
                non_adapted_model, last_adapted_model, best_adapted_model, all_labels, all_predictions = self.train_model()

                # 将当前的all_labels和all_predictions追加到总列表中
                all_labels_total.extend(all_labels)
                all_predictions_total.extend(all_predictions)

                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model, last_adapted_model,
                                     best_adapted_model)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')
            # end_time = time.time()  # 记录每个scenario循环的结束时间
            # scenario_time = end_time - start_time  # 计算每个scenario循环的时间
            # print(f"Scenario {src_id}_to_{trg_id} completed in {scenario_time:.2f} seconds")  # 打印每个scenario的时间

            # if len(all_labels_total) > 0 and len(all_predictions_total) > 0:
            #     cm = confusion_matrix(all_labels_total, all_predictions_total)
            #
            #     # 计算归一化混淆矩阵并乘以100
            #     row_sums = cm.sum(axis=1, keepdims=True)
            #     row_sums[row_sums == 0] = 1  # 防止零除
            #     cm_normalized = cm.astype('float64') / row_sums * 100  # 归一化后乘以100
            #
            #     # 验证每行总和
            #     for i in range(cm_normalized.shape[0]):
            #         assert np.isclose(cm_normalized[i].sum(), 100.0, atol=0.01), f"第{i}行总和异常: {cm_normalized[i].sum():.2f}"
            #
            #     class_names = ['0', '1', '2', '3', '4', '5']  # WISDM HAR
            #     class_names = ['0', '1', '2']
            #     disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
            #
            #     plt.figure(figsize=(25, 25))
            #     disp.plot(cmap=plt.cm.Blues)
            #     plt.title('Normalized confusion matrix', fontsize=20)
            #     plt.xticks(fontsize=12)
            #     # 设置混淆矩阵中数字的大小
            #     for text in disp.ax_.texts:
            #         text.set_fontsize(12)  # 更改这里的数字来调整数字大小
            #     plt.yticks(fontsize=12)
            #     plt.savefig(f'D:\\Users\\lhj\\混淆矩阵\\HAR\\{src_id}-to-{trg_id}-normalized.png')
            #     #plt.show()
            #
            #     # 导出混淆矩阵数据为CSV文件
            #
            #     cm_normalized_df = pd.DataFrame(cm_normalized, columns=class_names, index=class_names)
            #     cm_normalized_df.to_csv(f'D:\\Users\\lhj\\混淆矩阵\\HAR\\{src_id}-to-{trg_id}-normalized.csv')
            #
            # else:
            #     print("No data for confusion matrix")




if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs', type=str,
                        help='Directory containing all experiments')
    parser.add_argument('-run_description', default=None, type=str, help='Description of run, if noneTemporal_Imputer, DA method name will be used')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='MAPU', type=str, help='SHOT, EVD, MAPU,')

    # ========= Select the DATASET ==============   default=r'D:\TS_Datsets\ADATIME_data'
    parser.add_argument('--data_path', default=r'D:\Users\lhj\python_project\data', type=str, help='Path containing datase2t')
    parser.add_argument('--dataset', default='FD', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR -FD)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN_tAPE_back', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
