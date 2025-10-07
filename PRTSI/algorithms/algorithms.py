# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from models.models import classifier, Temporal_Imputer, masking, PatchEmbed
# from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl
# from scipy.spatial.distance import cdist
# from torch.optim.lr_scheduler import StepLR
# from copy import deepcopy
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from datetime import datetime
# import scipy.io
#
#
# image_counter = 0  # 使用全局计数器
#
# def get_algorithm_class(algorithm_name):
#     """Return the algorithm class with the given name."""
#     if algorithm_name not in globals():
#         raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
#     return globals()[algorithm_name]
#
#
# class Algorithm(torch.nn.Module):
#     """
#     A subclass of Algorithm implements a domain adaptation algorithm.
#     Subclasses should implement the update() method.
#     """
#
#     def __init__(self, configs):
#         super(Algorithm, self).__init__()
#         self.configs = configs
#         self.cross_entropy = nn.CrossEntropyLoss()
#
#     def update(self, *args, **kwargs):
#         raise NotImplementedError
#
#
# class MAPU(Algorithm):
#
#     def __init__(self, backbone, configs, hparams, device):
#         super(MAPU, self).__init__(configs)
#
#         self.feature_extractor = backbone(configs)
#         self.classifier = classifier(configs)
#         self.temporal_verifier = Temporal_Imputer(configs)
#         self.patchEmbed = PatchEmbed(128)
#         self.network = nn.Sequential(self.feature_extractor, self.classifier)
#
#         self.optimizer = torch.optim.Adam(
#             self.network.parameters(),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )
#
#         self.pre_optimizer = torch.optim.Adam(
#             self.network.parameters(),
#             lr=hparams["pre_learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )
#         self.tov_optimizer = torch.optim.Adam(
#             self.temporal_verifier.parameters(),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )
#         # device
#         self.device = device
#         self.hparams = hparams
#
#         self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
#
#         # losses
#         self.mse_loss = nn.MSELoss()
#         self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
#
#     def pretrain(self, src_dataloader, avg_meter, logger):
#
#         for epoch in range(1, self.hparams["num_epochs"] + 1):
#             for step, (src_x, src_y, _) in enumerate(src_dataloader):#循环信息
#                 # input src data
#                 src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)
#                 #加载数据及其标签
#                 self.pre_optimizer.zero_grad()
#                 self.tov_optimizer.zero_grad()
#                 #梯度清零
#
#                 src_feat, seq_src_feat = self.feature_extractor(src_x)
#                 #源数据特征提取
#                 # masked_data:(32,9,128) mask:(32,9,16)
#                 # change
#                 masked_data, mask = masking(src_x, num_splits=8, num_masked=1)#遮掩数据
#                 masked_src_feat,masked_seq_src_feat = self.feature_extractor(masked_data)#提取遮掩数据的特征
#                 masked_data = self.patchEmbed(masked_seq_src_feat)#将遮掩处理后的特征进行嵌入即使用icb asb进行处理
#                 ''' Temporal order verification  '''
#                 masked_data=masked_data.transpose(1, 2)
#                 tov_predictions = self.temporal_verifier(masked_data.detach())
#
#                 tov_predictions=tov_predictions.transpose(1, 2)
#                 tov_loss = self.mse_loss(tov_predictions, seq_src_feat)
#
#                 src_pred = self.classifier(src_feat)
#                 #使用分类器对原特征进行预测，得到概率分布
#                 # normal cross entropy
#                 src_cls_loss = self.cross_entropy(src_pred, src_y)
#                 total_loss = src_cls_loss + tov_loss
#                 total_loss.backward()
#                 self.pre_optimizer.step()
#                 self.tov_optimizer.step()
#                 losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
#                 # acculate loss
#                 for key, val in losses.items():
#                     avg_meter[key].update(val, 32)
#
#             # logging
#             logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
#             for key, val in avg_meter.items():
#                 logger.debug(f'{key}\t: {val.avg:2.4f}')
#             logger.debug(f'-------------------------------------')
#         src_only_model = deepcopy(self.network.state_dict())
#         return src_only_model
#
#     import numpy as np
#     import scipy.io
#     from copy import deepcopy
#
#     def update(self, trg_dataloader, avg_meter, logger):
#         # 定义最佳和最后模型
#         best_src_risk = float('inf')
#         best_model = self.network.state_dict()
#         last_model = self.network.state_dict()
#
#         # 冻结分类器和OOD探测器
#         for k, v in self.classifier.named_parameters():
#             v.requires_grad = False
#         for k, v in self.temporal_verifier.named_parameters():
#             v.requires_grad = False
#
#         # 初始化用于存储真实标签和预测标签
#         all_labels = []
#         all_predictions = []
#         last_trg_feats = []  # 存储最后一个 epoch 的 trg_feat 用于 tSNE
#         last_trg_labels = []  # 存储最后一个 epoch 的 trg_y 用于 tSNE
#         # class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']#HAR
#         # class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']#WISDM
#         class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']  # HHAR
#
#         # 获取伪标签
#         for epoch in range(1, self.hparams["num_epochs"] + 1):
#             for step, (trg_x, trg_y, trg_idx) in enumerate(trg_dataloader):
#                 trg_x = trg_x.float().to(self.device)
#                 trg_y = trg_y.to(self.device)  # 真实标签
#
#                 self.optimizer.zero_grad()
#                 self.tov_optimizer.zero_grad()
#
#                 trg_feat, trg_feat_seq = self.feature_extractor(trg_x)
#                 # 仅在最后一个 epoch 收集特征和标签
#                 if epoch == self.hparams["num_epochs"]:
#                     last_trg_feats.append(trg_feat.detach().cpu().numpy())  # 收集特征
#                     last_trg_labels.extend(trg_y.cpu().numpy())  # 收集标签
#                 masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
#                 masked_trg_feat_mask, masked_seq_trg_feat = self.feature_extractor(masked_data)
#                 masked_data = self.patchEmbed(masked_seq_trg_feat)
#
#                 masked_data = masked_data.transpose(1, 2)
#                 tov_predictions = self.temporal_verifier(masked_data)
#                 tov_predictions = tov_predictions.transpose(1, 2)
#                 tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)
#
#                 trg_pred = self.classifier(trg_feat)
#                 trg_prob = nn.Softmax(dim=1)(trg_pred)
#
#                 # 记录真实标签和预测标签
#                 all_labels.extend(trg_y.cpu().numpy())
#                 all_predictions.extend(torch.argmax(trg_prob, dim=1).cpu().numpy())
#
#                 trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))
#                 trg_ent -= self.hparams['im'] * torch.sum(
#                     -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))
#
#                 loss = trg_ent + self.hparams['TOV_wt'] * tov_loss
#                 loss.backward()
#                 self.optimizer.step()
#                 self.tov_optimizer.step()
#
#                 losses = {
#                     'entropy_loss': trg_ent.detach().item(),
#                     'Masking_loss': tov_loss.detach().item()
#                 }
#                 for key, val in losses.items():
#                     avg_meter[key].update(val, 32)
#
#             self.lr_scheduler.step()
#
#             # 保存最佳模型
#             if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
#                 best_src_risk = avg_meter['Src_cls_loss'].avg
#                 best_model = deepcopy(self.network.state_dict())
#
#             logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
#             for key, val in avg_meter.items():
#                 logger.debug(f'{key}\t: {val.avg:2.4f}')
#             logger.debug(f'-------------------------------------')
#
#         # 在最后一个 epoch 收集特征并保存
#         last_trg_feats = np.concatenate(last_trg_feats, axis=0)
#         last_trg_labels = np.array(last_trg_labels)
#
#         # 将特征和标签保存为 .mat 文件到指定位置
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         save_path = f'D:/Users/lhj/类别特征分布/feature_map/FD_change/{timestamp}.mat'
#         data_to_save = {
#             'features': last_trg_feats,
#             'labels': last_trg_labels
#         }
#         scipy.io.savemat(save_path, data_to_save)
#
#         # 训练结束后进行tSNE分析
#         # self.perform_tsne(last_trg_feats, last_trg_labels, class_names)
#
#         return last_model, best_model, all_labels, all_predictions
#
#     # def update(self, trg_dataloader, avg_meter, logger):
#     #     # 定义最佳和最后模型
#     #     best_src_risk = float('inf')
#     #     best_model = self.network.state_dict()
#     #     last_model = self.network.state_dict()
#     #
#     #     # 冻结分类器和OOD探测器
#     #     for k, v in self.classifier.named_parameters():
#     #         v.requires_grad = False
#     #     for k, v in self.temporal_verifier.named_parameters():
#     #         v.requires_grad = False
#     #
#     #     # 初始化用于存储真实标签和预测标签
#     #     all_labels = []
#     #     all_predictions = []
#     #     # all_trg_feats = []  # 存储 trg_feat 用于 tSNE
#     #     class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
#     #
#     #     # 获取伪标签
#     #     for epoch in range(1, self.hparams["num_epochs"] + 1):
#     #         for step, (trg_x, trg_y, trg_idx) in enumerate(trg_dataloader):
#     #             trg_x = trg_x.float().to(self.device)
#     #             trg_y = trg_y.to(self.device)  # 真实标签
#     #
#     #             self.optimizer.zero_grad()
#     #             self.tov_optimizer.zero_grad()
#     #
#     #             trg_feat, trg_feat_seq = self.feature_extractor(trg_x)
#     #             # all_trg_feats.append(trg_feat.detach().cpu().numpy())  # 收集特征
#     #             masked_data, mask = masking(trg_x, num_splits=8, num_masked=5)
#     #             masked_trg_feat_mask, masked_seq_trg_feat = self.feature_extractor(masked_data)
#     #             masked_data = self.patchEmbed(masked_seq_trg_feat)
#     #
#     #             masked_data = masked_data.transpose(1, 2)
#     #             tov_predictions = self.temporal_verifier(masked_data)
#     #             tov_predictions = tov_predictions.transpose(1, 2)
#     #             tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)
#     #
#     #             trg_pred = self.classifier(trg_feat)
#     #             trg_prob = nn.Softmax(dim=1)(trg_pred)
#     #
#     #             # 记录真实标签和预测标签
#     #             all_labels.extend(trg_y.cpu().numpy())
#     #             all_predictions.extend(torch.argmax(trg_prob, dim=1).cpu().numpy())
#     #
#     #             trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))
#     #             trg_ent -= self.hparams['im'] * torch.sum(
#     #                 -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))
#     #
#     #             loss = trg_ent + self.hparams['TOV_wt'] * tov_loss
#     #             loss.backward()
#     #             self.optimizer.step()
#     #             self.tov_optimizer.step()
#     #
#     #             losses = {
#     #                 'entropy_loss': trg_ent.detach().item(),
#     #                 'Masking_loss': tov_loss.detach().item()
#     #             }
#     #             for key, val in losses.items():
#     #                 avg_meter[key].update(val, 32)
#     #
#     #         self.lr_scheduler.step()
#     #
#     #         # 保存最佳模型
#     #         if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
#     #             best_src_risk = avg_meter['Src_cls_loss'].avg
#     #             best_model = deepcopy(self.network.state_dict())
#     #
#     #         logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
#     #         for key, val in avg_meter.items():
#     #             logger.debug(f'{key}\t: {val.avg:2.4f}')
#     #         logger.debug(f'-------------------------------------')
#     #
#     #     # 训练结束后进行tSNE分析
#     #     # self.perform_tsne(np.concatenate(all_trg_feats, axis=0), np.array(all_labels), class_names)
#     #
#     #     return last_model, best_model, all_labels, all_predictions
#
#     # def perform_tsne(self, features, labels, class_names):
#     #     # 使用tSNE进行降维
#     #     tsne = TSNE(n_components=2, random_state=42)
#     #     features_2d = tsne.fit_transform(features)
#     #
#     #     # 可视化
#     #     plt.figure(figsize=(10, 10))
#     #     scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
#     #     plt.title('t-SNE visualization')
#     #
#     #     # 创建一个图例，显示类别名及其颜色
#     #     legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_name,
#     #                                   markerfacecolor=scatter.cmap(scatter.norm(label_id)), markersize=10)
#     #                        for label_id, class_name in enumerate(class_names)]
#     #     plt.legend(handles=legend_elements, loc='upper right')
#     #
#     #     plt.colorbar(scatter, label='Class label')
#     #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     #     plt.savefig(f'D:/Users/lhj/类别特征分布/HAR_update_trg_feat_change/{timestamp}.png')  # 使用时间戳进行命名
#     #     plt.close()




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from models.models import classifier, Temporal_Imputer, masking, PatchEmbed
# from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl
# from scipy.spatial.distance import cdist
# from torch.optim.lr_scheduler import StepLR
# from copy import deepcopy
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from datetime import datetime
# import scipy.io
#
#
# image_counter = 0  # 使用全局计数器
#
# def get_algorithm_class(algorithm_name):
#     """Return the algorithm class with the given name."""
#     if algorithm_name not in globals():
#         raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
#     return globals()[algorithm_name]
#
#
# class Algorithm(torch.nn.Module):
#     """
#     A subclass of Algorithm implements a domain adaptation algorithm.
#     Subclasses should implement the update() method.
#     """
#
#     def __init__(self, configs):
#         super(Algorithm, self).__init__()
#         self.configs = configs
#         self.cross_entropy = nn.CrossEntropyLoss()
#
#     def update(self, *args, **kwargs):
#         raise NotImplementedError
#
#
# class MAPU(Algorithm):
#
#     def __init__(self, backbone, configs, hparams, device):
#         super(MAPU, self).__init__(configs)
#
#         self.feature_extractor = backbone(configs)
#         self.classifier = classifier(configs)
#         self.temporal_verifier = Temporal_Imputer(configs)
#         self.patchEmbed = PatchEmbed(128)
#         self.network = nn.Sequential(self.feature_extractor, self.classifier)
#
#         self.optimizer = torch.optim.Adam(
#             self.network.parameters(),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )
#
#         self.pre_optimizer = torch.optim.Adam(
#             self.network.parameters(),
#             lr=hparams["pre_learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )
#         self.tov_optimizer = torch.optim.Adam(
#             self.temporal_verifier.parameters(),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"]
#         )
#         # device
#         self.device = device
#         self.hparams = hparams
#
#         self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
#
#         # losses
#         self.mse_loss = nn.MSELoss()
#         self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
#
#     def pretrain(self, src_dataloader, avg_meter, logger):
#         loss_file_path = f'D:/Users/lhj/误差损失/FD_change/pretrain_losses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
#         for epoch in range(1, self.hparams["num_epochs"] + 1):
#             for step, (src_x, src_y, _) in enumerate(src_dataloader):#循环信息
#                 # input src data
#                 src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)
#                 #加载数据及其标签
#                 self.pre_optimizer.zero_grad()
#                 self.tov_optimizer.zero_grad()
#                 #梯度清零
#
#                 src_feat, seq_src_feat = self.feature_extractor(src_x)
#                 #源数据特征提取
#                 # masked_data:(32,9,128) mask:(32,9,16)
#                 # change
#                 masked_data, mask = masking(src_x, num_splits=8, num_masked=5)#遮掩数据
#                 masked_src_feat,masked_seq_src_feat = self.feature_extractor(masked_data)#提取遮掩数据的特征
#                 masked_data = self.patchEmbed(masked_seq_src_feat)#将遮掩处理后的特征进行嵌入即使用icb asb进行处理
#                 ''' Temporal order verification  '''
#                 masked_data=masked_data.transpose(1, 2)
#                 tov_predictions = self.temporal_verifier(masked_data.detach())
#
#                 tov_predictions=tov_predictions.transpose(1, 2)
#                 tov_loss = self.mse_loss(tov_predictions, seq_src_feat)
#
#                 src_pred = self.classifier(src_feat)
#                 #使用分类器对原特征进行预测，得到概率分布
#                 # normal cross entropy
#                 src_cls_loss = self.cross_entropy(src_pred, src_y)
#                 total_loss = src_cls_loss + tov_loss
#                 total_loss.backward()
#                 self.pre_optimizer.step()
#                 self.tov_optimizer.step()
#                 losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
#                 # acculate loss
#                 for key, val in losses.items():
#                     avg_meter[key].update(val, 32)
#
#             # 将损失写入文件
#             with open(loss_file_path, 'a') as f:
#                 f.write(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]\n')
#                 for key, val in avg_meter.items():
#                     f.write(f'{key}\t: {val.avg:2.4f}\n')
#                 f.write(f'-------------------------------------\n')
#
#             # logging
#             logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
#             for key, val in avg_meter.items():
#                 logger.debug(f'{key}\t: {val.avg:2.4f}')
#             logger.debug(f'-------------------------------------')
#         src_only_model = deepcopy(self.network.state_dict())
#         return src_only_model
#
#
#     def update(self, trg_dataloader, avg_meter, logger):
#         # 定义保存损失的文件路径
#         loss_file_path = f'D:/Users/lhj/误差损失/FD_change/update_losses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
#         # 定义最佳和最后模型
#         best_src_risk = float('inf')
#         best_model = self.network.state_dict()
#         last_model = self.network.state_dict()
#
#         # 冻结分类器和OOD探测器
#         for k, v in self.classifier.named_parameters():
#             v.requires_grad = False
#         for k, v in self.temporal_verifier.named_parameters():
#             v.requires_grad = False
#
#         # 初始化用于存储真实标签和预测标签
#         all_labels = []
#         all_predictions = []
#         last_trg_feats = []  # 存储最后一个 epoch 的 trg_feat 用于 tSNE
#         last_trg_labels = []  # 存储最后一个 epoch 的 trg_y 用于 tSNE
#         # class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']#HAR
#         # class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']#WISDM
#         #class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']  # HHAR
#
#         # 获取伪标签
#         for epoch in range(1, self.hparams["num_epochs"] + 1):
#             for step, (trg_x, trg_y, trg_idx) in enumerate(trg_dataloader):
#                 trg_x = trg_x.float().to(self.device)
#                 trg_y = trg_y.to(self.device)  # 真实标签
#
#                 self.optimizer.zero_grad()
#                 self.tov_optimizer.zero_grad()
#
#                 trg_feat, trg_feat_seq = self.feature_extractor(trg_x)
#                 # 仅在最后一个 epoch 收集特征和标签
#                 if epoch == self.hparams["num_epochs"]:
#                     last_trg_feats.append(trg_feat.detach().cpu().numpy())  # 收集特征
#                     last_trg_labels.extend(trg_y.cpu().numpy())  # 收集标签
#                 masked_data, mask = masking(trg_x, num_splits=8, num_masked=5)
#                 masked_trg_feat_mask, masked_seq_trg_feat = self.feature_extractor(masked_data)
#                 masked_data = self.patchEmbed(masked_seq_trg_feat)
#
#                 masked_data = masked_data.transpose(1, 2)
#                 tov_predictions = self.temporal_verifier(masked_data)
#                 tov_predictions = tov_predictions.transpose(1, 2)
#                 tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)
#
#                 trg_pred = self.classifier(trg_feat)
#                 trg_prob = nn.Softmax(dim=1)(trg_pred)
#
#                 # 记录真实标签和预测标签
#                 all_labels.extend(trg_y.cpu().numpy())
#                 all_predictions.extend(torch.argmax(trg_prob, dim=1).cpu().numpy())
#
#                 trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))
#                 trg_ent -= self.hparams['im'] * torch.sum(
#                     -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))
#
#                 loss = trg_ent + self.hparams['TOV_wt'] * tov_loss
#                 loss.backward()
#                 self.optimizer.step()
#                 self.tov_optimizer.step()
#
#                 # 将总损失保存在字典中
#                 losses = {
#                     'total_loss': loss.detach().item(),
#                     'entropy_loss': trg_ent.detach().item(),
#                     'Masking_loss': tov_loss.detach().item()
#                 }
#                 for key, val in losses.items():
#                     avg_meter[key].update(val, 32)
#
#             self.lr_scheduler.step()
#             # 将损失写入文件
#             with open(loss_file_path, 'a') as f:
#                 f.write(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]\n')
#                 for key, val in avg_meter.items():
#                     f.write(f'{key}\t: {val.avg:2.4f}\n')
#                 f.write(f'-------------------------------------\n')
#
#             # 保存最佳模型
#             if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
#                 best_src_risk = avg_meter['Src_cls_loss'].avg
#                 best_model = deepcopy(self.network.state_dict())
#
#             logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
#             for key, val in avg_meter.items():
#                 logger.debug(f'{key}\t: {val.avg:2.4f}')
#             logger.debug(f'-------------------------------------')
#
#         # 在最后一个 epoch 收集特征并保存
#         last_trg_feats = np.concatenate(last_trg_feats, axis=0)
#         last_trg_labels = np.array(last_trg_labels)
#
#         # 将特征和标签保存为 .mat 文件到指定位置
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         save_path = f'D:/Users/lhj/类别特征分布/feature_map/FD_change/{timestamp}.mat'
#         data_to_save = {
#             'features': last_trg_feats,
#             'labels': last_trg_labels
#         }
#         return last_model, best_model, all_labels, all_predictions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import classifier, Temporal_Imputer, masking, PatchEmbed
from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
import scipy.io


image_counter = 0  # 使用全局计数器

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class MAPU(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(MAPU, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        self.patchEmbed = PatchEmbed(128)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        loss_file_path = f'D:/Users/lhj/误差损失/FD_change/pretrain_losses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):#循环信息
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)
                #加载数据及其标签
                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                #梯度清零

                src_feat, seq_src_feat = self.feature_extractor(src_x)
                #源数据特征提取
                # masked_data:(32,9,128) mask:(32,9,16)
                # change
                masked_data, mask = masking(src_x, num_splits=8, num_masked=1)#遮掩数据
                masked_src_feat,masked_seq_src_feat = self.feature_extractor(masked_data)#提取遮掩数据的特征
                masked_data = self.patchEmbed(masked_seq_src_feat)#将遮掩处理后的特征进行嵌入即使用icb asb进行处理
                ''' Temporal order verification  '''
                masked_data=masked_data.transpose(1, 2)
                tov_predictions = self.temporal_verifier(masked_data.detach())

                tov_predictions=tov_predictions.transpose(1, 2)
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                src_pred = self.classifier(src_feat)
                #使用分类器对原特征进行预测，得到概率分布
                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)
                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()
                # 损失
                losses = {
                    'cls_loss': src_cls_loss.detach().item(),  # 分类器的交叉熵损失
                    'making_loss': tov_loss.detach().item(),  # 时间顺序验证的均方误差损失
                    'total_loss': total_loss.detach().item()  # 总损失
                }
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # 将损失写入文件
            with open(loss_file_path, 'a') as f:
                f.write(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]\n')
                for key, val in avg_meter.items():
                    f.write(f'{key}\t: {val.avg:2.4f}\n')
                f.write(f'-------------------------------------\n')

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model


    def update(self, trg_dataloader, avg_meter, logger):
        # 定义保存损失的文件路径
        loss_file_path = f'D:/Users/lhj/误差损失/FD_change/update_losses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        # 定义最佳和最后模型
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # 冻结分类器和OOD探测器
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        # 初始化用于存储真实标签和预测标签
        all_labels = []
        all_predictions = []
        last_trg_feats = []  # 存储最后一个 epoch 的 trg_feat 用于 tSNE
        last_trg_labels = []  # 存储最后一个 epoch 的 trg_y 用于 tSNE
        # class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']#HAR
        # class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']#WISDM
        #class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']  # HHAR

        # 获取伪标签
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (trg_x, trg_y, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                trg_y = trg_y.to(self.device)  # 真实标签

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)
                # 仅在最后一个 epoch 收集特征和标签
                if epoch == self.hparams["num_epochs"]:
                    last_trg_feats.append(trg_feat.detach().cpu().numpy())  # 收集特征
                    last_trg_labels.extend(trg_y.cpu().numpy())  # 收集标签
                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                masked_trg_feat_mask, masked_seq_trg_feat = self.feature_extractor(masked_data)
                masked_data = self.patchEmbed(masked_seq_trg_feat)

                masked_data = masked_data.transpose(1, 2)
                tov_predictions = self.temporal_verifier(masked_data)
                tov_predictions = tov_predictions.transpose(1, 2)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                trg_pred = self.classifier(trg_feat)
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # 记录真实标签和预测标签
                all_labels.extend(trg_y.cpu().numpy())
                all_predictions.extend(torch.argmax(trg_prob, dim=1).cpu().numpy())

                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss
                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {
                    'total_loss': loss.detach().item(),
                    'entropy_loss': trg_ent.detach().item(),  # 目标数据的熵损失
                    'Masking_loss': tov_loss.detach().item()  # 时间顺序验证的均方误差损失
                }
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()
            # 将损失写入文件
            with open(loss_file_path, 'a') as f:
                f.write(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]\n')
                for key, val in avg_meter.items():
                    f.write(f'{key}\t: {val.avg:2.4f}\n')
                f.write(f'-------------------------------------\n')

            # 保存最佳模型
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')


        return last_model, best_model, all_labels, all_predictions





