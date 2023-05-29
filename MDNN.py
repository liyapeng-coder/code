import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import argparse
import matplotlib.pyplot as plt
import logging


class HFLayer(nn.Module):
    def __init__(self, dimension, TargetSimilarityMatrix, EnzymeSimilarityMatrix, SubstructureSimilarityMatrix):
        super().__init__()
        self.pca = PCA(n_components=dimension)
        self.TargetEmbeding = torch.tensor(self.pca.fit_transform(TargetSimilarityMatrix.cpu())).to('cuda')
        self.EnzymeEmbeding = torch.tensor(self.pca.fit_transform(EnzymeSimilarityMatrix.cpu())).to('cuda')
        self.SubstructureEmbeding = torch.tensor(self.pca.fit_transform(SubstructureSimilarityMatrix.cpu())).to('cuda')
        # self.TargetEmbeding = TargetSimilarityMatrix
        # self.EnzymeEmbeding = EnzymeSimilarityMatrix
        # self.SubstructureEmbeding = SubstructureSimilarityMatrix

    def forward(self, X):
        return torch.cat([self.TargetEmbeding, self.EnzymeEmbeding, self.SubstructureEmbeding], 1), X


class GNNLayer(nn.Module):
    def __init__(self, dimension, sampleSize, DKG, pLayers):
        super().__init__()
        self.DKG = DKG
        self.sampleSize = sampleSize
        self.drugNumber = len(torch.unique(DKG[:, 0]))
        self.relationNumber = len(torch.unique(DKG[:, 2]))
        self.tailNumber = len(torch.unique(DKG[:, 1]))
        self.dimension = dimension
        self.drugEmbeding = nn.Embedding(num_embeddings=self.drugNumber, embedding_dim=dimension)
        self.relationEmbeding = nn.Embedding(num_embeddings=self.relationNumber, embedding_dim=dimension)
        self.tailEmbeding = nn.Embedding(num_embeddings=self.tailNumber, embedding_dim=dimension)
        fullConnectionLayers = []
        for i in range(pLayers):
            if (i < pLayers - 1):
                fullConnectionLayers.append(nn.Linear(dimension, dimension))
                fullConnectionLayers.append(nn.Sigmoid())
            else:
                fullConnectionLayers.append(nn.Linear(dimension, dimension))
        self.fullConnectionLayer = nn.Sequential(*fullConnectionLayers)
        self.fullConnectionLayer2 = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.BatchNorm1d(dimension))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, arguments):
        HFEmbeding, X = arguments
        hadamardProduct = self.drugEmbeding(self.DKG[:, 0]) * self.relationEmbeding(self.DKG[:, 2])
        semanticsFeatureScore = torch.sum(self.fullConnectionLayer(hadamardProduct), dim=1).reshape((-1, 1))
        tempEmbedding = semanticsFeatureScore * self.tailEmbeding(self.DKG[:, 1])
        neighborhoodEmbedding = torch.zeros(self.drugNumber, self.dimension).to('cuda')

        for i in range(self.drugNumber):
            # 采样
            length = torch.sum(self.DKG[:, 0] == i)
            if length >= self.sampleSize:
                index = list(data.WeightedRandomSampler(self.DKG[:, 0] == i, self.sampleSize, replacement=False))
                neighborhoodEmbedding[i] = torch.sum(tempEmbedding[index], dim=0)
            else:
                neighborhoodEmbedding[i] = torch.sum(tempEmbedding[self.DKG[:, 0] == i], dim=0)
                index = list(
                    data.WeightedRandomSampler(self.DKG[:, 0] == i, int(self.sampleSize - length), replacement=True))
                neighborhoodEmbedding[i] = neighborhoodEmbedding[i] + torch.sum(tempEmbedding[index], dim=0)

        concatenate = torch.cat([self.drugEmbeding.weight, neighborhoodEmbedding], 1)
        return HFEmbeding, self.fullConnectionLayer2(concatenate), X


class FusionLayer(nn.Module):
    def __init__(self, dimension, drop_out1, drop_out2):
        super().__init__()
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(dimension, 2048),
            nn.Sigmoid(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=drop_out1),
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=drop_out2),
            nn.Linear(1024, 65),
            nn.BatchNorm1d(65),
            nn.Softmax(dim=1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        HFEmbeding, GNNEmbeding, X = arguments
        Embeding = torch.cat([HFEmbeding, GNNEmbeding], 1)
        drugA = X[:, 0]
        drugB = X[:, 1]
        finalEmbedding = torch.cat([Embeding[drugA], Embeding[drugB]], 1).float()
        return self.fullConnectionLayer(finalEmbedding)


def train(net, train_iter, num_epochs, lr, wd, momentum, X_test, y_test, fold=0, device=torch.device(f'cuda')):
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = nn.CrossEntropyLoss()
    train_acc_list = []
    test_acc_list = []
    for epoch in range(num_epochs):
        train_result = []
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.float())
            train_result += ((torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).int()).tolist()
            # train_acc = accuracy_score(torch.argmax(y_hat, dim=1).cpu(), torch.argmax(y, dim=1).cpu())
            # print(f'loss {l:.4f}, train acc {train_acc:.4f}')
            l.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            y_hat = net(X_test)
            train_acc = sum(train_result) / len(train_result)
            test_acc = accuracy_score(torch.argmax(y_hat, dim=1).cpu(), torch.argmax(y_test, dim=1).cpu())
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f'train acc {train_acc:.4f}')
            print(f'test acc {test_acc:.4f}')
            print(f'max train acc {max(train_acc_list):.4f}')
            print(f'max test acc {max(test_acc_list):.4f}')
            print("epoch:{}...".format(epoch))
            print("fold:{}...".format(fold))
            print("----------------------------------------------")
            logging.info(f'train acc {train_acc:.4f}')
            logging.info(f'test acc {test_acc:.4f}')
            logging.info(f'max train acc {max(train_acc_list):.4f}')
            logging.info(f'max test acc {max(test_acc_list):.4f}')
            logging.info("epoch:{}...".format(epoch))
            logging.info("fold:{}...".format(fold))
            logging.info("----------------------------------------------")
    # print("----------------------------------------------")
    # print(f'max train acc {max(train_acc_list):.4f}')
    # print(f'max test acc {max(test_acc_list):.4f}')
    x = range(len(train_acc_list))
    y1 = train_acc_list
    y2 = test_acc_list
    plt.plot(x, y1, color='r', label="train_acc")  # s-:方形
    plt.plot(x, y2, color='g', label="test_acc")  # o-:圆形
    # plt.plot(x, y1, 's-', color='r', label="train_acc")  # s-:方形
    # plt.plot(x, y2, 'o-', color='g', label="test_acc")  # o-:圆形
    plt.xlabel("epoch")  # 横坐标名字
    plt.ylabel("accuracy")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.show()
    return max(train_acc_list), max(test_acc_list)


def train_KFold(net, lr, wd, features, labels, n_splits, num_epochs, batch_size, momentum):
    train_acc_list = []
    test_acc_list = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(features, labels)):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = torch.tensor(pd.get_dummies(labels, columns=['event']).values)[train_index], \
                          torch.tensor(pd.get_dummies(labels, columns=['event']).values)[test_index]
        dataset = data.TensorDataset(X_train, y_train)
        train_iter = data.DataLoader(dataset, batch_size, shuffle=True)
        net.load_state_dict(torch.load("./data/model_parameter.pkl"))
        train_acc, test_acc = train(net, train_iter, num_epochs, lr, wd, momentum, X_test, y_test, i,
                                    device=torch.device(f'cuda'))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'fold {i},  max_train_acc:{train_acc:.4f},  max_test_acc:{test_acc:.4f}')
        print("---------------------------------------")
        logging.info(f'fold {i},  max_train_acc:{train_acc:.4f},  max_test_acc:{test_acc:.4f}')
        logging.info("---------------------------------------")
    print(f'avg train acc {sum(train_acc_list) / n_splits:.4f}')
    print(f'avg test acc {sum(test_acc_list) / n_splits:.4f}')
    logging.info(f'avg train acc {sum(train_acc_list) / n_splits:.4f}')
    logging.info(f'avg test acc {sum(test_acc_list) / n_splits:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型参数设置')

    parser.add_argument('--weight_decay', type=float, default=1e-4, choices=[1e-4], help="weight_decay")
    parser.add_argument('--lr', type=float, default=0.05, choices=[0.1, 1], help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, choices=[0.9, 0.95], help="momentum")

    parser.add_argument('--epoch', type=int, default=300, choices=[100], help="epochs_number")
    parser.add_argument('--batch_size', type=int, default=512, choices=[512], help="batch_size")
    parser.add_argument('--GNNdimension', type=int, default=64, choices=[64], help="GNNdimension")
    parser.add_argument('--HFdimension', type=int, default=512, choices=[], help="HFdimension")
    parser.add_argument('--sample_number', type=int, default=7, choices=[7], help="sample_number")
    parser.add_argument('--drop_out1', type=float, default=0, choices=[0], help="drop_out1")
    parser.add_argument('--drop_out2', type=float, default=0, choices=[], help="drop_out2")
    parser.add_argument('--pLayers', type=int, default=1, choices=[1], help="pLayers")
    parser.add_argument('--n_splits', type=int, default=5, choices=[5], help="n_splits")

    net_args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename='./data/mdnn.log', filemode='w', format="%(message)s")

    TargetSimilarityMatrix = torch.tensor(np.array(pd.read_csv("TargetSimilarityMatrix.csv"))).to('cuda')
    EnzymeSimilarityMatrix = torch.tensor(np.array(pd.read_csv("EnzymeSimilarityMatrix.csv"))).to('cuda')
    SubstructureSimilarityMatrix = torch.tensor(np.array(pd.read_csv("SubstructureSimilarityMatrix.csv"))).to('cuda')

    df = pd.read_csv("FinalData.csv").dropna(axis=0, how='any').drop_duplicates()
    df.drop('1', axis=1, inplace=True)
    df = df.groupby("3").filter(lambda x: (len(x) > 1))

    df.drop(df.loc[(df['3'] == 'include')].index, axis=0, inplace=True)

    drug_number = pd.read_csv("drug_number.csv")
    key_list = list(drug_number.iloc[:, 0])
    value_list = list(drug_number.iloc[:, 1])
    drugmap = dict(zip(key_list, value_list))

    entitymap = {}
    entitylist = np.array(df.iloc[:, 1]).tolist()
    for entity in entitylist:
        if entity not in entitymap:
            entitymap[entity] = len(entitymap)

    relationmap = {}
    relationlist = np.array(df.iloc[:, 2]).tolist()
    for relation in relationlist:
        if relation not in relationmap:
            relationmap[relation] = len(relationmap)

    df.iloc[:, 0] = df.iloc[:, 0].map(drugmap)
    df.iloc[:, 1] = df.iloc[:, 1].map(entitymap)
    df.iloc[:, 2] = df.iloc[:, 2].map(relationmap)

    DKG = torch.tensor(df.to_numpy()).to('cuda')

    eventdf = pd.read_csv("event_encode.csv")
    # print(len(eventdf.iloc[:,2].unique()))
    # eventdf = pd.get_dummies(eventdf, columns=['event'])
    # features = torch.tensor(eventdf.iloc[:, 0:2].values)
    # labels = torch.tensor(eventdf.iloc[:, 2:].values)
    features = torch.tensor(eventdf.iloc[:, 0:2].values)
    labels = eventdf.iloc[:, 2:]

    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, stratify=labels)
    # dataset = data.TensorDataset(X_train, y_train)
    # train_iter = data.DataLoader(dataset, net_args.batch_size, shuffle=True)

    net = nn.Sequential(
        HFLayer(net_args.HFdimension, TargetSimilarityMatrix, EnzymeSimilarityMatrix, SubstructureSimilarityMatrix),
        GNNLayer(net_args.GNNdimension, net_args.sample_number, DKG, net_args.pLayers),
        FusionLayer(net_args.HFdimension * 6 + net_args.GNNdimension * 2, net_args.drop_out1, net_args.drop_out2))
    torch.save(net.state_dict(), "./data/model_parameter.pkl")

    train_KFold(net, net_args.lr, net_args.weight_decay, features, labels, net_args.n_splits, net_args.epoch,
                net_args.batch_size, net_args.momentum)
    # train(net, train_iter, net_args.epoch, net_args.lr, net_args.weight_decay, net_args.momentum, X_test, y_test)
