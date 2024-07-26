import mlflow
import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset

from dgl.data import CoraFullDataset, PubmedGraphDataset, CoraGraphDataset, FlickrDataset
from dgl.data import CoauthorPhysicsDataset, YelpDataset, AmazonCoBuyComputerDataset
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
import tqdm
from sklearn.metrics import accuracy_score
import time


class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.h_feats = h_feats
        self.dropout = nn.Dropout(dropout)

    def forward(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        h = self.conv1(mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        h = self.dropout(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]  # <---
        h = self.conv2(mfgs[1], (h, h_dst)) # <---
        h = self.dropout(h)
        return h


@torch.no_grad()
def test(model, device, data_loader):
    model.eval()
    predictions = []
    labels = []
    for input_nodes, output_nodes, mfgs in data_loader:
        inputs = mfgs[0].srcdata['feat'].to(device)
        labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
        predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    accuracy = accuracy_score(labels, predictions)

    return accuracy

datasets = ['ogbn-arxiv', 'pubmed', 'cora', 'flickr', 'coauthor-physics', 'yelp', 'amazon_comp', 'cora_full']

dataset_map = {
        'ogbn-arxiv': lambda: DglNodePropPredDataset('ogbn-arxiv'),
        'pubmed': PubmedGraphDataset(),
        'cora': CoraGraphDataset(),
        'flickr': FlickrDataset(),
        'coauthor-physics': CoauthorPhysicsDataset(),
        'yelp': YelpDataset(),
        'amazon_comp': AmazonCoBuyComputerDataset(),
        'cora_full': CoraFullDataset()
    }

i = 0 # Change this to the index of the dataset you want to use

dataset = dataset_map[datasets[i]]

if datasets[i] !=  'ogbn-arxiv':
    dataset = AsNodePredDataset(dataset,  split_ratio=[0.8, 0.1, 0.1])
    graph = dataset[0]
    if datasets[i] == 'cora' or datasets[i] == 'flickr' or datasets[i] == 'pubmed' :
        graph = dgl.add_reverse_edges(graph)
    node_labels = graph.ndata['label']
    print(graph)
    print(node_labels)
    train_nids = dataset.train_idx
    valid_nids = dataset.val_idx
    test_nids = dataset.test_idx
else:
    graph, node_labels = dataset[0]
    # Add reverse edges since the dataset is unidirectional.
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]
    print(graph)
    print(node_labels)
    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # change to 'cuda' for GPU

print(dataset)

node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()
print('Number of classes:', num_classes)



name = dataset.name

# Set sample sizes and lr ranges
lr_list = [0.01, 0.001, 0.0001]
sample_sizes_layer_1 = list(range(1, 21))
sample_sizes_layer_2 = list(range(1, 16))
for learning_rate in lr_list:
    for s1 in sample_sizes_layer_1:
        for s2 in sample_sizes_layer_2:
            with mlflow.start_run() as run:  # Use MLFlow for logging the values
                # Log hyperparameters and results with MLFlow

                epochs = 10
                #learning_rate = 0.01
                dropout = 0.0
                mlflow.log_param("sample_size_layer_1", s1)
                mlflow.log_param("sample_size_layer_2", s2)
                mlflow.set_tag("mlflow.runName", f"run_{s1}_{s2}")
                mlflow.log_param("num_epochs", epochs)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("dropout", dropout)
                mlflow.log_param("dataset", name[:-12])
                Neighbors = [s1, s2]
                sampler = dgl.dataloading.NeighborSampler(Neighbors)
                train_dataloader = dgl.dataloading.DataLoader(
                    # The following arguments are specific to DGL's DataLoader.
                    graph,  # The graph
                    train_nids,  # The node IDs to iterate over in minibatches
                    sampler,  # The neighbor sampler
                    device=device,  # Put the sampled MFGs on CPU or GPU
                    # The following arguments are inherited from PyTorch DataLoader.
                    batch_size=512,  # Batch size
                    shuffle=True,  # Whether to shuffle the nodes for every epoch
                    drop_last=False,  # Whether to drop the last incomplete batch
                    num_workers=0  # Number of sampler processes
                )

                input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))

                mfg_0_src = mfgs[0].srcdata[dgl.NID]
                mfg_0_dst = mfgs[0].dstdata[dgl.NID]

                model = Model(num_features, 128, num_classes, dropout).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
                valid_dataloader = dgl.dataloading.DataLoader(
                    graph, valid_nids, sampler,
                    batch_size=512,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                    device=device
                )

                #best_accuracy = 0
                train_loss = 0.0
                count_t, count_v = [0,0]
                # best_model_path = 'model.pt'
                start_time = time.time()
                for epoch in range(epochs):
                    model.train()

                    with tqdm.tqdm(train_dataloader) as tq:
                        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                            # feature copy from CPU to GPU takes place here
                            inputs = mfgs[0].srcdata['feat']
                            labels = mfgs[-1].dstdata['label']

                            predictions = model(mfgs, inputs)

                            loss = F.cross_entropy(predictions, labels)
                            train_loss += loss.item()
                            count_t += 1
                            opt.zero_grad()
                            loss.backward()
                            opt.step()

                            train_accuracy = accuracy_score(labels.cpu().numpy(),
                                                            predictions.argmax(1).detach().cpu().numpy())

                            tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)
                        train_loss /= count_t
                        mlflow.log_metric("train_loss", train_loss, step=epoch)
                    model.eval()
                    valid_loss = 0.0
                    predictions = []
                    labels = []
                    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
                        for input_nodes, output_nodes, mfgs in tq:
                            inputs = mfgs[0].srcdata['feat']
                            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())

                            if datasets[i] == 'yelp':
                                predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                            else:
                                outputs = model(mfgs, inputs)
                                loss = F.cross_entropy(outputs, mfgs[-1].dstdata['label'])
                                valid_loss += loss.item()
                                count_v += 1
                                predictions.append(outputs.argmax(1).cpu().numpy())



                        predictions = np.concatenate(predictions)
                        labels = np.concatenate(labels)
                        if datasets[i] == 'yelp':
                            labels = np.argmax(labels, axis=1)
                        valid_accuracy = accuracy_score(labels, predictions)

                        valid_loss /= count_v
                        mlflow.log_metric("val_loss", valid_loss, step=epoch)
                        #if best_accuracy < accuracy:
                        #    best_accuracy = accuracy
                            # torch.save(model.state_dict(), best_model_path)
                        # break
                    if epoch % 2 == 0:
                        print(
                            f'Epoch {epoch} | Train Loss: {train_loss:.5f} | Valid Loss: {valid_loss:.5f} | Train Accuracy: {train_accuracy:.5f} | Valid Accuracy: {valid_accuracy:.5f}')
                test_dataloader = dgl.dataloading.DataLoader(
                    # The following arguments are specific to DGL's DataLoader.
                    graph,  # The graph
                    test_nids,  # The node IDs to iterate over in minibatches
                    sampler,  # The neighbor sampler
                    device=device,  # Put the sampled MFGs on CPU or GPU
                    # The following arguments are inherited from PyTorch DataLoader.
                    batch_size=512,  # Batch size
                    shuffle=True,  # Whether to shuffle the nodes for every epoch
                    drop_last=False,  # Whether to drop the last incomplete batch
                    num_workers=0  # Number of sampler processes
                )
                # Log results with MLFlow

                #mlflow.log_metric("loss", loss)
                train_acc = test(model, device, train_dataloader)
                val_acc = test(model, device, valid_dataloader)
                test_accuracy = test(model, device, test_dataloader)
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("valid_accuracy", val_acc)
                mlflow.log_metric("test_accuracy", test_accuracy)
                elapsed_time = time.time() - start_time
                mlflow.log_metric("runtime_seconds", elapsed_time)
