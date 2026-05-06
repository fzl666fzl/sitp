from GNN_QATTEN_online import conf, train


if __name__ == "__main__":
    conf.zero_gnn_embedding = True
    print("zero-GNN ablation: graph embedding/action bias is forced to all zeros.")
    train()
