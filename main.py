import model as ml
import utils
import dataloader
import numpy as np
import torch
import argparse



np.random.seed(123456789)
torch.random.manual_seed(123456789)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--features', type=int, default="10")
    parser.add_argument('--gnn_hidden_dim', type=int, default="10")
    parser.add_argument('--lstm_hidden_dim', type=int, default="64")
    parser.add_argument('--gnn_num_layers', type=int, default="1")
    parser.add_argument('--lstm_num_layers', type=int, default="1")
    parser.add_argument('--look_back_window', type=int, default="15")
    parser.add_argument('--graph_file', type=str, default="dataset/stock_matrix.csv")
    parser.add_argument('--learning_rate', type=float, default="0.001")
    parser.add_argument('--train_epochs', type=int, default="1")
    parser.add_argument('--early_stopping_patience', type=int, default="20")
    parser.add_argument('--weight_decay', type=float, default="0.001")
    parser.add_argument('--scheduler_factor', type=float, default="0.1")
    parser.add_argument('--scheduler_patience', type=int, default="5")

    args = parser.parse_args()



    # load data
    train_loader, val_loader, test_loader, graph_data = dataloader.get_data(windows=args.look_back_window,graph_file=args.graph_file)

    # create model
    model = ml.COGRASP(features=args.features, 
                        gnn_hidden_dim=args.gnn_hidden_dim,
                        lstm_hidden_dim=args.lstm_hidden_dim, 
                        gnn_num_layers=args.gnn_num_layers, 
                        lstm_num_layers=args.lstm_num_layers)
    
    # train model

    utils.train_model(model, train_loader, val_loader, graph_data,
                       learning_rate=args.learning_rate, num_epochs=args.train_epochs, 
                       patience=args.early_stopping_patience, weight_decay=args.weight_decay, 
                       scheduler_factor=args.scheduler_factor, scheduler_patience=args.scheduler_patience)
    
    # test model
    utils.evaluate_model(model, test_loader, graph_data)









if __name__ == "__main__":
    main()
