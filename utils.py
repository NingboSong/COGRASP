import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
import time


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.best_loss = val_loss
        torch.save(model.state_dict(), 'checkpoint.pt')


def train(model, train_loader, optimizer, criterion, device, graph_data):
    graph_data = (graph_data[0].to(device), graph_data[1].to(device), graph_data[2].to(device))
    model.train()
    total_loss = 0
    for sequence_data, labels in train_loader:
        sequence_data, labels = sequence_data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(graph_data, sequence_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
 
def validate(model, val_loader, criterion, device, graph_data):
    graph_data = (graph_data[0].to(device), graph_data[1].to(device), graph_data[2].to(device))
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sequence_data, labels in val_loader:
            sequence_data, labels = sequence_data.to(device), labels.to(device)
            outputs = model(graph_data, sequence_data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)
    
def train_model(model, train_loader, val_loader, graph_data, learning_rate, num_epochs, patience, weight_decay, scheduler_factor, scheduler_patience):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)

    early_stopping = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device, graph_data)
        val_loss = validate(model, val_loader, criterion, device, graph_data)
        early_stopping(val_loss, model)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        end_time = time.time()
        run_time = end_time - start_time
        if epoch % 1 == 0:
            print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.8f}, Val Loss = {val_loss:.8f}, Early Stop:{early_stopping.counter},time:{run_time:.4f}',flush=True)
        if early_stopping.early_stop:
            print("Early stopping",flush=True) 
            break
    model.load_state_dict(torch.load('checkpoint.pt'))

def evaluate_model(model, test_loader, graph_data):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_data = (graph_data[0].to(device), graph_data[1].to(device), graph_data[2].to(device))
    model.eval()
    
    ic_list = []
    rank_ic_list = []

    with torch.no_grad():
        for sequence_data, labels in test_loader:
            sequence_data, labels = sequence_data.to(device), labels.to(device)
            outputs = model(graph_data, sequence_data)
            predictions = outputs.detach().cpu().numpy()
            actuals = labels.detach().cpu().numpy()

            predictions_flat = predictions.flatten()
            actuals_flat = actuals.flatten()

           

            ic = np.corrcoef(predictions_flat, actuals_flat)[0, 1]
            rank_ic = spearmanr(predictions_flat, actuals_flat).correlation

            ic_list.append(ic)
            rank_ic_list.append(rank_ic)
            

 
    avg_ic = np.mean(ic_list)
    ic_std = np.std(ic_list)
    icir = avg_ic / ic_std
    avg_rank_ic = np.mean(rank_ic_list)
    rank_ic_std = np.std(rank_ic_list)
    rank_icir = avg_rank_ic / rank_ic_std
    print(f' Avg IC: {avg_ic:.4f}, Avg Rank IC: {avg_rank_ic:.4f}, ICIR:{icir:.4f}, Rank ICIR:{rank_icir:.4f}', flush=True)
    
    return avg_ic, avg_rank_ic, icir, rank_icir  
