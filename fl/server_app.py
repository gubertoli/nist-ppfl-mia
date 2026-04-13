import torch
from collections import OrderedDict
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.app import Context, ArrayRecord, ConfigRecord, MetricRecord
from task import GenomicCNN, get_data_loaders

app = ServerApp()

def get_evaluate_fn(privacy_tier):
    """Generates an evaluation function for the server using the 20% holdout data."""
    val_loader = get_data_loaders(privacy_tier, is_server=True)
    use_dp = privacy_tier != "no-dp"
    
    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = GenomicCNN(use_dp=use_dp)
        
        model.load_state_dict(arrays.to_torch_state_dict(), strict=True)
        
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        loss, correct = 0.0, 0
        
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                loss += criterion(outputs, y).item()
                correct += (torch.max(outputs.data, 1)[1] == y).sum().item()
                
        accuracy = correct / len(val_loader.dataset)
        avg_loss = loss / len(val_loader)
        
        return MetricRecord({"accuracy": accuracy, "loss": avg_loss})
        
    return evaluate

@app.main()
def main(grid, context: Context) -> None:
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    tier = context.run_config["privacy-tier"]
    
    global_model = GenomicCNN(use_dp=(tier != "no-dp"))
    initial_arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(
        fraction_train=1.0, 
        min_train_nodes=4, 
        min_available_nodes=4, 
        fraction_evaluate=0.0, 
    )

    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(tier)
    )
