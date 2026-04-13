from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict
from opacus.accountants.utils import get_noise_multiplier
import torch
from task import GenomicCNN, get_data_loaders

app = ClientApp()

def calculate_tier_noise(epsilon, target_delta, epochs, sample_rate):
    return get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        epochs=epochs
    )

@app.train()
def train(msg: Message, context: Context):
    privacy_type = context.run_config["privacy-tier"]
    local_epochs = int(context.run_config["local-epochs"])
    server_rounds = int(context.run_config["num-server-rounds"])
    use_dp = privacy_type != "no-dp"
    
    model = GenomicCNN(use_dp=use_dp)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    partition_id = context.node_config["partition-id"]
    
    trainloader = get_data_loaders(privacy_type=privacy_type, partition_id=partition_id, is_server=False)
    
    batch_size = trainloader.batch_size
    dataset_size = len(trainloader.dataset)
    sample_rate = batch_size / dataset_size
    total_epochs = server_rounds * local_epochs
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(local_epochs):
        for x, y in trainloader:
            optimizer.zero_grad()
            
            if use_dp:
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                target_eps = 10 if privacy_type == "high-dp" else 200
                sigma = calculate_tier_noise(
                    epsilon=target_eps, 
                    target_delta=1e-5, 
                    epochs=total_epochs, 
                    sample_rate=sample_rate
                )
                
                for param in model.parameters():
                    noise = torch.randn_like(param.grad) * sigma * 2.0
                    param.grad.add_(noise)
            else:
                criterion(model(x), y).backward()
            
            optimizer.step()

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(model.state_dict()), 
            "metrics": MetricRecord({"num-examples": dataset_size})
        }), 
        reply_to=msg
    )
