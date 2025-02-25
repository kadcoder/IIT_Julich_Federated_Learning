import torch
import model as m
import config

def update_model_weights(avg_results, global_path):

    model = m.AgePredictor(input_size=1073).to(config.DEVICE)
    
    try:
        model.load_state_dict(torch.load(global_path, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Warning: Global model not found at {global_path}, initializing new model")
        torch.save(model.state_dict(), global_path)
    
    if not avg_results or not isinstance(avg_results, dict):
        raise ValueError("Invalid average_results - must be non-empty dictionary")
    
    # Initialize optimizer with current model's parameters
    #optimizer = torch.optim.Adam(model.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)
    # Alternatively, use SGD: Ensure the optimizer matches your intended configuration
    optimizer = torch.optim.SGD(model.parameters(), lr=config.INIT_LR, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
    
    old_weights = {name: param.clone() for name, param in model.named_parameters()}
    
    for name, param in model.named_parameters():
        if name in avg_results:
            grad = avg_results[name]
            if not isinstance(grad, torch.Tensor):
                grad = torch.tensor(grad, device=config.DEVICE)
            if grad.shape != param.shape:
                raise ValueError(f"Gradient shape mismatch for {name}: Expected {param.shape}, got {grad.shape}")
            param.grad = grad.to(config.DEVICE)
        else:
            # Ensure gradients are set to zero if not provided to prevent None gradients
            param.grad = torch.zeros_like(param.data).to(config.DEVICE)
    
    # Update weights using the local optimizer
    optimizer.step()
    optimizer.zero_grad()  # Clear gradients for next update
    
    # Check weight changes
    for name, param in model.named_parameters():
        if not torch.equal(old_weights[name], param):
            print(f"Layer {name} has changed.")
        else:
            print(f"Layer {name} remains unchanged.")

    
    return model