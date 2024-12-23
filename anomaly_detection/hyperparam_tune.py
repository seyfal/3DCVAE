import wandb
import torch
import argparse

# from anomaly_detection.models.cvae3d_flex import CVAE3D
from anomaly_detection.models.cvae3d import CVAE3D
from anomaly_detection.data.data_loader import get_data_loader
from anomaly_detection.training.train import train_epoch
from anomaly_detection.config.config_handler import get_config
from anomaly_detection.utils.utils import visualize_reconstructions

def objective():
    with wandb.init() as run:
        config = wandb.config
        base_config = get_config('anomaly_detection/config/config.yaml')
        base_config.update(config)
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # model = CVAE3D(input_shape=(24, 24, 240), latent_dim=base_config['latent_dim'], hidden_dims=[64, 128, 256]).to(device)
        model = CVAE3D(base_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(base_config['learning_rate']))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        train_loader = get_data_loader(base_config)
        
        for epoch in range(base_config['epochs']):
            loss, elbo, recon_loss, kl_loss, sam_loss, auc_loss = train_epoch(model, train_loader, optimizer, scheduler, base_config, epoch)
            
            wandb.log({
                "total_loss": loss,
                "elbo": elbo,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "sam_loss": sam_loss,
                "auc_loss": auc_loss
            })
   
            # Generate and log other visualizations every N epochs
            if epoch % base_config['visualization_interval'] == 0:
                model.eval()
                with torch.no_grad():  
                    recon_fig = visualize_reconstructions(model, train_loader, device)
                    wandb.log({"reconstructions": wandb.Image(recon_fig)})

        return loss

def create_sweep():
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        },
        'parameters': {
            'kl_weight': {
                'distribution': 'uniform',
                'min': 0,
                'max': 10.0
            },
            'sam_weight': {
                'distribution': 'uniform',
                'min': 0,
                'max': 10.0
            },
            'auc_weight': {
                'distribution': 'uniform',
                'min': 0,
                'max': 10.0
            }
        }
    }
    
    base_config = get_config('anomaly_detection/config/config.yaml')
    sweep_id = wandb.sweep(sweep_config, project=base_config['wandb_project_name'])
    print(f"Created sweep with ID: {sweep_id}")
    return sweep_id

def run_agent(sweep_id):
    base_config = get_config('anomaly_detection/config/config.yaml')
    wandb.agent(sweep_id, function=objective, project=base_config['wandb_project_name'], count=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run W&B sweep or agent")
    parser.add_argument("--create-sweep", action="store_true", help="Create a new sweep")
    parser.add_argument("--run-agent", type=str, help="Run an agent for the given sweep ID")
    args = parser.parse_args()

    if args.create_sweep:
        create_sweep()
    elif args.run_agent:
        run_agent(args.run_agent)
    else:
        print("Please specify either --create-sweep or --run-agent <sweep_id>")