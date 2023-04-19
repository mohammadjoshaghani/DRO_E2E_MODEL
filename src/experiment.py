import time, os, logging
from logger import logger
from runner import Runner
import wandb

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class Optimizer:
    def __init__(self, model, distance, ExpId):
        self.model = model
        self.distance = distance
        self.ExpId = ExpId
    
    def objective(self):      
        
        wandb.init()
        epochs, weightDecay, lr = wandb.config.epochs, wandb.config.weightDecay, round(wandb.config.learning_rate,ndigits=4)
        modes_loss = {}

        # run model
        self.runner = Runner('train', epochs, self.model, self.distance, weightDecay, self.ExpId)
        for mode in ['train', 'valid', 'test']:
            self.runner._init(mode, epochs, lr=lr)
            logger.info(f"\n### start {mode} phase for {model}_{distance}_wd_{weightDecay}_lr_{lr}_epochs_{epochs}:\n")
            self.runner.run()
            # save loss for each mode
            modes_loss[mode] = -self.runner.portfolio_Final
        # validation and test loss:
        valid_loss, test_loss = modes_loss['valid'], modes_loss['test']
        # Log model performance metrics to W&B
        wandb.log({"valid_loss": valid_loss, "test_loss": test_loss})

os.environ['WANDB_SILENT']="true"
os.environ['WANDB_MODE']="offline"

logger.info("\nstart:\n")
s_time = time.time()

# models ={"WaveCorr", "WaveCorr_Casual", "Equally_weighted"} 
model   = "WaveCorr"  
distance= "HL"   # "KL"
ExpId = '02'

optimizer = Optimizer(model, distance, ExpId)
sweep_configs = {
    "method": "grid",
    "metric": {"name": "valid_loss", "goal": "minimize"},
    "parameters": {
        "epochs": {"values": [5]},
        "weightDecay": {"values": [0.001, 0.005, 0.009]},
        # "learning_rate": {"distribution": "uniform", "min": 0.0100, "max": 0.0400},
        "learning_rate":{"values": [0.01, 0.02, 0.03, 0.05, 0.06]},
    },
}

sweep_id = wandb.sweep(sweep_configs, project=f"{model}_{distance}_{ExpId}")
wandb.agent(sweep_id=sweep_id, function=optimizer.objective, count=15)

logger.info(f"\n total time: {time.time()-s_time :.2f} seconds.")
logger.info("\n finish.")