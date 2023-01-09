from logger import logger
import time

from runner import Runner

s_time = time.time()

# models ={"WaveCorr", "WaveCorr_Casual", "Equally_weighted"} 
model   = "WaveCorr"  
distance= "HL"   # "KL"
epochs  = 32
weightDecay = 1e-3
experiment_id = "001"

runner = Runner('train', 1, model, distance, weightDecay, experiment_id)
for epoch in range(1, epochs+1):
    for mode in ['train', 'valid', 'test']:
        runner._init(mode, epoch)
        logger.info(f"\n### start {mode} phase for {model}_{distance}:\n")
        runner.run()

logger.info(f"\n total time: {time.time()-s_time :.2f} seconds.")
logger.info("\n finish.")
# Todo: regularization term for overfitting
#! the sharp ratio in objective is different from portfolio evaluation
#! the final portfolio sharpe ratio in CMD is different from resulst.csv