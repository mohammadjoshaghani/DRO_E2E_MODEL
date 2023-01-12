from logger import logger
import time

from runner import Runner

s_time = time.time()

# models ={"WaveCorr", "WaveCorr_Casual", "Equally_weighted"} 
# model   = "WaveCorr"  
distance= "HL"   # "KL"
epochs  = 32
# weightDecay = 0.2*1e-2
experiment_id = "010"

for model in ["WaveCorr", "WaveCorr_Casual"]:
    for weightDecay in [0.5*1e-2]:
        weightDecay = round(weightDecay,3)
        runner = Runner('train', 1, model, distance, weightDecay, experiment_id)
        for epoch in range(1, epochs+1):
            for mode in ['train', 'valid', 'test']:
                runner._init(mode, epoch)
                logger.info(f"\n### start {mode} phase for {model}_{distance}_WD_{weightDecay}_epoch_{epoch}:\n")
                runner.run()

logger.info(f"\n total time: {time.time()-s_time :.2f} seconds.")
logger.info("\n finish.")
# Todo: regularization term for overfitting
#! the sharp ratio in objective is different from portfolio evaluation
#! the final portfolio sharpe ratio in CMD is different from resulst.csv