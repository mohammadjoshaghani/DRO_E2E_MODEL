from logger import logger
import time

from runner import Runner

s_time = time.time()

# models ={"WaveCorr", "WaveCorr_Casual", "Equally_weighted"} 
model   = "WaveCorr"  
distance= "HL"   # "KL"
epochs  = 32
weightDecay = 1e-3

mode ='train' 
logger.info(f"\n### start {mode} phase for {model}_{distance}:\n")

runner = Runner(mode, epochs, model, distance, weightDecay)
runner.run()

mode='test'
runner._init(mode, epochs)
logger.info(f"\n### start {mode} phase for {model}_{distance}:\n")
runner.run()

logger.info(f"\n total time: {time.time()-s_time :.2f} seconds.")
logger.info("\n finish.")
# Todo: regularization term for overfitting
#! the sharp ratio in objective is different from portfolio evaluation
#! the final portfolio sharpe ratio in CMD is different from resulst.csv