from logger import logger
import time

from runner import Runner

s_time = time.time()

# models ={"WaveCorr", "WaveCorr_Casual", "Equally_weighted"} 
model   = "WaveCorr"  
distance= "HL"   # "KL"
epochs  = 1
weightDecay = 1e-3
experiment_id = "01"

mode ='train' 
logger.info(f"\n### start {mode} phase for {model}_{distance}:\n")

runner = Runner(mode, epochs, model, distance, weightDecay, experiment_id)
runner.run()

for mode in ['valid', 'test']:
    runner._init(mode, epochs)
    logger.info(f"\n### start {mode} phase for {model}_{distance}:\n")
    runner.run()

logger.info(f"\n total time: {time.time()-s_time :.2f} seconds.")
logger.info("\n finish.")