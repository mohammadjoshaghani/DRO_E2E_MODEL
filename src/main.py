from logger import logger
import time

from runner import Runner

s_time = time.time()

model   = "MLP-K-GP"#"MLP"
distance= "HL"   # "KL"
epochs  = 100

mode ='test' 
logger.info(f"\n### start {mode} phase for {model}_{distance}:\n")

runner = Runner(mode, epochs, model, distance)
runner.run()

# mode='test'
# runner._init(mode, epochs)
# logger.info(f"\n### start {mode} phase for {model}_{distance}:\n")
# runner.run()

logger.info(f"\n total time: {time.time()-s_time :.2f} seconds.")
logger.info("\n finish.")