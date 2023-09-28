from logger import logger
import time

from runner import Runner

s_time = time.time()
distance = "HL"  # "KL"
epochs = 4
experiment_id = "01"

for model in ["WaveCorr", "WaveCorr_Casual"]:
    for weightDecay in [0.005, 0.007, 0.009]:
        weightDecay = round(weightDecay, 3)
        runner = Runner("train", 1, model, distance, weightDecay, experiment_id)
        for epoch in range(1, epochs + 1):
            for mode in ["train", "valid", "test"]:
                runner._init(mode, epoch)
                logger.info(
                    f"\n### start {mode} phase for {model}_{distance}_WD_{weightDecay}_epoch_{epoch}:\n"
                )
                runner.run()

logger.info(f"\n total time: {time.time()-s_time :.2f} seconds.")
logger.info("\n finish.")
