import time
import optuna, plotly, os, json
optuna.logging.set_verbosity(optuna.logging.WARNING)

from logger import logger
from runner import Runner

class Optimizer:
    def __init__(self,model, distance, ExpId):
        self.model = model
        self.distance = distance
        self.ExpId = ExpId
    
    def objective(self, trial):
        #the parameters to tune
        weightDecay = trial.suggest_float('weightDecay', 0.001, 0.01, step=0.003)
        # epochs = trial.suggest_int('epochs', 10, 70, step=20)
        epochs = trial.suggest_int('epochs', 1, 3, step=1)
        
        # run model
        self.runner = Runner('train', epochs, self.model, self.distance, weightDecay, self.ExpId)
        for mode in ['train', 'valid']:
            self.runner._init(mode, epochs)
            # logger.info(f"\n### start {mode} phase for {model}_{distance}:\n")
            self.runner.run()
        return self.runner.portfolio_Final



    def optimize(self):
        study = optuna.create_study(study_name=f"Optimization of {self.model}")
        study.optimize(self.objective, n_trials = 10)
        self.path = os.path.join(os.getcwd(), f'results/ExpId_{self.ExpId}/')
        json.dump(study.best_params, open(self.path + "best_params.json", 'w'))
        logger.info('#####################')
        logger.info('best hyper-paramters:')
        logger.info(f'{study.best_params}')
        self.save_opt_plots(study)



    def save_opt_plots(self, study):
            # create plots of optimization
            fig1 = optuna.visualization.plot_parallel_coordinate(study, params=['weightDecay','epochs'])
            fig2 = optuna.visualization.plot_optimization_history(study)
            fig3 = optuna.visualization.plot_slice(study, params=['weightDecay','epochs'])
            fig4 = optuna.visualization.plot_contour(study, params=['weightDecay','epochs'])
            data= [fig1,fig2,fig3,fig4]

            #save the plots
            with open(self.path + 'optresult.html', 'w') as f:
                for fig_i in data:
                    f.write(fig_i.to_html(full_html=False, include_plotlyjs='cdn'))

logger.info("\nstart:\n")
s_time = time.time()

# models ={"WaveCorr", "WaveCorr_Casual", "Equally_weighted"} 
model   = "WaveCorr"  
distance= "HL"   # "KL"
ExpId = '01'

h_param_optimizer = Optimizer(model, distance, ExpId)
h_param_optimizer.optimize()

logger.info(f"\n total time: {time.time()-s_time :.2f} seconds.")
logger.info("\n finish.")
