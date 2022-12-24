import torch
import gpytorch



## DEEP KERNEL
class MlpLayer(torch.nn.Sequential):
    """this takes input of size: 215*8 -> 215*1
    """
    def __init__(self, input_dim=8, out_dim=1):
        super(MlpLayer, self).__init__()
        midle_dim = int((out_dim+input_dim)/2)
        self.add_module('linear1', torch.nn.Linear(input_dim, midle_dim))         
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(midle_dim, out_dim))         
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(out_dim, out_dim))


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_tasks = 20, num_latents = 10, inducing_points=""):
        """multi-task svgp model.
        it use LMC model to creates multi noisy observations from latant GP.

        Args:
            num_tasks (int, optional): number of assets. Defaults to 20.
            num_latents (int, optional): number of factors. Defaults to 10.
        """
        # Let's use a different set of inducing points for each latent function
        self.inducing_points = inducing_points
        
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            self.inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        
        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                                    gpytorch.variational.VariationalStrategy(
                                        self, self.inducing_points, variational_distribution, learn_inducing_locations=True),
                                    num_tasks=num_tasks,
                                    num_latents=num_latents,
                                    latent_dim=-1)
        
        super().__init__(variational_strategy)
        
        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )
        
    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# main deep kernel gaussian process model
class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor,num_tasks = 20, num_latents = 10, inducing_points=""):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = MultitaskGPModel(num_tasks, num_latents, inducing_points)
        
    def forward(self, x):
        features = self.feature_extractor(x)     # assets * features
        res = self.gp_layer(features)
        return res        


class MainGp(torch.nn.Module):
    def __init__(self, in_dim=8, out_dim=1,
                 num_tasks = 20, num_latents = 10, num_data = "", inducing_points=""):
        super(MainGp,self).__init__()
        feature_extractor = MlpLayer(in_dim, out_dim)
        self.modelgp = DKLModel(feature_extractor, num_tasks, num_latents, inducing_points)                                # main model
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.modelgp.gp_layer, num_data=num_data)  

    def forward(self,x_batch,y_batch):
        # calculate loss for one batch
        f_preds = self.modelgp(x_batch)
        loss = -self.mll(f_preds, y_batch)#.mean()
        # get covariance and mean for one batch
        y_preds = self.likelihood(f_preds)
        y_mean = y_preds.mean
        y_var = y_preds.variance
        # y_covar = y_preds.covariance_matrix
        return y_var, y_mean, loss