from typing import Literal
import torch
import numpy as np
import pandas as pd
from operator import add

#from itwinai.torch.inference import TorchPredictor
from itwinai.components import monitor_exec, Predictor

import model
from torch.utils.data import DataLoader
from engine import evaluate
from initialization import beta, criterion, n_avg, pixel_wise_criterion

# TODO: itwinai doesnt support distributed inference at the moment!!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pick the season to study among:
# '' (none, i.e. full dataset), 'winter_', 'spring_', 'summer_', 'autumn_'

# This is now moved to the configuration file
# choose wether to evaluate train and test data, and/or projections
#past_evaluation = False
#future_evaluation = True

class XTClimPredictor(Predictor):
    def __init__(
            self,
            evaluation: Literal["past", "future"] = 'past'
    ):
        super().__init__(model)
        self.evaluation = evaluation

    @monitor_exec
    def execute(self):

        if self.evaluation=='past':
            for season in ['winter_', 'spring_', 'summer_', 'autumn_']:

                # load previously trained model
                cvae_model = model.ConvVAE().to(device)
                cvae_model.load_state_dict(torch.load(f'outputs/cvae_model_{season}1d_1memb.pth')['model_state_dict'], strict=False)

                # train set and data loader
                train_time = pd.read_csv(f"input/dates_train_{season}data_1memb.csv")
                train_data = np.load(f"input/preprocessed_1d_train_{season}data_1memb.npy")
                n_train = len(train_data)
                trainset = [ ( torch.from_numpy(np.reshape(train_data[i], (2, 32, 32))),
                        train_time['0'][i] ) for i in range(n_train) ]
                trainloader = DataLoader(
                    trainset, batch_size=1, shuffle=False
                )

                # test set and data loader
                test_time = pd.read_csv(f"input/dates_test_{season}data_1memb.csv")
                test_data = np.load(f"input/preprocessed_1d_test_{season}data_1memb.npy")
                n_test = len(test_data)
                testset = [ ( torch.from_numpy(np.reshape(test_data[i], (2, 32, 32))),
                        test_time['0'][i] ) for i in range(n_test) ]
                testloader = DataLoader(
                    testset, batch_size=1, shuffle=False
                )

                # average over a few iterations
                # for a better reconstruction estimate
                train_avg_losses, _, tot_train_losses, _ = evaluate(cvae_model, trainloader,
                                                        trainset, device,
                                                        criterion,
                                                        pixel_wise_criterion)
                test_avg_losses, _, tot_test_losses, _ = evaluate(cvae_model, testloader,
                                                    testset, device, criterion,
                                                    pixel_wise_criterion)
                for i in range(1, n_avg):
                    train_avg_loss, _, train_losses, _ = evaluate(cvae_model, trainloader,
                                                    trainset, device, criterion,
                                                    pixel_wise_criterion)
                    tot_train_losses = list(map(add, tot_train_losses, train_losses))
                    train_avg_losses += train_avg_loss
                    test_avg_loss, _, test_losses, _ = evaluate(cvae_model, testloader,
                                                    testset, device, criterion,
                                                    pixel_wise_criterion)
                    tot_test_losses = list(map(add, tot_test_losses, test_losses))
                    test_avg_losses += test_avg_loss
                tot_train_losses = np.array(tot_train_losses)/n_avg
                tot_test_losses = np.array(tot_test_losses)/n_avg
                train_avg_losses = train_avg_losses/n_avg
                test_avg_losses = test_avg_losses/n_avg

                pd.DataFrame(tot_train_losses).to_csv(f"outputs/train_losses_{season}1d_allssp.csv")
                pd.DataFrame(tot_test_losses).to_csv(f"outputs/test_losses_{season}1d_allssp.csv")
                print('Train average loss:', train_avg_losses)
                print('Test average loss:', test_avg_losses)


        else:
            for season in ['winter_', 'spring_', 'summer_', 'autumn_']:

                # load previously trained model
                cvae_model = model.ConvVAE().to(device)
                cvae_model.load_state_dict(torch.load(f'outputs/cvae_model_{season}1d_1memb.pth')['model_state_dict'], strict=False)

                for scenario in ['585', '245']:

                    # projection set and data loader
                    proj_time = pd.read_csv(f"input/dates_proj_{season}data_1memb.csv")
                    proj_data = np.load(f"input/preprocessed_1d_proj{scenario}_{season}data_1memb.npy")
                    n_proj = len(proj_data)
                    projset = [ ( torch.from_numpy(np.reshape(proj_data[i], (3, 32, 32))),
                            proj_time['0'][i] ) for i in range(n_proj) ]
                    projloader = DataLoader(
                        projset, batch_size=1, shuffle=False
                    )

                    # get the losses for each data set
                    # on various experiments to have representative statistics
                    proj_avg_losses, _, tot_proj_losses, _ = evaluate(cvae_model, projloader,
                                                        projset, device, criterion,
                                                        pixel_wise_criterion)

                    for i in range(1, n_avg):
                        proj_avg_loss, _, proj_losses, _ = evaluate(cvae_model, projloader,
                                                        projset, device, criterion,
                                                        pixel_wise_criterion)
                        tot_proj_losses = list(map(add, tot_proj_losses, proj_losses))
                        proj_avg_losses += proj_avg_loss

                    tot_proj_losses = np.array(tot_proj_losses)/n_avg
                    proj_avg_losses = proj_avg_losses/n_avg

                    # save the losses time series
                    pd.DataFrame(tot_proj_losses).to_csv(f"outputs/proj{scenario}_losses_{season}1d_allssp.csv")
                    print(f'SSP{scenario} Projection average loss:', proj_avg_losses, 'for', season[:-1])
