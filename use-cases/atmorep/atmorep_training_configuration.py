import os
import json
from pathlib import Path
from itwinai.torch.config import TrainingConfiguration

class AtmoRepTrainingConfiguration(TrainingConfiguration):
    """AtmoRep TrainingConfiguration"""

################################################
  
    def get_self_dict(self):
       return self.__dict__

################################################

    def write_json( self, wandb) :

        if not hasattr( wandb.run, 'id') :
            return

        json_str = json.dumps(self.get_self_dict() )

        # save in directory with model files
        dirname = Path( self.path_results, 'models/id{}'.format( wandb.run.id))
        if not os.path.exists(dirname):
            os.makedirs( dirname)
        fname =Path(self.path_results,'models/id{}/model_id{}.json'.format(wandb.run.id,wandb.run.id))
        with open(fname, 'w') as f :
            f.write( json_str)

        # also save in results directory
        dirname = Path( self.path_results,'id{}'.format( wandb.run.id))
        if not os.path.exists(dirname):
            os.makedirs( dirname)
        fname = Path( dirname, 'model_id{}.json'.format( wandb.run.id))
        with open(fname, 'w') as f :
            f.write( json_str)

################################################

    def load_json(self, wandb_id) :
        if '/' in wandb_id :   # assumed to be full path instead of just id
            fname = wandb_id
        else :
            fname = Path( self.path_models, 'id{}/model_id{}.json'.format( wandb_id, wandb_id))
        try :
            with open(fname, 'r') as f :
                json_str = f.readlines() 
        except (OSError, IOError) as e:
            # try path used for logging training results and checkpoints
            try :
                fname = Path( self.path_results, '/models/id{}/model_id{}.json'.format(wandb_id,wandb_id))
                with open(fname, 'r') as f :
                    json_str = f.readlines()
            except (OSError, IOError) as e:
                print( f'Could not find fname due to {e}. Aborting.')
                quit()

        self.__dict__ = json.loads( json_str[0])

        # fix for backward compatibility
        if not hasattr( self, 'model_id') :
            self.model_id = self.wandb_id

        return self

####################################################################################################
 
    def get_model_filename(self, model = None, model_id = '', epoch=-2, with_model_path = True) :

        if isinstance( model, str) :
            name = model 
        elif model :
            name = model.__class__.__name__
        else : # backward compatibility
            name = 'mod'

        mpath = 'id{}'.format(model_id) if with_model_path else ''

        if epoch > -2 :
            model_file = Path( self.path_models, mpath, '{}_id{}_epoch{}.mod'.format(
                                                                name, model_id, epoch))
        else :
            model_file = Path( self.path_models, mpath, '{}_id{}.mod'.format( name, model_id))
            
        return model_file

 ####################################################################################################
 
    def add_backward_compatibility(self):
      
        if hasattr( self, 'loader_num_workers') :
            self.num_loader_workers = self.loader_num_workers
        if not hasattr( self, 'n_size'):
            self.n_size = [36, 0.25*9*6, 0.25*9*12] 
        if not hasattr(self, 'num_samples_per_epoch'):
            self.num_samples_per_epoch = 1024
        if not hasattr(self, 'num_samples_validate'):
            self.num_samples_validate = 128
        if not hasattr(self, 'with_mixed_precision'):
            self.with_mixed_precision = True
        if not hasattr(self, 'years_val'):
            self.years_val = self.years_test
        if not hasattr(self, 'batch_size'):
            self.batch_size = self.batch_size_max

        return self
