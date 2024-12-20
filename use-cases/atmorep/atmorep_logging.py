####################################################################################################
#
#  Copyright (C) 2022
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
# 
#  description :
#
#  license     :
#
####################################################################################################

# import torch
# import torchinfo
# import numpy as np
# import time
# import code

# from pathlib import Path
# import os
# import datetime
# import functools

# import wandb
# # import horovod.torch as hvd
# import torch.distributed as dist
# from torch.distributed.optim import ZeroRedundancyOptimizer
# import torch.utils.data.distributed

# import atmorep.config.config as config

# from atmorep.core.atmorep_model import AtmoRep
# from atmorep.core.atmorep_model import AtmoRepData
# from atmorep.training.bert import prepare_batch_BERT_multifield
# from atmorep.transformer.transformer_base import positional_encoding_harmonic

# import atmorep.utils.token_infos_transformations as token_infos_transformations

# from atmorep.utils.utils import Gaussian, CRPS, kernel_crps, weighted_mse, NetMode, tokenize, detokenize, unique_unsorted
# from atmorep.datasets.data_writer import write_forecast, write_BERT, write_attention
# from atmorep.datasets.normalizer import denormalize


# class Logging(object):
#   def __init__(self, cf, sources_info, sources_idxs, fields_prediction_idx, normalizers):

#       self.cf = cf
#       self.sources_info = sources_info
#       self.sources_idxs = sources_idxs
#       self.fields_prediction_idx = fields_prediction_idx
#       self.normalizers = normalizers

#   ###################################################
#   def log_validate( self, epoch, bidx, log_sources, log_preds) :
#     '''Hook for logging: output associated with concrete training strategy.'''

#     if not hasattr( self.cf, 'wandb_id') :
#       return

#     if 'forecast' in self.cf.BERT_strategy :
#       self.log_validate_forecast( epoch, bidx, log_sources, log_preds)
#     elif 'BERT' in self.cf.BERT_strategy or 'temporal_interpolation' == self.cf.BERT_strategy :
#       self.log_validate_BERT( epoch, bidx, log_sources, log_preds)
#     else :
#       assert False
  
#   ###################################################
#   def log_validate_forecast( self, epoch, batch_idx, log_sources, log_preds) :
#     '''Logging for BERT_strategy=forecast.'''

#     cf = self.cf

#     # save source: remains identical so just save ones
#     (sources, targets, _) = log_sources

#     sources_out, targets_out, preds_out, ensembles_out = [ ], [ ], [ ], [ ] 
#     batch_size = len(self.sources_info)  
#     # reconstruct geo-coords (identical for all fields)
#     forecast_num_tokens = 1
#     if hasattr( cf, 'forecast_num_tokens') :
#       forecast_num_tokens = cf.forecast_num_tokens
  
#     coords = []
#     for fidx, field_info in enumerate(cf.fields) :
#       # reshape from tokens to contiguous physical field
#       num_levels = len(field_info[2])
#       source = detokenize( sources[fidx].cpu().detach().numpy())
#       # recover tokenized shape
#       target = detokenize( targets[fidx].cpu().detach().numpy().reshape( [ num_levels, -1, 
#                            forecast_num_tokens, *field_info[3][1:], *field_info[4] ]).swapaxes(0,1))
     
#       coords_b = []
  
#       for bidx in range(batch_size):
#         dates   = self.sources_info[bidx][0]
#         lats    = self.sources_info[bidx][1]
#         lons    = self.sources_info[bidx][2]
#         dates_t = self.sources_info[bidx][0][ -forecast_num_tokens*field_info[4][0] : ]
        
#         lats_idx = self.sources_idxs[bidx][1]
#         lons_idx = self.sources_idxs[bidx][2]

#         for vidx, _ in enumerate(field_info[2]) :
#           normalizer, year_base = self.get_normalizer(self.normalizers, fidx, vidx, lats_idx, lons_idx) 
#           source[bidx,vidx] = denormalize( source[bidx,vidx], normalizer, dates, year_base)
#           target[bidx,vidx] = denormalize( target[bidx,vidx], normalizer, dates_t, year_base)

#         coords_b += [[dates, 90.-lats, lons, dates_t]]

#       # append
#       sources_out.append( [field_info[0], source])
#       targets_out.append( [field_info[0], target])
#       coords.append(coords_b)

#     # process predicted fields
#     for fidx, fn in enumerate(cf.fields_prediction) :
#       field_info = cf.fields[ self.fields_prediction_idx[fidx] ]
#       num_levels = len(field_info[2])
#       # predictions
#       pred = log_preds[fidx][0].cpu().detach().numpy()
#       pred = detokenize( pred.reshape( [ num_levels, -1, 
#                                     forecast_num_tokens, *field_info[3][1:], *field_info[4] ]).swapaxes(0,1))
#       # ensemble
#       ensemble = log_preds[fidx][2].cpu().detach().numpy().swapaxes(0,1)
#       ensemble = detokenize( ensemble.reshape( [ cf.net_tail_num_nets, num_levels, -1, 
#                                             forecast_num_tokens, *field_info[3][1:], *field_info[4] ]).swapaxes(1, 2)).swapaxes(0,1)
      
#       # denormalize
#       for bidx in range(batch_size) : 
#         lats  = self.sources_info[bidx][1]
#         lons  = self.sources_info[bidx][2]
#         dates_t = self.sources_info[bidx][0][ -forecast_num_tokens*field_info[4][0] : ]
       
#         for vidx, vl in enumerate(field_info[2]) :
#           normalizer, year_base = self.get_normalizer(self.normalizers, self.fields_prediction_idx[fidx], vidx, lats_idx, lons_idx)
#           pred[bidx,vidx] = denormalize( pred[bidx,vidx], normalizer, dates_t, year_base)
#           ensemble[bidx,:,vidx] = denormalize(ensemble[bidx,:,vidx], normalizer, dates_t, year_base)

#       # append
#       preds_out.append( [fn[0], pred])
#       ensembles_out.append( [fn[0], ensemble])

#     levels = np.array(cf.fields[0][2])
    
#     write_forecast( cf.wandb_id, epoch, batch_idx,
#                                  levels, sources_out,
#                                  targets_out, preds_out,
#                                  ensembles_out, coords)
  
#   ###################################################
  
#   def split_data(self, data, idx_list, token_size) :
#     lens_batches = [[len(t) for t in tt] for tt in idx_list]
#     lens_levels = [torch.tensor( tt).sum() for tt in lens_batches]
#     data_b = torch.split( data, lens_levels)    
#     # split according to batch
#     return [torch.split( data_b[vidx], lens) for vidx,lens in enumerate(lens_batches)]

#   def get_masked_data(self, field_info, data, idx_list, ensemble = False):
  
#     cf = self.cf
#     batch_size = len(self.sources_info)  
#     num_levels = len(field_info[2])
#     num_tokens = field_info[3]
#     token_size = field_info[4]
#     data_b  =  self.split_data(data, idx_list, token_size)
   
#     # recover token shape
#     if ensemble:
#       return [[data_b[vidx][bidx].reshape([-1, cf.net_tail_num_nets, *token_size])
#                                                             for bidx in range(batch_size)]
#                                                             for vidx in range(num_levels)]
#     else:
#       return [[data_b[vidx][bidx].reshape([-1, *token_size]) for bidx in range(batch_size)]
#                                                              for vidx in range(num_levels)]

#   ###################################################
#   def log_validate_BERT( self, epoch, batch_idx, log_sources, log_preds) :
#     '''Logging for BERT_strategy=BERT.'''

#     cf = self.cf
#     batch_size = len(self.sources_info) 

#     # save source: remains identical so just save ones
#     (sources, targets, tokens_masked_idx_list) = log_sources

#     sources_out, targets_out, preds_out, ensembles_out = [ ], [ ], [ ], [ ]
#     coords = []

#     for fidx, field_info in enumerate(cf.fields) : 

#       # reconstruct coordinates
#       is_predicted = fidx in self.fields_prediction_idx
#       num_levels = len(field_info[2])
#       num_tokens = field_info[3]
#       token_size = field_info[4]
#       sources_b = detokenize( sources[fidx].numpy())
     
#       if is_predicted :
#         targets_b   = self.get_masked_data(field_info, targets[fidx], tokens_masked_idx_list[fidx])
#         preds_mu_b  = self.get_masked_data(field_info, log_preds[fidx][0], tokens_masked_idx_list[fidx])
#         preds_ens_b = self.get_masked_data(field_info, log_preds[fidx][2], tokens_masked_idx_list[fidx], ensemble = True)

#       # for all batch items
#       coords_b = []
#       for bidx in range(batch_size):
#         dates = self.sources_info[bidx][0]
#         lats  = self.sources_info[bidx][1]
#         lons  = self.sources_info[bidx][2]

#         lats_idx = self.sources_idxs[bidx][1]
#         lons_idx = self.sources_idxs[bidx][2]

#         # target etc are aliasing targets_b which simplifies bookkeeping below
#         if is_predicted :
#           target   = [targets_b[vidx][bidx] for vidx in range(num_levels)]
#           pred_mu  = [preds_mu_b[vidx][bidx] for vidx in range(num_levels)]
#           pred_ens = [preds_ens_b[vidx][bidx] for vidx in range(num_levels)]

#         coords_mskd_l = []
#         for vidx, _ in enumerate(field_info[2]) :

#           normalizer, year_base = self.model.normalizer( fidx, vidx, lats_idx, lons_idx)
#           sources_b[bidx,vidx] = denormalize(sources_b[bidx,vidx], normalizer, dates, year_base = 2021) 

#           if is_predicted :
#             idx = tokens_masked_idx_list[fidx][vidx][bidx]
#             grid = np.flip(np.array( np.meshgrid( lons, lats)), axis = 0) #flip to have lat on pos 0 and lon on pos 1
#             grid_idx = np.flip(np.array( np.meshgrid( lons_idx, lats_idx)), axis = 0) #flip to have lat on pos 0 and lon on pos 1
       
#             # recover time dimension since idx assumes the full space-time cube
#             grid = torch.from_numpy( np.array( np.broadcast_to( grid,
#                                 shape = [token_size[0]*num_tokens[0], *grid.shape])).swapaxes(0,1))
#             grid_lats_toked = tokenize( grid[0], token_size).flatten( 0, 2)
#             grid_lons_toked = tokenize( grid[1], token_size).flatten( 0, 2)
          
#             idx_loc = idx - np.prod(num_tokens) * bidx
#             #save only useful info for each bidx. shape e.g. [n_bidx, lat_token_size*lat_num_tokens]
#             lats_mskd = np.array([unique_unsorted(t) for t in grid_lats_toked[ idx_loc ].numpy()])
#             lons_mskd = np.array([unique_unsorted(t) for t in grid_lons_toked[ idx_loc ].numpy()])

#             #time: idx ranges from 0->863 12x6x12 
#             t_idx = (idx_loc // (num_tokens[1]*num_tokens[2])) * token_size[0]
#             #create range from t_idx-2 to t_idx
#             t_idx = np.array([np.arange(t, t + token_size[0]) for t in t_idx])
#             dates_mskd = dates[t_idx]
            
#             for ii,(t,p,e,da,la,lo) in enumerate(zip( target[vidx], pred_mu[vidx], pred_ens[vidx],
#                                                     dates_mskd, lats_mskd, lons_mskd)) :
#               normalizer_ii = normalizer
#               if len(normalizer.shape) > 2: #local normalization                                     
#                 lats_mskd_idx = np.where(np.isin(lats,la))[0]
#                 lons_mskd_idx = np.where(np.isin(lons,lo))[0]
#                 #normalizer_ii = normalizer[:, :, lats_mskd_idx, lons_mskd_idx] problems in python 3.9
#                 normalizer_ii = normalizer[:, :, lats_mskd_idx[0]:lats_mskd_idx[-1]+1, lons_mskd_idx[0]:lons_mskd_idx[-1]+1]
             
#               targets_b[vidx][bidx][ii]   = denormalize(t, normalizer_ii, da, year_base)  
#               preds_mu_b[vidx][bidx][ii]  = denormalize(p, normalizer_ii, da, year_base) 
#               preds_ens_b[vidx][bidx][ii] = denormalize(e, normalizer_ii, da, year_base)
            
#             coords_mskd_l += [[dates_mskd, 90.-lats_mskd, lons_mskd] ]
       
#         coords_b += [ [dates, 90. - lats, lons] + coords_mskd_l ]
      
#       coords += [ coords_b ]
#       fn = field_info[0]
#       sources_out.append( [fn, sources_b])

#       targets_out.append([fn, [[t.numpy(force=True) for t in t_v] for t_v in targets_b]] if is_predicted else [fn, []])
#       preds_out.append( [fn, [[p.numpy(force=True) for p in p_v] for p_v in preds_mu_b]] if is_predicted else [fn, []] )
#       ensembles_out.append( [fn, [[p.numpy(force=True) for p in p_v] for p_v in preds_ens_b]] if is_predicted else [fn, []] )

#     levels = [[np.array(l) for l in field[2]] for field in cf.fields]
#     write_BERT( cf.wandb_id, epoch, batch_idx, 
#                 levels, sources_out, targets_out,
#                 preds_out, ensembles_out, coords )

# ######################################################

#   def log_attention( self, epoch, bidx, attention) : 
#     '''Hook for logging: output attention maps.'''
#     cf = self.cf

#     attn_out = []
#     for fidx, field_info in enumerate(cf.fields) : 
     
#       # coordinates 
#       coords_b = []
#       for bidx in range(batch_size):
#         dates = self.sources_info[bidx][0]
#         lats  = 90. - self.sources_info[bidx][1]
#         lons  = self.sources_info[bidx][2]
#         coords_b += [ [dates, lats, lons] ]

#       is_predicted = fidx in self.fields_prediction_idx
#       attn_out.append([field_info[0], attention[fidx]] if is_predicted else [fn, []])
      
#     levels = [[np.array(l) for l in field[2]] for field in cf.fields]
#     write_attention(cf.wandb_id, epoch,
#                     bidx, levels, attn_out,  coords_b )

# ######################################################

# def get_normalizer( self, normalizers, field, vl_idx, lats_idx, lons_idx ) :

#     if isinstance( field, str) :
#       for fidx, field_info in enumerate(self.cf.fields) :
#         if field == field_info[0] :
#           break
#       assert fidx < len(self.cf.fields), 'invalid field'
#       normalizer = normalizers[fidx]

#     elif isinstance( field, int) :
#       normalizer = self.dataset_train.normalizers[field][vl_idx]
#       if len(normalizer.shape) > 2:
#         normalizer = np.take( np.take( normalizer, lats_idx, -2), lons_idx, -1)
#     else :
#       assert False, 'invalid argument type (has to be index to cf.fields or field name)'
    
#     year_base = self.dataset_train.year_base

#     return normalizer, year_base