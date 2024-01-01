# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
import json
import pickle
import pandas as pd
import numpy as np
import torch
import sys
import os
sys.path.append("..") 
sys.path.insert(0, os.path.dirname(__file__))

from torch.utils.data import DataLoader
from utils import  load_model
from utils import get_configure, \
    collate_molgraphs, load_model, predict, load_dataset



def run_an_eval_epoch(smiles_list,args, model, data_loader):
    model.eval()
    predictions = []
    pka_acidic_predictions = []
    pka_basic_predictions = []
    with torch.no_grad():
        for _, batch_data in enumerate(data_loader):
            _, bg, labels, masks = batch_data
            prediction, pka_acidic_prediction, pka_basic_prediction = predict(args, model, bg)
            prediction = prediction.detach().cpu() * args['train_std'].cpu()+args['train_mean'].cpu()
            predictions.append(prediction)
            pka_acidic_predictions.append(pka_acidic_prediction)
            pka_basic_predictions.append(pka_basic_prediction)
        predictions = torch.cat(predictions, dim=0)
        pka_acidic_predictions = torch.cat(pka_acidic_predictions, dim=0)
        pka_basic_predictions = torch.cat(pka_basic_predictions, dim=0)

        output_data = {'canonical_smiles': smiles_list}
        if args['task'] is None:
            args['task'] = ['task_{:d}'.format(t) for t in range(1, args['n_tasks'] + 1)]
        else:
            pass
        for task_id, task_name in enumerate(args['task']):
            output_data[task_name] = predictions[:, task_id]
        output_data['pKa_acidic'] = pka_acidic_predictions[:, 0]
        output_data['pKa_basic'] = pka_basic_predictions[:, 0]
        df = pd.DataFrame(output_data)
        # out=pd.read_csv("C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\original_data\logp.csv")

        print(f"Predicted pKa acid: {output_data['pKa_acidic'].item()}")
        print(f"Predicted pKa amine: {output_data['pKa_basic'].item()}")
        print(f"Predicted logP: {output_data['logp'].item()}")
        # out.to_csv('C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\original_data\logp_evaluated.csv', index=False)



def main(smiles_list,args, exp_config, test_set):
    # Record settings
    exp_config.update({
        'model': args['model'],
        'mode':args['mode'],
        'n_tasks': args['n_tasks'],
        'atom_featurizer_type': args['atom_featurizer_type'],
        'bond_featurizer_type': args['bond_featurizer_type']
    })
    if args['atom_featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()+2
    if args['edge_featurizer'] is not None and args['bond_featurizer_type'] != 'pre_train':
        exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()

    t0 = time.time()
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    model = load_model(exp_config).to(args['device'])
    # checkpoint = torch.load(r"C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/model_pretrain_76.pth",map_location=torch.device('cpu'))#my_model/CRT76-logD/model_pretrain_76.pth"
    checkpoint = torch.load(r"C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model\RTlogD\model_pretrain_76.pth",map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    run_an_eval_epoch(smiles_list,args, model, test_loader)

    print('It took {:.4f}s to complete the task'.format(time.time() - t0))

SMILE = 'OC(C1C2(CC2)C1)=O'
   
if __name__ == '__main__':
    import torch
    from utils import setup
    with open(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/args.pickle', 'rb') as file:
        args =pickle.load(file)
    with open(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args = setup(args)
    test_set_dict = {'smiles': [SMILE]}
    df_test_set = pd.DataFrame(test_set_dict)
    df_test_set['logp']=np.nan
    df_test_set['standard_value']=np.nan
    df_test_set['exp']=np.nan
    test_set = df_test_set.copy()
    smiles_list=test_set['smiles'].to_list()
    test_set = load_dataset(args,test_set,"test")
    exp_config = get_configure(args['model'],"test")
    main(smiles_list,args, exp_config,test_set)