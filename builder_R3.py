import corda
import pandas as pd
import biomart
from cobra import Reaction, Metabolite
import cobrababel
import numpy as np
from interruptingcow import timeout


def process_table(table):

    print('Translating probes_id...')
    
    table.columns = ['NCBI_gene_id', 'Exp_value', 'PA','pval']
    table.drop(['PA','pval'],axis=1,inplace=True )
    probe_list = [id.split('_')[0] for id in table['NCBI_gene_id'].values]
    table['NCBI_gene_id'].values = probe_list

    return table
        
def rxn_confidence(model, table, translator , gene_id, q):
    # Creates confidence score for each reaction of the model       
        
    # initialize confidence dictionary 
    confidence = {r.id: r.gene_reaction_rule for r in model.reactions}
    
    # Average duplicated genes (avereps)
    table = table.groupby('NCBI_gene_id').mean().reset_index()
    
    # log normalize expression 
    table['Exp_value'] = table['Exp_value'].transform(np.log1p)

    #initialize confidence in table df, assign score = 3 to probes with exp_value > q
    table['confidence']=-1
    idx = [table['Exp_value'] > table['Exp_value'].quantile(q)]
    table.loc[idx[0],'confidence'] = 3
    
    gene_confidence=pd.Series(table.confidence.values,index=table.iloc[:,0].values).to_dict()
    
    ## Evaluate gprs
    for k , v  in confidence.items():
        if isinstance(v, str):
            confidence[k]=(corda.util.reaction_confidence(v,gene_confidence))
        else:
            confidence[k]=-1  ## Reaction with no associated GPR rule have confidence score = -1
    for k,v in confidence.items():
        if v == 0:
            confidence[k]=-1  ### all reactions except high confidence ones have score = -1 
    return confidence 

def build_model(model, confidence):
    with timeout(1800, exception= TimeoutError):
    try:
        opt = corda.CORDA(model, confidence , n=3)
        opt.tflux = 1e-3
        opt.build()
        print(opt)
        newmodel=opt.cobra_model()        
        return newmodel
    except TimeoutError:
        raise TimeoutError