#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:25:47 2018

@author: acabbia
"""

### RECON3 version

import cobra
import GEOparse
from builder_R3 import Builder 

#set paths
path = '/home/acabbia/Documents/Muscle_Model/models'
ref_model = '/home/acabbia/Documents/Muscle_Model/models/Recon3D.xml'
#ref2 =  '/home/acabbia/Documents/Muscle_Model/models/recon2.2.xml'

output_folder = path + "/library_muscle_R3/"

# import reference model (RECON 3.01)
recon3 = cobra.io.read_sbml_model(ref_model)
#recon2 = cobra.io.read_sbml_model(ref2)

#Gene expression data GEO ID (Raue 2012 dataset1)
GEO_accession_nr = "GSE28422"

#get data from GEO

#serie=GEOparse.get_GEO(geo=GEO_accession_nr)
serie=GEOparse.get_GEO(filepath='/home/acabbia/Documents/Muscle_Model/tissueModels-distance/GSE28422_family.soft.gz')

# required reactions  
#'ATPS4mi','ENO','PDHm','PYK','G3PD1','G6PDH2r','AKGDm','CYOOm3','r0913' + exchanges

requirements = ['EX_glc(e)','EX_hdca(e)','EX_ocdca(e)','EX_arach(e)','EX_doco13ac_','EX_lgnc(e)','EX_ala_L(e)',
                         'EX_arg_L(e)','EX_asn_L(e)','EX_asp_L(e)','EX_cys_L(e)','EX_gln_L(e)','EX_glu_L(e)',
                         'EX_gly(e)','EX_his_L(e)','EX_ile_L(e)','EX_leu_L(e)','EX_lys_L(e)','EX_met_L(e)',
                         'EX_phe_L(e)','EX_pro_L(e)','EX_ser_L(e)','EX_thr_L(e)','EX_trp_L(e)','EX_tyr_L(e)',
                         'EX_val_L(e)','EX_fru(e)','EX_ppa(e)','EX_but(e)','EX_hx(e)','EX_octa(e)','EX_ttdca(e)',
                         'EX_h2o(e)','EX_hco3(e)','EX_co2(e)','EX_h(e)']


#%%
# Extract list of models to build (only old patients, timepoints 1 and 3)

for n in serie.gsms:

    name = serie.gsms[n].metadata['title'][0]
     
    if 'Old' in name and ('T1' in name or 'T3' in name):
        
         print('Building model ', name)

        #build

        eset = serie.gsms[gsm].table
        confidence = Builder.rxn_confidence(recon3, eset, translator,'hgnc_id', q = 0.6)

        for r_id in requirements:
            confidence[r_id] = 3

        newmodel = Builder.build_model(recon3,confidence)
        newmodel.name = 'Aging Skeletal Muscle '+ gsm + label[1]+label[2]

        '''
        # fix some reactions if present
        try:
            # Reactions to be corrected (h_i)
            newmodel.reactions.CYOOm3.add_metabolites({'h_m':-7.9,'h_i':4 ,'h_c':0}, combine=False)
            
            # Fix PDH reversibility/directionality
            newmodel.reactions.PDHm.bounds = 0, 1000
        except:
            pass 
        
        '''
        #prune unused metabolites
        remov = cobra.manipulation.delete.prune_unused_metabolites(newmodel)    
        while len(remov) != 0:
            remov = cobra.manipulation.delete.prune_unused_metabolites(newmodel)
        
        #write newmodel in SBML
        cobra.io.write_sbml_model(newmodel,filename= output_folder+'ASK_'+label[1][0]+label[2][0]+'_'+gsm[-2:]+'.xml')
        
        print('===============================')

    else:
        continue

#%%
#################################################################################################################
# main loop builds model for each subject in serie
print('===============================')
for gsm in serie.gsms:

    label = str(serie.gsms[gsm].metadata['title']).split('_')

    print('Building model '+ 'ASK_'+label[1][0]+label[2][0]+'_'+gsm[-2:])

    eset = serie.gsms[gsm].table
    confidence = Builder.rxn_confidence(recon3, eset, translator,'hgnc_id', q = 0.6)

    for r_id in requirements:
        confidence[r_id] = 3

    newmodel = Builder.build_model(recon3,confidence)
    newmodel.name = 'Aging Skeletal Muscle '+ gsm + label[1]+label[2]

    # fix some reactions if present
    try:
        # Reactions to be corrected (h_i)
        newmodel.reactions.CYOOm3.add_metabolites({'h_m':-7.9,'h_i':4 ,'h_c':0}, combine=False)
        
        # Fix PDH reversibility/directionality
        newmodel.reactions.PDHm.bounds = 0, 1000
    except:
        pass 
    
    #prune unused metabolites
    remov = cobra.manipulation.delete.prune_unused_metabolites(newmodel)    
    while len(remov) != 0:
        remov = cobra.manipulation.delete.prune_unused_metabolites(newmodel)
    
    #write newmodel in SBML
    cobra.io.write_sbml_model(newmodel,filename= output_folder+'ASK_'+label[1][0]+label[2][0]+'_'+gsm[-2:]+'.xml')
    
    print('===============================')

###############################################################################################################

for r in recon3.reactions:
    print(r.genes)