#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:52:40 2018
@author: acabbia
"""

import cobra
import GEOparse
from reconstruction import affyprobe_translator , rxn_confidence , build_model
from functions import req_list

# set paths

ref_model = 'models/recon2.2.xml'
library_folder = 'models/muscle_old_sedentary_trained/'

# Gene expression data GEO ID (Raue 2012)
GEO_accession = 'GSE28422'

# cutoff level for gene expression
expr_cutoff = 0.7

# load resources
ref_model = cobra.io.read_sbml_model(ref_model)
serie = GEOparse.get_GEO(geo=GEO_accession)

# build translator dict
table = serie.gsms[list(serie.gsms.keys())[0]].table
translator = affyprobe_translator(table, 'hgnc_id')

# make list of required reactions

requirements_int = ['ATPS4m', 'ENO', 'PDHm', 'PYK',
                    'G3PD1', 'G6PDH2r', 'AKGDm', 'CYOOm3', 'r0913', 'GLCt2_2']
requirements_ex = req_list(ref_model, 'fluxes.tsv')
requirements = requirements_int + requirements_ex

# %%
#################################################################################################################
# main loop

for gsm in serie.gsms:

    name = serie.gsms[gsm].metadata['title'][0]

    if 'Old' in name and ('T1' in name or 'T3' in name):

        label = str(serie.gsms[gsm].metadata['title']).split(' ')

        print('Building model ' + 'SKM_' + label[0].split('_')[0]+label[1])

        eset = serie.gsms[gsm].table

        confidence = rxn_confidence(
            ref_model, eset, translator, 'hgnc_id', expr_cutoff)

        # Ensure requred reactions are included
        for r_id in requirements:
            confidence[r_id] = 3

        newmodel = build_model(ref_model, confidence)
        newmodel.name = 'Skeletal Muscle ' + label[0].split('_')[0]+label[1]

        # Reactions to be corrected (h_i)
        newmodel.reactions.CYOOm3.add_metabolites(
            {'h_m': -7.9, 'h_i': 4, 'h_c': 0}, combine=False)

        # Fix PDH reversibility/directionality
        newmodel.reactions.PDHm.bounds = 0, 1000

        # prune unused metabolites
        remov = cobra.manipulation.delete.prune_unused_metabolites(newmodel)

        while len(remov[1]) != 0:
            remov = cobra.manipulation.delete.prune_unused_metabolites(
                newmodel)

        # write newmodel in SBML
        cobra.io.write_sbml_model(
            newmodel, filename=library_folder+'SKM_old'+label[0].split('_')[0]+label[1]+'.xml')
        print('===============================')

    ###############################################################################################################


# %%
