import cobra
path = '/home/acabbia/Documents/Muscle_Model/models/muscle_old_sedentary_trained/'
newpath = '/home/acabbia/Documents/Muscle_Model/models/muscle_old_sedentary_trained+EX_glc/'

carbon_sources = ['EX_glc(e)','EX_fru(e)','EX_octa(e)','EX_lnlnca(e)','EX_hdcea(e)','EX_lnlc(e)',
                      'EX_ttdca(e)','EX_doco13ac_','EX_hdca(e)','EX_ocdca(e)','EX_arach(e)','EX_lgnc(e)']

ref_model = cobra.io.read_sbml_model('/home/acabbia/Documents/Muscle_Model/models/recon2.2.xml')

for filename in sorted(os.listdir(path)):
        model = cobra.io.read_sbml_model(path+filename)

        for c in carbon_sources:
            try:
                model.reactions.get_by_id(c)

            except:
                print(c, 'not in model')
                model.add_reaction(ref_model.reactions.get_by_id(c))
                print('reaction ', c, ' added')
                try:
                    model.reactions.get_by_id(c)
                    print('OK')
                    cobra.io.write_sbml_model(model,newpath+filename)
                except:
                    print('something is wrong...')

            print('--------------------------------')  