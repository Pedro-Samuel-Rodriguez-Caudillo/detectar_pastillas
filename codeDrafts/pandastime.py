import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pills= pd.read_csv("Pillbox.csv", low_memory = False)
pills.drop(labels=['ID', 'Enabled?', 'created at', 'updated at', 
'spp', 'setid', 'splsize', 'pillbox_size',
'splscore', 'pillbox_score', 'splimprint', 'pillbox_imprint', 
 'spl_strength', 'spl_ingredients', 
'spl_inactive_ing', 'source', 'rxtty', 'rxstring', 'rxcui', 'RxNorm Update time', 
'product_code', 'part_num', 'ndc9', 'ndc_labeler_code', 
'ndc_product_code','splshape','splcolor', 'marketing_act_code', 'effective_time', 'file_name', 
'equal_product_code', 'dosage_form', 'document_type', 'dea_schedule_code', 'dea_schedule_name', 
'author_type', 'author', 'approval_code', 'image_source', 'splimage', 'has_image', 'epc_match',
 'version_number', 'pillbox_shape_text','pillbox_color_text','part_medicine_name','laberer_code', 'application_number', 'updated', 'stale', 'new', 'Pillbox Value']
          ,axis = 1, inplace= True)
col = pills.pop('medicine_name')
pills.set_index(col, inplace = True)
refinedpills = (pills[(pills['splshape_text'] == 'ROUND') & (pills['splcolor_text'] == "BLUE")])
print(refinedpills)