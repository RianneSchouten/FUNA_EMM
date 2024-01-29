import pandas as pd
import preprocess_functions as pf

def into_desc(dd=None, db=None, info=None):

    print('transform into desc')

    #print(dd.keys()) # the four descriptor datasets
    #print(db.head()) # the 3 basic descriptors
    dddesc, dddesc_adapt_to_target = transform_into_desc(dd=dd, info=info)     

    # add basic descriptors    
    descriptive_desc = pd.merge(dddesc, db.drop_duplicates(), how = 'left') # there is only one similar column: IDCode

    descriptive_desc_adapt_to_target = pd.merge(dddesc_adapt_to_target, db.drop_duplicates(), how = 'left') # there is only one similar column: IDCode

    info.update({'shape_desc_0': descriptive_desc.shape[0], 'shape_desc_1': descriptive_desc.shape[1], 'na_desc_rows': descriptive_desc.isnull().any(axis=1).sum(), 'na_desc_overall': descriptive_desc.isnull().sum().sum()})

    info.update({'shape_desc18_0': descriptive_desc_adapt_to_target.shape[0], 'shape_desc18_1': descriptive_desc_adapt_to_target.shape[1], 'na_desc18_rows': descriptive_desc_adapt_to_target.isnull().any(axis=1).sum(), 'na_desc18_overall': descriptive_desc_adapt_to_target.isnull().sum().sum()})

    return descriptive_desc, info, descriptive_desc_adapt_to_target

def transform_into_desc(dd=None, info=None):

    # first on entire dataset
    # groupby and create desc columns
    ddgrouped = groupby_and_create_desc_columns(dd=dd, selection=False)    
    dddesc = merge_into_desc(dd=ddgrouped)   

    # then for the items corresponding with target
    ddgrouped18 = groupby_and_create_desc_columns(dd=dd, selection=True, info=info)  
    dddesc_adapt_to_target = merge_into_desc(dd=ddgrouped18)   

    return dddesc, dddesc_adapt_to_target

def groupby_and_create_desc_columns(dd=None, selection=None, info=None):

    ddgrouped = {}
    
    for key in dd.keys():
        data = dd[key].copy()
        if selection:
            datadm = info['DMIDCodeDMPreOrd']
            datasel = data[data.set_index(['IDCode',key+'PreOrd']).index.isin(datadm.set_index(['IDCode','DMPreOrd']).index)].reset_index(drop=True)
            data = datasel.copy()
            #data = data[data[key+'PreOrd']<19]
        datagrouped = data.groupby('IDCode').apply(pf.desc_columns_per_case, name_task=key)
        datagrouped_scaled = pf.min_max_scaling(df=datagrouped)
        #print(datagrouped_scaled)
        #print(datagrouped_scaled.shape)
        datagrouped_scaled.reset_index(inplace=True)
        ddgrouped[key] = datagrouped_scaled

    return ddgrouped

def merge_into_desc(dd=None):

    # start with NC
    data = dd['NC'].copy()

    for key in dd.keys():
        if not key == 'NC':
            adddata = dd[key].copy()
            data = pd.merge(data, adddata, how = 'left') # there is only one similar column: IDCode

    return data

