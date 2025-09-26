import pandas as pd
import random   
import string

def load_data(dir=None, prob=None, seed=None, tasks=None):

    # load IDs
    IDs = pd.read_parquet(dir + 'IDs.pq')   
    print(f"Total number of IDs: {len(IDs)}")
    # sample IDs
    IDs = IDs['ID'].sample(frac=prob, random_state=seed).tolist()
    print(f"Number of selected IDs: {len(IDs)}")

    # create mapping for re-anonymization
    id_mapping = re_anonymize(IDs=IDs, seed=seed)

    # loop over tasks, load data and select IDs
    # for every task, print head of the data
    for task in tasks:

        # make an exception for task == DM
        if task == 'DM':
            data = pd.read_csv("C:/Users/20200059/Documents/Data/FUNA/Received/FUNA ExceptionalModelMiningData2021 DM.txt", sep='\t')
        elif task == 'StudentDescs':
            data = pd.read_parquet(dir + 'descriptive_basic.pq')
            # keep first row for every student
            data = data.drop_duplicates(subset=['IDCode'], keep='first')
        else:
            data = pd.read_parquet(dir + f'{task}.pq')

        # select only the sampled IDs
        data = data[data['IDCode'].isin(IDs)]

        # prepare columns
        data = prepare_columns(task=task, data=data)

        # re-anonymize the data using the mapping
        data['Student'] = data['Student'].map(id_mapping)
        # count number of unique students
        print(f"Number of unique students in {task}: {data['Student'].nunique()}, {data.shape}")

        # sort by Student and Item if Item exists
        if 'Item' in data.columns:
            data = data.sort_values(by=['Student', 'Item'])
        else:
            data = data.sort_values(by=['Student'])

        #print(data.head())
    
        # store data as txt
        data.to_csv(dir + 'Takeoff/' + f'{task}.txt', sep='\t', index=False)

    # store the mapping as txt
    mapping_df = pd.DataFrame(list(id_mapping.items()), columns=['OriginalID', 'AnonymizedID'])
    mapping_df.to_csv('C:/Users/20200059/Documents/Data/FUNA/Received/' + 'Takeoff/' + 'ID_Mapping.txt', sep='\t', index=False)

def prepare_columns(task=None, data=None):
        
    # if column name starts with the task name, do something
    if any(col.startswith(f'{task}') for col in data.columns):
        data = data.rename(columns=lambda x: x.replace(f'{task}', '') if x.startswith(f'{task}') else x) 
    # if column name exist, do something
    if 'IDCode' in data.columns:
        data = data.rename(columns={'IDCode': 'Student'}) 
    if 'PreOrd' in data.columns:
        data = data.rename(columns={'PreOrd': 'Item'})  
    # if column names contains one of list of substrings, remove them
    cols_to_remove = [col for col in data.columns if any(substring in col for substring in ['Start', 'End'])]
    # add some more column names to be removed
    cols_to_remove.extend(['CourseName', 'TeacherIDCode', 'Cycle'])
    data = data.drop(columns=[col for col in cols_to_remove if col in data.columns], errors='ignore')
    # if there are any column names that do no start with a capital letter, capitalize the first letter
    data.columns = [col.capitalize() if col and not col[0].isupper() else col for col in data.columns]

    return(data)

def re_anonymize(IDs=None, seed=None):

    # create new list of ID strings based on random sampling from ascii letters and digits
    random.seed(seed)
    id_length = 8

    newIDs = {''.join(random.choices(string.ascii_letters + string.digits, k=id_length)) for _ in range(len(IDs))}

    id_mapping = {old_id: new_id for old_id, new_id in zip(IDs, newIDs)}

    return id_mapping    
       
load_data(dir="C:/Users/20200059/Documents/Data/FUNA/FUNA_EMM/",
    prob=0.1, #15486 Students, for now we can work with about 1500, as if it were one school
    seed=26092025,
    tasks=['NC','CA','SA','SS','DM','StudentDescs'])