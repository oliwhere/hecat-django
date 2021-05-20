from dss import dex
# from dss import esco_utils
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import bisect
import networkx as nx

import myapp.utils as utils


# FIX THIS
dexmodel = dex.DEXModel('./data/SKP Evaluation version 3.xml')
# esco = esco_utils.ESCOUtil()
occupations = pd.read_excel('./data/SKP_ESCO.xlsx')
napoved_year = 2018
napoved_period = 'I'

skp_skills = utils.get_skp_skills()#pd.read_pickle('./data/skp_skills_%d-%s.pcl' % (napoved_year,napoved_period) )
res = pd.read_pickle('./data/res_merged_2018.pcl')
id_distance_time = pickle.load( open('./data/id_dist_time.pcl','rb') )
# job_contract_mer = pd.read_pickle('./data/elise/job_contract_type.pcl')
# job_working_hours_mer = pd.read_pickle('./data/elise/job_job_working_hours.pcl')
DG = nx.read_gpickle('./data/elise/career_graph.pcl')
# FIX END

# NEW data_qq
complete_skills_dict = pickle.load(open('./data/complete_skills_dict.pcl','rb'))
#NEW end

data = dict()
data['Languages'] = 'yes'
data['Driving licence'] = 'yes'
data['Age appropriateness'] = 'yes'
data['Disability appropriateness'] = 'yes'

#SKP Wish
#SKPvsESCO
# data['BO wishes for contract type'] = 0
#Job contract type

# data['BO career wishes'] = 0
#Job career advancement

# data['BO working hours wishes'] = 0
#Job working hours
#MSO Location
# data['BO wish location'] = 'yes'


skp_code = 3434.01 # Input
up_enota = 50 # Sk.Loka
data['BO wishes for contract type'] = 'full time'
# Career
data['BO career wishes'] = '*'
data['BO working hours wishes'] = 'daily shift'

wishes = [5120,5151,3322]
wishes_location = [59,60,61]


# Data preparation
bo_lang = ['AN','NE']
bo_driving_lic = ['B1','B']

sel_cols_lic = ['DLIC_%s' % x for x in bo_driving_lic]
sel_cols_lang = ['LANG_%s' % x for x in bo_lang]



df_lang_dict = pd.read_pickle('./data/skp_lang.pcl')

inter = utils.get_intermediate_results()


lic_lang = pd.DataFrame(data={'SKP-6':inter['SKP-6'],
                   'dlic': inter.loc[:,sel_cols_lic].sum(axis=1)/inter.loc[:,'SUM_DLIC'],
                  'dlang': inter.loc[:,sel_cols_lang].sum(axis=1)/inter.loc[:,'SUM_LANG'],
                             'delovni čas':inter['delovni čas'],
                             'koda Urnik dela':inter['koda Urnik dela'],
                             'skills':inter['skills']})#.fillna(1)

lic_lang[['dlic','dlang']] = lic_lang[['dlic','dlang']].fillna(1)

#End data prepration



default = dexmodel.get_intput_attributes()




# There is a case when SKP4 is listed. SKP-6 has two 'decimal' digits. This line checks whether the code is SKP4 or SKP6
# There is a case when SKP4 is listed. SKP-6 has two 'decimal' digits. This line checks whether the code is SKP4 or SKP6
col_name = 'SKP koda-4' if int(skp_code) == skp_code else 'SKP koda-6'
#     bo_skills = esco.get_all_skills_SKP2ESCO(occupations,  skp_code, col_name)
bo_skills = np.array([])
for uri in occupations[occupations[col_name] == skp_code]["URI"].unique():
    bo_skills = np.union1d(complete_skills_dict[uri]['basic'],complete_skills_dict[uri]['optional'])

dif_skills = skp_skills.apply(lambda x: len(np.setdiff1d(x.skills, bo_skills)), axis=1)
resSKPs = utils.select_positions(res, up_enota, id_distance_time)
dif_df = pd.DataFrame({'SKP-6':skp_skills['SKP-6'],'diff':dif_skills})
dex_df = pd.merge(resSKPs, dif_df, on='SKP-6')
dex_df = pd.merge(dex_df, lic_lang, how='left', on='SKP-6')

# rem_SL = langmer['koda Tuji jeziki'] != 'SL'
# langmer_fix = langmer[rem_SL]

mask_dlic = inter.columns.str.contains('DLIC_.*')
mask_lang = inter.columns.str.contains('LANG_.*')







def check(r):
    data_val = dict()
    all_eval = dict()
    all_qq = dict()
    vals = r[1]
    if 0 in wishes:
        data['SKP Wish'] = '*'
    else:
        data['SKP Wish'] = 'yes' if vals['SKP-6'] in wishes else 'no'

    ind = bisect.bisect_left([10,30], 100-vals['diff'])
    #### FIX THIS!!! This is for the case when the SKP2ESCO is too broad
    if np.isnan(vals['diff']):
        ind = -1
    data['SKPvsESCO'] = np.flipud(default['SKPvsESCO'])[ind]
#     ind = bisect.bisect_left([5,10], vals['diff'])
#     #### FIX THIS!!! This is for the case when the SKP2ESCO is too broad
#     if (vals['diff'] == 0):
#         ind = -1
#     data['SKPvsESCO'] = np.flipud(default['SKPvsESCO'])[ind]

    ind = bisect.bisect_left([10,50], vals['weight_num'] - vals['number of BO'])
    data['Available positions'] = default['Available positions'][ind]

    ind = bisect.bisect_left([10,20], vals['distance_km'])
    data['MSO'] = np.flipud(default['MSO'])[ind]


    # Language
    data['Languages'] = 'yes'
    data['Driving licence'] = 'yes'
    if pd.isnull(vals['delovni čas']):
        data['Job contract type'] = '*'
    elif 'Kraj' in vals['delovni čas']:
        data['Job contract type'] = 'part time'
    else:
        data['Job contract type'] = 'full time'

    if pd.isnull(vals['koda Urnik dela']):
        data['Job working hours'] = '*'
    elif float(vals['koda Urnik dela']) == 5:
        data['Job working hours'] = 'daily shift'
    else:
        data['Job working hours'] = 'daily/night shift'

    data['Driving licence'] = 'yes' if vals['dlic'] >= 0.5 else 'no'
    data['Languages'] = 'yes' if vals['dlang'] >= 0.5 else 'no'

    # Job career advancement
    try:
        adv = len(list(nx.all_simple_paths(DG, source=skp_code, target=int(vals['SKP-4']), cutoff=3)))
    except:
        adv = -1

    try:
        degrade = len(list(nx.all_simple_paths(DG, target=skp_code, source=int(vals['SKP-4']), cutoff=3)))
    except:
        degrade = -1

    if adv > degrade:
        data['Job career advancement'] = 'up'
    elif adv < degrade:
        data['Job career advancement'] = 'down'
    elif adv == -1 and degrade == -1:
        data['Job career advancement'] = '*'
    else:
        data['Job career advancement'] = 'same'

    # Job working hours

    # print(vals['IDupEnote'].astype(int))
    #BO Wish location
    data['BO wish location'] = '*'
    if 0 not in wishes_location:
        data['BO wish location'] = 'yes' if int(vals['IDupEnote']) in wishes_location else 'no'

    # data_val[r[0]] = data

    return data
    eval_res, qq_res = dexmodel.evaluate_model(data)

    all_eval[r[0]] = eval_res
    all_qq[r[0]] = qq_res

#         break
    # len(bo_skills)


# import multiprocessing as mp
from multiprocessing import Pool
from datetime import datetime
import itertools

def eval(key, data, dexmodel):
    eval_res, qq_res = dexmodel.evaluate_model(data, skip_qq=False)
    return eval_res, qq_res, key

if __name__ == '__main__':
    pbar = tqdm()
    pbar.reset(total=len(dex_df))
    data_val = dict()
    for r in  dex_df.iterrows():
        data_val[r[0]] = dict(check(r))
        pbar.update()

    # pbar.reset(total=len(dex_df))
    # all_eval = dict()
    # all_qq = dict()
    # mp.set_start_method('spawn')
    # for k in data_val:
    #     eval_res, qq_res = dexmodel.evaluate_model(data_val[k])
    #
    #     all_eval[k] = eval_res
    #     all_qq[k] = qq_res
    #     pbar.update()
    start = datetime.now()
    with Pool(processes=4) as pool:
        x = pool.starmap(eval,zip(data_val.keys(),
                data_val.values(),
                itertools.repeat(dexmodel)))

    # x = dexmodel.evaluate_model_multi_proc(data_val)

    # p = Pool(10)
    # x=p.map(check, dex_df.iterrows())
    #
    print(datetime.now() - start)

    all_eval, all_qq, keys = zip(*x)
    all_eval = dict(zip(keys, all_eval))
    all_qq = dict(zip(keys, all_qq))
    print(datetime.now() - start)

    # df_qq = pd.DataFrame.from_dict(all_qq,orient='index').reset_index()
    df_eval = pd.DataFrame.from_dict(all_eval,orient='index').reset_index()
    print(df_eval['SKP Evaluation'].to_numpy())
    # df_qq['Eval_min'] = df_qq.apply(lambda x: np.min(x['SKP Evaluation']) ,axis=1)
    # df_qq['Eval_max'] = df_qq.apply(lambda x: np.max(x['SKP Evaluation']) ,axis=1)
    # final_index = df_qq.sort_values(by='Eval_max',ascending=False).head(10).index
    #
    # intermediate = pd.merge(dex_df.loc[final_index],df_eval.loc[final_index], right_on=df_eval.loc[final_index].index, left_on=dex_df.loc[final_index].index)
    # final_df = pd.merge(intermediate, occupations.loc[:,['SKP koda-6','SKP poklic']], left_on='SKP-6', right_on='SKP koda-6')
    # final_df = pd.merge(final_df,df_qq.loc[final_index,['Eval_min', 'Eval_max']],  left_on='key_0', right_on=df_qq.loc[final_index].index)
    # final_df = final_df.drop(columns='skills')
    # print(datetime.now() - start)
