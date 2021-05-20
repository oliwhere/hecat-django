from django.shortcuts import redirect, render

# from .models import Document
from .forms import DocumentForm
from .forms import *
from django.http import JsonResponse
import os
from dss.dex import DEXModel
from dss.esco_utils import ESCOUtil
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
from django.core.serializers.json import DjangoJSONEncoder
from django.core.serializers import serialize
import numpy as np
from django.shortcuts import render

# from .utils import select_positions
import myapp.utils as utils
import pickle
import networkx as nx
import bisect
from datetime import datetime
import json

import logging
from multiprocessing import Pool
import multiprocessing as mp
import itertools

logger = logging.getLogger("hecat")


from django.conf import settings
from django.core.cache import cache
import pandas as pd
import numpy as np
import os

from plotly.offline import plot
from plotly.graph_objs import Scatter, Scatterpolar
import plotly.graph_objs as go



class NumpyEncoder(DjangoJSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(self, obj)


@require_http_methods(["POST"])
def eval_hecat_dex(request):
    logger.info("POST")
    start_time = datetime.now()
    dexmodel = DEXModel(settings.DEX_MODEL)
    esco = ESCOUtil()
    occupations = utils.get_occupation()
    napoved_year = 2018
    napoved_period = "I"

    skp_skills = (
        utils.get_skp_skills()
    )  # pd.read_pickle('./data/skp_skills_%d-%s.pcl' % (napoved_year,napoved_period) )
    res = utils.get_merged()
    complete_skills_dict = utils.get_complete_skills_dict()
    df_lang_dict = utils.get_skp_lang()
    id_distance_time = utils.get_id_dist_time()
    # job_contract_mer = pd.read_pickle(os.path.join(settings.DATA_ROOT, 'elise/job_contract_type.pcl'))
    # job_working_hours_mer = pd.read_pickle(os.path.join(settings.DATA_ROOT, 'elise/job_job_working_hours.pcl'))
    # langmer = utils.get_language()
    # dmer = utils.get_driver_lic()
    DG = nx.read_gpickle(os.path.join(settings.DATA_ROOT, "elise/career_graph.pcl"))
    logger.debug("Load done %s" % (datetime.now() - start_time))

    # Dummy default
    data = dict()
    # data['Languages'] = 'yes'
    # data['Driving licence'] = 'yes'
    data["Age appropriateness"] = "yes"
    data["Disability appropriateness"] = "yes"

    default = dexmodel.get_intput_attributes()

    # Form data
    form = DexForm2(request.POST)
    if form.is_valid():

        skp_code = int(form.cleaned_data["skp_code"])
        up_enota = int(form.cleaned_data["up_enota"])  # Sk.Loka

        wishes = np.array(form.cleaned_data["wishes"]).astype(int)
        wishes_location = np.array(form.cleaned_data["wishes_location"]).astype(int)

        bo_lang = form.cleaned_data["bo_lang"]
        bo_driving_lic = form.cleaned_data["bo_driving_lic"]
        # logger.debug(skp_code, up_enota)
    else:
        return HttpResponse("Error")

    data["BO wishes for contract type"] = "full time"
    data["BO career wishes"] = "*"
    data["BO working hours wishes"] = "daily shift"
    # Form end

    sel_cols_lic = ["DLIC_%s" % x for x in bo_driving_lic]
    sel_cols_lang = ["LANG_%s" % x for x in bo_lang]
    inter = utils.get_intermediate_results()
    lic_lang = pd.DataFrame(
        data={
            "SKP-6": inter["SKP-6"],
            "dlic": inter.loc[:, sel_cols_lic].sum(axis=1) / inter.loc[:, "SUM_DLIC"],
            "dlang": inter.loc[:, sel_cols_lang].sum(axis=1) / inter.loc[:, "SUM_LANG"],
            "delovni čas": inter["delovni čas"],
            "koda Urnik dela": inter["koda Urnik dela"],
            "skills": inter["skills"],
        }
    )  # .fillna(1)
    lic_lang[["dlic", "dlang"]] = lic_lang[["dlic", "dlang"]].fillna(1)

    # There is a case when SKP4 is listed. SKP-6 has two 'decimal' digits. This line checks whether the code is SKP4 or SKP6
    col_name = "SKP koda-4" if int(skp_code) == skp_code else "SKP koda-6"
    #     bo_skills = esco.get_all_skills_SKP2ESCO(occupations,  skp_code, col_name)
    bo_skills = np.array([])
    for uri in occupations[occupations[col_name] == skp_code]["URI"].unique():
        bo_skills = np.union1d(
            complete_skills_dict[uri]["basic"], complete_skills_dict[uri]["optional"]
        )

    # dif_skills = skp_skills.apply(lambda x: len(np.setdiff1d(x.skills, bo_skills)), axis=1)
    len_skill = skp_skills.apply(lambda x: len(x.skills), axis=1)
    wh = len_skill != 0
    dif_skills = skp_skills.apply(
        lambda x: len(np.intersect1d(x.skills, bo_skills)), axis=1
    )
    dif_skills[~wh] = np.nan
    dif_skills[wh] = dif_skills[wh] / len_skill[wh] * 100

    resSKPs = utils.select_positions(res, up_enota, id_distance_time)
    dif_df = pd.DataFrame({"SKP-6": skp_skills["SKP-6"], "diff": dif_skills})
    dex_df = pd.merge(resSKPs, dif_df, on="SKP-6")
    dex_df = pd.merge(dex_df, lic_lang, how="left", on="SKP-6")

    mask_dlic = inter.columns.str.contains("DLIC_.*")
    mask_lang = inter.columns.str.contains("LANG_.*")

    all_eval = dict()
    all_qq = dict()

    # logger.debug(data)
    data_val = dict()
    for r in dex_df.iterrows():
        vals = r[1]
        if 0 in wishes:
            data["SKP Wish"] = "*"
        else:
            data["SKP Wish"] = "yes" if vals["SKP-6"] in wishes else "no"

        ind = bisect.bisect_left([10, 30], 100 - vals["diff"])
        #### FIX THIS!!! This is for the case when the SKP2ESCO is too broad
        if np.isnan(vals["diff"]):
            ind = -1
        data["SKPvsESCO"] = np.flipud(default["SKPvsESCO"])[ind]
        #     ind = bisect.bisect_left([5,10], vals['diff'])
        #     #### FIX THIS!!! This is for the case when the SKP2ESCO is too broad
        #     if (vals['diff'] == 0):
        #         ind = -1
        #     data['SKPvsESCO'] = np.flipud(default['SKPvsESCO'])[ind]

        ind = bisect.bisect_left([10, 50], vals["weight_num"] - vals["number of BO"])
        data["Available positions"] = default["Available positions"][ind]

        ind = bisect.bisect_left([10, 20], vals["distance_km"])
        data["MSO"] = np.flipud(default["MSO"])[ind]

        # Language
        data["Languages"] = "yes"
        data["Driving licence"] = "yes"
        if pd.isnull(vals["delovni čas"]):
            data["Job contract type"] = "*"
        elif "Kraj" in vals["delovni čas"]:
            data["Job contract type"] = "part time"
        else:
            data["Job contract type"] = "full time"

        if pd.isnull(vals["koda Urnik dela"]):
            data["Job working hours"] = "*"
        elif float(vals["koda Urnik dela"]) == 5:
            data["Job working hours"] = "daily shift"
        else:
            data["Job working hours"] = "daily/night shift"

        data["Driving licence"] = "yes" if vals["dlic"] >= 0.5 else "no"
        data["Languages"] = "yes" if vals["dlang"] >= 0.5 else "no"

        # Job career advancement
        try:
            adv = len(
                list(
                    nx.all_simple_paths(
                        DG, source=skp_code, target=int(vals["SKP-4"]), cutoff=3
                    )
                )
            )
        except:
            adv = -1

        try:
            degrade = len(
                list(
                    nx.all_simple_paths(
                        DG, target=skp_code, source=int(vals["SKP-4"]), cutoff=3
                    )
                )
            )
        except:
            degrade = -1

        if adv > degrade:
            data["Job career advancement"] = "up"
        elif adv < degrade:
            data["Job career advancement"] = "down"
        elif adv == -1 and degrade == -1:
            data["Job career advancement"] = "*"
        else:
            data["Job career advancement"] = "same"

        # Job working hours

        # print(vals['IDupEnote'].astype(int))
        # BO Wish location
        data["BO wish location"] = "*"
        if 0 not in wishes_location:
            data["BO wish location"] = (
                "yes" if int(vals["IDupEnote"]) in wishes_location else "no"
            )
        # eval_res, qq_res = dexmodel.evaluate_model(data)
        # all_eval[r[0]] = eval_res
        # all_qq[r[0]] = qq_res
        data_val[r[0]] = dict(data)
    #
    #
    #
    with Pool(processes=mp.cpu_count()) as pool:
        x = pool.starmap(
            dex_eval,
            zip(data_val.keys(), data_val.values(), itertools.repeat(dexmodel)),
        )

    all_eval, all_qq, keys = zip(*x)
    all_eval = dict(zip(keys, all_eval))
    all_qq = dict(zip(keys, all_qq))
    logger.debug(datetime.now() - start_time)
    df_qq = pd.DataFrame.from_dict(all_qq,orient='index').reset_index()
    df_eval = pd.DataFrame.from_dict(all_eval,orient='index').reset_index()
    df_qq['Eval_min'] = df_qq.apply(lambda x: np.min(x['SKP Evaluation']) ,axis=1)
    df_qq['Eval_max'] = df_qq.apply(lambda x: np.max(x['SKP Evaluation']) ,axis=1)
    final_index = df_qq.sort_values(by='Eval_max',ascending=False).index#.head(10).index

    sif_skp = utils.get_jobs()
    intermediate = pd.merge(dex_df.loc[final_index],df_eval.loc[final_index], right_on=df_eval.loc[final_index].index, left_on=dex_df.loc[final_index].index)
    final_df = pd.merge(intermediate, sif_skp.loc[:,['SFpoklicaSKP','Naziv']].drop_duplicates(subset='SFpoklicaSKP'), left_on='SKP-6', right_on='SFpoklicaSKP', how='left')
    # final_df = pd.merge(intermediate, occupations.loc[:,['SKP koda-6','SKP poklic']], left_on='SKP-6', right_on='SKP koda-6')
    final_df = pd.merge(final_df,df_qq.loc[final_index,['Eval_min', 'Eval_max']],  left_on='key_0', right_on=df_qq.loc[final_index].index)
    final_df = pd.merge(final_df,df_qq.loc[final_index], left_on='key_0', right_on=df_qq.loc[final_index].index, suffixes=['','_qq'])


    logger.debug(datetime.now() - start_time)
    # logger.debug(len(df_qq))
    # logger.debug(len(df_eval))
    # logger.debug(len(final_index))
    # logger.debug(len(intermediate))
    # logger.debug(len(final_df))
    # logger.debug(final_index)

    final_df['SKP-4'] = final_df['SKP-6'].astype(int)

    new_names = [ x.replace(' ','_') for x in final_df.columns]
    final_df.rename(columns=dict(zip(final_df.columns, new_names)),inplace=True)
    final_df.drop(inplace=True, columns=['skills','SFpoklicaSKP','key_0',
                        'index','weight_num','IDupEnote',
                        'distance_km','travel_min','number_of_BO'])
    tuple_convert = ['Skill_appropriateness', 'Personal_characteristics',
                 'SKP_Evaluation','Contract_type','Skills','Career_advance',
                'Working_hours','Location','Job_appropriateness','Work_type']

    sel_cols = ['Eval_min','Eval_max','delovni_čas','SKPvsESCO','Available_positions','Skill_appropriateness',
           'Job_contract_type','Job_working_hours','MSO','Career_advance','SKP_Evaluation']
    for col in tuple_convert:
        final_df[col] = final_df[col].apply(tuple)
        final_df['%s_qq' % col] = final_df['%s_qq' % col].apply(tuple)

    final_df.drop_duplicates(subset=sel_cols, inplace=True)

    # dj_res = dict()
    # for skp in final_df['SKP-4'].unique():
    #     wh = final_df['SKP-4'] == skp
    #     dj_res[skp] = final_df[wh]#.to_dict(orient='list')
    #     # break
    groups = final_df.groupby((final_df['SKP-4'].shift() != final_df['SKP-4']).cumsum())
    context = {}
    list(groups.groups.keys())[:5]
    for k in list(groups.groups.keys())[:5]:
        v = groups.get_group(k)
        context[v.iloc[0]['Naziv']] = v

    request.session['gen'] = True
    final_df.to_pickle(os.path.join(settings.SESSION_FILE_PATH, request.session.session_key + "_df.pcl"))
    logger.debug(datetime.now() - start_time)
    return render(request, "dex_res.html", {"groups": context})



def dex_eval(key, data, model):
    eval_res, qq_res = model.evaluate_model(data)
    return eval_res, qq_res, key


def get_file():
    filess = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            filess.append(filename)
    return filess


def get_skills(request, skp_code=0):
    sif_skp = cache.get("sif_skp")
    skp_code = float(skp_code)
    if sif_skp is None:
        logger.info("Loading sif_skp")
        sif_skp = pd.read_csv(os.path.join(settings.DATA_ROOT, "dimSKP08.csv"))
        cache.set("sif_skp", sif_skp, timeout=None)

    occupations = cache.get("occ")
    if occupations is None:
        logger.info("Loading occupations")
        occupations = pd.read_excel(os.path.join(settings.DATA_ROOT, "SKP_ESCO.xlsx"))
        cache.add("occ", occupations, timeout=None)

    esco = ESCOUtil()
    col_name = "SKP koda-4" if int(skp_code) == skp_code else "SKP koda-6"
    # who =  occupations[col_name] == skp_code
    res = esco.get_all_skills_SKP2ESCO(occupations, skp_code, col_name)
    retVal = []
    for r in res:
        retVal.append({"uri": r, "label": esco.getLabel(r)})

    return JsonResponse(retVal, safe=False, encoder=NumpyEncoder)
    return HttpResponse(res)
    skp_code = sif_skp[
        sif_skp.IDpoklicaSKP == rows[1]["IDpoklicaSKP08"]
    ].SFpoklicaSKP.iloc[0]

    # There is a case when SKP4 is listed. SKP-6 has two 'decimal' digits. This line checks whether the code is SKP4 or SKP6


@require_http_methods(["POST"])
def handeval(request):
    dex = DEXModel(settings.DEX_MODEL)
    f = DexForm(dex.get_intput_attributes(), request.POST)  # , dex_attributes={})
    if f.is_valid():
        res, qq_res = dex.evaluate_model(f.cleaned_data)
        dex_res = dict()
        dex_res["quantitative"] = res
        dex_res["qualitative"] = qq_res

        return JsonResponse(res, safe=True, encoder=NumpyEncoder)
        # return HttpResponse(str(res))
    return render(request, "name.html", {"form": f})


def hand(request):
    dex = DEXModel(settings.DEX_MODEL)
    f = DexForm(dex.get_intput_attributes())
    return render(request, "name.html", {"form": f})


def dex_local_input(request):
    if request.method == "GET":
        files = get_file()
        dex = DEXModel(f"{folder}{files[len(files) - 1]}")
        return dex.get_intput_attributes()


@require_http_methods(["GET"])
def dex_input(request):
    dex = DEXModel(settings.DEX_MODEL)
    return JsonResponse(dex.get_intput_attributes(), safe=True)


@csrf_exempt
@require_http_methods(["POST"])
def dex_evaluate(request):
    dex = DEXModel(settings.DEX_MODEL)
    data = json.loads(request.body)
    res, qq_res = dex.evaluate_model(data)

    dex_res = dict()
    dex_res["quantitative"] = res
    dex_res["qualitative"] = qq_res

    return JsonResponse(dex_res, safe=True, encoder=NumpyEncoder)


def my_view(request):
    message = "Upload as many files as you want!"
    if request.method == "POST":
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES["docfile"])
            newdoc.save()

            return redirect("my-view")
        else:
            message = "The form is not valid. Fix the following error:"
    else:
        form = DocumentForm()

    documents = Document.objects.all()

    context = {"documents": documents, "form": form, "message": message}
    return render(request, "list.html", context)


def skp_view(request):
    if request.method == "POST":
        form = DexForm2(request.POST)
        if form.is_valid():
            data = form.cleaned_data
    else:
        form = DexForm2()
        request.session['gen'] = True
    return render(request, "name copy.html", {"form": form})


def occupation_similarty_skp6(request):
    skp6 = float(request.GET.get("skp6"))
    skp2esco = utils.get_occupation()
    wh = skp2esco["SKP koda-6"] == skp6

    occupations = utils.get_mapping_occ()
    W_combined = utils.get_similarity_matrix()

    w_nesta = occupations.concept_uri == skp2esco[wh].URI.to_numpy()[0]
    occupation_id = occupations[w_nesta].id.to_numpy()[0]
    most_similar = np.flip(np.argsort(W_combined[occupation_id, :]))
    similarity = np.flip(np.sort(W_combined[occupation_id, :]))

    df = occupations[["preferred_label", "concept_uri"]].copy().loc[most_similar]
    df["similarity"] = similarity
    df = (
        pd.merge(
            df,
            skp2esco[["URI", "SKP koda-6"]],
            how="left",
            left_on="concept_uri",
            right_on="URI",
        )
        .drop(columns=["URI"])
        .drop_duplicates()
    )
    # df.head(100)

    # return HttpResponse(df.head(10).to_json(), content_type = 'application/json')
    return HttpResponse(df.head(100).to_html())


def occupation_similarity(request):
    uri = request.GET.get("uri")
    occupations = utils.get_mapping_occ()
    W_combined = utils.get_similarity_matrix()
    wh = occupations.concept_uri == uri
    occupation_id = occupations[wh].index[0]
    most_similar = np.flip(np.argsort(W_combined[occupation_id, :]))
    similarity = np.flip(np.sort(W_combined[occupation_id, :]))

    df = occupations[["preferred_label"]].copy().loc[most_similar]
    df["similarity"] = similarity

    # return HttpResponse(df.head(10).to_json(), content_type = 'application/json')
    return HttpResponse(df.head(10).to_html())

def view_test(request):
    final_df = pd.read_pickle(os.path.join(settings.DATA_ROOT, 'dex_testing.pcl'))
    final_df['SKP-4'] = final_df['SKP-6'].astype(int)

    new_names = [ x.replace(' ','_') for x in final_df.columns]
    final_df.rename(columns=dict(zip(final_df.columns, new_names)),inplace=True)
    final_df.drop(inplace=True, columns=['skills','SFpoklicaSKP','key_0',
                        'index','weight_num','IDupEnote',
                        'distance_km','travel_min','number_of_BO'])
    tuple_convert = ['Skill_appropriateness', 'Personal_characteristics',
                 'SKP_Evaluation','Contract_type','Skills','Career_advance',
                'Working_hours','Location','Job_appropriateness','Work_type']

    sel_cols = ['Eval_min','Eval_max','delovni_čas','SKPvsESCO','Available_positions','Skill_appropriateness',
           'Job_contract_type','Job_working_hours','MSO','Career_advance','SKP_Evaluation']
    for col in tuple_convert:
        final_df[col] = final_df[col].apply(tuple)
        final_df['%s_qq' % col] = final_df['%s_qq' % col].apply(tuple)

    final_df.drop_duplicates(subset=sel_cols, inplace=True)

    # dj_res = dict()
    # for skp in final_df['SKP-4'].unique():
    #     wh = final_df['SKP-4'] == skp
    #     dj_res[skp] = final_df[wh]#.to_dict(orient='list')
    #     # break
    groups = final_df.groupby((final_df['SKP-4'].shift() != final_df['SKP-4']).cumsum())
    context = {}
    list(groups.groups.keys())[:5]
    for k in list(groups.groups.keys())[:5]:
        v = groups.get_group(k)
        context[v.iloc[0]['Naziv']] = v

    request.session['gen'] = True
    final_df.to_pickle(os.path.join(settings.SESSION_FILE_PATH, request.session.session_key + "_df.pcl"))
    return render(request, "dex_res.html", {"groups": context})



def index_plotly(request):

    json_string = request.GET.get('index')
    all_index = np.unique(np.array(json.loads(json_string)).astype(int))

    final_df = pd.read_pickle(os.path.join(settings.SESSION_FILE_PATH, request.session.session_key + "_df.pcl"))

    sel_cols = ['SKPvsESCO','Available_positions','Skill_appropriateness',
           'Job_contract_type','Job_working_hours','MSO','Career_advance','SKP_Evaluation']

    fig = go.Figure()
    for index in all_index:
        r = [np.max(final_df.loc[int(index),'%s_qq' % c]) for c in sel_cols]
        fig.add_trace(go.Scatterpolar(
          r=r,
          theta=sel_cols,
          fill='toself',
          name=final_df.loc[int(index),'Naziv']
        ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True
        ),
      ),
      showlegend=True
    )

    plot_div = plot(fig,
               output_type='div',include_plotlyjs=False,
               show_link=False, link_text="",config = {'displaylogo': False})

    return HttpResponse(plot_div)
    # return render(request, "index.html", context={'plot_div': plot_div})
