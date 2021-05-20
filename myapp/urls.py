from django.urls import path, re_path
from .views import *

urlpatterns = [
    path("", my_view, name="my-view"),
    path("dexeval", dex_evaluate, name="dex-eval"),
    path("dexinput", dex_input, name="dex-input"),
    path("hand", hand, name="hand"),
    re_path("^skills/(?P<skp_code>\d+\.\d+)/$", get_skills, name="get_skills"),
    path("similarity", occupation_similarity, name = 'occupation_similarity'),
    path("similarity_skp6", occupation_similarty_skp6, name = 'occupation_similarty_skp6'),
    path("handeval/", handeval, name="handeval"),
    path("test", skp_view, name = 'skp-view'),
    path("eval_hecat_dex", eval_hecat_dex, name = 'eval_hecat_dex'),
    path("view_test", view_test, name = 'view_test'),
    path("index_plotly", index_plotly, name = 'index_plotly'),
    # re_path("^index_plotly/(?P<index>\d+)/$", index_plotly, name = 'index_plotly'),


]
