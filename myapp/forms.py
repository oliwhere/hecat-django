from django import forms
import pandas as pd
import os
from django.conf import settings
import myapp.utils as utils


class DocumentForm(forms.Form):
    docfile = forms.FileField(label="Select a file")


class DexForm(forms.Form):
    def __init__(self, dex_attributes={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in dex_attributes.keys():
            vals = dex_attributes[k]
            vals.insert(0, "*")
            choices = list(zip(vals, vals))
            self.fields[k] = forms.ChoiceField(
                choices=choices, widget=forms.Select(attrs={"class": "form-control"})
            )


class DexForm2(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        occupations = utils.get_occupation()

        occ = occupations.loc[:,["SKP koda-4","SKP poklic"]].drop_duplicates().sort_values(by="SKP poklic")

        ue = utils.get_ue()

        lang = utils.get_language()
        lang = lang[lang['koda Tuji jeziki'] != 'SL']

        dlic = utils.get_driver_lic()

        vals = list(occ["SKP koda-4"])
        vals1 = list(occ["SKP poklic"])
        idup = list(ue["IDupEnote"])
        naziv = list(ue["Naziv"])
        choices = list(zip(vals, vals1))

        vals.insert(0,0)
        vals1.insert(0,'*')
        wish_choices = list(zip(vals, vals1))


        choices2 = list(zip(idup, naziv))
        idup.insert(0,0)
        naziv.insert(0,'*')
        wish_choices2 = list(zip(idup, naziv))

        # bo_lang = form.cleaned_data['bo_lang']
        # bo_driving_lic = form.cleaned_data['bo_driving_lic']

        lic_type = dlic['koda Vozniško dovoljenje'].unique()
        choice_driver_lic = list(zip(lic_type,lic_type))

        lang_type = lang['koda Tuji jeziki'].unique()
        choice_lang = list(zip(lang_type,lang_type))


        self.fields["skp_code"] = forms.ChoiceField(
            choices=choices, widget=forms.Select(attrs={"class": "form-control"})
        )
        self.fields["up_enota"] = forms.ChoiceField(
            choices=choices2, widget=forms.Select(attrs={"class": "form-control"})
        )
        self.fields["wishes"] = forms.MultipleChoiceField(
            choices=wish_choices, widget=forms.SelectMultiple(attrs={"class": "form-control"}), initial = "1"
        )
        self.fields["wishes_location"] = forms.MultipleChoiceField(
            choices=wish_choices2, widget=forms.SelectMultiple(attrs={"class": "form-control"}), initial = "1"
        )

        self.fields["bo_driving_lic"] = forms.MultipleChoiceField(
            choices=choice_driver_lic, widget=forms.SelectMultiple(attrs={"class": "form-control"}), initial = "1"
        )

        self.fields["bo_lang"] = forms.MultipleChoiceField(
            choices=choice_lang, widget=forms.SelectMultiple(attrs={"class": "form-control"}), initial = "1"
        )
