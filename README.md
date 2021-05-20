# hecat_django

Project of using DEX eval class.
The project depends on several pre-processed files available in the ```data``` direcotry.

Creating the DEX model can be done using:

```python
dexmodel = dex.DEXModel('../data/SKP Evaluation version 3.xml')
```

The JSON describing the input parameters can be obtained as:
```
dexmodel.get_intput_attributes()
```


## Example of usage is the following:

```python
import pandas as pd
import numpy as np
import dex
from esco_utils import ESCOUtil
import pickle
import bisect
from tqdm.notebook import tqdm


esco = ESCOUtil()

pdist = pickle.load( open( "./data/dist.pcl", "rb" ) )
pload = pickle.load( open( "./data/all_geo.pcl", "rb" ) )
id_distance = pickle.load( open('./data/id_dist.pcl','rb') )
occupations = pd.read_excel('./data/SKP_ESCO.xlsx')


data = dict()
# data['Available positions']
data['Languages'] = 'yes'
data['Driving licence'] = 'yes'
data['Age appropriateness'] = 'yes'
data['Disability appropriateness'] = 'yes'

# data['SKP Wish'] = 
data['SKP Wish'] = 'yes'
data['BO wishes for contract type'] = '*'
data['Job contract type'] = '*'

data['BO career wishes'] = '*'
data['Job career advancement'] = '*'

data['BO working hours wishes'] = '*'
data['Job working hours'] = '*'

# data['MSO'] = x
# data['BO wish location'] = BO_wish_distance
data['BO wish location'] = 'yes'

dexmodel = dex.DEXModel('../data/external/SKP Evaluation version 3.xml')
# dexmodel.get_intput_attributes()

ind = bisect.bisect_left([5,10], vals['diff'])
data['SKPvsESCO'] = np.flipud(default['SKPvsESCO'])[ind]

ind = bisect.bisect_left([10,50], vals['diff'])
data['Available positions'] = default['Available positions'][ind]

ind = bisect.bisect_left([10,20], vals['diff'])
data['MSO'] = np.flipud(default['MSO'])[ind]

eval_res, qq_res = dexmodel.evaluate_model(data)

```
