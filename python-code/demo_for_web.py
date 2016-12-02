from __future__ import print_function

import sys
import re
import os
import glob


import numpy as np
import utils_
import my_plots

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from RegressionMatrix import regression_matrix
from Forecasting import frc_class

# from django.template import Template, Context
# from django.conf import settings
# from django.core.management import setup_environ
# settings.configure()
template = """
<html>
<h1>Multiscale forecasting</h1>
<body>
<h4>Loaded time series: {{ts_name}} dataset</h4>
{{readme}} <br>
Time series summary:
{{ts_summary}}
<br>
<h4>Forecasting results (MAE and MAPE):</h4>
Forecasting model: {{frc_model}}, feature generation: {{gnt}}, feature selection: {{slt}} <br>
{{frc_res}}

<!---
{{images}}
-->

<br>
<br>
<form id  =  "back"
action    =  "/"
method    =  "get"
>
Back to data submission:
<input id='submit' type="submit" value="Start" name="submit">
</form>
</body>
</html>
"""

MAX_TS_LEN = 50000

print(os.listdir(os.curdir))

file_name, frc_model, n_hist, n_req, train_test_ratio, pars, msg = utils_.read_web_arguments(sys.argv[1:])
ts = utils_.safe_read_iot_data(file_name, "all", True, default="poisson")
ts.history = n_hist
ts.request = n_req

ts_html_summary = ts.summarize_ts().to_html()

if frc_model.lower() == "lstm":
    from Forecasting import LSTM
    frc_model = frc_class.CustomModel(LSTM.LSTM, name="LSTM", **pars)
elif frc_model.lower() == "lasso":
    frc_model = frc_class.CustomModel(Lasso, name="Lasso", **pars)
elif frc_model.lower() == "rf":
    frc_model = frc_class.CustomModel(RandomForestRegressor, name="RandomForest", **pars)
elif frc_model.lower() == "gbr":
    pars["estimator"] = GradientBoostingRegressor()
    frc_model = frc_class.CustomModel(MultiOutputRegressor, name="Gradient boosting", **pars)
else:
    raise ValueError


ts.replace_nans()
if np.any([len(s) > MAX_TS_LEN for s in ts.data]):
    ts.align_time_series(max_history=MAX_TS_LEN)

data = regression_matrix.RegMatrix(ts)
data.create_matrix(nsteps=1, norm_flag=True)
data.train_test_split(train_test_ratio)

model, frc, gnt, slt = data.train_model(frc_model=frc_model, generator=None, selector=None)
data.forecast(model, replace=True)
frc_res = utils_.train_test_errors_table(data).to_html()

template = re.sub("\{\{ts_name\}\}", ts.name, template)
template = re.sub("\{\{readme\}\}", ts.readme, template)
template = re.sub("\{\{ts_summary\}\}", ts_html_summary, template)
template = re.sub("\{\{frc_model\}\}", frc_model.name, template)
template = re.sub("\{\{gnt\}\}", gnt.name, template)
template = re.sub("\{\{slt\}\}", slt.name, template)
template = re.sub("\{\{frc_res\}\}", frc_res, template)


images = data.plot_frc(folder="fig")


img_template = "<img rel src='./fig/{}'/><br>\n"
html_images = ""
# images = glob.glob(os.path.join("fig", "*.png"))

# print(os.listdir(os.path.curdir))
for img in images:
    fname = img.split(os.sep)[-1]
    html_images += img_template.format(fname)

template = re.sub("\{\{images\}\}", html_images, template)

with open("res.html", "wb") as f:
    f.write(template)

print("Done!")
