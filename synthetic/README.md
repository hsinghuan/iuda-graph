# SBMB \& SBMF

To create datasets, modify the data_dir strings in data_create.py and run:
```
python data_create.py
```

To run experiments, do:
```
python main.py --shift sbmb --data_dir path/to/data_dir/sbmb --method gcst-upl
```
Options for shift types include:
* sbmb
* sbmf
* sbmb-ys
* sbmf-ys

Options for method include:
* gcst-fpl
* gcst-upl
* gst
* cbst
* crst
* dann
* jan
* deep-coral
* uda-gcn
* fixed