# Elliptic

To preprocess the dataset, modify the data_dir string in data_process.py and run:
```
python data_process.py
```

To run experiments, do:
```
python main.py --data_dir path/to/data_dir/elliptic_bitcoin_dataset --method gcst-upl
```

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