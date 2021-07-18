First column is 'CLASS ID'(URI: uniform resource identifier) which is ontology class or instances. it is unique

Second column is class label or instance(product)name

8th col is parents id which is class id of parents separated by '|'.
there could be more than one parents to class/instance

Entity is the root class (BFO_0000001) but its parent is #Thing

we are working subtree rooted at (foodon product type) 'http://purl.obolibrary.org/obo/FOODON_00001002' Hit this url.

foodonpairs.txt columns are exact same class label as in CSV file no preprocessing no phrase detection is used.

Download "FDC dataset" and then create foodon.csv file using code in fdc_data.py

Use Sysnonyms column split by '|' and then add the words which are not in label.