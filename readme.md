
Requirements:
 - python 2.7 #currently the only supported version for tensorflow serving
 - tensorfow tensorflow-serving-api keras flask h5py
 

Command for launching server
```
tensorflow_model_server --port=9000 --model_base_path=/home/mkowoods/mobilenet-alpha-1-228-228-export-2 &> tf_serv.log &
```

# TODO:
 - Compile tensorflow serving based on GCP CPU architecture 
 - switch to using docker images to handle set-up and configure
 
 https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-2-containerize-it-db0ad7ca35a7
 https://github.com/llSourcell/How-to-Deploy-a-Tensorflow-Model-in-Production/blob/master/demo.ipynb
 https://github.com/aaxwaz/Serving-TensorFlow-Model