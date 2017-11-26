
Requirements:
 - python 2.7 #currently the only supported version for tensorflow serving
 - tensorfow tensorflow-serving-api keras flask h5py
 
 

Build Tf Serving from source
 - https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md
 - NOTE: you have to run `tensorflow/configure` before compiling Need to automate flags
 
 ```
bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-O3 tensorflow_serving/model_servers/tensorflow_model_server
```

Command for launching server
```
tensorflow_model_server --port=9000 --model_base_path=/home/mkowoods/mobilenet-alpha-1-228-228-export &> tf_serv.log &

or below if compiled from source 

serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_base_path=/home/mkowoods/mobilenet-alpha-1-228-228-export &

serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_conf=/home/mkowoods/tf-serving-flask/tf_serving_config.conf
```

# TODO:
 - Compile tensorflow serving based on GCP CPU architecture 
 https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions
 https://gist.github.com/venik/9ba962c8b301b0e21f99884cbd35082f
 - switch to using docker images to handle set-up and configure
 
 https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-2-containerize-it-db0ad7ca35a7
 https://github.com/llSourcell/How-to-Deploy-a-Tensorflow-Model-in-Production/blob/master/demo.ipynb
 https://github.com/aaxwaz/Serving-TensorFlow-Model
 
 