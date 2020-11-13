#!/bin/bash



tensorflowjs_converter --input_format=tfjs_layers_model --output_format=keras_saved_model ./model.json ./

# all default to using the above output ./saved_model.pb

tflite_convert --keras_model_file ./ --output_file ./model.tflite
xxd -i model.tflite model.h



tflite_convert --saved_model_dir=./ --inference_type=QUANTIZED_UINT8 --inference_output_type=tf.uint8 --mean_value=128 --std_value=127 --output_file=./model_Q_UINT8.tflite
xxd -i model_Q_UINT8.tflite model_Q_UINT8.h


tflite_convert --saved_model_dir=./ --inference_type=tf.uint8 --inference_output_type=tf.uint8 --mean_value=128 --std_value=127 --output_file=./model_tf_UINT8.tflite
xxd -i model_tf_UINT8.tflite model_tf_UINT8.h


tflite_convert --saved_model_dir=./ --inference_type=tf.int8 --inference_output_type=tf.int8 --mean_value=128 --std_value=127 --output_file=./model_tf_INT8.tflite
xxd -i model_tf_INT8.tflite model_tf_INT8.h



## tensorflowjs_converter --quantization_bytes 1 --input_format=tf_frozen_model --output_node_names=logits/BiasAdd --saved_model_tags=serve ./model/input_graph.pb ./web_model
