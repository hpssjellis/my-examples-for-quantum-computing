  
#!/bin/bash

echo "Problem: When you change the file name of a TFJS model with it's shard files, you also must change the PATHS in the .json file"
echo "Easier to not change the models file names"
echo "Enter the name of your TensorflowJS .json saved model with the .json extension, example: model.json"
read myFile
echo $myFile

tensorflowjs_converter --input_format=tfjs_layers_model --output_format=keras_saved_model ./$myFile ./
tflite_convert --keras_model_file ./ --output_file ./model.tflite
xxd -i model.tflite model.h
