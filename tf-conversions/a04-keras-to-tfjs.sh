#!/bin/bash

#tensorflowjs_converter --input_format=tfjs_layers_model --output_format=keras_saved_model ./model.json ./



tensorflowjs_converter --output_format=tfjs_layers_model --input_format=keras_saved_model ./ ./
