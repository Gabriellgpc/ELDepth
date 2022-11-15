# -*- coding: utf-8 -*-
# @Author: Condados
# @Date:   2022-10-07 15:20:49
# @Last Modified by:   Condados
# @Last Modified time: 2022-10-07 15:22:08
import tensorflow as tf
import os

def export_to_tflite(saved_model_dir, export_dest_dir):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    tflite_file = os.path.join(export_dest_dir, 'model.tflite')
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)