import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import contrails_predictions as cpd
import gradio as gr
from gradio.themes.base import Base
from PIL import Image
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

block_head_image = 'contrails_gradio_head.jpg'

description = """
            # Contrails Detection Tool
            
            This is a simple app designed to showcase the performance of a model 
            that can identify condensation trails that planes leave in the sky.  
            For detailed information please refer to:  
            https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming       
            """


class Seafoam(Base):
    pass


seafoam = Seafoam()


example_test = 'example_1'


def read_files(files):
    band_paths = []
    mask_path = None
    for file in files:
        if 'human_pixel_masks.npy' in file.name:
            mask_path = file.name
        elif 'band_' in file.name:
            band_paths.append(file.name)
    print('MASK PATH')
    print(mask_path)
    print('BAND PATHS')
    for path in band_paths:
        print(path)
    return cpd.visualize_prediction_avg(mask_path, band_paths)


with gr.Blocks(theme=seafoam) as iface:
    gr.Image(block_head_image)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            input_files = gr.File(file_count="multiple", file_types=[".npy"], label="Upload band files and a mask")
            with gr.Row():
                gr.ClearButton(input_files)
                predict_btn = gr.Button(value='Submit')
    with gr.Row():
        bands = gr.Image(type="pil", label="Infrared bands")
    with gr.Row():
        contrails_mask = gr.Image(type="pil", label="Contrails masks")
    with gr.Row():
        metrics = gr.Textbox(label='Metrics')

    predict_btn.click(read_files, inputs=input_files, outputs=[bands, contrails_mask, metrics])
    # gr.Examples(examples_dir=example_test, inputs=input_files, cache_examples=True, fn=read_files,
    #             outputs=[bands, contrails_mask, metrics])

if __name__ == "__main__":
    iface.launch(share=True)
