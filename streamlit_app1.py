import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import skimage.filters
import torchxrayvision as xrv
import os
from app import get_img,get_output,convert_to_text,create_histogram,plot_heatmap

# Create the 'temp' folder if it doesn't exist
if not os.path.exists('temp'):
    os.makedirs('temp')

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("X-Ray Image Analysis")

    uploaded_file = st.file_uploader("Upload an X-Ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Get the path of the uploaded file
        image_path = f"temp/{uploaded_file.name}"  # Save the uploaded image temporarily in the "temp" folder

        # Save the uploaded image to the temporary folder
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the image and get the output dictionary
        img_val = get_img(image_path)
        label, prob_value, output_dictonary = get_output(img_val)

        # Display the original uploaded image
        st.header("Original Uploaded Image")
        st.image(uploaded_file, use_column_width=True)

        # Display the output as text
        st.header("Results")
        st.write("Results:")
        st.write(output_dictonary)

        # Find the maximum probability and diseases with probability > 0.55
        # Find the maximum probability and diseases with probability > 0.55
        # max_probability = max(output.values())
        # selected_disease = [key for key, value in output.items() if value == max_probability][0]
        max_probability = max(output_dictonary.values())
        max_disease = [key for key, value in output_dictonary.items() if value == max_probability][0]

        st.write("\n\n")  # Adding some space

        # Display maximum probability and disease with highest probability in two rows
        st.markdown("<p style='text-align: center; color: yellow; font-size: 18px;'>Details of Diseases which have maximum probabilities are: </p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: yellow; font-size: 18px;'>1.Disease Name: {max_disease}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: yellow; font-size: 18px;'>2. Probability value: {max_probability:.2f}</p>", unsafe_allow_html=True)
        


        selected_diseases = [key for key, value in output_dictonary.items() if 0.55 < value <= 1.0]
        
        st.markdown("<p style='text-align: center; color: yellow ;font-size: 18px;'>Details od Diseases which have probability value between 0.55 to 1:</p>", unsafe_allow_html=True)
        if len(selected_diseases) > 0:
            for disease in selected_diseases:
                st.markdown(f"<p style='text-align: center; color: yellow;font-size: 18px'>{disease}: {output_dictonary[disease]:.2f}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align: center; color: yellow;font-size: 18px'>No diseases have probability range between 0.55 and 1.0.</p>", unsafe_allow_html=True)
        # # Convert output to text and display
        # text_result = convert_to_text(output_dictonary)
        # st.write(text_result)

        # Create a histogram and display
        st.header("Histogram")
        fig = create_histogram(output_dictonary)
        st.pyplot(fig)

        # Plot heatmap and display
        st.header("Heatmap")
        heatmap = plot_heatmap(img_val, output_dictonary)
        st.pyplot(heatmap)

       

if __name__ == '__main__':
    main()
