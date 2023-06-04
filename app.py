import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
InceptionV3_98_Accuracy = tf.keras.models.load_model('model_inception.h5')
Resenet50_72_Accuracy= tf.keras.models.load_model('classifier_resnet_model.h5')
VGG16_92_Accuracy = tf.keras.models.load_model('classifier_vgg16_model.h5')


# Create a dropdown menu to select the model
selected_model = st.selectbox("Select a model to use", ("InceptionV3_98_Accuracy", "Resenet50_72_Accuracy", "VGG16_92_Accuracy"))

# Load the selected model
if selected_model == "InceptionV3_98_Accuracy":
    model = InceptionV3_98_Accuracy
elif selected_model == "Resenet50_72_Accuracy":
    model = Resenet50_72_Accuracy
elif selected_model == "VGG16_92_Accuracy":
    model =  VGG16_92_Accuracy

# Create a list of classes
class_names = ['Not Pothole', 'Pothole']

@st.cache(allow_output_mutation=True)
def predict(image):
    # Preprocess the image
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    
    # Make the prediction
    predictions = model.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    class_idx = np.argmax(predictions[0])
    class_name = class_names[class_idx]
    
    return class_name, 100 * np.max(score)

def main():    
    st.title('Pothole Detector using Deep Convolutional Neural Networks')
    st.write('This app uses improved deep Convolutional Neural models to classify images as either containing a pothole or not.')
    
    # Allow the user to upload an image file
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make a prediction and display the result
        class_name, score = predict(image)
        st.write(f'Prediction: {class_name} ({score:.2f}%)')
        st.markdown("<hr>", unsafe_allow_html=True)
   
   # Add a footer to the app
    st.sidebar.markdown("Dissertation Research Submission by:")
    st.sidebar.write("Name: Astley Masvingise")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.write("Reg Number: R198957T")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.write("Supervisor: Mr Mlambo")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.write("Research Topic: Evaluating recent deep learning algorithms (CNNs) in pothole detection and repair prioritization for  Harare City Council’s road maintenance program.")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
if __name__ == '__main__':
    main()
