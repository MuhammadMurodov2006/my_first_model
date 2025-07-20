import streamlit as st
from fastai.vision.all import *


# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

#title
st.title('Image Classification with FastAI')

#image upload 
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if file:
    #showing image
    st.image(file, caption='Uploaded Image', use_container_width=True)

    #convert to PIL image
    img = PILImage.create(file)

    #model
    model = load_learner('My_model.pkl')

    # prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f'Prediction: {pred}')
    st.info(f'Probability: {probs[pred_id]*100:.1f}%')

    # #plotting
    # px.bar(x=probs*100, y=model.vocab)
    # st.plotly_chart(fig)