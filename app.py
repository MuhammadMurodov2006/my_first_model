import streamlit as st
from fastai.vision.all import *

# import platform
# import pathlib
# plt = platform.system()
# if plt == 'Linux':
#     pathlib.PosixPath = pathlib.WindowsPath

#title
st.title('Image Classification with FastAI')

# Model information notice
st.info("""
📋 **Model Information**: This model is trained to classify images into the following categories only:
- 🧸 **Toys**
- 🏠 **Home Appliances** 
- 🚗 **Auto Parts**
- ⚽ **Balls**
- 🚲 **Bicycles**

Please upload images from these categories for accurate predictions.
""")

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
