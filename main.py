import cv2 as cv, numpy as np, streamlit as st, os
from numpy.linalg import svd


st.title("SVD Demonstration")


# Loads the images from the folder specified in the path
def load_images(path):
    imgs = []
    for filename in os.listdir(path):
        if(os.path.isfile(os.path.join(path,filename))):
            imgs.append(cv.imread(os.path.join(path,filename)))
    return imgs
path = 'images'
Images = load_images(path)


with st.sidebar:
    st.subheader("Select an Image")
    img_num = st.slider(label="Image_Number", min_value=1, max_value=len(Images), step=1)
image = Images[img_num-1]
st.image(image, caption="Original Image", width=None, use_column_width=None, channels='BGR', output_format='auto', clamp=True)


with st.sidebar:
    k = int( st.slider( label="K-value (Here, it is minimum of number of pixels either in height or width)", 
                        min_value=int(min([2,image.shape[0],image.shape[1]])), 
                        max_value=int(min([image.shape[0],image.shape[1]])), 
                        step=1))
    # A library is not working with k = 1, hence absolute minimum k = 2 



# Calc svd for a channel (2D Matrix) and return compressed channel
def SVD_Compress_Channel(image,k):
    U,S,V = svd(image, full_matrices=False)
    # print("U",U,"\nS",S,"\nV",V,"\n")
    ans = (U[:,:k]).copy() @ (np.diag(S[:k])).copy() @ (V[:k,:]).copy()
    # print(ans)
    return ans

# Calc svd for a 3-channel image and return a compressed 3-channel image
def SVD_Compress(image,k):
    shp = image.shape
    new_img = np.zeros(shape=shp)
    for i in range(3): new_img[:,:,i] = (SVD_Compress_Channel(image=image[:,:,i].copy(), k=k))
    comp_ratio = round(100*(k/min(shp[0],shp[1])), 2)  # (k*max(shp[0],shp[1])) / (shp[0]*shp[1])
    return new_img, comp_ratio


op_img, comp_ratio = SVD_Compress(image=image,k=k)
st.image(op_img.astype('uint8'), caption="Compressed Image", channels='BGR', output_format='auto', clamp=True)

with st.sidebar:
    st.write(f"Compressed ratio = {comp_ratio} %")
    st.write("")
    st.write("Some compression error may be there due to internal type-conversion/type-casting.")

# print("img",image,"\n")
# print(type(new_img[0][0][0]))