#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


# In[2]:


#img1=cv2.imread("sushi.png")
#img2=cv2.imread("cat_meme.png")


# In[3]:


img1=Image.open("sushi.jpg")
img2=Image.open("cat_meme.jpg")


# In[4]:


img1_array=np.array(img1)
img2_array=np.array(img2)


# In[5]:


plt.imshow(img1_array)


# In[7]:


img1_array.shape


# In[8]:


plt.imshow(img2_array)


# In[9]:


img2_array.shape


# In[10]:


img2_resized=img2.resize((1200,630))
#img2_resized.save('cat_meme_resized')


# In[11]:


img2_array=np.array(img2_resized)
img2_array.shape


# In[12]:


add_images=cv2.add(img1_array,img2_array)
plt.imshow(add_images)


# # Adding Weighted Images

# In[13]:


#0.5,1 ->weights of respective images
#-100 ->Gamma 
weighted_img=cv2.addWeighted(img1_array,0.5,img2_array,1,-100)
plt.imshow(weighted_img)


# # NOT operator 

# In[14]:


not_img1 = cv2.bitwise_not(img1_array)
plt.imshow(not_img1)


# # OR operator 

# In[15]:


or_images = cv2.bitwise_or(img1_array, img2_array)
plt.imshow(or_images)


# In[16]:


black_canvas1=np.zeros(shape=(500,500,3))


# In[17]:


black_canvas1.shape


# In[18]:


plt.imshow(black_canvas1)


# In[19]:


cv2.rectangle(black_canvas1,pt1=(200,100),pt2=(300,300),color=(0,255,0),thickness=-1)
plt.imshow(black_canvas1)


# In[20]:


black_canvas2=np.zeros(shape=(500,500,3))


# In[21]:


black_canvas2.shape


# In[22]:


plt.imshow(black_canvas2)


# In[23]:


cv2.circle(black_canvas2,center=(200,300),radius=100,color=(0,255,0),thickness=-1)
plt.imshow(black_canvas2)


# In[24]:


and_canvases=cv2.bitwise_and(black_canvas1,black_canvas2)
plt.imshow(and_canvases)


# In[25]:


or_canvases=cv2.bitwise_or(black_canvas1,black_canvas2)
plt.imshow(or_canvases)


# In[26]:


xor_canvases=cv2.bitwise_xor(black_canvas1,black_canvas2)
plt.imshow(xor_canvases)


# In[28]:


add_canvases=cv2.add(black_canvas1,black_canvas2)
plt.imshow(add_canvases)


# In[29]:


subtract_canvases=cv2.subtract(black_canvas1,black_canvas2)
plt.imshow(subtract_canvases)


# In[ ]:




