# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# color of irland flag

yellow =np.array((255,200,0))
black=np.array((0,0,0))
red=np.array((255,0,0))
flag = np.zeros((12,12,3))

# %%
flag[:4] = black  
flag[4:8] =  red
flag[8:] = yellow  


plt.imshow(flag)
plt.axis('off')  
plt.show()

# %%


# %%



