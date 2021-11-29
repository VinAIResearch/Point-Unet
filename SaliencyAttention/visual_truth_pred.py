import SimpleITK as itk 
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML, Image

def OverLayLabelOnImage(ImgIn,Label,W):
    #ImageIn is the image
    #Label is the label per pixel
    # W is the relative weight in which the labels will be marked on the image
    # Return image with labels marked over it
    Img=ImgIn.copy()
    TR = [0,1, 0, 0,   0, 1, 1, 0, 0,   0.5, 0.7, 0.3, 0.5, 1,    0.5]
    TB = [0,0, 0, 1,   1, 0, 1, 0, 0.5, 0,   0.2, 0.2, 0.7, 0.5,  0.5]
    TG = [0,0, 1, 0, 1, 1, 0, 1, 0.7, 0.4, 0.7, 0.2, 0,   0.25, 0.5]
    R = Img[:, :, 0].copy()
    G = Img[:, :, 1].copy()
    B = Img[:, :, 2].copy()
    for i in range(1, 4, 1):
        if i<len(TR): #Load color from Table
            R[Label == i] = TR[i] * 255
            G[Label == i] = TG[i] * 255
            B[Label == i] = TB[i] * 255
        else: #Generate random label color
            R[Label == i] = np.mod(i*i+4*i+5,255)
            G[Label == i] = np.mod(i*10,255)
            B[Label == i] = np.mod(i*i*i+7*i*i+3*i+30,255)
    Img[:, :, 0] = Img[:, :, 0] * (1 - W) + R * W
    Img[:, :, 1] = Img[:, :, 1] * (1 - W) + G * W
    Img[:, :, 2] = Img[:, :, 2] * (1 - W) + B * W
    return Img

def fillImage(image, maxSize):
    if min(image.shape) == maxSize:
        return image
    result = np.zeros((maxSize,maxSize), dtype=np.uint8)
    startX = (maxSize - image.shape[0])//2
    startY = (maxSize - image.shape[1])//2
    endX = startX + image.shape[0]
    endY = startY + image.shape[1]
    result[startX:endX,startY:endY] = image
    return result
def convert(image,seg_gt=None, seg_pd=None):
    maxSize = max(image.shape)
    image = fillImage(image,maxSize)
    if seg_gt is not None:
        seg_gt = fillImage(seg_gt,maxSize)
    if seg_pd is not None:
        seg_pd = fillImage(seg_pd,maxSize)
    img1 = np.stack((image,)*3, axis=-1)
    img2 = img1.copy()
    img3 = img1.copy()
    img1 = OverLayLabelOnImage(img1,seg_gt,0.4)
    img3 = OverLayLabelOnImage(img3,seg_pd,0.4)

    output = np.concatenate((img2,img1,img3),axis=1)
    return output
def showImage(image,f=0,seg_gt=None, seg_pd=None):
    # img = image[f].copy()
    img = image
    img = img / np.amax(img) * 255
    img = img.astype(np.uint8)
    fig = plt.figure(figsize = (15,15))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ims = []
    maxSize = max(img.shape)
    for i in tqdm(range(maxSize)):
        if i >= img.shape[0]:
            top = np.zeros([maxSize,3*maxSize,3],dtype=np.uint8)
        else:
            top = convert(img[i,:,:],seg_gt[i,:,:],seg_pd[i,:,:])
        if i >= img.shape[1]:
            mid = np.zeros([maxSize,3*maxSize,3],dtype=np.uint8)
        else:
            mid = convert(img[:,i,:],seg_gt[:,i,:],seg_pd[:,i,:])
        if i >= img.shape[2]:
            bottom = np.zeros([maxSize,3*maxSize,3],dtype=np.uint8)
        else:
            bottom = convert(img[:,:,i],seg_gt[:,:,i],seg_pd[:,:,i])
        merge = np.concatenate((top,mid,bottom),axis=0)
        im = plt.imshow(merge, animated=True,aspect='auto')
        plt.axis('off')
        ims.append([im])
        
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                repeat_delay=1000)
#         
    return ani

# def convert5(image,seg_gt=None, seg_pd=None):
#     maxSize = max(image.shape)
#     image = fillImage(image,maxSize)
#     if seg_gt is not None:
#         seg_gt = fillImage(seg_gt,maxSize)
#     if seg_pd is not None:
#         seg_pd = fillImage(seg_pd,maxSize)
#     img1 = np.stack((image,)*3, axis=-1)
#     img2 = img1.copy()
#     img3 = img1.copy()

#     if seg_gt is not None:
#         img1[np.where(seg_gt==1)] = np.array([255,0,0])
#         img1[np.where(seg_gt==2)] = np.array([0,255,0])
#         img1[np.where(seg_gt==3)] = np.array([0,0,255])
#     if seg_pd is not None:
#         img3[np.where(seg_pd==1)] = np.array([255,0,0])
#         img3[np.where(seg_pd==2)] = np.array([0,255,0])
#         img3[np.where(seg_pd==3)] = np.array([0,0,255])
#     output = np.concatenate((img2,img1,img3, img3,img3),axis=1)
#     return output

# def showImage5(image,f=0,seg_gt=None, seg_pd=None):
#     # img = image[f].copy()
#     img = image
#     img = img / np.amax(img) * 255
#     img = img.astype(np.uint8)
#     fig = plt.figure(figsize = (25,15))
#     fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
#     ims = []
#     maxSize = max(img.shape)
#     for i in tqdm(range(maxSize)):
#         if i >= img.shape[0]:
#             top = np.zeros([maxSize,5*maxSize,3],dtype=np.uint8)
#         else:
#             top = convert5(img[i,:,:],seg_gt[i,:,:],seg_pd[i,:,:])
#         if i >= img.shape[1]:
#             mid = np.zeros([maxSize,5*maxSize,3],dtype=np.uint8)
#         else:
#             mid = convert5(img[:,i,:],seg_gt[:,i,:],seg_pd[:,i,:])
#         if i >= img.shape[2]:
#             bottom = np.zeros([maxSize,5*maxSize,3],dtype=np.uint8)
#         else:
#             bottom = convert5(img[:,:,i],seg_gt[:,:,i],seg_pd[:,:,i])
#         merge = np.concatenate((top,mid,bottom),axis=0)
#         im = plt.imshow(merge, animated=True,aspect='auto')
#         plt.axis('off')
#         ims.append([im])
        
#     ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
#                                 repeat_delay=1000)
# #         
#     return ani

def convert5(image,seg_gt=None, seg_pd_1=None,seg_pd_2=None):
    maxSize = max(image.shape)
    image = fillImage(image,maxSize)
    if seg_gt is not None:
        seg_gt = fillImage(seg_gt,maxSize)
    if seg_pd_1 is not None:
        seg_pd_1 = fillImage(seg_pd_1,maxSize)
    if seg_pd_2 is not None:
        seg_pd_2 = fillImage(seg_pd_2,maxSize)
    img1 = np.stack((image,)*3, axis=-1)
    img2 = img1.copy()
    img3 = img1.copy()
    img4 = img1.copy()

    if seg_gt is not None:
        img2[np.where(seg_gt==1)] = np.array([255,0,0])
        img2[np.where(seg_gt==2)] = np.array([0,255,0])
        img2[np.where(seg_gt==3)] = np.array([0,0,255])
    if seg_pd_1 is not None:
        img3[np.where(seg_pd_1==1)] = np.array([255,0,0])
        img3[np.where(seg_pd_1==2)] = np.array([0,255,0])
        img3[np.where(seg_pd_1==3)] = np.array([0,0,255])
    if seg_pd_2 is not None:
        img4[np.where(seg_pd_2==1)] = np.array([255,0,0])
        img4[np.where(seg_pd_2==2)] = np.array([0,255,0])
        img4[np.where(seg_pd_2==3)] = np.array([0,0,255])
    output = np.concatenate((img1,img2,img3, img4,img4),axis=1)
    return output

def showImage5(image,f=0,seg_gt=None, seg_pd_1=None,seg_pd_2=None):
    # img = image[f].copy()
    img = image
    img = img / np.amax(img) * 255
    img = img.astype(np.uint8)
    fig = plt.figure(figsize = (25,15))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ims = []
    maxSize = max(img.shape)
    for i in tqdm(range(maxSize)):
        if i >= img.shape[0]:
            top = np.zeros([maxSize,5*maxSize,3],dtype=np.uint8)
        else:
            top = convert5(img[i,:,:],seg_gt[i,:,:],seg_pd_1[i,:,:],seg_pd_2[i,:,:])
        if i >= img.shape[1]:
            mid = np.zeros([maxSize,5*maxSize,3],dtype=np.uint8)
        else:
            mid = convert5(img[:,i,:],seg_gt[:,i,:],seg_pd_1[:,i,:],seg_pd_2[:,i,:])
        if i >= img.shape[2]:
            bottom = np.zeros([maxSize,5*maxSize,3],dtype=np.uint8)
        else:
            bottom = convert5(img[:,:,i],seg_gt[:,:,i],seg_pd_1[:,:,i],seg_pd_2[:,:,i])
        merge = np.concatenate((top,mid,bottom),axis=0)
        im = plt.imshow(merge, animated=True,aspect='auto')
        plt.axis('off')
        ims.append([im])
        
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                repeat_delay=1000)
#         
    return ani


def getPredictlabel(path):
    b = np.load(path)
    b = np.argmax(b,-1)
    return b


def reshape(img,seg_gt,dst_shape=(144,240,240)):
    X = np.empty((4, *dst_shape))
    S = list(img.shape[1:])
    for i in range(len(S)):
        if S[i]%2 == 1:
            S[i] -= 1
    _x = (S[0] - dst_shape[0])//2
    _y = (S[1] - dst_shape[1])//2
    _z = (S[2] - dst_shape[2])//2
    for i in range(img.shape[0]):
        tmp = img[i][_x:S[0]-_x,_y:S[1]-_y,_z:S[2]-_z]
        tmp = tmp/np.amax(tmp)
        X[i] = tmp
    seg_gt = seg_gt[_x:S[0]-_x,_y:S[1]-_y,_z:S[2]-_z]
    return X, seg_gt

# ID = 'BRATS_429.nii.gz'
# img = itk.GetArrayFromImage(itk.ReadImage('../BraTS2017/imagesTr/'+ID))
# seg_gt = itk.GetArrayFromImage(itk.ReadImage('../BraTS2017/labelsTr/'+ID))


# img, seg_gt =reshape(img,seg_gt)



# seg_pd =getPredictlabel('BRATS_429.nii.gz_predict.npy')

# print(img.shape)
# print(seg_gt.shape)




# ani = showImage(img,0,seg_gt,seg_gt)
# ani.save('2020BRATS_429.gif', writer='imagemagick', fps=5)