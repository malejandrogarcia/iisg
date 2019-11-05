def mostrar_semillas(imagenC, im_O, im_B):
    imagenS = np.zeros(imagenC.shape)
    imagenS[:,:,0] = np.maximum(imagenC[:,:,0], im_O*255)
    imagenS[:,:,1] = imagenC[:,:,1]
    imagenS[:,:,2] = np.maximum(imagenC[:,:,2], im_B*255)
    plt.figure(figsize=(7,7))
    plt.imshow(imagenS.astype(int))
    plt.show()
