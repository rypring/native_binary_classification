import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_loss(loss,val_loss, name):
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('%s_loss.png' % name)

def plot_acc(acc, val_acc, name):
    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.title('Model Accuracy')
    plt.savefig('%s_acc.png' % name)

# DO NOT RUN TWICE
def flipImages(jpgpath):
    i = 0
    for jpg in jpgpath:
        image = Image.open(jpg)
        img_arr = np.array(image)
        img_arr = np.flip(img_arr, axis=1)
        image = Image.fromarray(img_arr)
        image.save('test/Images/native/img%d.jpeg' % i)
        i+=1


# DO NOT RUN TWICE
def removeWatermark(jpgpath):
    for jpg in jpgpath:
        image = Image.open(jpg)
        img_arr = np.array(image)
        img_arr = img_arr[:-40, :]
        image = Image.fromarray(img_arr)
        image.save(jpg)