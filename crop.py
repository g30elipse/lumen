from PIL import Image
from scipy import misc

SIZE = (120, 120)

img = misc.imread("data/test/myimage/img.jpg")
img = img[:,:,0]
misc.imsave('data/test/myimage/img-gray.jpg', img)


img = Image.open("data/test/myimage/img-gray.jpg")

size = img.size

sx = int(size[0]/SIZE[0])
sy = int(size[1]/SIZE[1])

x = 0
for i in range(1, sx):
    for j in range(1, sy):
        cropped = img.crop(((i-1)*SIZE[0], (j-1)*SIZE[1], i*SIZE[0], j*SIZE[1]))
        cropped.save('data/test/IIITA/{}.jpg'.format(x))
        x += 1


