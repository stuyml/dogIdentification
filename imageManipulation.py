from PIL import Image
from PIL import ImageFile
import numpy
import csv

def convertImg(url,folder,counts):
    im = Image.open("./"+folder+"/"+url+".jpg")
    #im.show()
    # width, height = im.size
    # if height < 600 and width < 600:
    #     wall = numpy.zeros((600,600,3),dtype=numpy.int)
    #     block = numpy.array(im)
    #
    #     x = 0
    #     y = 0
    #     wall[x:x+block.shape[0], y:y+block.shape[1]] = block
    #     #print(wall)
    #     newimage = Image.fromarray(wall.astype('uint8'), 'RGB')
    destination = "./"+folder+"ModifiedGrainy/"+url+".jpg"
    #     try:
    #         newimage.save(destination, "JPEG", quality=80, optimize=True, progressive=True)
    #         if counts < 5:
    #             newimage.show()
    #     except IOError:
    #         ImageFile.MAXBLOCK = newimage.size[0] * newimage.size[1]
    w, h = im.size
    im = im.crop((w/5, h/5, 4*w/5,4*h/5))
    im = im.resize((128,128))
    # im = im.convert("L")
    # newimage = im.point(lambda x: 0 if x<128 else 255, '1')
    newimage = im.convert("1")
    newimage.save(destination, "PNG", quality=80, optimize=True, progressive=True)
    if counts < 5:
        newimage.show()

f  = open("labels.csv")
reader = csv.reader(f)
pixelArrays = []

counter = 0


for row in reader:
    if counter == 0:
        counter +=1

    else:
        convertImg(row[0],"train",counter)
        counter +=1

    if counter%100 == 0:
        print(counter)
