from keras.preprocessing.image import load_img, img_to_array

target_image_path = "hedef_resim/asd.jpg" # hedef resim
style_reference_image_path = "referans_resim/asdd.jpg" # referans resim

# resmi 400px yüksekliğe yeniden boyutlandırma
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

import numpy as np
from keras.applications import vgg19

# yardımcı fonksiyonlar
def preprocess_image(image_path):

	img = load_img(image_path, target_size=(img_height, img_width))
	img = img_to_array(img)
	
	img = np.expand_dims(img, axis=0)
	img = vgg19.preprocess_input(img)

	return img

def deprocess_image(x):

	# ImageNet veri setinin ortalamasını çıkarıp sıfır merkezli yapmak.
	# Böylece vgg19.preprocess_input() un yaptıklarını geri alır.

	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68


	# resmi BGR den RGB ye dönüştürür.
	# vgg19.preprocess_input() un yaptıklarını geri alır.

	x = x[:, :, ::-1]

	x = np.clip(x, 0, 255).astype('uint8')

	return x

# öneğitimli vgg19 ağını yüklemek ve üç resme uygulamak(referans resim, hedef resim, üretilen resim)

from keras import backend as K 

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))

# yer tutucu üretilen resmi içerecek
combination_image =  K.placeholder((1, img_height, img_width, 3))


# üç resmi tekbir yığında birleştirir
input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor, weights="imagenet", include_top=False)

print("model dahil edildi !!")

# hedef resimle üretilecek resmin birbirine benzemesini sağlayacak içerik kaybı
def content_loss(base, combination):

	return K.sum(K.square(combination - base))


# girdi matrisinin gram matrisini hesaplamak
def gram_matrix(x):

	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

	gram = K.dot(features, K.transpose(features))

	return gram

# sitil kaybı
def style_loss(style, combination):

	S = gram_matrix(style)

	C = gram_matrix(combination)

	chanels = 3
	size = img_height * img_width

	return K.sum(K.square(S - C)) / (4. * (chanels**2) * (size**2))


# toplam değişim kaybı

def total_variation_loss(x):

	a = K.square(
		x[: , :img_height - 1, :img_width - 1 , :] -
		x[:, 1:, :img_width - 1 , :])

	b = K.square(
		x[: , :img_height - 1, :img_width - 1 , :] -
		x[:, :img_height - 1, 1:, :])

	return K.sum(K.pow(a+b, 1.25))


# enküçülteceğimiz kaybı tanımlamak

# katman isimlerini aktivasyon tensörlerine eşleyen sölük
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# içerik kaybı için kullanılacak katman
content_layer = 'block5_conv2'
# sitil kaybında kullanılacak katmanlar
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# tüm kayıp bileşenlerini skaler kayıp değerine eklemek
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * content_loss(target_image_features, # içerik kaybı
                                      combination_features)

for layer_name in style_layers:

    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss = loss + (style_weight / len(style_layers)) * sl

# toplam değişim kaybı
loss = loss + total_variation_weight * total_variation_loss(combination_image)


grads = K.gradients(loss, combination_image)[0]

# mevcut kayıp değerini ve gradyanları getirecek fonksiyon
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

# bu sınıf fetch_loss_and_grads fonksiyonunu kullanarak
# SciPy tarafından istendiği gibi kaybın ve gradyanların
# iki ayrı çağrıda getirilmesini sağlar
class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

result_prefix = 'cikti_resim/style_transfer_result'
iterations = 20

# resim scipy.optimize.fmin_1_bfgs_b sadece düz vektörlere
# işlem yapabildiğinden düzleştirilir.
x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    print('döngü başlıyor', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    print('kayıp:', min_val)
    # her iterasyonda elde edilen resim kaydedilir.
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('resim kaydedildi', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))