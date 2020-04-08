![licence](https://img.shields.io/badge/Keras-V2.3.1-red)
![licencee](https://img.shields.io/badge/Tensorflow-V2.0-yellow)
![licence](https://img.shields.io/badge/demir-ai-blueviolet)
![licence](https://img.shields.io/badge/Ahmet%20Furkan-DEM%C4%B0R-blue)


# Neural Style Transfer 

* Sinirsel stil aktarımı, hedef resmin içeriğini koruyarak referans resmin stilini aktarmaktır.
* stil: farklı uzamsal ölçeklerde resmin dokuları, renkleri ve görsel örüntüleri anlamına gelmektedir.
* özgün resmin içeriğini kaybetmeden referans resmin stilini uydurmak istiyoruz.
* Eğer içerik ve stii matematiksel olarak ifade edebilirsek, aşağıdaki gibi bir kayıp fonksiyonu enküçültülebilir.
       
      loss = distance(style(reference_image) - style(generated_image)) +
            distance(content(original_image) - content(generated_image))
            
* Burda distance L2 gibi bir norm  fonksiyonu content resmi alıp onun içeriğini hesaplayan bir fonksiyon ve style resmi alıp onun stilinin gösterimini hesaplayan fonksiyondur.
* Gatsys vd. derin evrişimli sinir ağlarının content ve style fonksiyonlarını tanımlamak için bir yol sunduğunu gözlemlemişlerdir. şimdi bunun nasıl olabileceğine bakalım.


# Content Kaybı

* İçerik kaybı olarak, hedef resim ve üretilen resmin öneğitimli bir evrişimli sinir ağının sonlarındaki katmanların aktivasyonlarının L2 normu iyi bir adaydır.
* Bu son katmanlardaki aktivasyonlara bakarak üretilen resmin hedef resme benzemesini sağlayacaktır.
* Eğer son katmanların gördüğü şeylerin girdi hedef resmin içeriği olduğunu kabul edersek, resmin içeriğini korumasını sağlar.


# Style kaybı

* Stilini alacağınız referans resmin evrişimli sinir ağı tarafından çıkarılan tüm ölçeklerdeki görünüşümü yakalamaya çalışırsınız.
* Stil kaybı referans resimle hedef resim arasında farklı katmanların aktivasyonlarının iç korelasyonlarını korumayı hedefler.


# Keras 'ta Sinirsel Stil Aktarımı

* Bu örneğimizde VGG19 öneğitimli evrişimli sinir ağını kullanarak Sinirsel Stil Aktarımını gerçekleştirdik. süreç şu şekilde olacaktır :  
* Referans resim, hedef resim ve üretilen resmin VGG19 aktivasyonlarını aynı anda hesaplayan bir ağ hesaplayalım.
* Enküçülterek stil aktarımını sağlayacağınız daha önce tanımladığınız kaybı tanımlamak için bu üç resim üzerinde hesaplanan katman aktivasyonlarını kullanalım.
*Kayıp fonksiyonunu enküçültmek için gradyan inişini kullanın


* Değişkenlerin Tanımlanması : 

![Screenshot_2020-04-08_23-39-37](https://user-images.githubusercontent.com/54184905/78831438-50087700-79f2-11ea-81b1-5d3cf0783c81.png)


* Yardımcı Fonksiyonlar :

![Screenshot_2020-04-08_23-41-52](https://user-images.githubusercontent.com/54184905/78831627-a07fd480-79f2-11ea-9645-d5fc55929883.png)


* Öneğitimli VGG19 ağını yüklemek ve üç resme uygulama : 

![Screenshot_2020-04-08_23-44-07](https://user-images.githubusercontent.com/54184905/78831807-ed63ab00-79f2-11ea-94d3-7a5946fe08ee.png)


* İçerik kaybı (hedef resimle üretilen resmin birbirine benzemesini sağlayacak içerik kaybı) : 

![Screenshot_2020-04-08_23-46-16](https://user-images.githubusercontent.com/54184905/78832070-54815f80-79f3-11ea-8175-82508aefd8a8.png)


* Stil kaybı : 

