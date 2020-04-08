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
