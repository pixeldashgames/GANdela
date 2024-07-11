# Resultados de las pruebas
El modelo fue probado con diferentes funciones de pérdidas tanto para el Discriminador como para el Generador

## Intentos de configuración
El problema principal de estos modelos es que durante las primeras iteraciones, el Discriminador era capaz de superar fácilmente al Generador, provocando que estuviera muy seguro de si la entrada era falsa o no. Dándole valores como 0.999 a las entradas reales y 0.001 a las falsas, restringiendo el aprendizaje del Generador.

### L1 con BCEwithLogits
Epoch : 0 | Gen Loss : 7779.90283203125 | Disc Loss : 0.7112271785736084

Epoch : 50 | Gen Loss : 7834.88427734375 | Disc Loss : 0.30028036236763

Epoch : 100 | Gen Loss : 7570.71240234375 | Disc Loss : 0.06394217163324356

### L1 with MSE
Epoch : 10 | Gen Loss : 8701.7890625 | Disc Loss : 0.33725571632385254

Epoch : 20 | Gen Loss : 8048.34912109375 | Disc Loss : 0.30385181307792664

Epoch : 30 | Gen Loss : 6809.49462890625 | Disc Loss : 0.2835003137588501

Epoch : 40 | Gen Loss : 7663.60498046875 | Disc Loss : 0.2863680422306061

Epoch : 50 | Gen Loss : 7830.93115234375 | Disc Loss : 0.22010774910449982

Epoch : 60 | Gen Loss : 8029.484375 | Disc Loss : 0.2207767814397812

Epoch : 70 | Gen Loss : 8668.478515625 | Disc Loss : 0.22636985778808594

Epoch : 80 | Gen Loss : 8916.2041015625 | Disc Loss : 0.21985182166099548

Epoch : 90 | Gen Loss : 7031.05810546875 | Disc Loss : 0.1898426115512848

Epoch : 100 | Gen Loss : 7979.99951171875 | Disc Loss : 0.22319374978542328

### MSE with BCEwithLogits
Epoch : 0 | Gen Loss : 798742.75 | Disc Loss : 0.7151925563812256

Epoch : 10 | Gen Loss : 99210.1484375 | Disc Loss : 0.7176052927970886

Epoch : 20 | Gen Loss : 82351.9453125 | Disc Loss : 0.5502198934555054

Epoch : 30 | Gen Loss : 107344.359375 | Disc Loss : 0.461466521024704

- After using 2 steps training

    Epoch : 10 | Gen Loss : 5682.86572265625 | Disc Loss : 0.10323390364646912

    Epoch : 20 | Gen Loss : 3714.49267578125 | Disc Loss : 0.10353855788707733

    Epoch : 30 | Gen Loss : 3954.466796875 | Disc Loss : 0.11623755097389221

    Epoch : 40 | Gen Loss : 5263.35107421875 | Disc Loss : 0.03681714087724686

    Epoch : 50 | Gen Loss : 4207.08984375 | Disc Loss : 0.04253549873828888

    Epoch : 60 | Gen Loss : 4234.42138671875 | Disc Loss : 0.06193991377949715

    Epoch : 70 | Gen Loss : 5359.783203125 | Disc Loss : 0.01920810341835022

    Epoch : 80 | Gen Loss : 3819.768798828125 | Disc Loss : 0.1254141926765442

    Epoch : 90 | Gen Loss : 4847.0 | Disc Loss : 0.033486366271972656

    Epoch : 100 | Gen Loss : 4832.95458984375 | Disc Loss : 0.027414310723543167

    Epoch : 110 | Gen Loss : 4499.0068359375 | Disc Loss : 0.018689129501581192

    Epoch : 120 | Gen Loss : 4022.912841796875 | Disc Loss : 0.03827538713812828

    Epoch : 130 | Gen Loss : 4988.22509765625 | Disc Loss : 0.040250711143016815

    Epoch : 140 | Gen Loss : 4469.6435546875 | Disc Loss : 0.06191401183605194

    Epoch : 150 | Gen Loss : 4215.97412109375 | Disc Loss : 0.09380695223808289

    Epoch : 160 | Gen Loss : 4244.1259765625 | Disc Loss : 0.014170262962579727

    Epoch : 170 | Gen Loss : 5543.23486328125 | Disc Loss : 0.025407154113054276

    Epoch : 180 | Gen Loss : 5570.6484375 | Disc Loss : 0.01545000821352005

    Epoch : 190 | Gen Loss : 4534.845703125 | Disc Loss : 0.04580564424395561

    Luego de aplicar el procedimiento se puede ver una marcada mejora de los valores de pérdida del generador y del Discriminador


# Modelo final
## MSE con MSE con dos tiempos
Luego de varias pruebas se determinó que el modelo con los mejores resultados es el que tanto para el Generador como el Discriminador se les asignaron la función de pérdida del Error de Mínimos Cuadrados (MSE)

Para esta solución se obtuvieron resultados que eran un poco mejores que las demás. Se pudo observar como aumentaba la puntuación dada por el Discriminador a la matriz generada, y a su vez como este disminuía en las siguientes épocas. Pudiéndose observar el como tanto el Generador como el Discriminador se "enfrentaban".

Al ser la mejor combinación encontrada hasta el momento, se utilizó un entrenamiento en dos tiempos. Este consiste en actualizar el generador más frecuentemente que el discriminador. Para este problema el Generador se entrenó dos veces por cada vez que lo hacía el Discriminador.

Esto se hizo con el fin de evitar el problema actual que era que el Disciminador superaba fácilmente al Generador.

### Resultados

Epoch : 10 | Gen Loss : 5161.53466796875 | Disc Loss : 0.23925688862800598

Epoch : 20 | Gen Loss : 4585.37939453125 | Disc Loss : 0.12140040844678879

Epoch : 30 | Gen Loss : 5591.15771484375 | Disc Loss : 0.09408846497535706

Epoch : 40 | Gen Loss : 4471.14892578125 | Disc Loss : 0.09517452120780945

Epoch : 50 | Gen Loss : 4268.10205078125 | Disc Loss : 0.15849535167217255

Epoch : 60 | Gen Loss : 5619.7939453125 | Disc Loss : 0.053437717258930206

Epoch : 70 | Gen Loss : 4717.66552734375 | Disc Loss : 0.08003553748130798

Epoch : 80 | Gen Loss : 4510.0126953125 | Disc Loss : 0.0807599350810051

Epoch : 90 | Gen Loss : 4117.96728515625 | Disc Loss : 0.14058037102222443

Epoch : 100 | Gen Loss : 5128.8388671875 | Disc Loss : 0.06015884503722191

Epoch : 110 | Gen Loss : 4116.8330078125 | Disc Loss : 0.10317926108837128

Epoch : 120 | Gen Loss : 5011.46630859375 | Disc Loss : 0.039441488683223724

Epoch : 130 | Gen Loss : 3917.461181640625 | Disc Loss : 0.07097624242305756

Epoch : 140 | Gen Loss : 4299.22998046875 | Disc Loss : 0.06659030169248581

Epoch : 150 | Gen Loss : 3580.409912109375 | Disc Loss : 0.07947041839361191

Epoch : 160 | Gen Loss : 5353.79736328125 | Disc Loss : 0.054755087941884995

Epoch : 170 | Gen Loss : 4320.47265625 | Disc Loss : 0.08891814202070236

Epoch : 180 | Gen Loss : 4033.01318359375 | Disc Loss : 0.04331081360578537

Epoch : 190 | Gen Loss : 4119.53125 | Disc Loss : 0.051476653665304184

Epoch : 200 | Gen Loss : 4353.91064453125 | Disc Loss : 0.05317274481058121

Epoch : 210 | Gen Loss : 4696.810546875 | Disc Loss : 0.04456945136189461

Durante el entrenamiento se puedo observar como el Discriminador, a diferencia de los otros modelos, no sobrepasó al Generador.
Se pudo observar cómo la puntuación dada al output generado variaba de 0.4 a 0.7 durante cada epoch.
Sin embargo se sigue evidenciando cómo el Discriminador está siempre un poco por encima al Generador.