# GANdela

# Integrantes
- Alfredo Montero López
- Leonardo Amaro Rodríguez
- Anthuan Montes de Oca
- Yisell Martínez Noa


# Introduccion
### Motivación
Se necesita poder generar imágenes territoriales realistas con la representacion de las alturas de cada punto en el mapa.

### Problemática
Los algoritmos matemáticos encargados de realizar operaciones similares, debido a su naturaleza no pueden crear pendientes muy pronunciadas. Por tanto la necesidad de buscar alternativas que intenten simular la realidad

### Objetivos generales
Creación de un GAN (Generative Adversarial Network) para la reacreación de un mapa junto alas elevaciones del terreno dada una imagen satelital

# Propuesta de solución
Se creó una pix2pix-GAN para la cual dada una imagen satelital, se generará una matriz de alturas de alturas.
- **Input**: Imagen satelital de 3x256x256 (RGB) en formato GeoTIFF .
- **Output**: Mapa de alturas de 1x256x256 .

Dada la imagen satelital y su combinación de colores(RGB), se puede calcular el tipo de terreno correspondiente.

Cada archivo (imagen y altura) se transforma y se le cambia su ancho y alto a 256x256 ya que debido a la naturaleza del algoritmo, ambos tienen que tener la misma dimensión. Luego, a la matriz resultante del modelo se le aplica una transformación para cambiar su dimensióna 32x32 para que pueda cumplir las restricciones del problema.

# Qué es pix2pix-GAN
Pix2Pix es un tipo de modelo de red generativa adversaria (GAN) diseñado para tareas de traducción de imágenes, es decir, convertir una imagen de un dominio a otro. Fue introducido en 2017 en el trabajo titulado "Image-to-Image Translation with Conditional Adversarial Networks". A diferencia de las GANs tradicionales, que generan imágenes a partir de un vector de ruido aleatorio, Pix2Pix toma una imagen de entrada y produce una imagen de salida correspondiente, lo que lo hace adecuado para aplicaciones como la conversión de bocetos en imágenes, la colorización de imágenes en blanco y negro, la conversión de fotos de día a noche, entre otros.

La arquitectura de Pix2Pix consta de dos componentes principales:
- Generador (G): Toma una imagen de entrada y genera una imagen que se espera se asemeje a una imagen realista del dominio objetivo. En Pix2Pix, el generador suele ser una red de tipo "U-Net", una arquitectura de red neuronal que permite la propagación de contextos a través de las diferentes escalas de la imagen.
- Discriminador (D): Su función es distinguir entre las imágenes reales del dominio objetivo y las imágenes generadas por el generador. El discriminador en Pix2Pix es una CNN (red neuronal convolucional) que clasifica segmentos de la imagen (en lugar de la imagen completa) como real o falsa, lo cual se conoce como PatchGAN.

El entrenamiento de Pix2Pix se realiza de manera adversaria: el generador intenta producir imágenes que el discriminador no pueda distinguir de las reales, mientras que el discriminador se mejora para distinguir entre imágenes reales y generadas. Este proceso se guía mediante una función de pérdida que combina la pérdida adversaria (para engañar al discriminador) y la pérdida L1 (para penalizar las diferencias pixel a pixel entre la imagen generada y la imagen real objetivo), ayudando así a producir resultados más fieles al dominio objetivo.

El resultado es un sistema capaz de transformar imágenes de un dominio a otro manteniendo la estructura y el contexto de la imagen de entrada, lo que lo hace muy efectivo para una amplia gama de aplicaciones de traducción de imágenes.

# Base de datos
La base de datos se obtuvo de Google Earth Engine.
- Para la parte de las imágenes satelitales se usó `LANDSAT/LC09/C02/T1`
- Para la parte de las alturas se usó `CGIAR/SRTM90_V4`

Se utilizó la API para la extracción de dichos datos. Los cuales son subidos al Drive de la cuenta de PixelCampione asociada con el ordenador actual.
Luego, descargarlos en la carpeta `dataset` para su posterior utilización.

La región se tuvo que distribuir en diferentes partes por restricciones de la misma API, por tanto se tuvieron que realizar diferentes llamados a la API.

# Resultados
Pendientes de calcular

# Trabajo futuro
Se puede entrenar otra instancia del modelo donde la salida seria el tipo de terreno correspondiente.
Esto se puede realizar de manera que a cada terreno le correspona un número. Incluso se puede modificar el número asignado para que terrenos similares tengan su representación numérica cercanas entre si.
Luego se tendrían dos GAN, una para la generación de alturas y otra para la clasificación de terrenos.

Otra idea seria que dadas las imágenes satelitales y su clasificación con respecto al tipo de terreno, se puede generar un mapa de alturas sin tener que ingresar una imagen satelital. Esto se puede lograr creando un vector por cada mapa en la base de datos tomando en cuenta la presencia o no de diferentes caracteristícas de terrenos. Eso, junto con la vectorización de la consulta, se puede obtener los mapas que más relación tienen con esta.
Permitiendo la redacción de consultas y su posterior interpretación usando técnicas de procesamiento de lenguaje natural o incluso de ChatGPT.

# Bibliografia
- [Pix2Pix](https://arxiv.org/abs/1611.07004)
- [GAN definition](https://arxiv.org/abs/1406.2661)
- [Google Earth catalog](https://developers.google.com/earth-engine/datasets/catalog)
- [Earth Engine API](https://developers.google.com/earth-engine/guides)
- [Dataset in personal Drive](https://drive.google.com/drive/folders/1RnP7gD8rTcWRNxTr9LC65Z3RDC_HgxgX?usp=sharing)