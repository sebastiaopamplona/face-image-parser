Descreveres os dados e pré-processamento, incluindo como dividiste em treino e validação, se fizeste data augmentation (e como) e a geração dos batches para treino:
  - **dados**:
  IMDB-WIKI (https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), dataset com 500K+ imagens de rosto, construído com *web scraping* dos sites IMDB e Wikipedia, com anotações da idade; dividido em 2 datasets:
    - WIKI: 62,328 imagens de rosto (32,399 depois do pré-processamento);
    - IMDB: 460,723 imagens de rosto (*TODO* depois do pré-processamento)
    - cada dataset está associado a um ficheiro de metadados (wiki.mat e imdb.mat) com anotações da data de nascimento, data da fotografia, posição do rosto na imagem, entre outros (nota: o ficheiro imdb.mat está corrompido, mas os nomes dos ficheiros das imagens contêm a data de nascimento e data da fotografia, por isso foi assim que contornei o problema e extraí as idades das pessoas nas fotografias);
    - para cada dataset existe uma versão *cropped*, onde os rostos já estão "recortados"; a versão *cropped* foi o meu ponto de partida;
  - **pré-processamento** (igual para o WIKI e IMDB, à exceção da extração da idade):
    1. removi as imagens corrompidas
    2. removi as imagens que não continham um rosto
    3. removi os outliers (18 <= idade <= 58)
    4. *data augmentation*
  - **data augmentation**:
    1. alinhei os rostos em relação aos olhos, utilizando a class *FaceAligner*, da library *imutils* (https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/facealigner.py)
    2. para cada imagem já alinhada e *cropped* pelos autores dos datasets, extraí o rosto utilizando a rede MTCNN (https://github.com/ipazc/mtcnn); esta extração estica também o rosto para o tamanho desejado (224x224 ou 160x160, dependendo da rede (VGG16 ou Facenet))
    eg. (1. | 2.):
    ![](https://i.ibb.co/HYHPfBR/0.jpg)
    ![](https://i.ibb.co/L5G8Lcm/1.jpg)
    ![](https://i.ibb.co/6NM4Mq0/2.jpg)
  - **como dividiste em treino e validação**: 
    - 90% treino
    - 5% validação
    - 5% teste
  - **geração dos batches para treino**:
    - utilizei os *data generators* do *Keras*, utilizados para os 3 conjuntos: treino, validação e teste;
    - tenho 3 [data generators](utils/data/data_generators.py):
        - AgeDG: critério de semelhança -> idade
        - AgeIntervalDG: critério de semelhança -> intervalo de idade          
        - EigenvaluesDG: critério de semelhança -> eigenvalue       

As arquitecturas que testaste e a função de loss que estás a usar (como a implementaste e como está encaixada no treino; nesta parte se calhar preciso que partilhes o código e digas quais as partes relevantes):
  - **arquitecturas que testaste**:
    - a arquitetura base consiste na concatenação de uma rede de *feature extraction* (eg. uma rede que gera embeddings, por exemplo VGG16 ou Facenet) e o label da imagem de input (eg. idade ou eigenvalue)
    ![](https://i.ibb.co/bz9MSyk/final-model.png)
  - **função de loss que estás a usar**:
    - **como a implementaste**:
    - **como está encaixada no treino**:

Os parâmetros de treino (épocas, parâmetros do optimizador) e os resultados (os plots da loss no treino e na validação):
  - **parâmetros de treino**:
    - **épocas**:
    - **parâmetros do optimizador**:
  - **resultados**:
    - **plots da loss no treino e validação**:


Diz-me também que testes já fizeste ao código para garantir que está tudo a funcionar:
