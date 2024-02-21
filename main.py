# Imports

# Imports para manipulação e visualização de dados
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Imports para manipulação de imagens
import os
import cv2
import itertools
import shutil
import imageio
import skimage
import skimage.io
import skimage.transform
from pathlib import Path

# Imports para Deep Learning
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.metrics import binary_accuracy

# Imports para cálculo de métricas e outras tarefas
import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Supress de warnings
import warnings
warnings.filterwarnings('ignore')

''' Extraindo a Variável Target '''


# Função para selecionar o 4º índice no final da string (nome do arquivo)
# Exemplo: CHNCXR_0470_1.png --> 1 é o label, significa que Tuberculose está presente na imagem.

def extrair_target(x):
    target = int(x[-5])

    if target == 0:
        return 'Normal'
    if target == 1:
        return 'Tuberculose'


''' Visualizando as Imagens '''


# Função para visualizar as imagens
def visualiza_images(col_name, figure_cols, df, caminho_imagens):
    # Define as categorias
    categories = (df.groupby([col_name])[col_name].nunique()).index

    # Prepara os subplots
    f, ax = plt.subplots(nrows=len(categories),
                         ncols=figure_cols,
                         figsize=(4 * figure_cols, 4 * len(categories)))

    # Desenha as imagens
    for i, cat in enumerate(categories):

        # Extrai uma amostra
        sample = df[df[col_name] == cat].sample(figure_cols)

        # Loop pelas colunas da figura
        for j in range(0, figure_cols):
            # Extrai o nome da imagem
            file = caminho_imagens + sample.iloc[j]['image_id']

            # Lê a imagem do disco
            im = imageio.imread(file)

            # Mostra a imagem em gray (preto e branco)
            ax[i, j].imshow(im, resample=True, cmap='gray')
            ax[i, j].set_title(cat, fontsize=14)

    plt.tight_layout()
    plt.show()

''' Ajustando e Organizando o Primeiro Dataset de Imagens de Raio-X '''


# Função para leitura dos metadados das imagens
def leitura_imagens(file_name):
    # Leitura da imagem
    image = cv2.imread(caminho_imagens + file_name)

    # Extração do número máximo e mínimo de pixels
    max_pixel_val = image.max()
    min_pixel_val = image.min()

    # image.shape[0] - largura da imagem
    # image.shape[1] - altura da imagem
    # image.shape[2] - número de canais
    # Se o shape não tiver um valor para num_channels (altura, largura) então atribuímos 1 ao número de canais.
    if len(image.shape) > 2:
        output = [image.shape[0], image.shape[1], image.shape[2], max_pixel_val, min_pixel_val]
    else:
        output = [image.shape[0], image.shape[1], 1, max_pixel_val, min_pixel_val]
    return output


# Função para a Matriz de Confusão
def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Matriz de Confusão',
                          cmap=plt.cm.YlOrRd):
    # Se normalize = True, obtemos a matriz de confusão com dados normalizados
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão Normalizada")
    else:
        print('Matriz de Confusão Sem Normalização')

    # Mostramos a Matriz de Confusão
    print(cm)

    # Plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Plot do texto
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Label Verdadeiro')
    plt.xlabel('Label Previsto')
    plt.tight_layout()

if __name__ == '__main__':

    ''' Definindo o Local de Armazenamento das Imagens de Raio-X '''

    # Lista o conteúdo do diretório
    os.listdir('dados')

    # Diretórios para os 2 grupos de imagens
    imagens_shen = 'dados/ChinaSet_AllFiles/CXR_png/'
    imagens_mont = 'dados/MontgomerySet/CXR_png/'

    # Grava a lista de imagens em cada pasta
    shen_image_list = os.listdir(imagens_shen)
    mont_image_list = os.listdir(imagens_mont)

    ''' Preparando e Carregando as Imagens de Raio-X '''

    # Prepara os dataframes com as listas das imagens
    df_shen = pd.DataFrame(shen_image_list, columns=['image_id'])
    df_mont = pd.DataFrame(mont_image_list, columns=['image_id'])

    # Remove da lista o nome 'Thumbs.db'
    df_shen = df_shen[df_shen['image_id'] != 'Thumbs.db']
    df_mont = df_mont[df_mont['image_id'] != 'Thumbs.db']

    # Reset do índice para e evitar erros mais tarde
    df_shen.reset_index(inplace=True, drop=True)
    df_mont.reset_index(inplace=True, drop=True)

    # Adicionando label aos dataframe
    df_shen['target'] = df_shen['image_id'].apply(extrair_target)
    df_mont['target'] = df_mont['image_id'].apply(extrair_target)

    # Shenzen Dataset
    df_shen['target'].value_counts()

    # Montgomery Dataset
    df_mont['target'].value_counts()

    ''' Ajustando e Organizando o Primeiro Dataset de Imagens de Raio-X '''

    # Define o caminho onde estão as imagens
    caminho_imagens = imagens_shen

    # Retorna os metadados das imagens
    meta_shen = np.stack(df_shen['image_id'].apply(leitura_imagens))

    # Grava o resultado em um dataframe
    df = pd.DataFrame(meta_shen, columns=['largura', 'altura', 'canais', 'maior_valor_pixel', 'menor_valor_pixel'])

    # Concatena com o dataset atual
    df_shen = pd.concat([df_shen, df], axis=1, sort=False)

    # Não precisamos mais desse dataframe. Removemos para liberar espaço na memória RAM.
    del df

    ''' Ajustando e Organizando o Segundo Dataset de Imagens de Raio-X '''

    # Define o caminho onde estão as imagens
    caminho_imagens = imagens_mont

    # Retorna os metadados das imagens
    meta_mont = np.stack(df_mont['image_id'].apply(leitura_imagens))

    # Grava o resultado em um dataframe
    df = pd.DataFrame(meta_mont, columns=['largura', 'altura', 'canais', 'maior_valor_pixel', 'menor_valor_pixel'])

    # Concatena com o dataset atual
    df_mont = pd.concat([df_mont, df], axis=1, sort=False)

    # Não precisamos mais desse dataframe. Removemos para liberar espaço na memória RAM.
    del df

    ''' Divisão dos Dados em Treino e Validação '''

    # Vamos combinar os 2 dataframes
    df_data = pd.concat([df_shen, df_mont], axis=0).reset_index(drop=True)

    # E "embaralhar (shuffle)" os dados
    df_data = shuffle(df_data)

    # Cria uma nova coluna chamada 'labels' que mapeia as classes para valores binários (0 ou 1)
    df_data['labels'] = df_data['target'].map({'Normal': 0, 'Tuberculose': 1})

    # Definimos y (saída)
    y = df_data['labels']

    # Definimos dados de treino e validação
    df_treino, df_val = train_test_split(df_data, test_size=0.15, random_state=101, stratify=y)

    ''' Separando as Imagens Organizadas Por Classe '''

    # Cria um novo diretório que servirá como base
    # Você deve alterar o base_dir para oo seu diretório, no Titan ou na sua máquina local
    base_dir = 'dados_final/'

    # Criamos o PATH (caminho)
    dir_base = Path(base_dir)

    # Verificamos se o diretório já existe e se não existir, criamos
    if dir_base.exists():
        print('O diretório já existe. Delete no SO e tente novamente.')
    else:
        os.mkdir(base_dir)

    # Preparamos a criação do diretório com dados de treino
    dados_treino = os.path.join(base_dir, 'dados_treino/')

    # Criamos o PATH (caminho)
    dir_treino = Path(dados_treino)

    # Verificamos se o diretório já existe e se não existir, criamos
    if dir_treino.exists():
        print('O diretório já existe. Delete no SO e tente novamente.')
    else:
        os.mkdir(dados_treino)

    # Preparamos a criação do diretório com dados de validação
    dados_val = os.path.join(base_dir, 'dados_val/')

    # Criamos o PATH (caminho)
    dir_val = Path(dados_val)

    # Verificamos se o diretório já existe e se não existir, criamos
    if dir_val.exists():
        print('O diretório já existe. Delete no SO e tente novamente.')
    else:
        os.mkdir(dados_val)

    # Diretório para imagens de raio-x Normais para treinamento
    Normal = os.path.join(dados_treino, 'Normal')

    # Criamos o PATH (caminho)
    dir_normal_treino = Path(Normal)

    # Verificamos se o diretório já existe e se não existir, criamos
    if dir_normal_treino.exists():
        print('O diretório já existe. Delete no SO e tente novamente.')
    else:
        os.mkdir(Normal)

    # Diretório com imagens de raio-x com Tuberculose para treinamento
    Tuberculose = os.path.join(dados_treino, 'Tuberculose')

    # Criamos o PATH (caminho)
    dir_tb_treino = Path(Tuberculose)

    # Verificamos se o diretório já existe e se não existir, criamos
    if dir_tb_treino.exists():
        print('O diretório já existe. Delete no SO e tente novamente.')
    else:
        os.mkdir(Tuberculose)

    # Diretório com imagens de raio-x Normais para validação
    Normal = os.path.join(dados_val, 'Normal')

    # Criamos o PATH (caminho)
    dir_normal_val = Path(Normal)

    # Verificamos se o diretório já existe
    if dir_normal_val.exists():
        print('O diretório já existe. Delete no SO e tente novamente.')
    else:
        os.mkdir(Normal)

    # Diretório com imagens de raio-x com Tuberculose para validação
    Tuberculose = os.path.join(dados_val, 'Tuberculose')

    # Criamos o PATH (caminho)
    dir_tb_val = Path(Tuberculose)

    # Verificamos se o diretório já existe
    if dir_tb_val.exists():
        print('O diretório já existe. Delete no SO e tente novamente.')
    else:
        os.mkdir(Tuberculose)

    # Define o image_id como o índice em df_data
    df_data.set_index('image_id', inplace=True)

    # Obtém uma lista de imagens em cada uma das duas pastas originais
    folder_1 = os.listdir(imagens_shen)
    folder_2 = os.listdir(imagens_mont)

    # Obtém uma lista de imagens de treino e validação
    lista_imagens_treino = list(df_treino['image_id'])
    lista_imagens_val = list(df_val['image_id'])

    ''' Pré-Processamento das Imagens '''

    # Resize das imagens
    IMAGE_HEIGHT = 96
    IMAGE_WIDTH = 96

    # Transfere as imagens de treino pré-processadas para o novo diretório

    print('\nPré-processamento dos dados de treino! Aguarde...')

    # Loop pela lista de imagens de treino
    for image in lista_imagens_treino:

        # Nome da imagem
        fname = image

        # Label da imagem
        label = df_data.loc[image, 'target']

        # Percorremos a folder_1 (imagens do dataset de shenzen) para buscar o caminho da imagem
        if fname in folder_1:
            # Diretório fonte da imagem
            src = os.path.join(imagens_shen, fname)

            # Diretório destino da imagem
            dst = os.path.join(dados_treino, label, fname)

            # Copia a imagem
            image = cv2.imread(src)

            # Aplica o redimensionamento
            image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # Salva a imagem no diretório de destino
            cv2.imwrite(dst, image)

        # Percorremos a folder_2 (imagens do dataset de montgomery) para buscar o caminho da imagem
        if fname in folder_2:
            # Diretório fonte da imagem
            src = os.path.join(imagens_mont, fname)

            # Diretório destino da imagem
            dst = os.path.join(dados_treino, label, fname)

            # Copia a imagem
            image = cv2.imread(src)

            # Aplica o redimensionamento
            image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # Salva a imagem no diretório de destino
            cv2.imwrite(dst, image)

    print('\nOs dados de treino estão prontos!')

    # Transfere as imagens de validação pré-processadas para o novo diretório

    print('\nPré-processamento dos dados de valiação/teste! Aguarde...')

    # Loop pela lista de imagens de validação/teste
    for image in lista_imagens_val:

        # Nome da imagem
        fname = image

        # Label da imagem
        label = df_data.loc[image, 'target']

        # Percorremos a folder_1 (imagens do dataset de shenzen) para buscar o caminho da imagem
        if fname in folder_1:
            # Diretório fonte da imagem
            src = os.path.join(imagens_shen, fname)

            # Diretório destino da imagem
            dst = os.path.join(dados_val, label, fname)

            # Copia a imagem
            image = cv2.imread(src)

            # Aplica o redimensionamento
            image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # Salva a imagem no diretório de destino
            cv2.imwrite(dst, image)

        # Percorremos a folder_2 (imagens do dataset de montgomery) para buscar o caminho da imagem
        if fname in folder_2:
            # Diretório fonte da imagem
            src = os.path.join(imagens_mont, fname)

            # Diretório destino da imagem
            dst = os.path.join(dados_val, label, fname)

            # Copia a imagem
            image = cv2.imread(src)

            # Aplica o redimensionamento
            image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # Salva a imagem no diretório de destino
            cv2.imwrite(dst, image)

    print('\nOs dados de validação/teste estão prontos!')

    ''' Dataset Augmentaion (Geração de Imagens Sintéticas) '''

    # Lista de classes
    class_list = ['Normal', 'Tuberculose']

    # Número de imagens sintéticas desejadas
    NUM_IMAGENS_SINTETICAS = 1000

    # Cria imagens sintéticas para aumentar o volume de dados de treino (não fazemos isso com dados de validação/teste)

    print('\nGeração de imagens sintéticas para treinamento! Aguarde...\n')

    # Loop pelas imagens de cada classe
    for item in class_list:

        # Estamos criando diretório temporário aqui porque o excluiremos posteriormente.
        # Criamos um diretório base
        aug_dir = base_dir + 'temp/'
        os.mkdir(aug_dir)

        # Criamos um diretório dentro do diretório base para armazenar imagens da mesma classe
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.mkdir(img_dir)

        # Escolhe a classe
        img_class = item

        # Listamos todas as imagens no diretório
        img_list = os.listdir(dados_treino + img_class)

        # Copiamos imagens do diretório de treino para a classe no loop, para o img_dir
        for fname in img_list:
            # Diretório fonte da imagem
            src = os.path.join(dados_treino + img_class, fname)

            # Diretório destino da imagem
            dst = os.path.join(img_dir, fname)

            # Copia a imagem da fonte para o destino
            shutil.copyfile(src, dst)

        # Apontamos para o diretório contendo as imagens que foram copiadas
        path = aug_dir
        save_path = dados_treino + img_class

        # Criamos um gerador de imagens
        datagen = ImageDataGenerator(rotation_range=10,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.1,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

        # Tamanho do batch
        batch_size = 50

        # Geração de dados
        aug_datagen = datagen.flow_from_directory(path,
                                                  save_to_dir=save_path,
                                                  save_format='png',
                                                  target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                  batch_size=batch_size)

        # Geramos as imagens aumentadas e adicionamos às pastas de treinamento
        num_files = len(os.listdir(img_dir))

        # Aqui criamos uma quantidade semelhante de imagens para cada classe
        num_batches = int(np.ceil((NUM_IMAGENS_SINTETICAS - num_files) / batch_size))

        # Executa o gerador e cria imagens aumentadas
        for i in range(0, num_batches):
            imgs, labels = next(aug_datagen)

        # Exclui o diretório temporário com os arquivos de imagem brutos
        shutil.rmtree(aug_dir)

    print('\nOs dados foram criados com sucesso!')

    ''' Construção do Modelo '''

    # Número de exemplos de treinamento
    num_amostras_treino = len(df_treino)

    # Número de exemplos de validação
    num_amostras_val = len(df_val)

    # Tamanho do batch de treino
    batch_size_treino = 10

    # Tamanho do batch de validação
    batch_size_val = 10

    # Aqui definimos o número de passos
    passos_treino = np.ceil(num_amostras_treino / batch_size_treino)
    passos_val = np.ceil(num_amostras_val / batch_size_val)

    # Aqui geramos os batches de dados
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Gera os batches de treino
    gen_treino = datagen.flow_from_directory(dados_treino,
                                             target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             batch_size=batch_size_treino,
                                             class_mode='categorical')

    # Gera os batches de validação
    gen_val = datagen.flow_from_directory(dados_val,
                                          target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                          batch_size=batch_size_val,
                                          class_mode='categorical')

    # Gera os batches de teste
    # Nota: shuffle = False faz com que o conjunto de dados de teste não seja "embaralhado"
    gen_teste = datagen.flow_from_directory(dados_val,
                                            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                            batch_size=batch_size_val,
                                            class_mode='categorical',
                                            shuffle=False)

    # Tamanho do kernel
    kernel_size = (3, 3)

    # Tamanho do Pool
    pool_size = (2, 2)

    # Número de neurônios da primeira camada
    num_neurons_1 = 32

    # Número de neurônios da primeira camada
    num_neurons_2 = 64

    # Número de neurônios da primeira camada
    num_neurons_3 = 128

    # Taxa de dropout nas camadas de convolução
    dropout_conv = 0.3

    # Taxa de dropout na camada densa
    dropout_dense = 0.3

    # Taxa de aprendizado
    taxa_aprendizado = 0.0001

    # Número de épocas de treinamento
    num_epochs = 50

    # Arquitetura do Modelo

    # Cria a sequência de camadas
    model = Sequential()

    # Adicionamos a primeira camada convolucional com 3 operações de convolução
    # Por que input_shape tem apenas 3 dimensões? Porque iremos alimentar uma imagem por vez durante o treinamento.
    model.add(Conv2D(num_neurons_1, kernel_size, activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(Conv2D(num_neurons_1, kernel_size, activation='relu'))
    model.add(Conv2D(num_neurons_1, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    # Adicionamos a segunda camada convolucional com 3 operações de convolução
    model.add(Conv2D(num_neurons_2, kernel_size, activation='relu'))
    model.add(Conv2D(num_neurons_2, kernel_size, activation='relu'))
    model.add(Conv2D(num_neurons_2, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    # Adicionamos a terceira camada convolucional com 3 operações de convolução
    model.add(Conv2D(num_neurons_3, kernel_size, activation='relu'))
    model.add(Conv2D(num_neurons_3, kernel_size, activation='relu'))
    model.add(Conv2D(num_neurons_3, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    # Camada de "achatamento"
    model.add(Flatten())

    # Camada densa com dropout
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout_dense))

    # Camada de saída
    model.add(Dense(2, activation="softmax"))

    # Sumário do modelo
    model.summary()

    # Compilação do modelo
    model.compile(Adam(taxa_aprendizado),
                  loss='binary_crossentropy',
                  metrics=['accuracy'],
                  sample_weight_mode=None)

    # Criamos um diretório para salvar o modelo treinado
    modelos_base_dir = '.'
    modelos_dir = os.path.join(modelos_base_dir, 'modelos/')

    # Define o Path
    dir_modelos = Path(modelos_dir)

    if dir_modelos.exists():
        print('O diretório já existe. Delete no SO e tente novamente.')
    else:
        os.mkdir(modelos_dir)

    # Nome completo do modelo a ser salvo
    modelo_salvo = modelos_dir + 'modelo_raiox.h5'

    # Definimos um checkpoint para verificar regularmente se a acurácia em validação melhorou
    # Se a performance melhorar em validação salvamos o modelo
    # Podemos ainda optar por salvar o modelo a cada número de épocas
    checkpoint = ModelCheckpoint(modelo_salvo,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    # Redução gradual da taxa de aprendizado (Reduce on Plateau)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                  factor=0.5,
                                  patience=2,
                                  verbose=1,
                                  mode='max',
                                  min_lr=0.00001)

    # Cria os callbacks que serão usados no treinamento
    callbacks_list = [checkpoint, reduce_lr]

    ''' Treinamento do Modelo '''

    history = model.fit(gen_treino,
                        steps_per_epoch=passos_treino,
                        validation_data=gen_val,
                        validation_steps=passos_val,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=callbacks_list)

    # Obtém os nomes das métricas do modelo
    model.metrics_names

    # Carregamos o modelo treinado
    model.load_weights('modelos/modelo_raiox.h5')

    # Extraímos as métricas de treinamento
    val_loss, val_acc = model.evaluate_generator(gen_val, steps=passos_val)

    # Imprimimos
    print('\nErro do Modelo em Validação (val_loss):', val_loss)
    print('Acurácia do Modelo em Validação (val_acc):', val_acc)

    # Extrai as métricas
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot

    plt.plot(epochs, acc, '-', label='Acurácia em Treinamento', color='blue')
    plt.title('Acurácia em Treinamento')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, '-', label='Erro em Treinamento', color='red')
    plt.title('Erro em Treinamento')
    plt.legend()
    plt.figure()

    # Plot

    plt.plot(epochs, val_acc, '-', label='Acurácia em Validação', color='green')
    plt.title('Acurácia em Validação')
    plt.legend()
    plt.figure()

    plt.plot(epochs, val_loss, '-', label='Erro em Validação', color='magenta')
    plt.title('Erro em Validação')
    plt.legend()
    plt.figure()

    # Vamos obter os labels dos dados de teste
    labels_teste = gen_teste.classes

    # Fazemos as previsões
    previsoes = model.predict_generator(gen_teste, steps=passos_val, verbose=1)

    # A função argmax() retorna o índice do valor máximo em uma linha
    matriz_conf = confusion_matrix(labels_teste, previsoes.argmax(axis=1))

    # Definimos os rótulos dos labels da classe. Eles precisam corresponder a ordem mostrada acima.
    matriz_conf_plot_labels = ['Normal', 'Tuberculose']

    # E então criamos o plot
    plot_confusion_matrix(matriz_conf, matriz_conf_plot_labels, title='Matriz de Confusão')

    # Geramos a sequência na qual o gerador processou as imagens de teste
    imagens_teste = gen_teste.filenames

    # Obtemos os rótulos verdadeiros
    y_true = gen_teste.classes

    # Obtemos os rótulos previstos
    y_pred = previsoes.argmax(axis=1)

    from sklearn.metrics import classification_report

    # Gera o relatório de classificação
    report = classification_report(y_true, y_pred, target_names=matriz_conf_plot_labels)
    print(report)


















