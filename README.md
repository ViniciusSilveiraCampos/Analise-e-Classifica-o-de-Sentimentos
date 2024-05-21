## Análise de Sentimento 😊😠

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"/>  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black"/>  <br> <br>


Este projeto utiliza redes neurais recorrentes (RNN) para realizar a análise de sentimento em textos. A implementação é feita utilizando PyTorch e outras bibliotecas de análise de dados.



## Índice 📚

- [Descrição](#descrição-)
- [Base de Dados](#base-de-dados-)
- [Modelo de Rede Neural](#modelo-de-rede-neural-)
- [Resultados](#resultados-)
- [Contato](#contato)

## Descrição 📝

Este projeto busca analisar sentimentos em textos utilizando uma rede neural recorrente (RNN). A base de dados contém várias emoções, excluindo a emoção 'neutral', e o modelo é treinado para classificar textos em 12 categorias emocionais diferentes.

## Base de dados 💾 

<p align='center'>
<img src="https://github.com/ViniciusSilveiraCampos/Analise-e-Classifica-o-de-Sentimentos/assets/108243297/26ae4235-d167-479c-8246-b15d3af6579e" width=50%></p>


<br>

- Disponivel [AQUI!](https://drive.google.com/file/d/1t_xztBbx3vsOnEOeMGnko6ufX2I59Zlz/view?usp=drive_link)

> A base de dados é carregada e as emoções são mapeadas para valores numéricos para facilitar o treinamento do modelo. A distribuição das emoções é visualizada utilizando gráficos.

> O conjunto de dados havia uma série de mais de cinco mil dados. Com classes e comentarios que expressavam os sentimentos de 'ódio', 'raiva', 'amor', 'preocupação', 'alívio', 'felicidade', 'diversão', 'vazio', 'entusiasmo', 'tristeza', 'surpresa' e 'tédio'.

<br>

## Modelo de Rede Neural 🧠

Uma RNN simples é implementada utilizando PyTorch. O modelo é composto por uma camada de embedding, uma camada RNN e uma camada totalmente conectada.


```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output 
```

## Resultados 📈

> A precisão do modelo é avaliada utilizando a métrica de acurácia e a matriz de confusão. A rede neural alcançou um porcentual de acerto de 73%. 


<p align='center'>
<img src='https://github.com/ViniciusSilveiraCampos/Analise-e-Classifica-o-de-Sentimentos/assets/108243297/eb807bca-a3e4-4381-b22a-51242c585d85' width=50%> 
</p>


## Contato

Email: vivico2005@gmail.com

Link do Codigo: https://github.com/ViniciusSilveiraCampos/Analise-e-Classifica-o-de-Sentimentos/blob/main/Analise_Sentimento.ipynb
