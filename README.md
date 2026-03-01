# Projeto de Recomendação com Aprendizado de Máquina

Este projeto tem como objetivo guiar os alunos através da construção de um sistema de recomendação simples, começando pelos fundamentos das redes neurais com o **Perceptron**.

## 🎯 Objetivo

Criar um ambiente prático e incremental para entender como os algoritmos de classificação funcionam "por baixo do capô". Iniciando com uma implementação manual do Perceptron para classificar músicas (ex: "Festa" vs "Relax") com base em suas características de áudio.

## 📂 Estrutura do Projeto

O projeto está organizado da seguinte forma:

- **`src/`**: Código fonte da aplicação.
  - **`models/`**: Implementações dos algoritmos (ex: `perceptron.py`).
- **`notebooks/`**: Jupyter Notebooks para aulas práticas e experimentação.
- **`data/`**: Conjunto de dados utilizado (baseado em atributos do Spotify).

## 🚀 Como Executar

### Pré-requisitos

1.  Ter o **Python** instalado.
2.  Ter o **uv** instalado (recomendado para gerenciamento de dependências).

### Instalação

1.  Clone este repositório (ou copie a pasta).
2.  Instale as dependências com o `uv`:

```bash
uv sync
```

### 3. Rodando a API

Para rodar o servidor de desenvolvimento (com hot-reload):
**Windows (PowerShell):**
```powershell
uv run fastapi dev
```

### 4. Testando a API

A documentação interativa (Swagger UI) está disponível em:
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)


### Rodando o Notebook

Para iniciar as aulas práticas, execute o Jupyter Notebook:


E abra o arquivo `notebooks/01_intro_dados.ipynb`.


