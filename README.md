# Classificador de Misoginia Multiclasse

App em Streamlit para classificar comentarios em portugues em categorias de misoginia usando um modelo fine-tuned baseado em BERTimbau.

## Visao geral

Este projeto recebe um comentario de texto e retorna:

- Tipo de classe prevista
- Confianca da predicao

O modelo e carregado diretamente do Hugging Face Hub:

- Repo do modelo: `adriellemarques/bertimbau-large-pt-misoginia-multiclass`

## Demo (local)

Interface simples:

- Titulo do app
- Caixa de texto para comentario
- Botao `Enviar`
- Resultado com classe e confianca

## Tecnologias

- Python 3.11+
- Streamlit
- Transformers (Hugging Face)
- PyTorch
- SentencePiece

## Estrutura do projeto

```text
.
|- streamlit_app.py
|- requirements.txt
|- README.md
```

## Como rodar localmente

1. Criar e ativar ambiente virtual (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

3. Iniciar app:

```powershell
streamlit run streamlit_app.py
```

## Deploy no Streamlit Community Cloud

Para publicar no Streamlit Cloud:

1. Suba este repositorio no GitHub.
2. No Streamlit Cloud, selecione:
	- Repository: este repo
	- Branch: `main`
	- Main file path: `streamlit_app.py`
3. Garanta que o `requirements.txt` esta na raiz.

### Observacoes importantes

- O modelo e baixado do Hugging Face na primeira execucao.
- A primeira inicializacao pode demorar mais por conta do download e cache.
- Nao versionar `.venv/` nem pesos locais grandes no GitHub.

## Modelo

O app usa o mapeamento de labels do arquivo `label_mapping.json` no repositorio do modelo.

Classes atuais:

- Ameaça Física
- Assédio Sexual
- Descrédito Intelectual
- Misoginia Geral
- Não-Misógino

## Requisitos fixados

As versoes estao fixadas em `requirements.txt` para maior reprodutibilidade no deploy.

## Possiveis melhorias

- Mostrar top-k classes com barras de probabilidade
- Adicionar historico de inferencias na sessao
- Incluir avaliacao de desempenho por classe

---

Feito com Streamlit + Hugging Face.