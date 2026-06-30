import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import json
from pathlib import Path
from torch.nn.functional import softmax

MODEL_PATH = "./modelo"


def load_label_mapping(model_path: str) -> dict:
	mapping_file = Path(model_path) / "label_mapping.json"
	if mapping_file.exists():
		with open(mapping_file, encoding="utf-8") as f:
			data = json.load(f)
		return {int(k): v for k, v in data["id2label"].items()}
	return {}


@st.cache_resource
def load_model():
	tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
	model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	return tokenizer, model, device


def predict(text: str, tokenizer, model, device):
	if not text:
		return None
	inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
	inputs = {k: v.to(device) for k, v in inputs.items()}
	with torch.no_grad():
		outputs = model(**inputs)
		logits = outputs.logits.squeeze(0)
		probs = softmax(logits, dim=-1).cpu().numpy()

	# load label mapping from file, fallback to model config
	id2label = load_label_mapping(MODEL_PATH)
	if not id2label:
		id2label = getattr(model.config, "id2label", {})
		id2label = {int(k): v for k, v in id2label.items()} if id2label else {i: f"LABEL_{i}" for i in range(len(probs))}

	pred_idx = int(np.argmax(probs))
	return {"label": id2label[pred_idx], "confidence": float(probs[pred_idx])}


def main():
	st.title("Classificador de misoginia multiclasse")

	st.write("Cole o comentário abaixo e clique em Enviar para classificar.")

	text = st.text_area("Digite o comentário", height=200)

	if st.button("Enviar"):
		if not text.strip():
			st.warning("Por favor digite um comentário.")
			return

		with st.spinner("Carregando modelo e classificando…"):
			try:
				tokenizer, model, device = load_model()
			except Exception as e:
				st.error(f"Erro ao carregar o modelo: {e}")
				return

			result = predict(text, tokenizer, model, device)
			if result is None:
				st.error("Nenhum resultado gerado.")
				return

		st.subheader("Resultado:")
		st.write(f"**Tipo de misoginia:** {result['label']}")
		st.write(f"**Confiança:** {result['confidence']:.1%}")


if __name__ == "__main__":
	main()

