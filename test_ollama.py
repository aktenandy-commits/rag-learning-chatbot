from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2")
antwort = llm.invoke("Was ist eine Vektordatenbank? Antworte in 2 Sätzen auf Deutsch.")
print(antwort)