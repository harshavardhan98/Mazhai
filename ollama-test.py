import ollama

stream = ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': 'how many classes are present in zinger ecommerce code repo?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
