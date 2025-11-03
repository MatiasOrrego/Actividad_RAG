# Mi RAG Práctico - Guía Rápida

Sistema RAG (Retrieval Augmented Generation) personalizado con experimentación de modelos, optimización de parámetros y pruebas de prompts.

---

## Inicio Rápido

### 1. Preparación

```bash
# Asegúrate de estar en el virtualenv (Python 3.11+)
source env313/Scripts/activate   # En Git Bash

# Instalar dependencias
python -m pip install -r requirements.txt

# Crear archivo .env
echo "API_KEY=tu_clave_de_google_generative_ai" > .env
```

### 2. Cambiar de Documento PDF

En el notebook `mi_rag_practico.ipynb`, busca esta línea (celda 3, "Configuración"):

```python
PDF_PATH = "mi_tp.pdf"  # ← CAMBIA ESTO
```

Reemplaza con:
```python
PDF_PATH = "tu_documento.pdf"
```

O si está en una carpeta:
```python
PDF_PATH = "./documentos/archivo.pdf"
```

### 3. Ejecutar el Notebook

```bash
jupyter notebook mi_rag_practico.ipynb
```

---

## Experimentación: Qué Cambiar

### Parámetro 1: Tamaño de Chunks (celda 6)

```python
# Pequeño (500 chars, más chunks, menos contexto)
chunks = text_splitter_small(raw_text)

# Mediano (1000 chars, equilibrio) - RECOMENDADO
chunks = text_splitter_medium(raw_text)

# Grande (2000 chars, menos chunks, más contexto)
chunks = text_splitter_large(raw_text)
```

**Efecto:**
- **Pequeño**: Búsquedas más específicas, pero menos contexto
- **Mediano**: Equilibrio entre precisión y contexto
- **Grande**: Más contexto, pero respuestas pueden ser genéricas

---

### Parámetro 2: Modelo de Embedding (celda 7)

```python
# Opción 1: nomic-embed-text (768D, rápido)
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Opción 2: mxbai-embed-large (1024D, más preciso) - RECOMENDADO
embedding = OllamaEmbeddings(model="mxbai-embed-large:latest")

# Opción 3: all-minilm (384D, ultraligero)
embedding = OllamaEmbeddings(model="all-minilm:latest")
```

**Efecto:**
- **nomic**: Rápido, buen balance
- **mxbai**: Más preciso pero lento (1024 dimensiones)
- **all-minilm**: Ultra rápido, menos precisión

---

### Parámetro 3: Template de Prompt (celda 10)

```python
# Opción 1: Simple
prompt = prompt_simple

# Opción 2: Con instrucciones (RECOMENDADO)
prompt = prompt_instructions

# Opción 3: Con fuentes y atribuciones
prompt = prompt_with_sources
```

**Efecto:**
- **Simple**: Respuestas cortas y directas
- **Con instrucciones**: Mejor adherencia al contexto
- **Con fuentes**: Indica dónde encontró la información

---

### Parámetro 4: Modelo LLM (celda 11)

En el loop interactivo (celda 12), cambia:

```python
# En response(...) cambia model_name:
for chunk in response(input_user, contexto=docs, model_name="gemini-2.0-flash"):
```

Opciones:
- `gemini-2.0-flash` (rápido, recomendado)
- `gemini-1.5-pro` (más potente)
- `gemini-1.5-flash` (alternativa)

---

### Parámetro 5: Número de Documentos Recuperados

En el loop (celda 12):

```python
docs = retrieval(input_user, k=4)  # ← Cambia 4 a más o menos
```

- `k=2`: Respuestas enfocadas, menos contexto
- `k=4`: Balance (recomendado)
- `k=8`: Muchos documentos, respuestas largas

---

## Preguntas Frecuentes

### P: ¿El PDF no se carga?
**R:** Verifica que:
1. El archivo existe en la ruta indicada
2. Está en la carpeta correcta o usa ruta absoluta
3. No está protegido/encriptado

### P: Error "No module named langchain_ollama"
**R:** Instala las dependencias:
```bash
python -m pip install -r requirements.txt
```

### P: ¿Ollama no está disponible?
**R:** Asegúrate de tener Ollama ejecutándose:
```bash
ollama serve
```
En otra terminal. O descarga desde: https://ollama.ai

### P: ¿Las respuestas no tienen contexto?
**R:** Prueba:
1. Aumentar `chunk_size` en text_splitter
2. Aumentar `k` en retrieval (más documentos)
3. Usar prompt con más instrucciones

### P: ¿Cómo cambio de tema para el prompt?
**R:** Edita el texto de instrucciones:
```python
prompt_instructions = PromptTemplate.from_template("""
Eres un asistente experto en [TU TEMA].
...
""")
```

---

## Estructura del Notebook

| Celda | Propósito |
|-------|-----------|
| 1-3 | Configuración e imports |
| 4 | Cargar PDF |
| 5-6 | Experimentar con chunks |
| 7 | Experimentar con embeddings |
| 8-9 | Base de datos vectorial |
| 10 | Experimentar con prompts |
| 11 | Experimentar con LLM |
| 12 | Loop interactivo (¡prueba aquí!) |
| 13 | Análisis y comparación |

---

## Workflow Recomendado

1. **Primera ejecución:** Corre todo con configuración por defecto
2. **Prueba el sistema:** Ejecuta celda 12 y haz preguntas
3. **Experimenta:** Cambia un parámetro a la vez
   - Celda 6: Cambia tamaño de chunks
   - Celda 7: Cambia modelo embedding
   - Celda 10: Cambia prompt
   - Celda 12: Prueba con nuevos parámetros
4. **Evalúa:** Usa celda 13 para análisis

---

## Referencias Útiles

- [LangChain Docs](https://python.langchain.com/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Google Generative AI](https://ai.google.dev/)
- [Ollama Models](https://ollama.ai/library)

---

## Tips Avanzados

### Reutilizar vector store (no volver a procesar PDF)

Si ya procesaste un PDF, puedes reutilizar su base de datos:

```python
# Solo recupera docs existentes (omite agregar_documentos)
vector_store = get_vector_store("langchain")
docs = retrieval("tu pregunta")
```

### Probar múltiples prompts rápidamente

```python
for prompt_name in ["simple", "instructions", "with_sources"]:
    prompt = eval(f"prompt_{prompt_name}")
    print(f"\n=== Probando {prompt_name} ===")
    for chunk in response(input_user, contexto=docs):
        print(chunk, end="")
```

### Medir velocidad de respuesta

```python
import time
start = time.time()
# tu código aquí
elapsed = time.time() - start
print(f"Tiempo: {elapsed:.2f}s")
```
