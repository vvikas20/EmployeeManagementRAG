# AI/ML Learning Guide for the Employee Management RAG System

This guide introduces the AI and machine-learning terminology used by this solution. It is written for readers who are new to AI/ML and connects each idea to the system's actual code.

## 1. What this system does

The application answers employee and company questions by combining three kinds of work:

1. **Conventional database lookup:** SQLite (`sqlite3`) provides structured facts such as employee profiles, leave balances, and department counts.
2. **RAG (Retrieval-Augmented Generation):** ChromaDB finds relevant pieces of uploaded policy documents; a language model uses those pieces to write an answer.
3. **Language-model tasks:** A configured model classifies requests, rewrites follow-up questions, drafts email text, and synthesizes answers.

The main path is:

```text
Question
  -> use conversation history to clarify it
  -> classify its intent
  -> retrieve SQL facts, policy passages, or both
  -> ask a language model to compose an answer when needed
  -> stream the answer to the UI
```

The code does **not** train or fine-tune a model. It calls already-trained models to perform inference. In `config.yaml`, chat generation is configurable across cloud and local providers (such as Groq for high-speed cloud inference, Google AI Studio Gemini, or local Ollama), while policy embeddings use `mxbai-embed-large`.

## 2. AI, machine learning, and language-model basics

| Term | Plain-language meaning | How it relates to this solution |
|---|---|---|
| **Artificial intelligence (AI)** | A broad name for software that performs tasks associated with human intelligence. | The application uses language models to interpret and generate language. |
| **Machine learning (ML)** | A way to build systems that learn patterns from example data rather than relying only on hand-written rules. | The language and embedding models were trained elsewhere. This application consumes them; it does not train them. |
| **Natural language processing (NLP)** | Techniques for working with human language, including classification, search, and generation. | Query routing, query rewriting, embeddings, and answer generation are NLP tasks. |
| **Generative AI (GenAI)** | AI that creates new content, such as text, code, or images. | The LLM generates policy answers and email drafts. |
| **Model** | A trained mathematical system that maps inputs to outputs. | Chat models produce text; the embedding model produces numeric vectors. |
| **Training** | The process of adjusting a model using training examples. | Not performed by this project. |
| **Fine-tuning** | Further training an existing model on a narrower dataset or task. | Not performed by this project. |
| **Inference** | Using a trained model to produce an output for a new input. | This is what happens when the app sends a prompt to a configured model. |
| **Provider** | The service or runtime that hosts a model. | Ollama can run models locally; the hosted profile sends chat requests to the configured Gemini provider. |
| **LLM (large language model)** | A model trained to understand and generate text, commonly by predicting likely next tokens. | Used for query rewriting, routing, response writing, and email drafting. |
| **Embedding model** | A model that maps text to a vector designed to represent its meaning. | Ollama's `mxbai-embed-large` maps policy chunks and user questions to vectors. |

### Common ML learning approaches

These are general ML terms, not training procedures implemented in this application:

- **Supervised learning:** learns from examples paired with known answers or labels.
- **Unsupervised learning:** finds structure in data without supplied labels.
- **Self-supervised learning:** creates learning signals from the data itself; commonly used to pre-train language models.
- **Reinforcement learning:** learns behavior from actions and rewards.

The solution uses pre-trained models regardless of which learning approaches were used to create them.

### Common model architecture families

- **Encoder models:** turn input into contextual representations; commonly used for classification or embeddings.
- **Decoder-only models:** generate text one token at a time; commonly used as chat/generation LLMs.
- **Encoder-decoder models:** encode an input and generate a transformed output; often used for translation or summarization.

These are useful model-family distinctions. The application selects models by configured provider/model name; it does not implement or alter their internal architecture.

## 3. Prompts, tokens, and generation settings

### Prompt and prompt roles

A **prompt** is the input given to a language model. In chat APIs, messages commonly have roles:

- **System message:** instructions about the model's role and constraints.
- **User message:** the request or task.
- **Assistant message:** a generated response.
- **Tool message:** the result returned by an external function/tool call.

For example, `LiteLLMClient.generate_completion` sends a system instruction and a user prompt:

```python
messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]
```

### Tokens and context window

- A **token** is a piece of text used by a language model. A token may be a whole word, part of a word, punctuation, or whitespace; it is not the same as a character or a word.
- The **context window** is the maximum amount of input and generated text a model can process in one request.
- A **tokenizer** converts text into tokens and back.
- **Token budget** is the portion of the context window available for prompts, retrieved context, conversation history, and output.

`MemoryAgent` uses `tiktoken` to estimate history length and tries to keep its context string near 1,000 tokens. The policy chunk setting of `500` is different: the current chunker measures **Python characters**, not model tokens.

### Generation controls

- **Temperature** controls variation in token selection. Lower settings tend to be more repeatable; higher settings allow more variation. It is not a direct accuracy setting.
- **Maximum output tokens** limits how much text the model may generate; it does not enlarge the context window.
- **Structured output** asks for a defined shape, often JSON. The router and email agent request JSON so the application can read fields such as `intent`, `subject`, and `body`.
- **Streaming** returns generated text incrementally rather than waiting for the entire answer. The application forwards answer events using Server-Sent Events (SSE).
- **Model alias/profile** is a configuration-level name or group that lets the application choose a model without hard-coding every call site. This project has `default-llm` and `fast-llm` aliases and profiles such as `ollama` and `hosted`.

Example configuration excerpt (the actual model and values are read from `config.yaml`):

```yaml
llm:
  active_profile: hosted
  profiles:
    ollama:
      default-llm:
        model: ollama/llama3.1
        temperature: 0.2
        max_tokens: 2048
```

**LiteLLM** is a model-provider abstraction in this project: its client lets application code make completion, streaming, and tool-call requests through one interface.

## 4. Documents and chunking

### Why split a document?

A policy document may be too large to embed or include in a model prompt as one piece. **Chunking** splits it into smaller passages. Chunks should be small enough to retrieve and fit into a prompt, but large enough to keep their meaning and surrounding details.

### Common chunking strategies

| Strategy | How it works | Typical trade-off |
|---|---|---|
| **Fixed-size character chunks** | Split every N characters. | Simple, but can break words, sentences, or sections. |
| **Fixed-size token chunks** | Split by token count. | Aligns better with model limits, but requires tokenization and may still split ideas. |
| **Sentence-based** | Group complete sentences until a size limit. | Preserves sentence boundaries; a sentence alone may be too long or lack context. |
| **Paragraph-based** | Keep paragraphs together and combine nearby paragraphs. | Often preserves local meaning; paragraph lengths can vary widely. |
| **Recursive separator-based** | Try larger structural separators first (for example, headings, paragraphs, lines, then sentences/words) and split more finely only when necessary. | Good general-purpose compromise; depends on separator choices. |
| **Structure-aware** | Use document structure such as Markdown headings, HTML sections, or PDF layout. | Can preserve section context; depends on reliable structure extraction. |
| **Semantic chunking** | Split where meaning changes, often using embedding similarity between adjacent text spans. | Can produce coherent chunks, but adds computation and design complexity. |
| **Sliding-window / overlapping chunks** | Move a fixed-size window across text and repeat some text between adjacent chunks. | Reduces lost context at boundaries, but duplicates content and increases storage. |

Strategies can be combined. For example, a splitter can use headings and paragraphs first, then use sentences for an oversized paragraph.

### The chunking strategy used here

`VectorDBManager.chunk_text` combines paragraph grouping with sentence splitting for oversized paragraphs:

```python
paragraphs = re.split(r'\n\s*\n', text)
...
if len(current_chunk) + len(p) <= self.chunk_size:
    current_chunk += ("\n\n" + p if current_chunk else p)
...
if len(p) > self.chunk_size:
    sentences = re.split(r'(?<=[.!?])\s+', p)
```

Important implementation details:

- The configured default is `chunk_size: 500`, and this value is counted in **characters**.
- The code comment says “Recursive chunking by headers, paragraphs, and sentences,” but the implementation does not split on Markdown headers as a separate level. It groups paragraphs and splits oversized paragraphs at sentence boundaries.
- `chunk_overlap: 50` is read from configuration, but `chunk_text` does not apply overlap. Chunks currently do **not** intentionally repeat 50 characters.
- When one sentence is itself longer than the limit, the current sentence-splitting loop does not further divide it.

This distinction matters when reading architecture notes or tuning configuration: a setting has no effect unless the implementation uses it.

## 5. Embeddings, vector search, and retrieval

### Embedding and vector

An **embedding** is a list of numbers (a **vector**) produced from text. Texts with related meanings are intended to be close together in the embedding model's vector space. The numbers are not a human-readable summary; they are useful for comparing meaning.

Both the policy passages and the user's question must be embedded in a compatible vector space. This is why this application calls the same configured embedding model for document chunks and searches.

```python
resp = requests.post(
    f"{self.ollama_base}/api/embeddings",
    json={"model": self.embed_model, "prompt": text},
    timeout=10
)
```

### Vector database and index

- A **vector database** stores vectors and associated data and supports similarity search. This solution uses persistent **ChromaDB**.
- A **collection** is a named group of stored items. The configured collection is `company_policies`.
- **Metadata** is descriptive information stored alongside a chunk, such as its document name, category, and chunk index.
- A **vector index** organizes vectors to make nearest-neighbor search faster. ChromaDB's HNSW index is an approximate-nearest-neighbor approach: it trades exactness for efficient search at scale.

### Similarity and distance

Search ranks stored vectors by how close they are to the query vector. Common comparison measures include:

- **Cosine similarity/distance:** compares the angle between vectors; commonly used for text embeddings.
- **Dot product / inner product:** compares vector alignment and magnitude (or just alignment when vectors are normalized).
- **Euclidean (L2) distance:** measures straight-line distance between vectors.

This solution configures ChromaDB with cosine space:

```python
metadata={"hnsw:space": "cosine"}
```

Chroma returns a cosine **distance**; the app presents `1 - distance` as a similarity-like score. The score is useful for ranking, not a calibrated probability that the answer is correct.

### Retrieval types

| Retrieval type | How it finds candidate text | Example / status |
|---|---|---|
| **Sparse / lexical retrieval** | Matches words or terms; BM25 is a common ranking algorithm. | Useful for exact names, IDs, and phrases. Not the policy vector lookup implemented here. |
| **Dense / semantic retrieval** | Compares query and document embeddings. | Used by this solution's policy search. |
| **Hybrid retrieval** | Combines sparse and dense results. | A common alternative, but not implemented in `VectorDBManager.search_policy`. |
| **Metadata-filtered retrieval** | Limits search by fields such as category or document. | Metadata is stored by this solution; the current policy search does not pass a metadata filter. |
| **Reranking** | Re-scores an initial candidate set with a separate model or stronger scoring step. | A possible improvement; not implemented in the current policy path. |

### Top-k, ranking, and relevance

- **Top-k** means return the `k` highest-ranked results. Policy-agent retrieval defaults to 4; the orchestrator can request 3, and the MCP policy tool defaults to 3.
- **Candidate set** means the chunks returned by the first retrieval stage.
- **Relevance** means how useful a result is for the question. Similarity is only a proxy for relevance; a highly similar chunk can still omit the needed fact.
- A **similarity threshold** is a minimum score below which a result is rejected. The current search returns its top-k results without applying such a threshold.

## 6. RAG: Retrieval-Augmented Generation

**RAG** combines search with text generation:

1. **Ingest:** read a policy file and extract text (PDF text extraction is supported).
2. **Chunk:** split the extracted text.
3. **Embed and index:** embed each chunk and store its vector, text, and metadata in ChromaDB.
4. **Retrieve:** embed a user's policy question and find the nearest chunks.
5. **Augment:** place the retrieved text into the generation prompt as context.
6. **Generate:** ask the LLM to answer from that context.

The policy agent builds a context block from retrieved passages and instructs the LLM to use only that context:

```python
context_text = "\n\n".join([
    f"--- Chunk (Similarity: {c['similarity']}) ---\n{c['text']}"
    for c in retrieved_chunks
])
```

Key RAG terms:

- **Corpus / knowledge base:** the set of documents available for retrieval. Here it is the ingested company-policy content.
- **Grounding:** giving the model trusted facts or source text to base its answer on.
- **Hallucination:** plausible-sounding generated content that is unsupported or incorrect. RAG prompts ask the model to say when the context does not contain an answer, but prompts cannot guarantee zero hallucinations.
- **Source attribution:** identifying where an answer's supporting information came from. The policy agent returns document names as sources; it does not currently cite page numbers or exact chunk locations in the answer.
- **Retrieval-augmented context:** the selected chunk text added to the LLM prompt. It is separate from the model's original training data.
- **Query transformation:** rewriting or enriching a search query before retrieval. This application rewrites conversational questions to stand alone before routing; this is not a specialized multi-query search step.

RAG is particularly useful when facts come from private, changing, or domain-specific documents that the model was not trained on. It does not retrain the model on those documents.

## 7. The SQL and hybrid-data paths

Not every question should use vector search:

- **Structured data** is organized into fields and tables, such as employee number, department, and leave balance. SQL is a natural way to query it.
- **Unstructured data** is free-form text such as policy documents. Semantic retrieval is useful for finding relevant passages.
- **Hybrid question** requires both sources, such as asking which policy entitlement applies to one employee's current balance.

This project has two database interaction patterns:

1. **Parameterized tools:** the LLM chooses from predefined functions such as retrieving a profile or leave balance. The application validates arguments and executes SQL. This narrows the model's actions to known operations.
2. **Text-to-SQL:** an LLM translates a natural-language analytics request into T-SQL. The application parses the query with `sqlglot`, permits a `SELECT`, applies a row limit if missing, and then executes it. This is more flexible than fixed tools and requires careful validation.

The difference:

- A normal **SQL query** retrieves exact structured records using database logic.
- A **RAG query** finds relevant unstructured passages by meaning.
- A **hybrid answer** may combine both results and ask an LLM to phrase them together.

## 8. Agents, routing, and tool calling

- An **agent** is a software component that uses a model and/or tools to pursue a task. In this project, “agent” refers to application components with distinct responsibilities; it does not mean an independently thinking person.
- **Orchestration** coordinates those components and decides the order of work.
- A **router / supervisor** classifies a request and selects a processing path. Here, intents include company statistics, employee SQL, policy questions, hybrid employee-policy questions, custom SQL, and email actions.
- **Intent classification** assigns a request to a category, often represented by a label such as `POLICY_QUERY`.
- **Entity extraction** identifies details such as an employee number (`2212`) from a request.
- A **tool call / function call** is a model request for the application to run a named function with arguments. The model does not directly execute SQL; the application dispatches the call to a pre-defined handler.
- A **tool schema** describes a tool's name, arguments, and expected structure so a compatible model can select and call it.
- A **tool loop** can send the tool result back to the model so it can decide whether it needs another tool or can answer.
- **MCP (Model Context Protocol)** is an interoperability protocol that exposes tools and context to compatible AI clients. It is not a language model or a retrieval algorithm. This project also exposes employee/policy tools through a local MCP server.

The request router uses a fast model for classification, while some answer-generation tasks use the default model. The names “fast” and “default” are configuration aliases, not universal model categories.

## 9. Conversation memory and query rewriting

People often use follow-ups such as “What about her manager?” without repeating the employee name. The system needs context to interpret such a question.

Memory approaches commonly include:

- **Buffer memory:** keep the full conversation.
- **Window memory:** keep only the most recent turns.
- **Summary memory:** summarize older turns to save tokens.
- **Entity memory:** retain key facts or entities such as the last employee mentioned.

This application combines a database-backed chat history with a small in-memory session state, keeps recent history, summarizes older history after a threshold, and tracks fields such as the last employee code and intent. The memory agent asks a fast LLM to rewrite the latest query as a standalone question and can request clarification when it detects ambiguity.

Example:

```text
Earlier: Show employee 2212's leave balance.
Follow-up: Who is his manager?
Rewritten search/request: Who is the manager of employee 2212?
```

**Pronoun resolution** is identifying what words such as “he,” “she,” or “their” refer to. Here, simple rules add hints for the last employee code, and an LLM receives the chat context to rewrite the question.

Memory is not the same as model training: retaining a chat history makes it available as prompt context; it does not update the model's learned weights.

## 10. Output quality and evaluation terminology

These terms help assess a RAG application even though the code does not currently implement a full evaluation pipeline:

- **Retrieval precision:** of the chunks returned, how many are relevant?
- **Retrieval recall:** of all relevant chunks available, how many were retrieved?
- **Hit rate / recall@k:** how often at least one relevant chunk appears in the top `k` results.
- **Faithfulness / groundedness:** whether generated claims are supported by the retrieved context.
- **Answer relevance:** whether the response actually answers the user's question.
- **Latency:** time from request to response. Streaming can show output sooner, even if total generation time is similar.
- **Evaluation set:** a collection of representative questions with expected answers or supporting sources, used to measure changes consistently.

Improving `top_k`, chunk size, or a prompt should be measured against a representative set rather than judged from one example.

## 11. Important implementation caveats

1. **The configured chunk overlap is not applied.** `chunk_overlap` is loaded but never used by the current `chunk_text` method.
2. **Chunk length means characters here.** It is not a tokenizer-aware token limit.
3. **Similarity is not certainty.** A score is a ranking signal and the search currently has no minimum score cutoff.
4. **A fallback embedding is not semantic.** If the Ollama embedding request fails, `generate_embedding` returns a deterministic dummy vector. Such vectors do not encode text meaning; relying on them can produce misleading search results. Ingestion and query embedding should both use the real embedding model for meaningful retrieval.
5. **Hosted chat and local embeddings are separate paths.** The active chat profile can be hosted while policy embeddings still call local Ollama.
6. **The application uses model prompts as instructions, not guarantees.** Validation, access controls, and deterministic code are still necessary for correctness and safety.

## 12. Quick glossary

| Term | Short definition |
|---|---|
| **API** | A software interface used by one component to request work from another. |
| **ChromaDB** | The vector-store database used for policy passages. |
| **Embedding space** | The numeric space in which an embedding model places text vectors. |
| **HNSW** | A graph-based approximate-nearest-neighbor indexing method. |
| **JSON** | A structured text format used for model and API data. |
| **Nearest neighbor** | A stored vector ranked close to a query vector. |
| **Ollama** | A runtime/API used here for local model inference and embeddings. |
| **PDF parsing** | Extracting text from a PDF before ingestion; scanned-image PDFs may require OCR, which this code does not add. |
| **Persistent storage** | Data saved so it remains available after the process ends; ChromaDB is configured as persistent. |
| **SQLite** | The embedded relational database (`./data/employee_management.db`) used for employee records, departments, and leave balances. |
| **SSE** | Server-Sent Events, a web mechanism used here to stream response events to the browser. |
| **Top-k** | The number of highest-ranked search results returned. |
| **Vector** | An ordered list of numbers; embeddings are vectors. |

## 13. Where to explore these concepts in the code

- [`database.py`](./database.py): SQLite schema, auto-seeding of 28 employees, safe schema introspection, and chat session logging.
- [`vector_db.py`](./vector_db.py): chunking, embeddings, ChromaDB indexing, cosine search, and top-k retrieval.
- [`agents/policy_agent.py`](./agents/policy_agent.py): RAG context construction and answer generation.
- [`agents/memory_agent.py`](./agents/memory_agent.py): conversation history, token counting, summarization, and query rewriting.
- [`agents/router_agent.py`](./agents/router_agent.py): intent classification and employee-code extraction.
- [`agents/tool_agent.py`](./agents/tool_agent.py): model-selected tool calls and the tool execution loop.
- [`agents/sql_tools.py`](./agents/sql_tools.py): parameterized SQL tool functions.
- [`agents/text_to_sql_agent.py`](./agents/text_to_sql_agent.py): natural-language-to-SQL generation and query validation.
- [`agents/orchestrator.py`](./agents/orchestrator.py): coordination of memory, routing, retrieval, tools, and response generation.
- [`litellm_client.py`](./litellm_client.py): provider abstraction, prompts, generation, streaming, structured output, and model tool calls.
- [`mcp_server.py`](./mcp_server.py): Model Context Protocol server exposing tool capabilities over stdio.
- [`config.yaml`](./config.yaml): model profiles, model parameters, chunk settings, and feature flags.
- [`main.py`](./main.py): API entry points for policy upload, query handling, and streaming.
