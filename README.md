# GPT-QA-doc-bot
This project is helpful when user needs to feed long pdf files in order to ask questions about the contents of pdf.
It implements similarity search on vector databases to reduce context size and inference time. Streamlit is used for local hosting.

![image](https://github.com/AyushModi123/GPT-QA-doc-bot/assets/99743679/2f1a081f-994d-47ed-a53d-c3e10192f16d)

In above image, a research paper pdf is uploaded by the user which is converted into vector database first then based on the query, similarity search on vector indexes outputs most relevant information which is then fed into GPT model through OpenAI API as preprompt + context + query.
