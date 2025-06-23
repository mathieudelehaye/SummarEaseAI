# LangChain/OpenAI summarization logic
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document

def summarize_article_with_limit(article_text, max_lines=30):
    llm = OpenAI(temperature=0.2)
    prompt_template = f"""Summarize the following text in at most {max_lines} lines:\n\n{{text}}\n\nSummary:"""
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt_template)
    docs = [Document(page_content=article_text)]
    summary = chain.run(docs)
    return summary
