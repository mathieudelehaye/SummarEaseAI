# Wikipedia fetching utility
import wikipediaapi

def fetch_article(topic):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(topic)
    return page.text if page.exists() else None
