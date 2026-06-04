import json
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')

prompt_template = """

  You are a product 
  copywriter.
  Write a short marketing 
  blurb for the following 
  product.

  Return ONLY valid JSON 
  with these keys:
  - "headline": a catchy 
  one-line headline (max 8 
  words)
  - "summary": a 2-sentence 
  description
  - "tags": a list of 3
  relevant keywords

  Product: {description}

"""

prompt = ChatPromptTemplate.from_template(prompt_template)

json_parser = SimpleJsonOutputParser()

chain = prompt | llm | json_parser

description = "A stainless-steel insulated water bottle that keeps drinks cold for 24 hours"

result = chain.invoke({"description": description})


print(result)