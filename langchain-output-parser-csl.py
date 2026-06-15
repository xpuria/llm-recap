from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

list_parser = CommaSeparatedListOutputParser()

format_instruction = list_parser.get_format_instructions()

prompt_template = """
List 10 popular chinese food.


"""

prompt = ChatPromptTemplate.from_template(prompt_template)

llm = ChatOpenAI()


chain = prompt | llm | list_parser


result = chain.invoke({})


print(result)