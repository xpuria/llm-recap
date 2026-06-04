from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

class PersonInfo(BaseModel):
    name: str = Field(description="The person's name")
    age: str = Field(description="The person's age")


parser = PydanticOutputParser(pydantic_object=PersonInfo)

prompt = PromptTemplate(
    template = "Extract the relevant information from the text \n{format_instructions}\nText:{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
model = ChatOpenAI(model = "gpt-4o-mini")
input_query = "Anna is 30 years old and works as a software engineer."

formatted_input = prompt.format_prompt(query=input_query)

output_str = model.invoke(formatted_input).content

print(output_str)