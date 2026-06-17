
from typing import Any, List

from langchain_core.prompts import BaseChatPromptTemplate

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage

from pydantic import Field

class DynamicStructureChatPrompt(BaseChatPromptTemplate):
    input_variables: List[str] = Field(default=["user_query"])

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        user_query = kwargs['user_query']

        messages : List[BaseMessage] = []

        if "list" in user_query.lower() or "table" in user_query.lower():
            messages.append(SystemMessage(content="Please provide the output as a structured list or table."))
        else:
            messages.append(SystemMessage(content="You are a helpful assistant"))
            
        messages.append(HumanMessage(content=user_query))

        return messages

    
    def _prompt_type(self) -> str:
        return "dynamic_structure_chat_prompt"
    

dynamic_prompt = DynamicStructureChatPrompt()

query1 = "Summarize the benefits of LCEL."
messages1 = dynamic_prompt.format_messages(user_query=query1)
print(messages1)

query2 = "Give me a list of vector stores supported by LangChain."
messages2 = dynamic_prompt.format_messages(user_query=query2)
print(messages2)