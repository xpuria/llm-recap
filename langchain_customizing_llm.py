import time
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import ChatResult

class CustomeLoggedChatWrapper(BaseChatModel):
    llm: BaseChatModel
    
    @property
    def _llm_type(self) -> str:
        return "custom_logged_chat_wrapper"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        start_time = time.time()

        processed_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                processed_messages.append(
                    HumanMessage(content=f"[User Inquiry] {message.content}", additional_kwargs=message.additional_kwargs)
                )
            else:
                processed_messages.append(message)
            
        result = self.llm._generate(processed_messages, stop=stop, run_manager=run_manager, **kwargs)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Custom Wrapper: Interaction took {duration:.2f} seconds.")

        return result
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        start_time = time.time()

        processed_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                processed_messages.append(
                    HumanMessage(content=f"[User Inquiry] {message.content}", additional_kwargs=message.additional_kwargs)
                )
            else:
                processed_messages.append(message)
            
        result = await self.llm._agenerate(processed_messages, stop=stop, run_manager=run_manager, **kwargs)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Custom Wrapper (Async): Interaction took {duration:.2f} seconds.")

        return result


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    
    load_dotenv()


    underlying_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


    custom_chat_model = CustomeLoggedChatWrapper(llm=underlying_model)


    print(" Testing Synchronous invoke()")
    sync_messages = [HumanMessage(content="Tell me a one-line joke about coding.")]
    sync_response = custom_chat_model.invoke(sync_messages)
    print(f"Result: {sync_response.content}\n")


    async def test_async():
        print("Testing Asynchronous ainvoke()")
        async_messages = [HumanMessage(content="Tell me a one-line joke about data science.")]
        async_response = await custom_chat_model.ainvoke(async_messages)
        print(f"Result: {async_response.content}\n")

    asyncio.run(test_async())

    


