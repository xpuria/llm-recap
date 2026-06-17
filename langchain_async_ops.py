"""
LangChain Async Operations

invoke() -> ainvoke()

stream() -> astream()

batch() -> abatch()

...
"""

# async def -> coroutine -> functions that can be paused and resumed
# await -> pauses the execution of the current coroutine until awaited operation completes

# asyncio.run(coroutine()) -> way to start asyncio event loop and run a top level coroutine until completion
# asyncio.gather(*coros) -> multiple coroutines

# event loop -> manage and distributes the execution of different async tasks




#without async

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

model = ChatOpenAI()

output_parser = StrOutputParser()

sync_chain = prompt | model | output_parser

#result = sync_chain.invoke({"topic": "AI"})

#print(result)


#with async

import asyncio
"""
async def run_async_chain():
    result = await sync_chain.ainvoke({"topic": "data scientist"})

    print(result)


async def main():
    await run_async_chain()


if __name__ == "__main__":
    asyncio.run(main())
"""

async def llm_calls(topic):
    result = await sync_chain.ainvoke({"topic": topic})
    print(result)
    return result

async def run_concurrent_chains():
    topics = ["AI", "data scientist", "machine learning", "deep learning"]

    tasks = [llm_calls(topic) for topic in topics] 

    results = await asyncio.gather(*tasks)

async def main():
    await run_concurrent_chains()

"""
# another possible way by using abatch

async def run_batch_chains():
    topics = [{"topic": "AI"}, {"topic": "data scientist"}, {"topic": "machine learning"}, {"topic": "deep learning"}]

    results = await sync_chain.abatch(topics)

    for result in results:
        print(result)

async def main():
    await run_batch_chains()


"""
if __name__ == "__main__":
    asyncio.run(main())

