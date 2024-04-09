from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()

handler = ChatModelStartHandler()
chat = ChatOpenAI(
    callbacks=[handler],
)

tables = list_tables()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are an AI that has access to SQLite database.\n"
                f"The database has the following tables: {tables}\n"
                "Do not make assumptions on what table or what column exists.\n"
                "Instead, use the describe_tables function."
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)

# `return_messages` is set to True to return the list of messages as objects instead of strings.
# Message as strings for completion based models.
# Messages as objects for message based models.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [run_query_tool, describe_tables_tool, write_report_tool]

agent = create_openai_functions_agent(
    llm=chat,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True,
    tools=tools,
    memory=memory,
)

agent_executor.invoke(
    {"input": "How many orders are there. Write the results to a report file."}
)

agent_executor.invoke({"input": "Repeat the exact same process for users."})
