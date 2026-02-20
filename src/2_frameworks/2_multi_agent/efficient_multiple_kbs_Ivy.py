"""Example code for planner-worker agent collaboration with multiple tools."""

import asyncio
from typing import Any, AsyncGenerator

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from langfuse import propagate_attributes

from src.utils import (
    oai_agent_stream_to_gradio_messages,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.agent_session import get_or_create_session
from src.utils.client_manager import AsyncClientManager
from src.utils.gradio import COMMON_GRADIO_CONFIG
from src.utils.langfuse.shared_client import langfuse_client
from src.utils.tools.gemini_grounding import (
    GeminiGroundingWithGoogleSearch,
    ModelSettings
)

from src.utils.tools.sql_database import (
    ReadOnlySqlDatabase
    )


import pandas as pd
import sqlite3

async def _main(
    query: str, history: list[ChatMessage], session_state: dict[str, Any]
) -> AsyncGenerator[list[ChatMessage], Any]:
    # Initialize list of chat messages for a single turn
    turn_messages: list[ChatMessage] = []

    # Construct an in-memory SQLite session for the agent to maintain
    # conversation history across multiple turns of a chat
    # This makes it possible to ask follow-up questions that refer to
    # previous turns in the conversation
    session = get_or_create_session(history, session_state)

    # Use the main agent as the entry point- not the worker agent.
    with (
        langfuse_client.start_as_current_observation(
            name="Orchestrator-Worker", as_type="agent", input=query
        ) as obs,
        propagate_attributes(
            session_id=session.session_id  # Propagate session_id to all child observations
        ),
    ):
        # Run the agent in streaming mode to get and display intermediate outputs
        result_stream = agents.Runner.run_streamed(
            EDA_Agent,
            input=query,
            session=session,
            max_turns=30,  # Increase max turns to support more complex queries
        )

        async for _item in result_stream.stream_events():
            turn_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(turn_messages) > 0:
                yield turn_messages

        obs.update(output=result_stream.final_output)


if __name__ == "__main__":
    load_dotenv(verbose=True)

    # Set logging level and suppress some noisy logs from dependencies
    set_up_logging()

    # Set up LangFuse for tracing
    setup_langfuse_tracer()

    # Initialize client manager
    # This class initializes the OpenAI and Weaviate async clients, as well as the
    # Weaviate knowledge base tool. The initialization is done once when the clients
    # are first accessed, and the clients are reused for subsequent calls.
    client_manager = AsyncClientManager()

    # Use smaller, faster model for focused search tasks
    worker_model = client_manager.configs.default_worker_model
    # Use larger, more capable model for complex planning and reasoning
    planner_model = client_manager.configs.default_planner_model

    gemini_grounding_tool = GeminiGroundingWithGoogleSearch(
        model_settings=ModelSettings(model=worker_model)
    )

    # Worker Agent: handles long context efficiently
    kb_agent = agents.Agent(
        name="KnowledgeBaseAgent",
        instructions="""
            You are an agent specialized in searching a product knowledge base.
            You will receive a single search query as input.
            Use the search_knowledgebase tool to perform the search.
            Search specifically within the product_description column to identify products that match the query.
            Return the results as:
            A LIST of objects containing in the order it appears in the dataset:
                - product_id
                - product_description
            After the list, return a separate string stating:
                - "Total count: X"
            where X is the number of matching product_ids.
            Requirements:
                -Return all matching products.
                -If no matches are found, return an empty list: []
                -Do not fabricate or infer information.
                -Do not return raw search results.
                -Do not include long quotations.
                -Only return structured, relevant results.
            If the tool returns no matches, return an empty LIST.
            Do NOT make up information. Do NOT return raw search results or long quotes.
        """,
        # instructions="""
        #     You are an agent specialized in searching a knowledge base.
        #     You will receive a single search query as input.
        #     Use the 'search_knowledgebase' tool to perform a search.
        #     Use the product_description column to find products that match the query.
        #     Then return the respective product_ids and product_decription as a LIST.
        #     Return all of the product_ids from product_description that match.
        #     Also return the final count of product_ids as a seperate string in the end.
        #     If the tool returns no matches, return an empty LIST.
        #     Do NOT make up information. Do NOT return raw search results or long quotes.
        # """,

            # LIST of product_ids that has a product_description that match the query.

        tools=[
            agents.function_tool(client_manager.knowledgebase.search_knowledgebase),
        ],
        # a faster, smaller model for quick searches
        model=agents.OpenAIChatCompletionsModel(
            model=worker_model, openai_client=client_manager.openai_client
        ),
    )

    ## EDA Agent

    transactions_data = pd.read_csv("/home/coder/agent-bootcamp/src/utils/mock_transactions.csv")
    # 2. Establish a SQLite connection
    database = "common_db"
    conn = sqlite3.connect(database)
    # 3. Save data into the SQLite database with the table name 'Users'
    transactions_data.to_sql(name='transactions_data', con=conn, if_exists='replace')


    calendar_data = pd.read_csv("/home/coder/agent-bootcamp/src/utils/mock_calendar.csv")
    # 2. Establish a SQLite connection
    database = "calendar_data"
    # conn = sqlite3.connect(database)
    # 3. Save data into the SQLite database with the table name 'Users'
    calendar_data.to_sql(name='calendar_data', con=conn, if_exists='replace')


    products_data = pd.read_csv("/home/coder/agent-bootcamp/src/utils/mock_products.csv")
    # 2. Establish a SQLite connection
    database = "products_data"
    # conn = sqlite3.connect(database)
    # 3. Save data into the SQLite database with the table name 'Users'
    products_data.to_sql(name='products_data', con=conn, if_exists='replace')


    segments_data = pd.read_csv("/home/coder/agent-bootcamp/src/utils/mock_segments.csv")
    # 2. Establish a SQLite connection
    database = "segments_data"
    # conn = sqlite3.connect(database)
    # 3. Save data into the SQLite database with the table name 'Users'
    segments_data.to_sql(name='segments_data', con=conn, if_exists='replace')


    stores_data = pd.read_csv("/home/coder/agent-bootcamp/src/utils/mock_stores.csv")
    # 2. Establish a SQLite connection
    database = "stores_data"
    # conn = sqlite3.connect(database)
    # 3. Save data into the SQLite database with the table name 'Users'
    stores_data.to_sql(name='stores_data', con=conn, if_exists='replace')

    x = ReadOnlySqlDatabase(connection_uri="sqlite:///common_db")


    # EDA Agent: handles long context efficiently
    EDA_Agent = agents.Agent(
        name="EDA_Agent",
        instructions="""

        You are a EDA agent who has access to a sql database and you should query the database to answer the user's questions.

        The SQL database has the name common_db and has the following tables. 
            - transactions_data
            - calendar_data
            - products_data
            - segments_data
            - stores_data

        

            ALWAYS FOLLOW THESE INSTRUCTIONS:
            - The sql code should not have ``` in beginning or end and sql word in output.
            - Make the query case insensitive.
            - JOIN tables ONLY when necessary.
            - If the tool returns no matches, return an empty STRING.
            - Do NOT make up information.
            
        """,
        # instructions="""
        #      You are an expert in converting English questions to BigQuery SQL query!
             
        #     The SQL database has the name transactions_data and has the following columns 
        #     - transaction_id,product_id,product_description,sales_quantity,transaction_date,unit_price,customer_id,banner_name,sales_amount,store_id
            
        #     The SQL database has the name calendar_data and has the following columns 
        #     - year,week,start_date,end_date
            
        #     The SQL database has the name products_data and has the following columns 
        #     - product_id,product_description,brand_description,brand_id,category_id,category_name,subcategory_id,subcategory_name

        #     The SQL database has the name segments_data and has the following columns 
        #     - customer_id,parent_segment,segment
            
        #     The SQL database has the name stores_data and has the following columns 
        #     - banner_name,store_id,division_name
            
        #     For example,
        #     Example 1 - How many entries of records are present?, 
        #     the SQL command will be something like this SELECT COUNT(*) FROM transactions_data ;
            
        #     Example 2 - Tell me all the transactions that of S/4 PINK FLOWER CANDLES IN BOWL?, 
        #     the SQL command will be something like this SELECT * FROM transactions_data 
        #     where product_description="S/4 PINK FLOWER CANDLES IN BOWL"; 

        #     Example 3 - Give me a breakdown of average unit price and average quantity by product description
        #     for the all hair product related transactions
        #     that occured between weeks 12-18 in year 2024
        #     The SQL command will be something like this:
        #     SELECT
        #         t.product_description,
        #         AVG(t.unit_price) AS average_unit_price,
        #         AVG(t.sales_quantity) AS average_sales_quantity
        #     FROM
        #         transactions_data AS t
        #     JOIN
        #         products_data AS p
        #     ON
        #         t.product_id = p.product_id
        #     JOIN
        #         calendar_data AS c
        #     ON
        #         t.transaction_date BETWEEN c.start_date AND c.end_date
        #     WHERE
        #         LOWER(p.product_description) LIKE "%hair%"
        #         OR LOWER (p.category_name) LIKE "%hair"
        #         OR LOWER (p.subcategory_name) LIKE "%hair"
        #         AND c.year = 2024
        #         AND c.week >= 12
        #         AND c.week <= 18
        #     GROUP BY
        #         t.product_description;         
            
        #     ALWAYS FOLLOW THESE INSTRUCTIONS:
        #     - The sql code should not have ``` in beginning or end and sql word in output.
        #     - Make the query case insensitive.
        #     - JOIN tables ONLY when necessary.
        #     - If the tool returns no matches, return an empty STRING.
        #     - Do NOT make up information.
        # """,


        # instructions="""
        #     You are an agent specialized in searching a knowledge base.
        #     You will receive a single search query as input.
        #     Use the 'search_knowledgebase' tool to perform a search.
        #     Use the product_description column to find products that match the query.
        #     Then return the respective product_ids and product_decription as a LIST.
        #     Return all of the product_ids from product_description that match.
        #     Also return the final count of product_ids as a seperate string in the end.
        #     If the tool returns no matches, return an empty LIST.
        #     Do NOT make up information. Do NOT return raw search results or long quotes.
        # """,

            # LIST of product_ids that has a product_description that match the query.
                        
        tools=[
            agents.function_tool(x.get_schema_info),
            agents.function_tool(x.execute),
        ],
        # a faster, smaller model for quick searches
        model=agents.OpenAIChatCompletionsModel(
            model=worker_model, openai_client=client_manager.openai_client
        ),
    )

            # You are a campaign planner agent and your goal is to find the best customers
            # to target with the given campaign requirements/description by breaking down complex product descriptions, 
            # using the provided tools and synthesizing the information into a list of customers_ids.

    # Main Agent: more expensive and slower, but better at complex planning
    main_agent = agents.Agent(
        name="MainAgent",
        instructions="""
            You are a retail marketing targeting orchestration agent.

            Your job is to:
            1. Interpret natural language targeting requests.
            2. Translate them into a structured targeting specification.
            3. Select and call ONLY provided tools.
            4. Produce a reproducible audience dataset containing fields: 
                - customer_id.
            5. Pass the dataset to the EDA agent when analysis is requested.

            You have access to the following tools:
            1. 'search_knowledgebase' - use this tool to search for information in a
                knowledge base. The knowledge base reflects the product descriptions.
            2. 'EDA_Agent ' - use this tool for current events,
                news, fact-checking or when the information in the knowledge base is
                not sufficient to answer the question.
        
            You must NEVER:
            - invent data
            - query undefined tables
            - assume business rules

            You are a deep research agent and your goal is to conduct in-depth, multi-turn
            research by breaking down complex queries, using the provided tools, and
            synthesizing the information into a comprehensive report.

            You have access to the following tools:
            1. 'KnowledgeBaseAgent' - use this tool to search for information in a
                knowledge base. The knowledge base reflects a subset of Wikipedia as
                of May 2025.
            2. 'get_web_search_grounded_response' - use this tool for current events,
                news, fact-checking or when the information in the knowledge base is
                not sufficient to answer the question.

            Both tools will not return raw search results or the sources themselves.
            Instead, they will return a concise summary of the key findings, along
            with the sources used to generate the summary.

            For best performance, divide complex queries into simpler sub-queries
            Before calling either tool, always explain your reasoning for doing so.

            Note that the 'get_web_search_grounded_response' tool will expand the query
            into multiple search queries and execute them. It will also return the
            queries it executed. Do not repeat them.

            **Routing Guidelines:**
            - When answering a question, you should first try to use the 'search_knowledgebase'
            tool, unless the question requires recent information after May 2025 or
            has explicit recency cues.
            - If either tool returns insufficient information for a given query, try
            reformulating or using the other tool. You can call either tool multiple
            times to get the information you need to answer the user's question.

            **Guidelines for synthesis**
            - After collecting results, write the final answer from your own synthesis.
            - Add a "Sources" section listing unique sources, formatted as:
                [1] Publisher - URL
                [2] Wikipedia: <Page Title> (Section: <section>)
            Order by first mention in your text. Every factual sentence in your final
            response must map to at least one source.
            - If web and knowledge base disagree, surface the disagreement and prefer sources
            with newer publication dates.
            - Do not invent URLs or sources.
            - If both tools fail, say so and suggest 2–3 refined queries.

            Be sure to mention the sources in your response, including the URL if available,
            and do not make up information.
        """,
        # Allow the planner agent to invoke the worker agent.
        # The long context provided to the worker agent is hidden from the main agent.
        tools=[
            kb_agent.as_tool(
                tool_name="search_knowledgebase",
                tool_description=(
                    "Search the knowledge base for a query and return a concise summary "
                    "of the key findings, along with the sources used to generate "
                    "the summary"
                ),
            ),

            EDA_Agent.as_tool(
                tool_name="EDA_Agent",
                tool_description=(
                    "Search the knowledge base for a query and return a concise summary "
                    "of the key findings, along with the sources used to generate "
                    "the summary"
                ),
            ),
            # agents.function_tool(
            #     gemini_grounding_tool.get_web_search_grounded_response,
            #     name_override="search_web",
            # ),

        ],
        # a larger, more capable model for planning and reasoning over summaries
        model=agents.OpenAIChatCompletionsModel(
            model=planner_model, openai_client=client_manager.openai_client
        ),
        # NOTE: enabling parallel tool calls here can sometimes lead to issues with
        # with invalid arguments being passed to the search agent.
        model_settings=agents.ModelSettings(parallel_tool_calls=False),
    )

    demo = gr.ChatInterface(
        _main,
        **COMMON_GRADIO_CONFIG,
        examples=[
            [
                "Write a structured report on the history of AI, covering: "
                "1) the start in the 50s, 2) the first AI winter, 3) the second AI winter, "
                "4) the modern AI boom, 5) the evolution of AI hardware, and "
                "6) the societal impacts of modern AI"
            ],
            [
                "Get a list of ALL product codes which has candle in its description"
            ],
            [
                """ Give me a breakdown of average unit price and average quantity by banner and product description 
                    for the all Vaseline brand candle transactions 
                    occuring between weeks 12-18 in year 2024 
                    from Australia?
                """
            ],

            [
                """ Give me a breakdown of average unit price and average quantity by product description 
                    for the all hair product related transactions 
                    that occured between weeks 12-18 in year 2024 
            
                """
            ],

        
           # [
           #    On average, how many customers purchase hair products and if they do, how much do they spend each month 
           # ],
            
            [
                """ On average, how many customers purchase hair products and if they do, how much do they spend each month 
                """
            ],
        ],
        title="2.2.3: Multi-Agent Orchestrator-worker for Retrieval-Augmented Generation with Multiple Tools",
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())



        