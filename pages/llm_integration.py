"""LLM Integration page - Fine-tuning and RAG showcase"""

import streamlit as st


def show():
    st.title("🤖 LLM Integration & Fine-tuning")

    st.markdown("""
    Advanced LLM techniques for baseball analytics including RAG,
    fine-tuning, and prompt engineering.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "RAG Pipeline",
        "Fine-tuning",
        "LangChain Agents",
        "Live Chat Demo"
    ])

    with tab1:
        show_rag()

    with tab2:
        show_finetuning()

    with tab3:
        show_langchain()

    with tab4:
        show_chat_demo()


def show_rag():
    """RAG pipeline implementation"""
    st.header("RAG (Retrieval Augmented Generation)")

    st.markdown("""
    ### Baseball Knowledge Base with RAG

    Enhancing LLM responses with real-time baseball data and statistics.
    """)

    st.code("""
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Document Processing
def create_knowledge_base(documents: list[str]) -> Chroma:
    \"\"\"Create vector store from baseball documents.\"\"\"

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\\n\\n", "\\n", ". ", " "]
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore

# 2. Custom Prompt Template
BASEBALL_PROMPT = PromptTemplate(
    template=\"\"\"You are an expert baseball analyst with deep knowledge
    of MLB statistics, sabermetrics, and player performance.

    Use the following context to answer the question. If you don't know
    the answer, say so - don't make up information.

    Context:
    {context}

    Question: {question}

    Provide a detailed analysis with specific statistics when available.
    \"\"\",
    input_variables=["context", "question"]
)

# 3. RAG Chain
def create_rag_chain(vectorstore: Chroma) -> RetrievalQA:
    \"\"\"Create RAG chain for baseball Q&A.\"\"\"

    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.3
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": BASEBALL_PROMPT},
        return_source_documents=True
    )

    return chain

# 4. Query with Sources
def query_baseball(chain, question: str) -> dict:
    \"\"\"Query the baseball knowledge base.\"\"\"
    result = chain({"query": question})

    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }

# Usage
vectorstore = create_knowledge_base(baseball_docs)
chain = create_rag_chain(vectorstore)

response = query_baseball(
    chain,
    "What makes Shohei Ohtani's 2023 season historically significant?"
)
    """, language="python")

    st.markdown("---")

    st.markdown("### Real-time Data Integration")

    st.code("""
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent

# Custom tool for live MLB data
class MLBDataTool:
    def __init__(self, fetcher):
        self.fetcher = fetcher

    def get_player_stats(self, player_name: str) -> str:
        \"\"\"Fetch current season stats for a player.\"\"\"
        player_id = self.fetcher.get_player_id(player_name)
        if not player_id:
            return f"Player {player_name} not found"

        logs = self.fetcher.get_player_game_logs(player_id)
        # Process and format stats...
        return formatted_stats

    def get_standings(self) -> str:
        \"\"\"Get current MLB standings.\"\"\"
        standings = self.fetcher.get_standings()
        # Format standings...
        return formatted_standings

# Create tools
mlb_tool = MLBDataTool(fetcher)

tools = [
    Tool(
        name="player_stats",
        func=mlb_tool.get_player_stats,
        description="Get current season statistics for an MLB player"
    ),
    Tool(
        name="standings",
        func=mlb_tool.get_standings,
        description="Get current MLB standings by division"
    ),
    Tool(
        name="knowledge_base",
        func=lambda q: chain({"query": q})["result"],
        description="Search baseball knowledge base for historical info"
    )
]

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    """, language="python")


def show_finetuning():
    """Model fine-tuning techniques"""
    st.header("LLM Fine-tuning")

    st.markdown("""
    ### LoRA Fine-tuning for Baseball Analysis

    Parameter-efficient fine-tuning for domain-specific performance.
    """)

    st.code("""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# 1. Load base model
model_name = "mistralai/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,  # QLoRA
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. LoRA Configuration
lora_config = LoraConfig(
    r=16,                          # LoRA rank
    lora_alpha=32,                 # Alpha parameter
    target_modules=[               # Modules to apply LoRA
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 3. Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 13,631,488 || all params: 7,254,843,392 || trainable%: 0.188%

# 4. Prepare training data
def format_instruction(sample):
    return f\"\"\"### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}\"\"\"

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./mlb-analyst-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit"
)

# 6. Train
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    formatting_func=format_instruction,
    args=training_args,
    max_seq_length=2048,
    packing=True
)

trainer.train()

# 7. Save adapter
model.save_pretrained("./mlb-analyst-adapter")

# 8. Merge and save (optional)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./mlb-analyst-merged")
    """, language="python")

    st.markdown("---")

    st.markdown("### Training Data Format")

    st.code("""
# baseball_training_data.jsonl
{
    "instruction": "Analyze the player's recent performance and predict future output",
    "input": "Player: Mike Trout\\nLast 10 games: .320 AVG, 3 HR, 8 RBI\\nOpp Pitcher: 4.50 ERA",
    "output": "Based on Mike Trout's recent hot streak (.320 AVG in last 10 games)
    and the favorable matchup against a pitcher with a 4.50 ERA, I predict
    above-average performance. His 3 HRs suggest good timing at the plate.
    Expected output: 1-2 hits with a 25% chance of extra-base hit."
}
{
    "instruction": "Explain the significance of this sabermetric stat",
    "input": "Stat: wOBA of .420",
    "output": "A wOBA (weighted On-Base Average) of .420 is elite-level production.
    The league average is typically around .320, so .420 represents performance
    approximately 31% above average. This player is among the top 5% of hitters
    in the league in terms of offensive value per plate appearance."
}
    """, language="json")


def show_langchain():
    """LangChain agent implementation"""
    st.header("LangChain Agents")

    st.markdown("""
    ### Multi-tool Baseball Analyst Agent

    Autonomous agent that can fetch data, analyze stats, and provide insights.
    """)

    st.code("""
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional

# Tool schemas
class PlayerStatsInput(BaseModel):
    player_name: str = Field(description="Full name of the MLB player")
    stat_type: str = Field(description="Type of stats: 'hitting' or 'pitching'")
    games: int = Field(default=10, description="Number of recent games to analyze")

class GamePredictionInput(BaseModel):
    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    date: Optional[str] = Field(default=None, description="Game date (YYYY-MM-DD)")

# Tool implementations
def get_player_stats(player_name: str, stat_type: str, games: int = 10) -> str:
    \"\"\"Fetch and analyze player statistics.\"\"\"
    # Implementation...
    return analysis

def predict_game(home_team: str, away_team: str, date: str = None) -> str:
    \"\"\"Predict game outcome using ML model.\"\"\"
    # Implementation...
    return prediction

# Create structured tools
tools = [
    StructuredTool.from_function(
        func=get_player_stats,
        name="player_stats",
        description="Get detailed statistics and analysis for an MLB player",
        args_schema=PlayerStatsInput
    ),
    StructuredTool.from_function(
        func=predict_game,
        name="game_prediction",
        description="Predict the outcome of an MLB game",
        args_schema=GamePredictionInput
    ),
    StructuredTool.from_function(
        func=lambda q: rag_chain({"query": q})["result"],
        name="baseball_knowledge",
        description="Search historical baseball knowledge and sabermetrics info"
    )
]

# Agent prompt
AGENT_PROMPT = \"\"\"You are an expert MLB analyst assistant.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question
Thought: think about what to do
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the detailed analysis

Question: {input}
{agent_scratchpad}\"\"\"

# Create agent
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
agent = create_react_agent(llm, tools, AGENT_PROMPT)

# Memory for conversation
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

# Agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# Usage
response = agent_executor.invoke({
    "input": "Compare Shohei Ohtani and Mike Trout's performance this season"
})
    """, language="python")


def show_chat_demo():
    """Interactive chat demo"""
    st.header("Baseball AI Assistant Demo")

    st.markdown("""
    Ask questions about MLB statistics, player performance, or game predictions.
    """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your MLB Analytics AI assistant. Ask me about player stats, game predictions, or sabermetrics!"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about MLB..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response (simulated for demo)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                import time
                time.sleep(1)

                # Simulated intelligent response
                if "ohtani" in prompt.lower():
                    response = """**Shohei Ohtani Analysis**

Based on current data:
- **Batting**: .304 AVG, 44 HR, 95 RBI
- **Pitching**: 10-5 W-L, 3.14 ERA, 167 K

Ohtani continues his historic two-way performance. His wRC+ of 175 ranks among the top 5 in MLB, while his pitching WAR of 4.2 would make him an ace on most staffs.

*Data fetched from MLB Stats API and analyzed using our ML models.*"""

                elif "predict" in prompt.lower():
                    response = """**Game Prediction Model**

Our PyTorch LSTM model analyzes:
- Team recent performance (rolling 10-game stats)
- Head-to-head historical matchups
- Pitching matchup analysis
- Park factors and weather

Confidence intervals are calculated using Monte Carlo simulation with 1000 iterations.

Would you like me to predict a specific matchup?"""

                elif "war" in prompt.lower() or "sabermetric" in prompt.lower():
                    response = """**WAR (Wins Above Replacement)**

WAR measures a player's total value compared to a replacement-level player.

**Calculation components:**
- Batting Runs
- Baserunning Runs
- Fielding Runs
- Positional Adjustment
- League Adjustment
- Replacement Level Runs

**Scale:**
- 0-1: Replacement level
- 2-3: Solid starter
- 4-5: All-Star
- 6+: MVP caliber
- 8+: Historic season

Which player's WAR would you like me to analyze?"""

                else:
                    response = f"""I can help you with:

1. **Player Statistics** - Current season stats, historical performance
2. **Game Predictions** - Win probabilities using our ML model
3. **Sabermetrics** - Advanced metrics like WAR, wOBA, FIP
4. **Team Analysis** - Standings, trends, matchup analysis

What would you like to know about "{prompt}"?"""

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! How can I help you?"}
        ]
        st.rerun()
