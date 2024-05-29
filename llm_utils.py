from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
import inspect
import json

import pandas as pd
import numpy as np
import re
import os
import shutil

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser, JSONAgentOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description
from langchain.globals import set_debug



class SeeWhat(BaseCallbackHandler):
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        print(f"the inputs of the chain is {inputs}\n\n")
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"the messages of the llm is {messages}\n\n")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"the prompts of the llm is {prompts}\n\n")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"the input_str of the tool is {input_str}\n\n")




def chat_with_model(prompt, input, llm,
                    temperature=1,
                    frequency_penalty=1,
                    tools=None):
    if tools:
        #准备大语言模型：这里需要 OpenAI，可以方便地按需停止推理
        llm_with_stop = llm.bind(stop=["\nObservation"], temperature=temperature, frequency_penalty=frequency_penalty)

        #处理prompt
        prompt = prompt.partial(
            tools=render_text_description(tools),
            tool_names=", ".join([t.name for t in tools]),
            )
        
        #agent_scratchpad表示思维链的记录
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_stop
            | JSONAgentOutputParser()
        )
        #构建Agent执行器：执行器负责执行Agent工作链，直至得到最终答案（的标识）并输出回答
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        output = agent_executor.invoke(input)

    else:
        chain = prompt | llm.bind(temperature=0.9) | StrOutputParser()
        output = chain.invoke(input)
    
    return output