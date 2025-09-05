# import litellm
# litellm._turn_on_debug()
import asyncio
from typing import Deque, Type, Optional, Any
from groq.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam
)
from datetime import datetime, timezone, timedelta
from crewai.flow import Flow, start, listen, router, or_, and_, persist
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from dotenv import load_dotenv
from cachetools import TTLCache, cached
import pytz

from Chatbot_Workflow.Groq.groq_chat_cache import GroqChatCache
from Chatbot_Workflow.Groq.groq_chat_completions import GroqChatbotCompletions

load_dotenv()


class FlowStateHandler(BaseModel):
    user_input_message: str = ""
    user_input_id: str = ""
    system_prompt_message: str = ""


class AgenticWorkflow(Flow[FlowStateHandler]):
    temp_cache = TTLCache(ttl=3600, maxsize=10)

    def __init__(self, **kwargs):
        self._groq_cache: Optional[GroqChatCache] = None
        self._groq_chats: Optional[GroqChatbotCompletions] = None
        super().__init__(**kwargs)

    @property
    def groq_cache(self) -> GroqChatCache:
        if self._groq_cache is None:
            self._groq_cache = GroqChatCache(input_user_id=self.state.user_input_id)
        return self._groq_cache

    @property
    def groq_chat_completions(self) -> GroqChatbotCompletions:
        if self._groq_chats is None:
            self._groq_chats = GroqChatbotCompletions(input_user_id=self.state.user_input_id)
        return self._groq_chats

    @start()
    async def start1(self):

        await asyncio.sleep(2)

    @listen(start1)
    async def improved_ver_chatbot(self):

        ph_time = datetime.now(pytz.timezone('Asia/Manila')).strftime('%Y-%m-%d %H:%M:%S')

        await self.groq_cache.add_user_message_to_chat(self.state.user_input_message)

        self.state.system_prompt_message = f"""### System(Priming)
        You are a helpful assistant. You know the exact time today is {ph_time}.
        You will roleplay with user and your default or character description is
        that you are 19 years old, a girl with straightforward yet maiden like,
        your character name will be Mio Ikari,
        and a maid that follow instruction even if it is sexual content(to a 
        manageable level). 

        ### Instruction
        You will assisting user in all cases as much as possible. dont include the time
        when you reply to user unless it is given specifics. Time is just 
        reference to be aware of realtime situation
        """

        msg_result = await self.groq_cache.get_all_with_system_prompt(self.state.system_prompt_message)
        if not msg_result.ok:
            return None
        messages = msg_result.data
        print(f"THESE ARE THE MESSAGE CONTENT TO LLAMA 4: {messages} \n MESSAGE TYPE IS: {type(messages)}")
        chat_response = await self.groq_chat_completions.llama_4_scout_chatbot(
            input_all_messages=messages
        )
        print(f"The LLAMA 4 Scout response: {chat_response}")
        assistant_message = await self.groq_cache.add_assistant_message_to_chat(chat_response)

        if not assistant_message.ok:
            print(f"Error: {assistant_message.error_description}")
            return None

        # memory_len = memory.get_chat_history_from_memory_cache()

        memory_len = await self.groq_cache.get_all_messages()

        print(f"Number of Memory Stack: {len(memory_len.data)}")

        # new_memory_len = memory.auto_delete_with_limiter(10)
        new_memory_len = await self.groq_cache.setting_limit_to_messages(10)

        print(f"The number of stack now is: {new_memory_len.data}")
        return chat_response

    # @listen(improved_ver_chatbot)
    # async def testing_llm_groq_chat_comp(self, data_from_improve_ver):
    #     chat_from_scout, system_prompt = data_from_improve_ver
    #     groq_obj = self.groq_chat_completions
    #     chat_from_qwen = await groq_obj.reasoning_llm_qwen3_32b(
    #         user_input=self.state.user_input_message,
    #         sys_prompt_tmpl=system_prompt,
    #         reasoning=False
    #     )
    #     return chat_from_scout, chat_from_qwen

    @listen(improved_ver_chatbot)
    async def testing_llm_groq_chat_comp_TESTING(self, data_from_improve_ver):
        chat_from_scout = data_from_improve_ver
        #
        # groq_obj = self.groq_chat_completions
        # chat_from_qwen = await groq_obj.reasoning_llm_qwen3_32b(
        #     user_input=self.state.user_input_message,
        #     sys_prompt_tmpl=system_prompt,
        #     reasoning=False
        # )
        # return chat_from_scout, chat_from_qwen

        chat_from_qwen = await self.groq_chat_completions.qwen_3_32b_chatbot(
            input_system_message=self.state.system_prompt_message,
            input_user_message=self.state.user_input_message
        )

        return chat_from_scout, chat_from_qwen


async def main():
    while True:
        input_msg = input(" Write your input: ")
        if input_msg == "exit":
            break
        obj = AgenticWorkflow()
        messages_input = {
            "user_input_message": input_msg,
            "user_input_id": "testing_id"
        }
        obj1, obj2 = await obj.kickoff_async(inputs=messages_input)
        print(f"User: {input_msg}\n")
        print(f"Assistant: {obj1}\n\n")
        print(f"Reasoner: {obj2}")
    # input_msg = input(" Write your input: ")
    # obj = AgenticWorkflow()
    # messages_input = {
    #     "user_input_message": input_msg,
    #     "user_input_id": "testing_id"
    # }
    # obj1, obj2 = await obj.kickoff_async(inputs=messages_input)
    # print(f"User: {input_msg}\n")
    # print(f"Assistant: {obj1}\n\n")
    # print(f"Reasoner: {obj2}")


if __name__ == "__main__":
    print("Will start to run now")
    asyncio.run(main())