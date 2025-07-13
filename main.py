import asyncio


async def main():
    while True:
        input_msg = input(" Write your input: ")
        if input_msg == "exit":
            break
        obj = AgenticWorkflow()
        messages_input = {
            "user_input_message":input_msg,
            "user_input_id":"testing_id"
        }
        obj1, obj2 = await obj.kickoff_async(inputs=messages_input)
        print(f"User: {input_msg}\n")
        print(f"Assistant: {obj1}\n\n")
        print(f"Reasoner: {obj2}")


if __name__ == "__main__":
    asyncio.run(main())