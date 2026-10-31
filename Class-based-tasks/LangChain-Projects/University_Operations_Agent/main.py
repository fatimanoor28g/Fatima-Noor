
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from university_tools import (
    get_student_profile,
    get_student_marks,
    get_course_schedule,
    calculate_percentage,
    get_fee_status
)


# ---------------------------------
# 1. Load Environment Variables
# ---------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env file!")


# ---------------------------------
# 2. Model Configuration
# ---------------------------------

MODEL_CONFIG = {
    "model": "openai/gpt-oss-120b",
    "temperature": 0,
    "max_retries": 2,
    "timeout": 60
}


# ---------------------------------
# 3. Create ChatGroq Model
# ---------------------------------

def create_model():

    model = ChatGroq(
        model=MODEL_CONFIG["model"],
        temperature=MODEL_CONFIG["temperature"],
        max_retries=MODEL_CONFIG["max_retries"],
        timeout=MODEL_CONFIG["timeout"],
        api_key=api_key
    )

    return model


# ---------------------------------
# 4. Register Tools
# ---------------------------------

tools = [
    get_student_profile,
    get_student_marks,
    get_course_schedule,
    calculate_percentage,
    get_fee_status
]


# ---------------------------------
# 5. Agent Instructions
# ---------------------------------

SYSTEM_PROMPT = """
You are a University Operations Assistant.

Your job is to answer university-related and general questions.

IMPORTANT RULES:

1. GENERAL QUESTIONS:
   Answer general educational questions directly.
   Do not call university tools for general knowledge.

2. UNIVERSITY DATA:
   Always use the appropriate tool when a question
   requires student profiles, marks, schedules,
   or fee status.

3. CALCULATIONS:
   Use calculate_percentage for percentage calculations.

4. NO FABRICATION:
   Never invent student information, marks,
   course schedules, or fee status.
   Only report information returned by tools.

5. TOOL ERRORS:
   If a tool returns an ERROR message,
   explain the problem clearly to the user.
   Never replace missing information with guesses.

6. CONVERSATION MEMORY:
   Remember relevant information from earlier
   messages in the current conversation thread.
   If the information is not available in this
   thread, clearly say that you do not know it.

7. PRIVACY:
   Never assume that information from one
   conversation thread belongs to another.

   
   8. INVALID STUDENT ID:
   If the user asks for marks and provides a student ID
   but no course, call get_student_marks with the
   student ID first to verify whether the student exists.
   If the student does not exist, explain that clearly.
   If the student exists, ask which course they mean.

Always provide clear and simple answers.
"""


# ---------------------------------
# 6. Create Agent with Memory
# ---------------------------------

def create_university_agent():

    model = create_model()

    memory = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )

    return agent


# ---------------------------------
# 7. Run Agent
# ---------------------------------

def run_agent(agent, user_input, thread_id):

    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 10
    }

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },
        config=config
    )

    final_message = response["messages"][-1]

    return final_message.content


# ---------------------------------
# 8. Main Function
# ---------------------------------

def main():

    print("\n===== University Operations Agent =====")

    agent = create_university_agent()

    thread_id = input(
        "\nEnter conversation thread ID: "
    ).strip()

    if not thread_id:
        print("Thread ID cannot be empty.")
        return

    print("\nAgent is ready!")
    print("Type /thread followed by an ID to switch conversations.")
    print("Type exit to stop.\n")

    while True:

        user_input = input("You: ").strip()

        if not user_input:
            print("Please enter a question.")
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Switch conversation thread
        if user_input.startswith("/thread "):

            new_thread_id = user_input[8:].strip()

            if not new_thread_id:
                print("Please provide a valid thread ID.")
                continue

            thread_id = new_thread_id

            print(f"Switched to thread: {thread_id}")
            continue

        try:

            answer = run_agent(
                agent,
                user_input,
                thread_id
            )

            print("\nAssistant:", answer, "\n")

        except Exception as error:

            print("\nAgent Error:", str(error))
            print("Please try again.\n")


# ---------------------------------
# 9. Program Entry Point
# ---------------------------------

if __name__ == "__main__":
    main()
