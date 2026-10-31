
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

MODEL_CONFIG = {
    "model": "openai/gpt-oss-120b",
    "temperature": 0,
    "timeout": 60,
    "max_retries": 2
}

VALID_LEVELS = ["beginner", "intermediate", "advanced"]

VALID_MODES = ["explanation", "revision notes", "quiz"]


# ==================================================
# 2. QUIZ SCHEMA
# ==================================================

class QuizQuestion(BaseModel):

    question: str = Field(
        description="Quiz question"
    )

    choices: list[str] = Field(
        min_length=4,
        max_length=4,
        description="Exactly four choices"
    )

    correct_answer: str = Field(
        description="Correct answer from the choices"
    )

    explanation: str = Field(
        description="Explanation of the correct answer"
    )


class QuizOutput(BaseModel):

    questions: list[QuizQuestion] = Field(
        min_length=1
    )


# ==================================================
# 3. CREATE MODEL
# ==================================================

def create_model():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    model = ChatGroq(
        api_key=api_key,
        **MODEL_CONFIG
    )

    return model



# ==================================================
# 4. CREATE PROMPT TEMPLATE
# ==================================================

def create_prompt():

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            You are an Adaptive AI Learning Assistant.

            Teach the student according to their level.

            Student Level: {level}
            Learning Goal: {goal}
            Output Mode: {mode}

            IMPORTANT:
            Generate ONLY the selected output mode.
            Never generate all three modes together.

            Follow these instructions:

            1. If Output Mode is "explanation":
               - Explain the topic step by step.
               - Use simple language according to student level.
               - Include relevant examples.
               - Do NOT include revision notes or quiz.

            2. If Output Mode is "revision notes":
               - Provide short and organized revision notes.
               - Include definitions and key points.
               - Do NOT include a full explanation or quiz.

            3. If Output Mode is "quiz":
               - Create educational quiz questions.
               - Each question must have exactly four choices.
               - Provide the correct answer.
               - Include an explanation of the correct answer.
               - Do NOT include revision notes or a separate explanation.

            Always follow the selected output mode.
            """

        ),

        (
            "human",
            "Teach me about this topic: {topic}"
        )

    ])

    return prompt


# ==================================================
# 5. CREATE RUNNABLE PIPELINES
# ==================================================

def create_pipelines(model, prompt):

    # Normal text pipeline
    text_pipeline = (
        prompt
        | model
        | StrOutputParser()
    )

    # Structured quiz pipeline
    quiz_model = model.with_structured_output(QuizOutput)

    quiz_pipeline = (
        prompt
        | quiz_model
    )

    return text_pipeline, quiz_pipeline


# ==================================================
# 6. VALIDATE INPUT
# ==================================================

def validate_input(topic, level, goal, mode):

    if not topic.strip():
        raise ValueError("Topic cannot be empty!")

    if level.lower() not in VALID_LEVELS:
        raise ValueError("Invalid student level!")

    if not goal.strip():
        raise ValueError("Learning goal cannot be empty!")

    if mode.lower() not in VALID_MODES:
        raise ValueError("Invalid output mode!")


# ==================================================
# 7. GET STUDENT INPUT
# ==================================================

def get_student_input():

    print("\n===== Adaptive AI Learning Assistant =====")

    topic = input("Enter topic: ").strip()

    level = input(
        "Enter level (beginner/intermediate/advanced): "
    ).strip().lower()

    goal = input("Enter learning goal: ").strip()

    print("\n1. Explanation")
    print("2. Revision Notes")
    print("3. Quiz")

    choice = input("Choose output mode: ").strip()

    mode_map = {
        "1": "explanation",
        "2": "revision notes",
        "3": "quiz"
    }

    if choice not in mode_map:
        raise ValueError("Invalid mode selection!")

    mode = mode_map[choice]

    return topic, level, goal, mode


# ==================================================
# 8. PREPARE RUNTIME VARIABLES
# ==================================================

def prepare_input(topic, level, goal, mode):

    return {
        "topic": topic,
        "level": level,
        "goal": goal,
        "mode": mode
    }


# ==================================================
# 9. GENERATE EXPLANATION WITH STREAMING
# ==================================================

def stream_explanation(model, prompt, values):

    print("\n===== Explanation =====\n")

    pipeline = prompt | model

    for chunk in pipeline.stream(values):

        if isinstance(chunk.content, str):
            print(chunk.content, end="", flush=True)
        else:
            print(chunk.content, end="", flush=True)

    print("\n")


# ==================================================
# 10. GENERATE TEXT OUTPUT
# ==================================================

def generate_text(text_pipeline, values):

    response = text_pipeline.invoke(values)

    print("\n===== AI Response =====\n")
    print(response)

    return response


# ==================================================
# 11. GENERATE QUIZ
# ==================================================

def generate_quiz(quiz_pipeline, values):

    result = quiz_pipeline.invoke(values)

    print("\n===== AI Quiz =====\n")

    for index, question in enumerate(result.questions, start=1):

        if question.correct_answer not in question.choices:
            raise ValueError(
                "Correct answer is not present in choices!"
            )

        print(f"Question {index}: {question.question}")

        for option_index, choice in enumerate(
            question.choices, start=1
        ):
            print(f"{option_index}. {choice}")

        print("Correct Answer:", question.correct_answer)
        print("Explanation:", question.explanation)
        print()

    return result


# ==================================================
# 12. MAIN APPLICATION
# ==================================================

def main():

    try:

        # Create AI model
        model = create_model()

        # Create prompt
        prompt = create_prompt()

        # Create pipelines
        text_pipeline, quiz_pipeline = create_pipelines(
            model,
            prompt
        )

        # Get student input
        topic, level, goal, mode = get_student_input()

        # Validate before model call
        validate_input(topic, level, goal, mode)

        # Prepare runtime variables
        values = prepare_input(
            topic,
            level,
            goal,
            mode
        )

        # Select output mode
        if mode == "explanation":

            stream_explanation(
                model,
                prompt,
                values
            )

        elif mode == "quiz":

            generate_quiz(
                quiz_pipeline,
                values
            )

        else:

            generate_text(
                text_pipeline,
                values
            )

    except ValueError as error:

        print("\nInput Error:", error)

    except Exception as error:

        print("\nApplication Error:", error)


# ==================================================
# 13. APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
