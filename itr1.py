import os
import asyncio
from dotenv import load_dotenv
import anthropic
import json

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

print("Client initialized successfully.")

user_inputs = [
    "She don't like coffee.",
    "How old are you?",
    "What is your favorite color?",
    "What is your favorite food?",
    "What is your favorite movie?",
    "What do you do for a living?"
]

grammar_prompt = (
    "You are a helpful grammar analyzer expert. Your task is to analyze the grammar of the user's input and provide feedback on any errors or improvements."
    "You just give JSON output of the form: {\"errors\": [\"error1\", \"error2\"], \"improvements\": [\"improvement1\", \"improvement2\"]}. "
    "Errors are the grammatically incorrect words while Improvements are the grammatically correct alternatives."
)

vocabulary_prompt = (
    "You are a helpful vocabulary analyzer expert. Your task is to analyze the vocabulary of the user's input and provide feedback. You do not include grammatical or spelling errors and overlook them. "
    "You just give JSON output of the form: {\"words\": [\"word1\", \"word2\"], \"alternatives\": [\"alternative1\", \"alternative2\"]}. "
    "Alternatives are the words that could be used to better express the user's meaning. Do not provide any grammatical feedback."
)

async def main():
    for user_input in user_inputs:
        response = await client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            temperature=0.5,
            system=grammar_prompt,
            messages=[{"role": "user", "content": user_input}]
        )
        response = response.content[0].text
        response = json.loads(response)
        response['sentence'] = user_input
        print("Grammar Analysis:", response)

        response = await client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            temperature=0.5,
            system=vocabulary_prompt,
            messages=[{"role": "user", "content": user_input}]
        )
        response = response.content[0].text
        response = json.loads(response)
        response['sentence'] = user_input
        print("Vocabulary Analysis:", response)
        break

if __name__ == "__main__":
    asyncio.run(main())
