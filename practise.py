import os
import asyncio
from dotenv import load_dotenv
import anthropic
import json

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

print("Client initialized successfully.")

user_data = json.load(open('sample_data/user_data.json', 'r'))

sys_prompt = (
    "You are a helpful grammar expert. Your task is to help the user improve the language proficiency by designing an exercise curated to the user's needs."
    "You just give JSON output of the form: {\"questions\": [\"{\"question_type\": \"One of MCQ, Fill in the blank, etc.\", \"question_text\": \"\", \"answer\": \"\"}\"]}. "
    "Devise the exercise based on user's day to day life and other relevant information. If the user is japanese use manga examples etc."
)

async def main():
    user_info = user_data['user']
    user_data_string = f"User is {user_info['name']}, aged {user_info['age']}, a {user_info['gender']}. He resides in {user_info['location']['city']}, {user_info['location']['country']}. And has interests in {user_info['interests']}."
    response = await client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=100,
        temperature=0.5,
        system=sys_prompt,
        messages=[{"role": "user", "content": user_data_string}, {"role":"user", "content":f"The current mistakes to work on include {user_data['sentences']}. Just give JSON output no other text."}]
    )
    response = response.content[0].text
    try:
        response = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
