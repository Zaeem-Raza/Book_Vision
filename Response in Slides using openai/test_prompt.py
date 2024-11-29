import openai  # type: ignore
import my_secrets


def call_openai_chat(prompt, model="gpt-3.5-turbo"):
    """Calls OpenAI's API to get a response based on the prompt."""
    openai.api_key = my_secrets.OPEN_AI_SECRET_KEY

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    response = call_openai_chat(user_prompt)
    print("GPT Response:\n", response)
