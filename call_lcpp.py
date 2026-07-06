import requests
import json

def query_llama_cpp(prompt, server_url="http://127.0.0.1:8080/completion", n_predict=400):
    """
    Queries llama.cpp llama-server with a given prompt and returns the response as a string.

    Args:
        prompt (str): The prompt to send to the server.
        server_url (str): The URL of the llama-server.

    Returns:
        str: The generated text from the server, or None if an error occurs.
    """
    try:
        payload = {
            "prompt": prompt,
            "stream": False,  # Set to False to get the full response at once
            "n_predict": n_predict, #adjust as needed.
            # add other parameters as needed, for example:
            # "temperature": 0.8,
            # "top_p": 0.95,
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(server_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        return data["content"]

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to llama-server: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None
    except KeyError as e:
        print(f"KeyError: {e}. Check the llama-server response format.")
        return None



if __name__ == "__main__":
    prompt = "Write a short story about a talking cat."
    result = query_llama_cpp(prompt)

    if result:
        print(result)

    prompt = "What is the capital of France?"
    result = query_llama_cpp(prompt)

    if result:
        print(result)