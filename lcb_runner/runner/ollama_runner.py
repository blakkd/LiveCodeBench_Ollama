import os
from time import sleep

try:
    import ollama
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


class OllamaRunner(BaseRunner):

    def __init__(self, args, model):
        super().__init__(args, model)
        # Prioritize the command-line argument, then environment variable, then default.
        host = args.ollama_host or os.getenv("OLLAMA_HOST")
        self.client = ollama.Client(host=host)
        
        self.client_kwargs: dict[str | str] = {
            "model": args.model,
            "options": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_predict": args.max_tokens,
                "stop": args.stop,
            }
        }

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list), "Ollama runner requires chat-formatted prompt"

        def __run_single(counter):
            try:
                response = self.client.chat(
                    messages=prompt,
                    **self.client_kwargs,
                )
                content = response['message']['content']
                return content
            except Exception as e:
                print(f"Exception: {repr(e)}. Retrying... ({10 - counter} retries left)")
                sleep(20 * (11 - counter))
                counter = counter - 1
                if counter == 0:
                    print(f"Failed to run model for {prompt}!")
                    print("Exception: ", repr(e))
                    raise e
                return __run_single(counter)

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(__run_single(10))
        except Exception as e:
            raise e

        return outputs