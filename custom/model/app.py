from mlserver import MLModel
from mlserver.codecs import decode_args
import pyjokes
from typing import List


class Joke(MLModel):
    async def load(self) -> bool:
        self.ready = True
        return self.ready

    # Logic for making predictions against our model
    @decode_args
    async def predict(self, payload: List[str]) -> List[str]:
        response = pyjokes.get_joke(language="en")
        return [response]
