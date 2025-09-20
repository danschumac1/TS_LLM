from pydantic import BaseModel

class ContextAnswerBM(BaseModel):
    restate: str
    answer: str