from abc import ABC, abstractmethod
from src.core.types import EnvStep


class VectorEnv(ABC):
    @abstractmethod
    def reset(self) -> EnvStep:
        ...

    @abstractmethod
    def step(self, actions):
        ...
