from typing import Optional
from pathlib import Path

class MemoryManager:
    """Handles memory storage and operations."""
    
    def __init__(self, memory_path: Optional[Path] = None):
        self.memory_path = memory_path
        self.index = 0
        self.memory: str = self._load()

    def _load(self) -> str:
        if self.memory_path and self.memory_path.exists() and (self.memory_path / f"segment_{self.index}.txt").exists():
            return (self.memory_path / f"segment_{self.index}.txt").read_text()
        return ""

    def _save(self):
        if self.memory_path:
            txt_path = self.memory_path / f"segment_{self.index}.txt"
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(self.memory)
            self.index += 1

    def overwrite(self, memory: str):
        self.memory = memory
        self._save()
        return "Memory overwritten."

    # def append(self, delta: str) -> str:
    #     """Append to memory. Returns tool result string."""
    #     if not delta:
    #         return "No content to add."
    #     self.memory = self.memory + "\n" + delta if self.memory else delta
    #     self._save()
    #     return f"Memory updated. Current memory:\n{self.memory}"

    def get(self) -> str:
        """Get current memory content."""
        return self.memory

    def reset(self):
        """Clear memory."""
        self.memory = ""
        self.index = 0
        # self._save()