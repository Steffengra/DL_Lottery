
class Job:
    size_rb: int
    delay_steps: int

    def __init__(
            self,
            size_rb: int,
    ) -> None:
        self.total_size_rb = size_rb
        self.size_rb = size_rb
        self.delay_steps = 0
        self.priority = 0

    def set_priority(
            self,
    ) -> None:
        self.priority = 1
