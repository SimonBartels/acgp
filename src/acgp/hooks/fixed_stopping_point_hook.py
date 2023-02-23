from acgp.hooks.abstract_hook import AbstractHook


class FixedStoppingPointHook(AbstractHook):
    def __init__(self, stopping_point: int):
        super().__init__()
        self.stopping_point = stopping_point
        self.iteration = stopping_point

    def pre_chol(self, idi: int, K_, y_):
        return idi >= self.stopping_point

    def post_chol(self, idi: int, K_, y_):
        pass
