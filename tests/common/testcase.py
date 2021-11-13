import unittest
from functools import singledispatchmethod

from tests.common.fixtures import DataFixture, ModelFixture


class KwargsHelper:
    def __init__(self, **kwargs):
        self.keys_set = []
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.keys_set.append(k)

    @property
    def kwargs(self):
        return self.as_kwargs()

    @kwargs.getter
    def kwargs(self):
        return self.as_kwargs()

    def as_kwargs(self):
        return {k: v for k, v in vars(self).items() if k in self.keys_set}


class TestCase(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    @singledispatchmethod
    def use_fixture(self, arg):
        raise NotImplementedError("Cant Use This Fixture Type")

    @use_fixture.register
    def _(self, arg: ModelFixture):
        self.model = arg.model

    @use_fixture.register
    def _(self, arg: DataFixture):
        self.transform = arg.transform
        self.train_dataset = arg.train_dataset
        self.test_dataset = arg.test_dataset
        self.train_loader = arg.train_loader
        self.test_loader = arg.test_loader
        self.x = arg.x
        self.y = arg.y

    @staticmethod
    def kwargs_helper(base=None, **kwargs):
        kwargs = {**kwargs, **{k: v for k, v in vars(base).items() if not k.startswith("_")}}
        return KwargsHelper(**kwargs)
