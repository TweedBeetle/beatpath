import functools
import pathlib
import time
from pathlib import Path
import random

from cachetools import Cache, LRUCache
from cachetools.keys import hashkey
import pickle

from prompt_toolkit.validation import Validator


def pdump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pload(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_list_validator(list, error_message=None):
    if error_message is None:
        error_message = 'the item you have selected is not in the list'

    validator = Validator.from_callable(
        lambda item: item in list,
        error_message=error_message,
        move_cursor_to_end=True
    )

    return validator


class PersistentLRUCache(LRUCache):  # @todo: this doesn't work

    def __init__(self, location, maxsize, save_interval=0.1, getsizeof=None):

        super().__init__(maxsize, getsizeof)

        self.save_interval = save_interval
        self.location = location
        if Path(location).is_file():
            # print('loading persistsent cache')
            self._Cache__data, self._Cache__currsize, self._Cache__maxsize, self._LRUCache__order = pload(self.location)

        self.last_save = time.time()

    def __setitem__(self, key, value, cache_setitem=Cache.__setitem__):
        super().__setitem__(key, value, cache_setitem)

        current_time = time.time()

        if current_time - self.last_save >= self.save_interval:

            pdump((self._Cache__data, self._Cache__currsize, self._Cache__maxsize, self._LRUCache__order), self.location)

            self.last_save = time.time()


def list_cached(cache: Cache, key=hashkey):
    def decorator(func):
        if cache is None:
            def wrapper(elems):
                return func(elems)
        else:
            def wrapper(elems):

                missed_elems = []

                for elem in elems:
                    k = key(elem)
                    if k not in cache:
                        missed_elems.append(elem)

                if missed_elems:
                    new_values = func(missed_elems)
                    for elem, new_value in zip(missed_elems, new_values):
                        try:
                            cache[key(elem)] = new_value
                        except ValueError:
                            # print('aaaa')
                            pass  # value too large

                return [cache[key(elem)] for elem in elems]

        return functools.update_wrapper(wrapper, func)

    return decorator


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def in_bounds(val, min, max):
    return max >= val >= min


def in_delta(val, target, delta):
    return target + delta >= val >= target - delta


def random_sample(pop, n):
    if n >= len(pop):
        return pop
    return random.sample(pop, n)


def execute_in_chunks(chunksize=99):
    def decorator(func):
        def wrapper(elems):
            results = []

            for chunk in chunks(elems, chunksize):
                results += func(chunk)

            return results

        return functools.update_wrapper(wrapper, func)

    return decorator
