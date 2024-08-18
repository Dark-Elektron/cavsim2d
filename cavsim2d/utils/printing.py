
from termcolor import colored


def error(*arg):
    print(colored(f'{arg[0]}', 'red'))


def warning(*arg):
    print(colored(f'{arg[0]}', 'yellow'))


def running(*arg):
    print(colored(f'{arg[0]}', 'cyan'))


def info(*arg):
    print(colored(f'{arg[0]}', 'blue'))


def done(*arg):
    print(colored(f'{arg[0]}', 'green'))