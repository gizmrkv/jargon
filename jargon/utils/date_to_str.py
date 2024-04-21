import datetime


def date_to_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")
