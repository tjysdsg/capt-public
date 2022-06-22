from enum import IntEnum, unique

__all__ = ['ErrorCode']


@unique
class ErrorCode(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    PERMISSION_DENIED = 2
    SERVER_ERROR = 3
