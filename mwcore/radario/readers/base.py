from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import serial

class BaseReader(Protocol):
    def read(self) -> Any:
        ...

class SerialReader(BaseReader):
    Data_port: 'serial.Serial'
    CLI_port: 'serial.Serial'

    def read(self) -> Any: ...

    def connect(self) -> None: ...